import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt 
import os
from sklearn.model_selection import train_test_split
import optuna
plt.ioff()  

#torch.manual_seed(0)
#np.random.seed(0)

# Function to load pickled data
def pickle_loader(file_path):
    return pickle.load(open(file_path, "rb"))

# Function to reset the weights of the model
def weight_reset(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.reset_parameters()
        
# Existing code for dataset and model definition
class TimeseriesDatasetOptimized(Dataset):
    def __init__(self, ghg_df, aer_df, nat_df, hist_df):
        self.ghg_data = ghg_df.drop(columns=["model"]).values
        self.aer_data = aer_df.drop(columns=["model"]).values
        self.nat_data = nat_df.drop(columns=["model"]).values
        self.hist_data = hist_df.drop(columns=["model"]).values

    def __len__(self):
        return len(self.ghg_data)

    def __getitem__(self, idx):
        X = torch.tensor(np.array([self.ghg_data[idx], self.aer_data[idx], self.nat_data[idx]]), dtype=torch.float32)
        Y = torch.tensor(self.hist_data[idx], dtype=torch.float32)
        return X, Y
    
# Class definition for the UNet model
class UNet1DWithCrop(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1DWithCrop, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)  
        
        # Decoder (upsampling)
        self.up4 = self.up_block(512, 256)   
        self.up3 = self.up_block(256, 128)
        self.up2 = self.up_block(128, 64)
        self.up1 = self.up_block(64, out_channels)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3) 
        
        # Decoder with cropping for skip connections
        d4 = self.up4(e4)
        d3 = self.up3(self.crop_and_add(d4, e3))
        d2 = self.up2(self.crop_and_add(d3, e2))
        d1 = self.up1(self.crop_and_add(d2, e1))
        
        return d1
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),  
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
    
    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),  
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    
    def crop_and_add(self, x1, x2):
        """
        Crop x2 to the size of x1 and add them.
        """
        diff = x2.size()[2] - x1.size()[2]
        x2 = x2[:, :, diff // 2: diff // 2 + x1.size()[2]]
        return x1 + x2

# Data Paths
train_data_paths = {
    'ghg': "/net/pallas/usr/neuro/com/habbar/projet/data_nc/stagelong/projetlong/data_source_dr/Region17/train-ghg_TTDGM2-S0-GAN85pc-N1000_z17v0_df.p",
    'aer': "/net/pallas/usr/neuro/com/habbar/projet/data_nc/stagelong/projetlong/data_source_dr/Region17/train-aer_TTDGM2-S0-GAN85pc-N1000_z17v0_df.p",
    'nat': "/net/pallas/usr/neuro/com/habbar/projet/data_nc/stagelong/projetlong/data_source_dr/Region17/train-nat_TTDGM2-S0-GAN85pc-N1000_z17v0_df.p",
    'hist': "/net/pallas/usr/neuro/com/habbar/projet/data_nc/stagelong/projetlong/data_source_dr/Region17/train-hist_TTDGM2-S0-GAN85pc-N1000_z17v0_df.p"
}

test_data_paths = {
    'ghg': "/net/pallas/usr/neuro/com/habbar/projet/data_nc/stagelong/projetlong/data_source_dr/Region17/test-ghg_TTDGM2-S0-GAN15pc-NMx1000_z17v0_df.p",
    'aer': "/net/pallas/usr/neuro/com/habbar/projet/data_nc/stagelong/projetlong/data_source_dr/Region17/test-aer_TTDGM2-S0-GAN15pc-NMx1000_z17v0_df.p",
    'nat': "/net/pallas/usr/neuro/com/habbar/projet/data_nc/stagelong/projetlong/data_source_dr/Region17/test-nat_TTDGM2-S0-GAN15pc-NMx1000_z17v0_df.p",
    'hist': "/net/pallas/usr/neuro/com/habbar/projet/data_nc/stagelong/projetlong/data_source_dr/Region17/test-hist_TTDGM2-S0-GAN15pc-NMx1000_z17v0_df.p"
}

# Load the training data
ghg_data_train = pickle_loader(train_data_paths['ghg'])
aer_data_train = pickle_loader(train_data_paths['aer'])
nat_data_train = pickle_loader(train_data_paths['nat'])
hist_data_train = pickle_loader(train_data_paths['hist'])

# Create an instance of the TimeseriesDatasetOptimized
dataset_optimized = TimeseriesDatasetOptimized(ghg_data_train, aer_data_train, nat_data_train, hist_data_train)

# Create a UNet model instance with cropping
unet_model_with_crop = UNet1DWithCrop(in_channels=3, out_channels=1)

model_names = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CESM2', 'CNRM-CM6-1', 'CanESM5', 'FGOALS-g3', 'GISS-E2-1-G',
               'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6', 'MRI-ESM2-0', 'NorESM2-LM']
model_names = ['CanESM5', 'FGOALS-g3',
               'HadGEM3-GC31-LL', 'IPSL-CM6A-LR']

# Initialize an empty dictionary to store MSE for each excluded model
mse_per_model = {}
criterion = nn.MSELoss()   
optimizer = torch.optim.Adam(unet_model_with_crop.parameters(), lr=0.00001)

num_epochs = 500
train_loader = DataLoader(dataset_optimized, batch_size=32, shuffle=True)

# 1. Séparation des données
train_indices, val_indices = train_test_split(np.arange(len(dataset_optimized)), test_size=0.1, random_state=42)
train_dataset = torch.utils.data.Subset(dataset_optimized, train_indices)
val_dataset = torch.utils.data.Subset(dataset_optimized, val_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

best_val_loss = float("inf")  # Initialisation de la meilleure perte de validation
patience = 10
no_improve_epochs = 0

def create_model_directory(trial, model_to_exclude):
    params = trial.params
    dir_name = f"unet_lr{params['lr']}_epochs{params['num_epochs']}_bs{params['batch_size']}_kernel{params['kernel_size']}_base{params['base_channels']}_reg{params['reg']}_patience{params['patience']}"
    model_dir = f"/net/pallas/usr/neuro/com/habbar/projet/data_nc/Figs/u-net/{model_to_exclude}/{dir_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def objective(trial):
    # Hyperparameters to be optimized
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    num_epochs = trial.suggest_int("num_epochs", 100, 1000)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    kernel_size = trial.suggest_int("kernel_size", 3, 7)
    base_channels = trial.suggest_int("base_channels", 16, 64)
    reg = trial.suggest_float("reg", 1e-5, 1e-3)
    patience = trial.suggest_int("patience", 5, 20)
    
    # Initialize an empty dictionary to store MSE for each excluded model
    mse_per_model = {}
    
    # Create a UNet model instance with cropping
    unet_model_with_crop = UNet1DWithCrop(in_channels=3, out_channels=1)
    
    # Setup optimizer and loss criterion
    optimizer = torch.optim.Adam(unet_model_with_crop.parameters(), lr=lr, weight_decay=reg) 
    criterion = nn.MSELoss()
    
    for model_to_exclude in model_names:
        print(f"lr: {lr}, num_epochs: {num_epochs}, batch_size: {batch_size}, kernel_size: {kernel_size}, base_channels: {base_channels}, reg: {reg}, patience: {patience}")
        print(f"Training for {model_to_exclude}")
        # Data loading and filtering
        ghg_data_train_filtered = ghg_data_train.query(f'model != "{model_to_exclude}"')
        aer_data_train_filtered = aer_data_train.query(f'model != "{model_to_exclude}"')
        nat_data_train_filtered = nat_data_train.query(f'model != "{model_to_exclude}"')
        hist_data_train_filtered = hist_data_train.query(f'model != "{model_to_exclude}"')
        
        dataset_filtered = TimeseriesDatasetOptimized(ghg_data_train_filtered, aer_data_train_filtered, nat_data_train_filtered, hist_data_train_filtered)
        
        train_indices, val_indices = train_test_split(np.arange(len(dataset_filtered)), test_size=0.1, random_state=42)
        train_data_filtered = torch.utils.data.Subset(dataset_filtered, train_indices)
        val_data_filtered = torch.utils.data.Subset(dataset_filtered, val_indices)

        train_loader = DataLoader(train_data_filtered, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data_filtered, batch_size=batch_size, shuffle=False)
        
        # Create model directory
        model_dir = create_model_directory(trial, model_to_exclude)
        
        # Reset model weights
        unet_model_with_crop.apply(weight_reset)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        no_improve_epochs = 0

        for epoch in range(num_epochs):
            unet_model_with_crop.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = unet_model_with_crop(inputs)

                labels_cropped = labels[:, :outputs.shape[2]]
                loss = criterion(outputs, labels_cropped.unsqueeze(1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataset_filtered)

            # 2. Évaluation sur l'ensemble de validation
            unet_model_with_crop.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = unet_model_with_crop(inputs)
                    labels_cropped = labels[:, :outputs.shape[2]]
                    loss = criterion(outputs, labels_cropped.unsqueeze(1))
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_data_filtered)

            # Mise à jour des listes avec les pertes courantes
            train_losses.append(epoch_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {epoch_loss:.4f} - Val loss: {val_loss:.4f}")
            
            # save epoch losses et val_losses
            np.save(f"{model_dir}/train_losses.npy", np.array(train_losses))
            np.save(f"{model_dir}/val_losses.npy", np.array(val_losses))

            # Vérifier si c'est le meilleur modèle et Arrêt précoce
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(unet_model_with_crop.state_dict(), f"{model_dir}/best_model.pth")
                no_improve_epochs = 0

            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print("Arrêt précoce!")
                break
            
        # Charger le meilleur modèle dans le dossier du modèle
        unet_model_with_crop.load_state_dict(torch.load(f"{model_dir}/best_model.pth"))
        
        # Mettre le modèle en mode d'évaluation pour l'inférence
        unet_model_with_crop.eval()
        
        # entregistrer les pertes de validation
        mse_per_model[model_to_exclude] = best_val_loss
        
        # Afficher les pertes de validation pour chaque modèle sur matplotlib
        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='val loss')
        plt.legend()
        plt.title(f"Loss curves U-net [Training for {model_to_exclude}][TTDGM2-S0-GAN85pc-N1000_z17v0]-epochs{num_epochs}-lr{lr}-bs{batch_size}-kernel{kernel_size}-base{base_channels}-reg{reg}-patience{patience}")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{model_dir}/losses.png")
        plt.close()
        
    # Retourner la moyenne des pertes de validation pour tous les modèles
    return np.mean(list(mse_per_model.values()))

# Create a study object and optimize the objective function.
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Print optimization results
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))