import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

#torch.manual_seed(0)
#np.random.seed(0)
logger = logging.getLogger()

if logger.hasHandlers():
    # Clear the logger handlers
    logger.handlers.clear()

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
        d3 = self.up3(self.crop_and_add(e3, d4))
        d2 = self.up2(self.crop_and_add(e2, d3))
        d1 = self.up1(self.crop_and_add(e1, d2))
        
        return d1
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Ajout d'une couche de Dropout
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
optimizer = torch.optim.Adam(unet_model_with_crop.parameters(), lr=0.001)  
scheduler = ReduceLROnPlateau(optimizer, 'min') # Définir le planificateur de réduction du taux d'apprentissage

num_epochs = 1000
train_loader = DataLoader(dataset_optimized, batch_size=32, shuffle=True)

# 1. Séparation des données
train_indices, val_indices = train_test_split(np.arange(len(dataset_optimized)), test_size=0.1, random_state=42)
train_dataset = torch.utils.data.Subset(dataset_optimized, train_indices)
val_dataset = torch.utils.data.Subset(dataset_optimized, val_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

best_val_loss = float("inf")  # Initialisation de la meilleure perte de validation
patience = 20
no_improve_epochs = 0
# Loop through each model to exclude it
for model_to_exclude in model_names:
    print(f"Excluding model: {model_to_exclude}")
    
    model_dir = f"/net/pallas/usr/neuro/com/habbar/projet/data_nc/Figs/u-net/{model_to_exclude}"
    best_model_path = f"{model_dir}/best_model.pth"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Check if the model is already trained and exists in the directory
    if os.path.exists(best_model_path):
        print(f"Model for {model_to_exclude} already exists. Loading the model and skipping training.")
        unet_model_with_crop.load_state_dict(torch.load(best_model_path))
        unet_model_with_crop.eval()

    else:
        # Load and filter the data
        ghg_data_train_filtered = pickle_loader(train_data_paths['ghg']).query(f'model != "{model_to_exclude}"')
        aer_data_train_filtered = pickle_loader(train_data_paths['aer']).query(f'model != "{model_to_exclude}"')
        nat_data_train_filtered = pickle_loader(train_data_paths['nat']).query(f'model != "{model_to_exclude}"')
        hist_data_train_filtered = pickle_loader(train_data_paths['hist']).query(f'model != "{model_to_exclude}"')

        ghg_data_test_filtered = pickle_loader(test_data_paths['ghg']).query(f'model == "{model_to_exclude}"')
        aer_data_test_filtered = pickle_loader(test_data_paths['aer']).query(f'model == "{model_to_exclude}"')
        nat_data_test_filtered = pickle_loader(test_data_paths['nat']).query(f'model == "{model_to_exclude}"')
        hist_data_test_filtered = pickle_loader(test_data_paths['hist']).query(f'model == "{model_to_exclude}"')

        # Create instances of TimeseriesDatasetOptimized
        dataset_train = TimeseriesDatasetOptimized(ghg_data_train_filtered, aer_data_train_filtered,
                                                nat_data_train_filtered, hist_data_train_filtered)
        dataset_test = TimeseriesDatasetOptimized(ghg_data_test_filtered, aer_data_test_filtered,
                                                nat_data_test_filtered, hist_data_test_filtered)

        # Reset lists for storing train and validation losses
        train_losses = []
        val_losses = []

        # Reset the best validation loss
        best_val_loss = float("inf")

        # Reset the model's parameters to their initial state
        unet_model_with_crop.apply(weight_reset)
        logging.basicConfig(filename=f"{model_dir}/training.log", level=logging.INFO,
                            format='%(asctime)s - %(levelname)-8s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

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
            
            epoch_loss = running_loss / len(dataset_optimized)

            # 2. Évaluation sur l'ensemble de validation
            unet_model_with_crop.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = unet_model_with_crop(inputs)
                    labels_cropped = labels[:, :outputs.shape[2]]
                    loss = criterion(outputs, labels_cropped.unsqueeze(1))
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_dataset)
            
            # Mettre à jour le planificateur de réduction du taux d'apprentissage
            scheduler.step(val_loss) 
            
            # Mise à jour des listes avec les pertes courantes
            train_losses.append(epoch_loss)
            val_losses.append(val_loss)
            
            logging.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            logging.info(f"Epoch: {epoch + 1}/{num_epochs}.. Training loss: {epoch_loss:.4f}.. Validation Loss: {val_loss:.4f}")
            
            print(f"Epoch: {epoch + 1}/{num_epochs}.. Training loss: {epoch_loss:.4f}.. Validation Loss: {val_loss:.4f}")

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

        test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = unet_model_with_crop(inputs)
                all_outputs.append(outputs.cpu().numpy())
                
                # Rogner les étiquettes pour correspondre à la taille de sortie
                labels_cropped = labels[:, :outputs.shape[2]]
                all_labels.append(labels_cropped.cpu().numpy())


        all_outputs = np.concatenate(all_outputs)
        all_labels = np.concatenate(all_labels)
        
        # Sauvegardez les valeurs des pertes dans le dossier du modèle
        np.save(f"{model_dir}/train_losses.npy", np.array(train_losses))
        np.save(f"{model_dir}/val_losses.npy", np.array(val_losses))

        # Calculer la MSE sur l'ensemble de test
        test_mse = ((all_outputs - all_labels) ** 2).mean()
        # Calculer la RMSE
        test_rmse = np.sqrt(test_mse)
        logging.info(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")

        # Stocker la RMSE pour ce modèle dans le dictionnaire
        mse_per_model[model_to_exclude] = test_rmse
        all_outputs = torch.from_numpy(all_outputs) if isinstance(all_outputs, np.ndarray) else all_outputs
        all_labels = torch.from_numpy(all_labels) if isinstance(all_labels, np.ndarray) else all_labels

        tolerance = 0.5
        correct_predictions = torch.abs(all_outputs - all_labels) <= tolerance
        accuracy = torch.mean(correct_predictions.float())
        
        # 3. Visualisation après l'entraînement
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        
        plt.title(f"Loss curves U-net [Training for {model_to_exclude}][TTDGM2-S0-GAN85pc-N1000_z17v0]-epochs{num_epochs}-lr{0.001}-bs{32}")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        plt.annotate(f"Test RMSE = {test_rmse:.4f}\nAccuracy = {accuracy:.4f}",
                        xy=(0.8, 0.1), xycoords='axes fraction',
                        fontsize=12, ha='center', va='center',
                        bbox=dict(boxstyle='round', fc='w'))
        plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{model_dir}/loss_curve.png")
        plt.close()

        
        print("=========================================================================\n-------------------------------------------------------------------------\n=========================================================================")
        
