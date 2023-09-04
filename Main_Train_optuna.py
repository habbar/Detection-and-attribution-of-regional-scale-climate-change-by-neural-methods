#!/usr/bin/env python
# coding: utf-8

# In[30]:

import matplotlib
matplotlib.use('Agg')

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime as dt

import pandas as pd

import torch
#from torch.utils.data import DataLoader

import regionmask


import optuna_experiment
from optuna.trial import TrialState

# In[2]:


# To load personal libraries:
sys.path.append('.')

# In[31]:


# To load personal libraries (currently in folder 'tools'):
sys.path.append('tools')

#import generic_tools_pl as gt   # like hexcolor(), ...
#import specific_nn_tools_pl as nnt  # like CustomDataset(), Net(), train_model() 
import generic_tools_pl as gt
from generic_tools_pl import region


# In[32]:


from training_proc_pl import train_procedure
import testing_proc_pl as tstp
import inverting_proc_pl as invp

plt.ioff()
# In[33]:


#get_ipython().system('nvidia-smi')
from torch.cuda import device_count

print(f'\nTorch version: {torch.__version__}')
if device_count() > 0:
    print('Torch has detected {0} GPUs compatible with CUDA:'.format(device_count()),end='')
    for device in np.arange(device_count()):
        print('\n  - torch cuda device=',torch.device(device),
              '\n    properties:',torch.cuda.get_device_properties(torch.device(device)))
else:
    print('Any Torch GPUs compatible with CUDA.')

if hasattr(torch.backends, 'mps') :
    # this verifies if the current current Pychch installation was built with MPS activated.
    if torch.backends.mps.is_built():# this ensures that the current current PyTorch installation was built with MPS activated.
        print("\nPyTorch installation was built with MPS activated!")
    else:
        print("\nPyTorch installation was not built with MPS activated. Maybe this is not a MacOS Monterrey platform with ARM processor?")
else:
    print("Current torch.backends doesn't have attribute 'mps' (this is usefull if you are in a MacOS Monterrey platform with ARM processor)")

# Call set_printoptions(threshold=x) to change the threshold where arrays are truncated. To prevent truncation, set the threshold to np.inf.
# Change linewidth to print wider lines ...
#display(np.get_printoptions())
if False:
    np.set_printoptions(threshold=np.inf,linewidth=180) # good for a 1920 pixels wide screen
    pd.set_option('display.max_columns', 18)            #
else:
    np.set_printoptions(threshold=np.inf,linewidth=300)   # good for an iMac 27" screen
    pd.set_option('display.max_columns', 30)              #

#pd.set_option('display.max_columns', 100)             # (this is maybe too much!)


# In[34]:


#verbose = False
verbose = True
#--------------------
save_figs = True
#save_figs = False
#--------------------
figs_dir = "/usr/home/habbar/Bureau/data_nc/Figs"

local_nb_label = "Nb5_TrainOptunaEssais-vPL0"   # label a appliquer aux noms des fichiers (figures) produits dans le notebook

fig_ext = 'png'
figs_defaults = { 'dpi' : 300, 'facecolor':'w', 'edgecolor' : 'w', 'format':fig_ext} # ajuter format='png' ou autre a l'appel de savefig()

if save_figs and not os.path.exists(figs_dir):
    os.makedirs(figs_dir)

list_of_regionabrevs = regionmask.defined_regions.ar6.all.abbrevs
list_of_regionnames = regionmask.defined_regions.ar6.all.names
nb_of_regions = len(list_of_regionnames)

# Quelques regions d'interet principales:
#
# |   17 |  WCE   | West&Central-Europe |
# |   46 |  ARO   | Arctic-Ocean |
# |   48 |  EPO   | Equatorial.Pacific-Ocean |
# |   50 |  NAO   | N.Atlantic-Ocean |

#for RegionID in [ 17, 46, 48, 50 ] :
# for epochs in [ 100, 200, 500, 1000 ] :
#     for RegionID in [ 17, 46, 48, 50 ] :
#for epochs in [ 100, 200 ] :
#for epochs in [ 1000 ] :
#    for RegionID in [ 17 ] : #, 46, 48, 50 ] :

# ### Paramètres du modèle

# In[35]:

for region in [17, 46]:
    gt.region = region
    print(f'\nRegion={region}')
    # identifiant du réseau entrainé et des données TRAIN et TEST
    #data_gener_method = 'TTDGM1'; train_set_label = f'{data_gener_method}-S0-GAN85pc-N132_v4';    test_set_label = f'{data_gener_method}-S0-GAN15pc-NMx100_v4'
    #data_gener_method = 'TTDGM1'; train_set_label = f'{data_gener_method}-S0-GAN85pc-N1000_v4';    test_set_label = f'{data_gener_method}-S0-GAN15pc-NMx1000_v4'

    #data_gener_method = 'TTDGM2'; train_set_label = f'{data_gener_method}-S0-GAN85pc-N132_v4';    test_set_label = f'{data_gener_method}-S0-GAN15pc-NMx100_v4'
    data_gener_method = 'TTDGM2'; train_set_label = f'{data_gener_method}-S0-GAN85pc-N1000_z{region}v0';   test_set_label = f'{data_gener_method}-S0-GAN15pc-NMx1000_z{region}v0'

    # identifiant des données INVERSION ---------------------------------------------------------------
    inversion_suffix = 'INVDGM1-S1-NMx100_z{}v0'.format(region)
    #inversion_suffix = 'INVDGM1-S1-NMx400_v4'   # *** (ce set de donnees n'existe plus, A VERIFIER Si BESOIN) ***
    #inversion_suffix = 'INVDGM1-S1-NMx1000_v4'   # *** (ce set de donnees n'existe plus, A VERIFIER Si BESOIN) ***

    # ether_dir = '/net/ether/data/varclim/ggalod/Constantin'

    #data_dir = '/datatmp/data/constantin_data/data_source'
    #data_dir = '/net/acratopotes/datatmp/data/constantin_data/data_source'
    #data_dir = 'data'

    # On lit le dataset de train
    # Repertoire des donnees
    try:
        # WORK dir carlos projet ryn sur Jean Zay
        data_dir = "/usr/home/habbar/Bureau/data_nc/stagelong/projetlong/data_source_dr/Region{}".format(region)
        if not os.path.isdir(data_dir):
            print(f" ** data_dir '{data_dir}' not found. Trying next...")
            
            # WORK dir Guillaume sur Jean Zay ** NON **
            #data_dir = '/gpfswork/rech/ryn/rces866/Constantin'
            #if not os.path.isdir(data_dir):
            #print(f" ** data_dir '{data_dir}' not found. Trying next...")
            
            # SSD sur Acratopotes au Locean
            data_dir = '/net/acratopotes/datatmp/data/constantin_data/data_source_pl'
            if not os.path.isdir(data_dir):
                print(f" ** data_dir '{data_dir}' not found. Trying next...")
                
                # sur Cloud SU (carlos)
                data_dir = os.path.expanduser('~/Clouds/SUnextCloud/Labo/Travaux/Theses-et-stages/These_Constantin/constantin_data/data_source_pl')
                if not os.path.isdir(data_dir):
                    print(f" ** data_dir '{data_dir}' not found. Trying next...")
                    
                    # en dernier recours, en esperant qu'il y a un repertoire 'data' present ...
                    data_dir = os.path.expanduser('data')
                    if not os.path.isdir(data_dir):
                        print(f" ** data_dir '{data_dir}' not found at all **\n")
                        raise Exception('data_dir not found')

    except Exception as e:
        print(f'\n *** Exception error "{e}" ***\n')
        raise

    print(f"data_dir found at '{data_dir}'")


    # In[36]:


    data_dir


    # ### Entrainement du modèle

    # In[37]:


    # Validation set:
    val_part_of_train_ok = True        # takes a percent of TRAIN (index_other from Train data) to be the VAL set (do nothing with index_model data)
    val_part_of_train_fraction = 0.15  # valid only if val_part_of_train_ok is True
    #val_part_of_train_ok = False      # takes all index_other from Train data for TRAIN and index_model for VAL

    seed_before_training = 0

    # ----------------------------------------------------------------------------------------------------
    # Flag to decide to train with all models or to train indivudually each model (experiences jumelles)
    do_train_with_all_models_flg = False
    #do_train_with_all_models_flg = False

    # ----------------------------------------------------------------------------------------------------
    # Flag to force flatting NAT and HIST profiles (a low-pass foltering is applied to those forcings)
    #do_try_lp_nathist_filtering = True
    do_try_lp_nathist_filtering = False
    if do_try_lp_nathist_filtering:
        # dictionary having the arguments for the scipy.signal.butter() function
        lp_nathist_filtering_dic = { 'n':4, 'Wn':[1./10.], 'btype':'lowpass' }
    else:
        lp_nathist_filtering_dic = None

    # ----------------------------------------------------------------------------------------------------
    # Which model are in training
    #models_to_train = ['ACCESS-ESM1-5', 'HadGEM3-GC31-LL', 'MRI-ESM2-0', 'CESM2', 'FGOALS-g3', 'IPSL-CM6A-LR']
    #models_to_train = ['FGOALS-g3', 'IPSL-CM6A-LR']
    models_to_train = None   # if None then is all models !
    #models_to_train = ['CNRM-CM6-1', 'CanESM5', 'FGOALS-g3']

    # 
    # if you want one or several neural nets trained with same conditions. 
    n_nnets = 1
    #n_nnets = 2
    #n_nnets = 6

    # ----------------------------------------------------------------------------------------------------
    # Variables changing architecture of the Net
    # ----------------------------------------------------------------------------------------------------
    # Sizes of CNNs in the architecture (and implicitly the number of CNN (chained each to other)
    # to be used in with the Net class function in specific_nn_tools.py file.
    kernel_sizes = [7,7,7]
    # ----------------------------------------------------------------------------------------------------
    # number of channels in the CNNs
    channel_sizes = 24;  # scalar or list same lenght as kernel_sizes

    # ----------------------------------------------------------------------------------------------------
    # Number of training epochs to execute
    #epochs = 300
    #epochs = 1000
    epochs = 100000

    # ----------------------------------------------------------------------------------------------------
    # Other training parameters
    batch_size = 100
    learning_rate = 0.001
    #learning_rate = 0.005

    # ----------------------------------------------------------------------------------------------------
    # various boooleens

    plot_loss_figs = True

    do_train = True
    #do_train = False

    do_test_experiences = True
    #do_test_experiences = False

    do_inversion_experiences = False
    #do_inversion_experiences = False

    local_train_extra_label = None
    models_to_train_with_all = None

    # ----------------------------------------------------------------------------------------------------
    # Particularities

    # In case of training once with all models we choose a model to act as identificator of the case, this
    # is needed due to programing choices but do not change anything in results. You can choose any model 
    if do_train_with_all_models_flg :
        models_to_train_with_all = 'IPSL-CM6A-LR'
        models_to_train = [ models_to_train_with_all ]  # we choose only one model in order to do only ONE learning.  In all cases and because do_train_with_all_models_flg is True, Trainig is composed of all data models.

        # This label will be added to the case identifier (and sub-folders where outputs in one side and figures
        # on the other will be saved). See the values of 'base_cases_list' returned by the train_procedure.
        local_train_extra_label = 'EssaiTRwAll2'


    # ### Courbe Loss

    # In[38]:

    # train_procedure arguments and options:
    # -------------------------------------
    #  data_gener_method, train_set_label, do_train=False, test_set_label=None,
    #  train_case_extra_label=None, models_to_train=None,
    #  train_with_all=False,
    #  epochs=100, ksizes=[7,7,7], lr=0.001,
    #  batch_size=100, channel_size=24, regul=0.0005, extrap="no-extrap", n_nnets=1,
    #  lp_nathist_filtering=False, lp_nathist_filtering_dictionary=None,
    #  lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
    #  seed_before_training=0, log_scale_fig_loss=True, normalization_ok=False,
    #  val_part_of_train_ok = False, val_part_of_train_fraction = 0.15,
    #  data_in_dir=None, data_out_dir=None, figs_dir=None, plot_loss_figs=True, save_loss_figs=True,
    #  local_nb_label="train_procedure", fig_ext='png',
    #  figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w', 'format':'png'},
    #  verbose=False,
    
    def objective(trial):
        # Paramètres à optimiser
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        n_cnn = trial.suggest_int("n_cnn", 2, 6)
        channel_sizes = trial.suggest_int('channel_sizes', 16, 64, step=8)
        kernel_sizes = []
        for i in range(n_cnn):
            #kernel_sizes.append(trial.suggest_int("ks{}".format(i), 3, 15, step=2))
            kernel_sizes.append(7)
        

        # Autres paramètres fixes
        data_gener_method = 'TTDGM2'
        train_set_label = f'{data_gener_method}-S0-GAN85pc-N1000_z{region}v0'
        n_nnets = 1
        epochs = 400
        lp_nathist_filtering = False
        lp_nathist_filtering_dictionary = None
        val_part_of_train_ok = True
        val_part_of_train_fraction = 0.15
        verbose = False

        base_cases_list, sub_cases_list, best_loss_list = train_procedure(data_gener_method, train_set_label, seed_before_training=seed_before_training,
                                                        train_case_extra_label=local_train_extra_label, models_to_train=models_to_train,
                                                        train_with_all=do_train_with_all_models_flg,
                                                        data_in_dir=data_dir,
                                                        epochs=epochs, n_nnets=n_nnets, batch_size=batch_size,
                                                        ksizes=kernel_sizes, channel_size=channel_sizes, lr=learning_rate,
                                                        val_part_of_train_ok=val_part_of_train_ok, val_part_of_train_fraction=val_part_of_train_fraction,
                                                        lp_nathist_filtering=lp_nathist_filtering, lp_nathist_filtering_dictionary=lp_nathist_filtering_dictionary,
                                                        figs_dir=figs_dir, plot_loss_figs=plot_loss_figs, save_loss_figs=True, local_nb_label=local_nb_label,
                                                        do_train=True,
                                                        default_device='cpu',
                                                        verbose=verbose)
        
        print(f"{'%'*132}\n% Cases trained:")
        for icas,(cas,scas) in enumerate(zip(base_cases_list, sub_cases_list)):
            print(f"%   - Base case({icas}) .. '{cas}'")
            print(f"%     sub-case ...... '{scas}'")
            
        print(f"Best loss list: {best_loss_list}")
        print(f"{'%'*132}")
        
        # On retourne les meilleurs résultats qui donne la meilleure loss 
        return np.mean(best_loss_list)
    
    study = optuna_experiment.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)


print(f"\nFIN d'Execution [{local_nb_label}]:\n Date/heure d'arret ..... {dt.now()}")

