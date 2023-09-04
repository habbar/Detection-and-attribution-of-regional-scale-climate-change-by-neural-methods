#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training models function for PL
Contains training function:
    
    # TRAIN_CNN_MODEL:
    #
    # Usage:
        model, loss_dic, pytorch_total_params = train_cnn_model ( ... )
    
    # TRAIN_PROCEDURE:
    #
    # Usage:
        base_cases_list, sub_cases_list = train_procedure (data_gener_method, train_set_label, ...)
Created on Tue Apr 18 11:32:47 2023
@author: hamza.abbar@etudiant.univ-lr.fr
"""

def train_cnn_model(data, data_val, cnn_model, dev,
                data_val2nd=None, lr=0.001, nb_epoch=100,
                taille=3, regularisation=None, momentum=None,
                exp_lr_scheduler=False, exp_lr_gamma=0.99, verbose_exp_lr=False,
                mstep_lr_scheduler=False, mstep_lr_milestones=[100,200], mstep_lr_gamma=0.1, verbose_mstep_lr=False,
                save_best_val=False,
                best_val_netfile='cnn_best_val_net.pt',
                save_best_val2nd=False,
                best_val2nd_netfile='cnn_best_val2nd_net.pt',
                normalization=False,
                verbose=False,

                ):
    """
    
    Returns three arguments:
        - cnn_model,
        - loss_dic .... dictionary containing loss, val_loss, etc.
        - pytorch_total_params,
    
    """
    import math
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
    from torch.optim import Adam

    pytorch_total_params = math.fsum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
    #print("pytorch_total_params=",pytorch_total_params)

    criterion = nn.MSELoss()
    
    do_val2nd = False
    if data_val2nd is not None :
        do_val2nd = True
    
    # parametres du regularisateur
    optimizer_parameters = { 'lr':lr }
    if regularisation is not None and regularisation != -1 :
        optimizer_parameters['weight_decay'] = regularisation
    if momentum is not None :   # MOMENTUM, parameter for SGD nor for ADAM
        optimizer_parameters['momentum'] = momentum
    
    # initialise le regularisateur
    optimizer = Adam(cnn_model.parameters(), **optimizer_parameters)
    
    if exp_lr_scheduler:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR
        exp_lr_scheduler = ExponentialLR(optimizer,
                                         gamma=exp_lr_gamma,
                                         verbose=verbose_exp_lr)
    if mstep_lr_scheduler:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR
        mstep_lr_scheduler = MultiStepLR(optimizer,
                                         milestones=mstep_lr_milestones,
                                         gamma=mstep_lr_gamma,
                                         verbose=verbose_mstep_lr)
    
    Loss_tab = []; Loss_val_tab = []; Loss_val2nd_tab = []
    
    if normalization :
        Loss_tab_N = []; Loss_val_tab_N = []; Loss_val2nd_tab_N = []

    best_val_loss = best_val2nd_loss = 1.0e4
    #best_val_loss_epoch = best_val2nd_loss_epoch = 0

    for n_iter in range(nb_epoch):
        
        # initialisation ...
        loss_total = loss_total_N = length = 0
        loss_total_val = loss_total_val_N = length_val = 0
        if do_val2nd:
            loss_total_val2nd = loss_total_val2nd_N = length_val2nd = 0

        # pour chaque batch 
        for ix,(x_train, y_train) in enumerate(data):
            optimizer.zero_grad()
            #print(ix,type(x_train),end='')
            y_hat = cnn_model(x_train)
            loss = criterion(y_hat, y_train)
            loss.backward()
            optimizer.step()
            loss_total += loss
            if normalization :
                loss_total_N += loss
            #optimizer.zero_grad()
            length += 1
        
        if exp_lr_scheduler:
            exp_lr_scheduler.step()
        if mstep_lr_scheduler:
            mstep_lr_scheduler.step()

        with torch.no_grad():
            # Validation
            for ix_v,(x_val, y_val) in enumerate(data_val):
                y_hat_val = cnn_model(x_val)
                loss_val = criterion(y_hat_val, y_val)
                loss_total_val += loss_val
                if normalization :
                    loss_total_val_N += loss_val
                length_val += 1

                # Validation (deuxieme ensemble)
                if do_val2nd:
                    for ix_v,(x_val, y_val) in enumerate(data_val2nd):
                        y_hat_val = cnn_model(x_val)
                        loss_val2nd = criterion(y_hat_val, y_val)
                        loss_total_val2nd += loss_val2nd
                        if normalization :
                            loss_total_val2nd_N += loss_val2nd
                        length_val2nd += 1
        
        current_loss = loss_total.item() / length
        current_val_loss = loss_total_val.item() / length_val
        if do_val2nd:
            current_val2nd_loss = loss_total_val2nd.item() / length_val2nd

        Loss_tab.append(current_loss)
        Loss_val_tab.append(current_val_loss)

        if normalization :
            Loss_tab_N.append(loss_total_N.item() / length)
            Loss_val_tab_N.append(loss_total_val_N.item() / length_val)

        if current_val_loss < best_val_loss :
            if save_best_val :
                if verbose:
                    print(f"best val loss {current_val_loss} at {n_iter} epochs, saving net in file '{best_val_netfile}'")
                torch.save(cnn_model, best_val_netfile)
            best_val_loss = current_val_loss

        if do_val2nd:
            current_val2nd_loss = loss_total_val2nd.item() / length_val2nd

            Loss_val2nd_tab.append(current_val2nd_loss)

            if normalization:
                Loss_val2nd_tab_N.append(loss_total_val2nd_N.item() / length_val2nd)

            if current_val2nd_loss < best_val2nd_loss :
                if save_best_val2nd :
                    if verbose:
                        print(f"best val2nd loss {current_val2nd_loss} at {n_iter} epochs, saving net in file '{best_val2nd_netfile}'")
                    torch.save(cnn_model, best_val2nd_netfile)
                best_val2nd_loss = current_val2nd_loss
        
    # met les differentes loss dans un dictionnaire: initialisation
    loss_dic = { 'loss':Loss_tab, 'val_loss':Loss_val_tab }
    
    # rajoute dans le dictionnaire des loss les loss sur données particulieres ...
    if normalization :
        loss_dic = { **loss_dic, 'loss_n':Loss_tab_N, 'val_loss_n':Loss_val_tab_N }
    if do_val2nd :
        loss_dic = { **loss_dic, 'val2nd_loss':Loss_val2nd_tab }
        if normalization :
            loss_dic = { **loss_dic, 'val2nd_loss_n':Loss_val2nd_tab_N }

    return cnn_model, loss_dic, pytorch_total_params


def train_procedure (data_gener_method, train_set_label, do_train=False, test_set_label=None,
                     train_case_extra_label=None, models_to_train=None,
                     train_with_all=False,
                     channel_size=24, ksizes=[7,7,7],
                     epochs=100, batch_size=100, lr=0.001, regul=0.0005, extrap="no-extrap", n_nnets=1,
                     lp_nathist_filtering=False, lp_nathist_filtering_dictionary=None,
                     lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
                     seed_before_training=0, log_scale_fig_loss=True, normalization_ok=False,
                     val_part_of_train_ok = False, val_part_of_train_fraction = 0.15,
                     data_in_dir=None, data_out_dir=None, figs_dir=None, plot_loss_figs=True, save_loss_figs=True,
                     loss_limits=None,
                     source_dirname='data_source_pl',
                     local_nb_label="train_procedure", fig_ext='png',
                     figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w', 'format':'png'},
                     default_device='gpu', ngpu=1,   # ngpu value is discarded if default_device is not 'gpu'
                     verbose=False,
                    ) :
    import os
    import numpy as np
    import pickle
    from datetime import datetime as dt
    import matplotlib.pyplot as plt

    import torch
    from torch.utils.data import DataLoader

    import generic_tools_pl as gt   # like hexcolor(), ...
    import specific_nn_tools_pl as nnt  # like CustomDataset(), Net(), ...


    if not plot_loss_figs:
        save_loss_figs = False
    
    # epochs should be a scalar or a list of scalars
    if np.isscalar(epochs) :
        list_of_epochs = [  epochs ]
    else:
        list_of_epochs = epochs
    
    # ksizes should be a list of scalars or a a list of listes
    if np.isscalar(ksizes) :
        ksizes = [ ksizes ]
    if np.isscalar(ksizes[0]) :
        list_of_kernel_sizes = [ ksizes ]
    else:
        list_of_kernel_sizes = ksizes
    n_kernel_sizes = len(list_of_kernel_sizes)
    
    # channel_size should be a list of scalars or a a list of listes
    if np.isscalar(channel_size) :
        channel_size = [ channel_size ]
    if np.isscalar(channel_size[0]) :
        list_of_channel_sizes = [ channel_size ]
    else:
        list_of_channel_sizes = channel_size
    n_channel_sizes = len(list_of_channel_sizes)
    
    # equalize lists lengths between kernel and channels sizes
    if len(list_of_channel_sizes) == 1 and len(list_of_kernel_sizes) > 1 :
        list_of_channel_sizes = list_of_channel_sizes * len(list_of_kernel_sizes)
    if len(list_of_kernel_sizes) == 1 and len(list_of_channel_sizes) > 1 :
        list_of_kernel_sizes = list_of_kernel_sizes * len(list_of_channel_sizes)
    if len(list_of_channel_sizes) != len(list_of_kernel_sizes):
        raise f"Different number of kernel sizes ({n_kernel_sizes}) and number of channel sizes ({n_channel_sizes}). Each should be scalar or list but if list thus must have same length."

    # learnrates should be a scalar or a list of scalars
    if np.isscalar(lr) :
        list_of_learnrates = [ lr ]
    else:
        list_of_learnrates = lr

    # analysing input arguments
    load_test_ok = test_set_label is not None

    # Repertoire des donnees
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        data_out_dir = './data_out'

    if figs_dir is None :
        figs_dir = '.'
    
    if verbose :
        print(f"data_in_dir found: '{data_in_dir}'")
        print(f"data_out_dir: '{data_out_dir}'")
        print(f"figs_dir: '{figs_dir}'")

    
    # ################################################################################################
    # Reading data basic parametres
    #
    # Lecture d'un dictionnaire contenant les tableaux d'indices, le nombre de simulations
    # par modele, la liste de modeles, liste de forçages et liste d'années. Il y a un
    # dictionnaire pour TRAIN et un autre pour TEST.
    # ################################################################################################

    train_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix='train', 
                                                       set_label=train_set_label,
                                                       verbose=verbose)
    #if load_test_ok :
    #    test_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix='test',
    #                                                  set_label=test_set_label,
    #                                                  verbose=verbose)
    
    # ################################################################################################
    # Case information
    #
    # Retrieved from TRAIN and TEST dictionaries.
    # ################################################################################################

    # On charge dans le device: gpu/mps/cpu
    dtype = torch.float
    device = nnt.get_device_auto(device=default_device, ngpu=ngpu)
    if verbose :
        print('Currently used device is :', device)

    forcing_names = train_combi_dic['forcings']
    #nb_forc=len(forcing_names)

    #all_years = train_combi_dic['years']
    train_years = np.arange(1900,2015)

    lenDS = len(train_years)

    if verbose :
        print(f" data in dir: {data_in_dir}")
        print(f" train set label: {train_set_label}")
        print(f" case forcings: {forcing_names}")
        print(f" case train years: [{lenDS} values from {train_years[0]} to {train_years[-1]}]")


    # ################################################################################################
    # Reading forcing data
    #
    # Retrieved from TRAIN and TEST data files.
    # ################################################################################################

    model_names_in_train, all_years, train_mod_df, \
        data_train_dic = gt.load_forcing_data(data_in_dir, file_prefix='train',
                                              set_label=train_set_label, forcing_names=forcing_names,
                                              verbose=verbose)

    if load_test_ok :
        _, _, test_mod_df, \
            data_test_dic = gt.load_forcing_data(data_in_dir, file_prefix='test',
                                                 set_label=test_set_label, forcing_names=forcing_names,
                                                 verbose=verbose)
    if models_to_train is not None:
        model_names = models_to_train
    else:
        model_names = model_names_in_train

    print(f"\nModel names ... {model_names}")
    print(f"All years ..... {all_years[0]} to {all_years[-1]}")

    train_NAT = data_train_dic['nat'];   train_GHG = data_train_dic['ghg']; train_AER = data_train_dic['aer']; train_HIST = data_train_dic['hist']
    if load_test_ok :
        test_NAT = data_test_dic['nat']; test_GHG = data_test_dic['ghg'];   test_AER = data_test_dic['aer'];   test_HIST = data_test_dic['hist']
    
    if verbose :
        print('train size:',train_GHG.shape)

    # Low-pass filtaring (NAT & HIST lissage)
    
    if lp_nathist_filtering :
        from scipy import signal

        b_lp_filter, a_lp_filter = gt.filtering_forcing_signal_f (lp_nathist_filtering_dictionary,
                                                                  verbose=verbose )
        if verbose :
            print("\nTraying Low-Pass HIST & NAT filtering:")

        if verbose :
            print("Filtering HIST & NAT Train data:")
        
        train_NAT = signal.filtfilt(b_lp_filter, a_lp_filter, train_NAT)
        train_HIST = signal.filtfilt(b_lp_filter, a_lp_filter, train_HIST)

        if load_test_ok :
            if verbose :
                print("Filtering HIST & NAT Test data:")
            test_NAT = signal.filtfilt(b_lp_filter, a_lp_filter, test_NAT)
            test_HIST = signal.filtfilt(b_lp_filter, a_lp_filter, test_HIST)

    # Preparing Forcing Tensors 

    if verbose :
        print("Train Forcings data set ... ",end='')
    NAT  = torch.tensor(train_NAT.copy(), dtype=dtype).to(device)
    GHG  = torch.tensor(train_GHG.copy(), dtype=dtype).to(device)
    AER  = torch.tensor(train_AER.copy(), dtype=dtype).to(device)
    HIST = torch.tensor(train_HIST.copy(), dtype=dtype).to(device)

    if verbose :
        print('size:',GHG.shape)

    if load_test_ok :
        # On lit le dataset de test
        if verbose :
            print("Test Forcings data set .... ",end='')
        NAT_T  = torch.tensor(test_NAT.copy(), dtype=dtype).to(device)
        GHG_T  = torch.tensor(test_GHG.copy(), dtype=dtype).to(device)
        AER_T  = torch.tensor(test_AER.copy(), dtype=dtype).to(device)
        HIST_T = torch.tensor(test_HIST.copy(), dtype=dtype).to(device)

        if verbose :
            print('size:',GHG_T.shape)


    # ################################################################################################
    # Training conditions
    # ################################################################################################

    #if verbose :
    print("\nTraining conditions:")
    print(f" - train_case_extra_label .. {train_case_extra_label if train_case_extra_label is not None else '<UNDEFINED>'}")
    print(f" - n_nnets ................. {n_nnets}")
    print(f" - epochs list ............. {list_of_epochs}")
    print(f" - kernel sizes list ....... {list_of_kernel_sizes}")
    print(f" - learnrates list ......... {list_of_learnrates}")
    print(f" - channel_sizes list ...... {list_of_channel_sizes}")
    print(f" - regul ................... {regul}")
    print(f" - extrap .................. {extrap}")
    print(f" - log_scale_fig_loss ...... {log_scale_fig_loss}")

    # ################################################################################################
    # Initialise for training
    # ################################################################################################

    #normalization_ok = False
    np.random.seed(seed_before_training)
    
    if loss_limits is not None:
        loss_min_limit,loss_max_limit = loss_limits
    
    if lp_nathist_filtering :
        abs_min4minloss = 1e-4
        if loss_limits is None:
            loss_min_limit,loss_max_limit = 0.015,0.13
    else:
        abs_min4minloss = 1e-4
        if loss_limits is None:
            loss_min_limit,loss_max_limit = 0.018,0.13

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    train_loss_ls      = '-'
    val_loss_ls        = (0, (4, 1))
    val2nd_loss_ls     = (0, (4, 1, 1, 1))  #'dashdot'
    val_min_loss_ls    = (0, (1, 1))
    val2nd_min_loss_ls = (0, (3, 1, 1, 1, 1, 1))  #'dashdot'

    verbose_cnn = True
    #verbose_cnn = False

    save_best_val = True
    #save_best_val = False

    case_train_extension   = None
    exp_lr_scheduler       = False; exp_lr_gamma = None
    verbose_exp_lr         = False
    mstep_lr_scheduler     = False; mstep_lr_milestones = None; mstep_lr_gamma = None
    verbose_mstep_lr       = False

    val2nd_ok = save_best_val2nd = False
    if val_part_of_train_ok :
        val2nd_ok = True
        #val2nd_ok = False
        #save_best_val2nd = val2nd_ok
        #save_best_val2nd = False
    

    time_start = dt.now()
    print(f"\nTRAINING START TIME: {time_start}\n")

    base_cases_list = []
    sub_cases_list  = []

    if train_case_extra_label is None:
        case_out_dir_extension = 'NewNet'
    else:
        case_out_dir_extension = train_case_extra_label

    # Définir une liste pour la loss optimale
    best_loss_list = []
    
    for i_train_case,(nb_epochs,kernel_size_list,channel_size_list,learnrate) in enumerate(zip(list_of_epochs,
                                                                                               list_of_kernel_sizes,
                                                                                               list_of_channel_sizes,
                                                                                               list_of_learnrates,
                                                                                              )) :
        current_case_out_dir_extension = case_out_dir_extension
        case_train_extension = f'Lr{learnrate}'

        if train_set_label.find('N1000_') > 0 :
            # si données TRAIN est fait avec 1000 patterns par modele on diminue le LR 
            learnrate /= 2

        kern_size_label = "-".join([str(k) for k in kernel_size_list])
        
        if np.alltrue([channel_size_list[i] == channel_size_list[0] for i in np.arange(len(channel_size_list))]):
            # if all the same size
            channel_size_label = f"{channel_size_list[0]}"
        else:
            # if different sizes
            channel_size_label = "-".join([str(c) for c in channel_size_list])

        #current_case_out_dir_extension = 'NewNet-MltpSced'
        mstep_lr_scheduler = False
        if mstep_lr_scheduler:
            if nb_epochs <= 200:
                exp_lr_gamma=0.98; mstep_lr_milestones=[80,160]; mstep_lr_gamma=0.30; verbose_mstep_lr = False #ikern == 0
            elif nb_epochs <= 2000:
                exp_lr_gamma=0.98; mstep_lr_scheduler=True; mstep_lr_milestones=[200,400]; mstep_lr_gamma=0.30; verbose_mstep_lr = False
            current_case_out_dir_extension += '_MltpSced'
            case_train_extension += f"_MltpSced{mstep_lr_gamma}at{'-'.join([str(m) for m in mstep_lr_milestones])}e"

        if lp_nathist_filtering :
            current_case_out_dir_extension += '_NHLpFilt'

        # identifiant de famille des reseaux entraines --------------------------------------------------------------------------
        case_out_dir_base = f'out_v5_nn{n_nnets}-{train_set_label}_{len(model_names)}mod'
        if current_case_out_dir_extension is not None:
            case_out_dir_base += f"_{current_case_out_dir_extension}"
        # -----------------------------------------------------------------------------------------------------------------------

        case_out_base_path = os.path.join(data_out_dir, case_out_dir_base)
        print(f"Repertoire de base de sortie pour tous les Cas: '{case_out_base_path}/'")
        if not os.path.exists(case_out_base_path):
            os.makedirs(case_out_base_path)

        if lp_nathist_filtering :
            filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)
            
            if not os.path.isfile(filtering_dic_filename):
                print(f"Saving filtering parameters in file '{filtering_dic_filename}'")
                pickle.dump( lp_nathist_filtering_dictionary, open( filtering_dic_filename, "wb" ) )
            else:
                print(f"\n ** filtering file '{filtering_dic_filename}'\n ** already exists. Filter parameters not saved **")

        # identifiant supplementaire du réseau entrainé (referent au nombre d'epochs, -------------------------------------------
        # au Batch size et, eventuellement, au Learning rate ou autres particularités
        case_train_suffix = f'e{nb_epochs}_bs{batch_size}'
        if case_train_extension is not None:
            case_train_suffix += f"_{case_train_extension}"
        if val_part_of_train_ok: 
            case_train_suffix += f"-VfT{int(val_part_of_train_fraction*100)}"
        if train_with_all: 
            case_train_suffix += f"-TwALL"
        # -----------------------------------------------------------------------------------------------------------------------

        if verbose :
            print(f"\nTaille des CNN: {channel_size_list}")

        # Identifiant global des cas (cummun a tous les NNets de l'ensemble)
        cnn_name_base = f'CNN_Ks{kern_size_label}_nCh{channel_size_label}_Reg{regul}_Xtr{extrap.upper()}_{case_train_suffix}'

        base_cases_list.append(case_out_dir_base)
        sub_cases_list.append(cnn_name_base)

        ################# VAL ON A MODEL ONLY TRAIN OTHERS ####################        
        # Model Names:
        #  [ 'ACCESS-ESM1-5', 'BCC-CSM2-MR',  'CESM2',       'CNRM-CM6-1', 'CanESM5', 'FGOALS-g3', 'GISS-E2-1-G', 'HadGEM3-GC31-LL',
        #    'IPSL-CM6A-LR',  'MIROC6',       'MRI-ESM2-0',  'NorESM2-LM' ]
        all_mod_for_training = model_names
        for imod, training_model in enumerate(all_mod_for_training) :
            #training_model = 'IPSL-CM6A-LR'

            print(f"\n{'-'*132}\nTraining case {i_train_case+1}) '{case_train_suffix}' for modele {imod+1}/{len(all_mod_for_training)})"+
                  f"'{training_model}' (it goes to Validation set, training with others)")

            print(f"\nTraining: {n_nnets} NNets, Batch Size: {batch_size}, Nb Epochs= {nb_epochs}, Learning Rate: {learnrate}")
            print(f"Case Train suffix ..... '{case_train_suffix}'")
            print(f"Case Output dir base .. '{case_out_dir_base}'")

            # On utilise tout sauf le modèle choisie pour l'entrainement (training_model)
            index_model = train_mod_df[lambda df: df['model'] == training_model].index.values.tolist()
            index_other = train_mod_df[lambda df: df['model'] != training_model].index.values.tolist()
            index_all = train_mod_df.index.values.tolist()

            if train_with_all :
                index_for_train = index_all
            else:
                index_for_train = index_other

            if val_part_of_train_ok :
                # Validation set is a part of 'index_for_train' Train data, Train set is the rest of it
                _n_all_train = len(index_for_train)
                _tmp_train_index = [index_for_train[i] for i in np.random.permutation(_n_all_train).tolist()]

                n_val = int(_n_all_train * val_part_of_train_fraction)
                n_train = _n_all_train - n_val

                index_for_train = _tmp_train_index[:n_train]
                index_for_val =_tmp_train_index[n_train:]
                if val2nd_ok:
                    index_for_val2nd = index_model
                    n_val2nd = len(index_for_val2nd)

                if verbose:
                    print(f"\nValidation is {val_part_of_train_fraction*100:.1f}% part of 'index_for_train' Train data:")
                    print(f" - TRAIN set has {n_train} patterns")
                    print(f" - VAL set has {n_val} patterns")
                    if val2nd_ok:
                        print(f" - VAL 2nd set has {n_val2nd} patterns")

            else:
                # Normal case: Train set is all the 'index_for_train' part of Train data, and
                #              validation set the of 'index_model' part of Train data

                index_for_train = index_for_train
                index_for_val = index_model

                n_train = len(index_for_train)
                n_val = len(index_for_val)

                if verbose:
                    print("\nTrain set are all 'index_for_train' data, Validation all 'index_model':")
                    print(f" - VAL set has {n_val} patterns")
                    print(f" - TRAIN set has {n_train} patterns")

            # On extrapole ou ajoute des points en t
            n_to_add = np.sum([k//2 for k in kernel_size_list])
            #print('n_to_add:',n_to_add_orig)
            if verbose :
                print('n_to_add:',n_to_add)
                print("size(NAT):",NAT.shape)
                print("lenDS:", lenDS,", n_to_add:",n_to_add)
            NAT2 = NAT[:,-(lenDS+n_to_add*2):]
            GHG2 = GHG[:,-(lenDS+n_to_add*2):]
            AER2 = AER[:,-(lenDS+n_to_add*2):]

            HIST2 = HIST[:,-lenDS:]

            experiment_name_sdir_lbl = f'Training for {training_model}'
            experiment_name_sdir_prnt = f'Training-for-mod_{training_model}'

            case_out_dir = os.path.join(data_out_dir, case_out_dir_base, f'{cnn_name_base}', experiment_name_sdir_prnt)

            suptitlelabel = f"{cnn_name_base} [{experiment_name_sdir_lbl}] [{train_set_label}] ({n_nnets} Nets)"

            case_figs_dir = os.path.join(figs_dir, case_out_dir_base, f'{cnn_name_base}', experiment_name_sdir_prnt)
            
            if not do_train :
                print(f"\n {'*'*16}\n *** NO TRAIN for Model ({imod+1}) '{training_model}' ***")
                print(f" *** Case output base folder name ...... '{case_out_dir_base}'")
                print(f" *** CNN case base subfolder name ...... '{cnn_name_base}'")
                print(f" *** Case output folder path ........... '{case_out_dir}' ["+\
                      ("ALREADY EXISTS" if os.path.exists(case_out_dir) else "DO NOT EXISTS")+"]")
                print(f" *** Case output figures folder path ... '{case_figs_dir}' ["+\
                      ("ALREADY EXISTS" if os.path.exists(case_out_dir) else "DO NOT EXISTS")+"]")
                print(' *** NEXT TRAIN CASE ...')
                
            else:
                print(f'Repertoire de sortie du cas: {case_out_dir}')
                if not os.path.exists(case_out_dir):
                    os.makedirs(case_out_dir)

                print(f'Repertoire des figures du cas: {case_figs_dir}')
                if save_loss_figs and not os.path.exists(case_figs_dir):
                    os.makedirs(case_figs_dir)

                if plot_loss_figs:
                    fig,axes = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False,figsize=(16,7),
                         gridspec_kw={'hspace': 0.05, 'wspace': 0.05, 
                                      'left': 0.06, 'right': 0.98,
                                      'top' : 0.92, 'bottom' : 0.08 })

                    ax = axes

                data = DataLoader(nnt.CustomDataset(GHG2[index_for_train,:],
                                                    AER2[index_for_train,:],
                                                    NAT2[index_for_train,:],
                                                    HIST2[index_for_train,:]),
                                  shuffle=True,
                                  batch_size=batch_size)

                data_valid = DataLoader(nnt.CustomDataset(GHG2[index_for_val,:],
                                                          AER2[index_for_val,:],
                                                          NAT2[index_for_val,:],
                                                          HIST2[index_for_val,:]),
                                        shuffle=False,
                                        batch_size=batch_size)
                if val2nd_ok:
                    data_2nd_valid = DataLoader(nnt.CustomDataset(GHG2[index_for_val2nd,:],
                                                                  AER2[index_for_val2nd,:],
                                                                  NAT2[index_for_val2nd,:],
                                                                  HIST2[index_for_val2nd,:]),
                                                shuffle=False,
                                                batch_size=batch_size)

                # Entrainement des NNets de l'ensemble
                for innet in range(n_nnets):
                    #######################################
                    loss = []; val_loss = []; val2nd_loss = []
                    if normalization_ok :
                        loss_n = []; val_loss_n = []; val2nd_loss_n = []

                    # On definie le reseau
                    Network = nnt.Net(channel_size_list, kernel_size_list, bias=True).to(device)
                    if verbose:
                        print(nnt.torch_summarize(Network))

                    # On nomme le réseau
                    cnn_name = f'{cnn_name_base}_ST_N{innet}'
                    print(f'Training for model {imod+1}/{len(all_mod_for_training)} - Net ({innet}/{n_nnets}) {cnn_name}:')

                    net_dir = os.path.join(case_out_dir,f'CNN_N{innet}')
                    print(' net_dir:',net_dir)
                    if not os.path.exists(net_dir):
                        os.makedirs(net_dir)

                    net_file_no_ext = 'Net'
                    net_filename = os.path.join(net_dir,f'{net_file_no_ext}.pt')

                    if os.path.isfile(net_filename):
                        print(f"\n ** Training innet {innet} skipped because Net file already exists **\n **   '{net_filename}' **'\n ** But loss loaded ...\n")

                        loss     = pickle.load(open(os.path.join(net_dir,'loss.p'), "rb" ),encoding="latin1")
                        val_loss = pickle.load(open(os.path.join(net_dir,'loss-valid.p'), "rb" ),encoding="latin1")
                        if val2nd_ok: 
                            val2nd_loss = pickle.load(open(os.path.join(net_dir,'loss-valid2nd.p'), "rb" ),encoding="latin1")

                        if normalization_ok :
                            loss_n     = pickle.load(open(os.path.join(net_dir,'loss_n.p'), "rb" ),encoding="latin1")
                            val_loss_n = pickle.load(open(os.path.join(net_dir,'loss-valid_n.p'), "rb" ),encoding="latin1")
                            if val2nd_ok : 
                                val2nd_loss_n = pickle.load(open(os.path.join(net_dir,'loss-valid2nd_n.p'), "rb" ),encoding="latin1")

                    else:
                        # ##############################################################################################
                        # Training parameters and train_cnn_model arguments
                        # ##############################################################################################

                        train_model_optargs = { 'lr':learnrate,
                                               'nb_epoch':nb_epochs,
                                               'regularisation':regul,
                                               'exp_lr_scheduler':exp_lr_scheduler,
                                               'exp_lr_gamma':exp_lr_gamma,
                                               'verbose_exp_lr':verbose_exp_lr,
                                               'mstep_lr_scheduler':mstep_lr_scheduler,
                                               'mstep_lr_milestones':mstep_lr_milestones,
                                               'mstep_lr_gamma':mstep_lr_gamma,
                                               'verbose_mstep_lr':verbose_mstep_lr,
                                               'verbose':verbose_cnn,
                                               'normalization':normalization_ok,
                                              }
                        if save_best_val :
                            # ajoute options
                            train_model_optargs = { **train_model_optargs,
                                                   'save_best_val':save_best_val,
                                                   'best_val_netfile':os.path.join(net_dir,f'{net_file_no_ext}_best-val.pt'),
                                                  }
                        if val2nd_ok : 
                            train_model_optargs = { **train_model_optargs,
                                                   'data_val2nd':data_2nd_valid,
                                                   'save_best_val2nd':save_best_val,
                                                   'best_val2nd_netfile':os.path.join(net_dir,f'{net_file_no_ext}_best-val2nd.pt'),
                                                  }

                        # ##############################################################################################
                        # Training
                        # ##############################################################################################
                        #
                        cnn, loss_dic, nbP = train_cnn_model(data, data_valid, Network, device, **train_model_optargs)
                        #
                        # ##############################################################################################
                        
                        loss = loss_dic['loss']
                        best_loss_list.append(np.min(loss))
                        val_loss = loss_dic['val_loss']
                        if normalization_ok :
                            loss_n = loss_dic['loss_n']
                            val_loss_n = loss_dic['val_loss_n']
                        if val2nd_ok :
                            val2nd_loss = loss_dic['val2nd_loss']
                            if normalization_ok :
                                val2nd_loss_n = loss_dic['val2nd_loss_n']

                        torch.save(cnn, net_filename)

                        # On sauve la loss dans des np.array
                        pickle.dump( loss, open( os.path.join(net_dir,'loss.p'), "wb" ) )
                        pickle.dump( val_loss, open( os.path.join(net_dir,'loss-valid.p'), "wb" ) )
                        if val2nd_ok : 
                            pickle.dump( val2nd_loss, open( os.path.join(net_dir,'loss-valid2nd.p'), "wb" ) )

                        if normalization_ok :
                            pickle.dump( loss_n, open( os.path.join(net_dir,'loss_n.p'), "wb" ) )
                            pickle.dump( val_loss_n, open( os.path.join(net_dir,'loss-valid_n.p'), "wb" ) )
                            if val2nd_ok : 
                                pickle.dump( val2nd_loss_n, open( os.path.join(net_dir,'loss-valid2nd_n.p'), "wb" ) )
                        #raise

                    min_loss = np.min(loss+val_loss+val2nd_loss); max_loss = np.max(loss+val_loss+val2nd_loss)
                    min_val_epoch = np.argmin(val_loss)
                    if val2nd_ok : 
                        min_val2nd_epoch = np.argmin(val2nd_loss)

                    if plot_loss_figs:
                        hp1, = ax.plot(loss, ls=train_loss_ls, label=f'train N{innet}')
                        ax.set_yscale('log')

                        current_cycle_color = hp1.get_c()
                        current_color_lighter = [np.min((1,c*1.4)) for c in gt.hexcolor(current_cycle_color)]  # for VAL
                        current_color_darker = [c*0.5 for c in gt.hexcolor(current_cycle_color)]    # for 2nd VAL

                        ax.plot(val_loss, ls=val_loss_ls, color=current_color_lighter,label=f'val N{innet} ({val_loss[min_val_epoch]:.3f}@{min_val_epoch})')
                        if val2nd_ok : 
                            ax.plot(val2nd_loss, ls=val2nd_loss_ls, color=current_color_darker, label=f'val2nd N{innet} ({val2nd_loss[min_val2nd_epoch]:.3f}@{min_val2nd_epoch})')
                            pass

                        if innet == 0:
                            lax = ax.axis()
                        lax = [lax[0], lax[1], lax[2] if min_loss > lax[2] else min_loss - (min_loss/50), lax[3] if max_loss < lax[3] else max_loss]

                        ax.plot([min_val_epoch,min_val_epoch], [abs_min4minloss,val_loss[min_val_epoch]], ls=val_min_loss_ls, color=current_color_lighter)  #, label=f'val N{innet} ({val_loss[min_val_epoch]:.3f}@{min_val_epoch})')
                        if val2nd_ok : 
                            ax.plot([min_val2nd_epoch,min_val2nd_epoch], [abs_min4minloss,val2nd_loss[min_val2nd_epoch]], ls=val2nd_min_loss_ls, color=current_color_darker) #, label=f'val2nd N{innet} ({val2nd_loss[min_val2nd_epoch]:.3f}@{min_val2nd_epoch})')
                            pass

                        ax.axis(lax)

                if plot_loss_figs:
                    ax.set_xlabel('epochs')
                    ax.legend(loc='upper right')  
                    ax.grid(True,lw=0.5,ls=':')

                    loss_title = f'Loss Curves {suptitlelabel}'
                    ax.set_title(loss_title,size="large",y=1.02)

                    figs_file = f"Fig{local_nb_label}_loss-curves_{n_nnets}NNets"

                    if save_loss_figs :
                        figs_filename = os.path.join(case_figs_dir,f"{figs_file}")
                        if val2nd_ok:
                            figs_filename += "+Val2nd"
                        if not os.path.isfile(f"{figs_filename}.{fig_ext}"):
                            print(f"-- saving figure in file ... '{figs_filename}.{fig_ext}'")
                            plt.savefig(f"{figs_filename}.{fig_ext}", **figs_defaults)
                        else:
                            print(f" ** loss figure already exists. Not saved ... '{figs_filename}.{fig_ext}'")

                    # Zoom
                    ylim = ax.get_ylim()
                    new_ylim = [ylim[0] if loss_min_limit is None else loss_min_limit, ylim[1] if loss_max_limit is None else loss_max_limit]

                    ax.set_ylim(new_ylim)
                    ax.set_title(f"{loss_title} [FIX-Yax]",size="large",y=1.02)

                    if save_loss_figs :
                        figs_filename += "_FIX-Yax"
                        if not os.path.isfile(f"{figs_filename}.{fig_ext}"):
                            print(f"-- saving figure in file ... '{figs_filename}.{fig_ext}'")
                            plt.savefig(f"{figs_filename}.{fig_ext}", **figs_defaults)
                        else:
                            print(f" ** loss figure already exists. Not saved ... '{figs_filename}.{fig_ext}'")

                    plt.show()

                time_interm = dt.now()
                print(f"\nTRAINING INTERMEDIARY TIME: {time_interm}\n")

    print('<train_procedure done>\n')

    time_end = dt.now()
    print(f"\nTRAINING START TIME at {time_start}"+\
          f"\n            and END at {time_end}\n")

    if not do_train :
        print("\nNO TRAIN Activated ! Next time you can do try option >> do_train=True << !\n")

    return base_cases_list, sub_cases_list, best_loss_list
    
    
    
        
    