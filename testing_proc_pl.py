#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing models function for PL
Contains testing functions:
    
    # TEST_CNN_MODEL:
    #
    # Usage:
        y_hat_out, Loss_test_tab = test_cnn_model (data_test, cnn_model, dev)
    # TEST_PROCEDURE:
    #
    # Usage:
        test_procedure (base_case_to_explore, sub_case_to_explore, ...)
    # PLOT_LOSS_CURVES:
    #
    # Usage:
        plot_loss_curves (base_case_to_explore, sub_case_to_explore, ...)
    # BUILD_TEST_TABLE_ALL_MODELS:
    #
    # Usage:
        test_results_all_mod_df = build_test_table_all_models (base_case_to_explore, sub_case_to_explore, ...)
    # PLOT_MSE_ALL_MODELS_AND_NETS:
    #
    # Usage:
        plot_mse_all_models_and_nets (base_case_to_explore, sub_case_to_explore, ...)
        plot_mse_all_models_and_nets (base_case_to_explore, sub_case_to_explore, col_to_plot='RMSE', ...)
    # PLOT_OUTPUT_UNIQ_HIST_PROFILS:
    #
    # Usage:
        plot_output_uniq_HIST_profils (base_case_to_explore, sub_case_to_explore, ...)
    # PLOT_MEAN_OUTPUT_BY_MOD_ON_HIST_PROFILS:
    #
    # Usage:
        plot_mean_output_by_mod_on_HIST_profils(base_case_to_explore, sub_case_to_explore, ...)
Created on Tue Aug 23 15:03:59 2022
@author: carlos.mejia@locean.ipsl.fr
"""
import numpy as np
from IPython.display import display


def test_cnn_model(data_test, cnn_model, dev, returntab=False, verbose=False):
    
    import numpy as np
    import torch
    import torch.nn as nn

    loss_test_list=[]
    criterion = nn.MSELoss()
    y_hat_out=[]
    
    with torch.no_grad():
        for (x_test, y_test) in data_test:
            x_test = x_test.to(dev)
            y_test = y_test.to(dev)
            y_hat_test = cnn_model(x_test)
            #loss_test = criterion(y_hat_test.float(), y_test.float())
            loss_test = criterion(y_hat_test, y_test)

            y_hat_out.append(y_hat_test.detach().cpu().clone().numpy())
            loss_test_list.append(loss_test.detach().cpu().clone().numpy())

            if verbose:
                print('loss:',loss_test)

    y_hat_out_tab = np.array(y_hat_out)

    Loss_test_tab = np.array(loss_test_list)

    Loss_test = Loss_test_tab.mean()
    
    if returntab :
        return y_hat_out_tab, Loss_test_tab
    else:
        return y_hat_out_tab, Loss_test


def test_procedure(base_case_to_explore, sub_case_to_explore, models_to_test=None,
                   trained_with_all=False, sample_model=None,
                   set_prefix_to_test='test', another_label_to_use=None,
                   lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
                   data_in_dir=None, data_out_dir=None,
                   source_dirname='data_source_pl',
                   test_set_=None, load_best_val=False, load_best_val2nd=False,
                   no_print_tables=False, save_net_results=True, force_write=False,
                   default_device='cpu', ngpu=1,   # ngpu value is discarded if default_device is not 'gpu'
                   verbose=False, loss_limits = None):
    
    import os
    import numpy as np
    import pickle

    import torch
    from torch.utils.data import DataLoader


    import generic_tools_pl as gt   # like hexcolor(), ...
    import specific_nn_tools_pl as nnt  # like CustomDataset(), Net(), train_model() 

    #from specific_nn_tools_pl import Net
    
    # Repertoire des donnees
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        #data_out_dir = './data_out'
        data_out_dir = "/usr/home/habbar/Bureau/data_nc/data_out"

    # Retrieving parameters from base_case_to_explore label:
    # (example: 'out_v5_nn6-TTDGM2-S0-GAN85pc-N132_v4_12mod_NewNet')
    #  - 'n_nnets',
    #  - 'data_gener_method',
    #  - 'seed_value',
    #  - 'gan_train_percent_value',
    #  - 'gan_test_percent_value',
    #  - 'data_and_training_label',
    #  - 'associated_test_label',
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = gt.retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    print("base_case_dic keys:",base_case_dic.keys())
    
    n_nnets = base_case_dic['n_nnets']
    data_gener_method = base_case_dic['data_gener_method']
    data_and_training_label = base_case_dic['data_and_training_label']
    test_set_label = base_case_dic['associated_test_label']
    lp_nathist_filtering = base_case_dic['lp_nathist_filtering']

    if verbose:
        print(f" - Number of CNN trained: {n_nnets}")
        print(f" - Data generation method: {data_gener_method}")
        print(f" - Data and Training Label: {data_and_training_label}")
        print(f" - Associated Test Label: {test_set_label}")
        print(f" - Low-pass NAT and HIST filtering: {lp_nathist_filtering}")

    if another_label_to_use is not None :
        set_label_to_use = another_label_to_use
    else:
        if set_prefix_to_test.lower() == 'test' :
            set_label_to_use = test_set_label
        elif set_prefix_to_test.lower() == 'train' :
            set_label_to_use = data_and_training_label
        else :
            set_label_to_use = '<UNKNOWN>'
    test_set_case_label = f'{set_prefix_to_test.upper()}_data_set'

    print(f"\n{test_set_case_label}: Testing on {set_prefix_to_test.upper()} data set:")
    print(f" - Test set Label: {set_label_to_use}")

    # ################################################################################################
    # Reading data basic parametres of Test data
    #
    # Lecture d'un dictionnaire contenant les tableaux d'indices, le nombre de simulations
    # par modele, la liste de modeles, liste de forçages et liste d'années. Il y a un
    # dictionnaire pour TRAIN et un autre pour TEST.
    # ################################################################################################

    test_combi_dic = gt.read_data_set_characteristics(data_in_dir,
                                                      file_prefix=set_prefix_to_test,
                                                      set_label=set_label_to_use,
                                                      verbose=verbose)
    
    # Retrieving parameters from specified sub_case_to_explore label:
    # (example: 'CNN_Ks7-7-7_nCh24_Reg0.0005_XtrNO-EXTRAP_e200_bs100_Lr0.001-VfT15')
    #  - 'kern_size_list',
    #  - 'channel_size',
    #  - 'regularization_weight_decay',
    #  - 'extrapolation_label',
    #  - 'epochs',
    #  - 'batch_size',
    #  - 'learning_rate',
    #  - 'val_part_of_train_fraction'
    if verbose:
        print(f"\nRetrieving parameters from specified sub case to explore '{sub_case_to_explore}':\n(a subdirectory of '{base_case_to_explore}')") # decomposing the sub case to explore 
    # kern_size_list, for instance
    sub_case_dic = gt.retrieve_param_from_sub_case(sub_case_to_explore, verbose=verbose)
    kern_size_list = sub_case_dic['kern_size_list']
    if verbose:
        print(f" - List of sizes of CNN hidden layers or (kernel sizes): {kern_size_list}")

    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore

    if models_to_test is None :
        models_to_test = test_combi_dic['models']
        if verbose:
            print(f"\nList of models found in Test set:\n  {models_to_test}")

        # list of folders os cases inside sub_case_to_explore (normally one folder is a training for a Climat model)
        #glob.glob(os.path.join(data_out_dir,base_case_to_explore,sub_case_to_explore,f"Training-for-mod_*"))
        #all_subcases_trained = np.sort(glob.glob(os.path.join(data_out_dir,base_case_to_explore,sub_case_to_explore,f"Training-for-mod_*"))).tolist()
        #models_to_test = [ s.split('_')[-1] for s in all_subcases_trained]
        #if verbose:
        #    print(f"\nList of models trained found in sub-case folder '{sub_case_to_explore}/':\n  {models_to_test}")

    # ################################################################################################
    # Case information
    #
    # Retrieved from TRAIN and TEST dictionaries.
    # ################################################################################################

    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    if lp_nathist_filtering :
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)
        
        if not os.path.isfile(filtering_dic_filename):
            print(f"\n ** FILTERING FILE NOT FOUD '{filtering_dic_filename}'\n ** and 'lp_nathist_filtering' is active !! IT'S AN ERROR ? **/n")
        else:
            print(f"Loading filtering parameters from file '{filtering_dic_filename}'")
            lp_nathist_filtering_dictionary = pickle.load(open( filtering_dic_filename, "rb" ), encoding="latin1")

    # On charge dans le device: gpu/mps/cpu
    dtype = torch.float
    device = nnt.get_device_auto(device=default_device)
    if verbose :
        print('Currently used device is :', device)

    forcing_names = test_combi_dic['forcings']

    #all_years = test_combi_dic['years']
    train_years = np.arange(1900,2015)

    lenDS = len(train_years)

    #print(f"\n case models: {model_names}")
    if verbose :
        print(f" case forcings: {forcing_names}")
        print(f" case train years: [{lenDS} values from {train_years[0]} to {train_years[-1]}]")

    model_names, all_years, test_mod_df, \
        data_test_dic = gt.load_forcing_data(data_in_dir, file_prefix=set_prefix_to_test,
                                             set_label=set_label_to_use,
                                             forcing_names=forcing_names, verbose=verbose)

    if verbose:
        print(f"\nModel names ... {model_names}")
        print(f"All years ..... {all_years[0]} to {all_years[-1]}")
        
    test_NAT = data_test_dic['nat']; test_GHG = data_test_dic['ghg']; test_AER = data_test_dic['aer']; test_HIST = data_test_dic['hist']

    if lp_nathist_filtering :
        from scipy import signal

        b_lp_filter, a_lp_filter = gt.filtering_forcing_signal_f (lp_nathist_filtering_dictionary,
                                                                  verbose=False )
        if verbose :
            print("Filtering HIST & NAT Train data having shapes: {test_NAT.shape} and {test_HIST.shape}")
        
        test_NAT = signal.filtfilt(b_lp_filter, a_lp_filter, test_NAT)
        test_HIST = signal.filtfilt(b_lp_filter, a_lp_filter, test_HIST)

    # On lit le dataset de test
    if verbose :
        print("Test Forcings data set .... ",end='')
    NAT_T  = torch.tensor(test_NAT.copy(), dtype=dtype).to(device)
    GHG_T  = torch.tensor(test_GHG.copy(), dtype=dtype).to(device)
    AER_T  = torch.tensor(test_AER.copy(), dtype=dtype).to(device)
    HIST_T = torch.tensor(test_HIST.copy(), dtype=dtype).to(device)

    if verbose :
        print('size:',GHG_T.shape)

    #n_to_add=list_k1[ir]//2+list_k2[ir]//2+list_k3[ir]//2
    n_to_add = np.sum([k//2 for k in kern_size_list])

    NAT2_T = NAT_T[:,-(lenDS+n_to_add*2):]
    GHG2_T = GHG_T[:,-(lenDS+n_to_add*2):]
    AER2_T = AER_T[:,-(lenDS+n_to_add*2):]

    #HIST2_T = HIST_T[:,-(lenDS+n_to_add*2):]
    HIST2_T = HIST_T[:,-lenDS:]
    
    tmp_Yi = HIST2_T.detach().cpu().clone().numpy()

    if load_best_val2nd :
        net_label = 'best-val2nd'
        net_filename = 'Net_best-val2nd.pt'
    elif load_best_val :
        net_label = 'best-val'
        net_filename = 'Net_best-val.pt'
    else:
        net_label = 'last'
        net_filename = 'Net.pt'

    #print(NAT2_T.shape,GHG2_T.shape,AER2_T.shape,HIST2_T.shape)
    
    case_allmod_out_dir = os.path.join(data_out_dir, base_case_to_explore, f'{cnn_name_base}')

    dic_tables_file = f"dic_test-tables_{len(models_to_test)}-models_{n_nnets}-nnets_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST.p"
    dic_tables_filename = os.path.join(case_allmod_out_dir,dic_tables_file)

    # On utilise tout les modèles
    data_test = DataLoader(nnt.CustomDataset(GHG2_T, AER2_T, NAT2_T, HIST2_T), 
                           batch_size=1)

    all_models_test_dic = {}
    for i_trained,trained_model in enumerate(models_to_test) :
        
        current_model_test_dic = { 'model': trained_model }
        
        experiment_name_in_sdir = f'Training-for-mod_{trained_model}'
        experiment_name_out_sdir = f'Test-trained-on_{trained_model}'

        case_in_dir = os.path.join(data_out_dir, base_case_to_explore, f'{cnn_name_base}', experiment_name_in_sdir)
        case_out_dir = os.path.join(data_out_dir, base_case_to_explore, f'{cnn_name_base}', experiment_name_out_sdir)

        print(f"Repertoire d'entrainement du cas: {case_in_dir}")
        if not os.path.exists(case_in_dir):
            print(f"\n *** Case training directory '{case_in_dir}/' not found. Skiping model case ...")
            continue

        print(f'Repertoire de sortie du cas: {case_out_dir}')
        if not os.path.exists(case_out_dir):
            os.makedirs(case_out_dir)

        yhat_file = f"hist-hat_{n_nnets}-nnets_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST.p"
        yhat_filename = os.path.join(case_out_dir,yhat_file)

        ttable_file = f"test-mean-table_{n_nnets}-nnets_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST.p"
        ttable_filename = os.path.join(case_out_dir,ttable_file)

        yhat = np.zeros((n_nnets,HIST2_T.shape[0],lenDS))
        loss_mean_list = []
        #model_test_tables = []
        model_test_dfs = []
        for innet in range(n_nnets):
            print(f'\nNet[{innet}]: ',end='')

            net_in_dir = os.path.join(case_in_dir,f'CNN_N{innet}')
            net_out_dir = os.path.join(case_out_dir,f'CNN_N{innet}')
            if not os.path.exists(net_out_dir):
                os.makedirs(net_out_dir)

            if verbose :
                print('\nnet_in_dir',net_in_dir)
                print('net_iout_dir',net_out_dir)

            print(f" - Net file: '{net_filename}' [{net_label}]",end='')

            # Lecture et chargement du CNN entraine
            cnn = torch.load(os.path.join(net_in_dir,net_filename), map_location=torch.device('cpu')).to(device)

            y_t, loss_test_tab = test_cnn_model(data_test,cnn,device,returntab=True)

            loss_mean_list.append(loss_test_tab)
            
            loss_test = loss_test_tab.mean()
            yhat[innet,:,:] = y_t.reshape((HIST2_T.shape[0],lenDS))
            #print(y_t.reshape((HIST2_T.shape[0],lenDS)).shape)
            # On sauve la loss dans des np.array
            print(" ... Loss_test=",loss_test)

            loss_test_filename = os.path.join(net_out_dir,f'loss-test_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST_net-{innet}.p')
            if force_write or not os.path.isfile(loss_test_filename):
                print(f" - saving test net({innet} file: '{loss_test_filename}'")
                pickle.dump(loss_test_tab, open(loss_test_filename, "wb" ) )
            else:
                print(f" ** loss test net({innet} file '{loss_test_filename}' already exists,  not saved **")
    
            print(f"\nLoss Table for Net [{innet}] trained on {trained_model} - {cnn_name_base} ({data_and_training_label})")

            mod_nnet_df = gt.print_loss_table(loss_test_tab, test_mod_df, tmp_Yi, yhat[innet,:],
                                              indexorder=model_names, no_print=no_print_tables)
            
            if save_net_results :
                df_net_result_table_file = f"df_test-net-result-table_mod-{trained_model}_{innet}-nnet_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST"
                
                df_net_result_table_csv_file = df_net_result_table_file+".p"
                df_net_result_table_csv_filename = os.path.join(net_out_dir,df_net_result_table_csv_file)
    
                if force_write or not os.path.isfile(df_net_result_table_csv_filename):
                    print(f" - saving df of test net({innet} result in a CSV file: '{df_net_result_table_csv_filename}'")
                    mod_nnet_df.to_csv(df_net_result_table_csv_filename, sep=',', float_format='%.5f')
                else:
                    print(f" ** df of test net({innet} result CSV file '{df_net_result_table_csv_filename}' already exists, not saved **")

                df_net_result_table_pkl_file = df_net_result_table_file+".pkl"
                df_net_result_table_pkl_filename = os.path.join(net_out_dir,df_net_result_table_pkl_file)
    
                if force_write or not os.path.isfile(df_net_result_table_pkl_filename):
                    print(f" - saving df of test net({innet} result in a Pickle file: '{df_net_result_table_pkl_filename}'")
                    mod_nnet_df.to_pickle(df_net_result_table_pkl_filename)
                else:
                    print(f" ** df of test net({innet} result Pickle file '{df_net_result_table_pkl_filename}' already exists, not saved **")

            model_test_dfs.append(mod_nnet_df)
        
        current_model_test_dic['innet_df'] = model_test_dfs

        loss_ensemble =np.array(loss_mean_list).mean(axis=0)
        yhat_ensemble = yhat.mean(axis=0) # mean for all NNets
        print(f"\nLoss Table for Ensemble {n_nnets}-Nets {cnn_name_base} ({data_and_training_label})")
        
        mod_allnnet_df = gt.print_loss_table(loss_ensemble, test_mod_df, tmp_Yi, yhat_ensemble, indexorder=model_names)

        current_model_test_dic['allnet_df'] = mod_allnnet_df

        if force_write or not os.path.isfile(yhat_filename) :
            print(f" - saving yhat output file: '{yhat_filename}'")
            pickle.dump(yhat, open(yhat_filename, "wb" ) )
        else:
            print(f" ** yhat output file '{yhat_filename}' already exists, not saved **")

        if force_write or not os.path.isfile(ttable_filename) :
            print(f" - saving ttable output file: '{ttable_filename}'")
            pickle.dump(yhat, open(ttable_filename, "wb" ) )
        else:
            print(f" ** ttable output file '{ttable_filename}' already exists, not saved **")

        #loss_t=np.append(loss_t,loss_test_tab)
    
        all_models_test_dic[trained_model] = current_model_test_dic
    
    if len(all_models_test_dic) != len(models_to_test) :
        # corects dic_tables_filename because number of models changed
        models_to_test = all_models_test_dic.keys()
        dic_tables_file = f"dic_test-tables_{len(models_to_test)}-models_{n_nnets}-nnets_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST.p"
        dic_tables_filename = os.path.join(case_allmod_out_dir,dic_tables_file)

    if force_write or not os.path.isfile(dic_tables_filename) :
        print(f" - saving dic of all models/nnets test result file: '{dic_tables_filename}'")
        pickle.dump(all_models_test_dic, open(dic_tables_filename, "wb" ) )
    else:
        print(f" ** dic of all models/nnets test result file '{dic_tables_filename}' already exists, not saved **")

    print(f" yhat shape: {yhat.shape}")

    print('<test_procedure done>\n')
 
    return


def plot_loss_curves(base_case_to_explore, sub_case_to_explore, 
                     abs_minloss=1e-6, loss_limits=None,
                     models_to_plot=None, data_in_dir=None, data_out_dir=None, figs_dir=None,
                     source_dirname='data_source_pl',
                     save_also_free_fig=False, save_figs=True,
                     force_plot=False, force_write=False,
                     log_scale=False,
                     local_nb_label="LossCurves", fig_ext='png',
                     figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w'},
                     verbose=False,
                     ) :
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    import generic_tools_pl as gt   # like hexcolor(), ...

    fixed_limits_ok = False
    if loss_limits is not None:
        fixed_limits_ok = True
        loss_min_limit,loss_max_limit = loss_limits
    
    # Repertoire des donnees et figures
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        #data_out_dir = './data_out'
        data_out_dir = "/usr/home/habbar/Bureau/data_nc/data_out"

    if figs_dir is None :
        figs_dir = '.'

    # Retrieving parameters from base_case_to_explore label:
    # (example: 'CNN_Ks7-7-7_nCh24_Reg0.0005_XtrNO-EXTRAP_e200_bs100_Lr0.001-VfT15')
    #  - 'n_nnets',
    #  - 'data_gener_method',
    #  - 'seed_value',
    #  - 'gan_train_percent_value',
    #  - 'gan_test_percent_value',
    #  - 'data_and_training_label',
    #  - 'associated_test_label',
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = gt.retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    n_nnets = base_case_dic['n_nnets']
    data_and_training_label = base_case_dic['data_and_training_label']
    test_set_label = base_case_dic['associated_test_label']
    lp_nathist_filtering = base_case_dic['lp_nathist_filtering']
    if verbose:
        print(f" - Number of CNN trained: {n_nnets}")
        print(f" - Data and Training Label: {data_and_training_label}")
        print(f" - Associated Test Label: {test_set_label}")
        print(f" - Low-pass NAT and HIST filtering: {lp_nathist_filtering}")

    # Retrieving parameters from specified sub_case_to_explore label:
    # (example: 'out_v5_nn6-TTDGM2-S0-GAN85pc-N132_v4_12mod_NewNet')
    #  - 'kern_size_list',
    #  - 'channel_size',
    #  - 'regularization_weight_decay',
    #  - 'extrapolation_label',
    #  - 'epochs',
    #  - 'batch_size',
    #  - 'learning_rate',
    #  - 'val_part_of_train_fraction'
    if verbose:
        print(f"\nRetrieving parameters from specified sub case to explore '{sub_case_to_explore}':\n(a subdirectory of '{base_case_to_explore}')") # decomposing the sub case to explore 
    # kern_size_list, for instance
    sub_case_dic = gt.retrieve_param_from_sub_case(sub_case_to_explore, verbose=verbose)
    kern_size_list = sub_case_dic['kern_size_list']
    if verbose:
        print(f" - List of sizes of CNN hidden layers or (kernel sizes): {kern_size_list}")

    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore

    if models_to_plot is None :
        # list of folders os cases inside sub_case_to_explore (normally one folder is a training for a Climat model)
        glob.glob(os.path.join(data_out_dir,base_case_to_explore,sub_case_to_explore,"Training-for-mod_*"))

        all_subcases_trained = np.sort(glob.glob(os.path.join(data_out_dir,base_case_to_explore,sub_case_to_explore,"Training-for-mod_*"))).tolist()
        models_to_plot = [ s.split('_')[-1] for s in all_subcases_trained]
        if verbose:
            print(f"\nList of models trained found in sub-case folder '{sub_case_to_explore}/':\n  {models_to_plot}")

    n_models_to_plot = len(models_to_plot)

    #test_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix=set_prefix_to_test, 
    #                                              set_label=test_set_label,
    #                                              verbose=verbose)

    y_ticks_list = None
    if log_scale :
        y_ticks_list = [1.e-04, 3.e-04, 1.e-03, 3.e-03, 1.e-02, 3.e-02, 1.e-01,  3.e-01, 1.e+00, 3.e+00, 1.e+01, 3.e+01]

    alpha_train  = 0.3
    alpha_val    = 0.9
    alpha_val2nd = 0.9

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    train_loss_ls      = '-'
    val_loss_ls        = (0, (4, 1))
    val2nd_loss_ls     = (0, (4, 1, 1, 1))  #'dashdot'
    val_min_loss_ls    = (0, (1, 1))
    val2nd_min_loss_ls = (0, (3, 1, 1, 1, 1, 1))  #'dashdot'

    val2nd_ok = False

    abs_min_all_loss = abs_min_all_val_loss = abs_min_all_val2nd_loss = 1e6
    abs_max_all_loss = abs_max_all_val_loss = abs_max_all_val2nd_loss = -1e6

    for i_trained,trained_model in enumerate(models_to_plot) : 

        #do_save_loss_curve_fig = True
        
        experiment_name_in_sdir = f'Training-for-mod_{trained_model}'
        experiment_name_out_sdir = f'Test-trained-on_{trained_model}'

        case_in_dir = os.path.join(data_out_dir, base_case_to_explore, f'{cnn_name_base}', experiment_name_in_sdir)
        case_out_dir = os.path.join(data_out_dir, base_case_to_explore, f'{cnn_name_base}', experiment_name_out_sdir)
        
        print(f"Repertoire d'entrainement du cas: {case_in_dir}")
        if not os.path.exists(case_in_dir):
            print(f"\n *** Case training directory '{case_in_dir}/' not found. Skiping model case ...")
            continue

        print(f'Repertoire de sortie du cas: {case_out_dir}')
        if not os.path.exists(case_out_dir):
            os.makedirs(case_out_dir)

        #suptitlelabel = f"{cnn_name_base} [{experiment_name_out_sdir}] [{data_and_training_label}] ({n_nnets} Nets)"

        case_figs_dir = os.path.join(figs_dir, base_case_to_explore, f'{cnn_name_base}', experiment_name_out_sdir)
        print(f'Repertoire des figures du cas: {case_figs_dir}')
        if save_figs and not os.path.exists(case_figs_dir):
            os.makedirs(case_figs_dir)
            
        train_loss_list = []
        val_loss_list = []
        val2nd_loss_list = []

        # read TRAIN & VAL loss files
        min_train_loss_list = []
        min_val_loss_list = []
        min_val_loss_epoch_list = []
        min_val2nd_loss_list = []
        min_val2nd_loss_epoch_list = []
        abs_min_mod_loss = abs_min_mod_val_loss = abs_min_mod_val2nd_loss = 1e6
        abs_max_mod_loss = abs_max_mod_val_loss = abs_max_mod_val2nd_loss = -1e6
        for innet in range(n_nnets):
            # On nomme le réseau
            #cnn_name = f'{cnn_name_base}_ST_N{innet}'
            net_in_dir = os.path.join(case_in_dir,f'CNN_N{innet}')
            print(net_in_dir)
            print(os.listdir(net_in_dir))

            # read saved loss and loss_valid
            loss_file = os.path.join(net_in_dir,'loss')
            loss = pickle.load(open(f"{loss_file}.p", "rb" ),encoding="latin1")
            loss_valid = pickle.load(open(f"{loss_file}-valid.p", "rb" ),encoding="latin1")

            if i_trained == 0 and \
                innet == 0 and \
                os.path.isfile(f"{loss_file}-valid2nd.p") :
                val2nd_ok = True

            if val2nd_ok :
                loss_valid2nd = pickle.load(open(f"{loss_file}-valid2nd.p", "rb" ),encoding="latin1")

            train_loss_list.append(loss)
            val_loss_list.append(loss_valid)
            if val2nd_ok :
                val2nd_loss_list.append(loss_valid2nd)

            min_val_loss_epoch_list.append(np.argmin(loss_valid))
            min_val_loss_list.append(np.min(loss_valid))
            min_train_loss_list.append(np.min(loss))

            if val2nd_ok :
                min_val2nd_loss_epoch_list.append(np.argmin(loss_valid2nd))
                min_val2nd_loss_list.append(np.min(loss_valid2nd))
            
            abs_min_mod_loss = min((abs_min_mod_loss, min(loss))); abs_max_mod_loss = max((abs_max_mod_loss, max(loss)))
            abs_min_mod_val_loss = min((abs_min_mod_val_loss, min(loss_valid))); abs_max_mod_val_loss = max((abs_max_mod_val_loss, max(loss_valid)))
            if val2nd_ok :
                abs_min_mod_val2nd_loss = min((abs_min_mod_val2nd_loss, min(loss_valid2nd))); abs_max_mod_val2nd_loss = max((abs_max_mod_val2nd_loss, max(loss_valid2nd)))
                    
            print(f"\n Min/max LOSS for model '{trained_model}', all Nets:")
            print(f"  - Loss ......... {abs_min_mod_loss} / {abs_max_mod_loss}")
            print(f"  - Val loss ..... {abs_min_mod_val_loss} / {abs_max_mod_val_loss}")
            if val2nd_ok :
                print(f"  - Val2nd loss .. {abs_min_mod_val2nd_loss} / {abs_max_mod_val2nd_loss}")
                print(f"  - All losses ... {min(abs_min_mod_loss,abs_min_mod_val_loss,abs_min_mod_val2nd_loss)} / {max(abs_max_mod_loss,abs_max_mod_val_loss,abs_max_mod_val2nd_loss)}")
            else:
                print(f"  - All losses ... {min(abs_min_mod_loss,abs_min_mod_val_loss)} / {max(abs_max_mod_loss,abs_max_mod_val_loss)}")
    
        figs_file = f"Fig{local_nb_label}_loss-and-val-loss-curve_{n_nnets}NNets"
        figs_filename = os.path.join(case_figs_dir,f"{figs_file}")
        
        if val2nd_ok:
            figs_filename += "+Val2nd"

        if not force_plot and save_figs and os.path.isfile(f"{figs_filename}.{fig_ext}"):
            print(f" ** {local_nb_label} figure already exists '{figs_filename}.{fig_ext}'. Figure not prepared")
            
        else:
            fig,axes = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False,figsize=(16,8),
                 gridspec_kw={'hspace': 0.05, 'wspace': 0.05, 
                              'left': 0.055, 'right': 0.99,
                              'top' : 0.90, 'bottom' : 0.065 })
            y_suptitle = 1.005 
            ax = axes
            # plot TRAIN loss
            loss_hh_list = []
            for innet,tmp_loss in enumerate(train_loss_list):
                hh, = ax.plot(tmp_loss, lw=1.5, ls=train_loss_ls, label=f'train N{innet}', alpha=alpha_train)
                loss_hh_list.append(hh)
    
            cycle_color = [ h.get_c() for h in loss_hh_list ]
            current_color_brighter = [[np.min((1,c*1.4)) for c in gt.hexcolor(hexcol)] for hexcol in cycle_color]  # for VAL
            current_color_darker = [[c*0.7 for c in gt.hexcolor(hexcol)] for hexcol in cycle_color]    # for 2nd VAL
    
            # plot VAL loss
            for innet,(tmp_loss,col) in enumerate(zip(val_loss_list,current_color_brighter)):
                ax.plot(tmp_loss, lw=1.5, ls=val_loss_ls, color=col, label=f'val N{innet}', alpha=alpha_val)
             
            if val2nd_ok :
                # plot 2nd VAL loss
                for innet,(tmp_loss,col) in enumerate(zip(val2nd_loss_list,current_color_darker)):
                    ax.plot(tmp_loss, lw=1.5, ls=val2nd_loss_ls, color=col, label=f'val2nd N{innet}', alpha=alpha_val2nd)
    
            if log_scale :
                ax.set_yscale('log')
    
            ax.set_xlabel('epochs')
            ax.set_ylabel('loss')
            ax.grid(True,lw=0.5,ls=':')
    
            #hl = ax.legend(ncol=3 if val2nd_ok else 2)
    
            lax = ax.axis()
    
            if y_ticks_list is not None :
                y_ticks = y_ticks_list
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_ticks)
    
            for tmp_epoch,tmp_loss,col in zip(min_val_loss_epoch_list,min_val_loss_list,current_color_brighter):
                ax.plot([tmp_epoch,tmp_epoch], [abs_minloss,tmp_loss], lw=2, ls=val_min_loss_ls, color=col, alpha=alpha_val) #,label=f'val N{innet}')
    
            if val2nd_ok :
                for tmp_epoch,tmp_loss,col in zip(min_val2nd_loss_epoch_list,min_val2nd_loss_list,current_color_darker):
                    ax.plot([tmp_epoch,tmp_epoch], [abs_minloss,tmp_loss], lw=2, ls=val2nd_min_loss_ls, color=col, alpha=alpha_val2nd) #,label=f'val N{innet}')
    
            lax = [lax[0], lax[1], lax[2] if np.min(min_val_loss_list) > lax[2] else np.min(min_val_loss_list), lax[3]]
            ax.axis(lax)
    
            #ax.set_title(f"Loss Curves {cnn_name_base} [{data_and_training_label}] ({n_nnets} Nets) cases trained"+\
            #             f"\n(min validation loss signaled for each Net by a vertical dotted lines)")
            loss_title = f'Loss Curves - Trained for: {trained_model} ({n_nnets} Nets)'+(" [NAT-HIST LP filt.]" if lp_nathist_filtering else "")
            loss_stitle1 = f"{base_case_to_explore} / {cnn_name_base}"
            loss_stitle2 = f'(min validation loss signaled for each Net by a vertical dotted lines) [{data_and_training_label}]'
    
            ax.set_title(f"{loss_title}\n{loss_stitle1}\n{loss_stitle2}",size="x-large",y=y_suptitle)
        
            if save_figs and (not fixed_limits_ok or fixed_limits_ok and save_also_free_fig):
                if force_write or not os.path.isfile(f"{figs_filename}.{fig_ext}"):
                    print(f"-- saving figure in file ... '{figs_filename}.{fig_ext}'")
                    plt.savefig(f"{figs_filename}.{fig_ext}", **figs_defaults)
                else:
                    print(f" ** loss figure already exists. Not saved ... '{figs_filename}.{fig_ext}'")
            
            if fixed_limits_ok :
                figs_filename += "_FIX-Yax"
    
                # Zoom
                ylim = ax.get_ylim()
                new_ylim = [ylim[0] if loss_min_limit is None else loss_min_limit, ylim[1] if loss_max_limit is None else loss_max_limit]
    
                ax.set_ylim(new_ylim)
                ax.set_title(f"{loss_title} [FIX-Yax]\n{loss_stitle1}\n{loss_stitle2}",size="x-large",y=y_suptitle)
    
                if save_figs :
                    if force_write or not os.path.isfile(f"{figs_filename}.{fig_ext}"):
                        print(f"-- saving figure in file ... '{figs_filename}.{fig_ext}'")
                        plt.savefig(f"{figs_filename}.{fig_ext}", **figs_defaults)
                    else:
                        print(f" ** loss figure already exists. Not saved ... '{figs_filename}.{fig_ext}'")
                else:
                    print(' ** figure not saved. Saving not active **')
    
            plt.show()

        abs_min_all_loss = min((abs_min_all_loss, abs_min_mod_loss)); abs_max_all_loss = max((abs_max_all_loss, abs_max_mod_loss))
        abs_min_all_val_loss = min((abs_min_all_val_loss, abs_min_mod_val_loss)); abs_max_all_val_loss = max((abs_max_all_val_loss, abs_max_mod_val_loss))
        if val2nd_ok :
            abs_min_all_val2nd_loss = min((abs_min_all_val2nd_loss, abs_min_mod_val2nd_loss)); abs_max_all_val2nd_loss = max((abs_max_all_val2nd_loss, abs_max_mod_val2nd_loss))

    print("\n Min/max LOSS for ALL models, all Nets:")
    print(f"  - Loss ......... {abs_min_all_loss} / {abs_max_all_loss}")
    print(f"  - Val loss ..... {abs_min_all_val_loss} / {abs_max_all_val_loss}")
    if val2nd_ok :
        print(f"  - Val2nd loss .. {abs_min_all_val2nd_loss} / {abs_max_all_val2nd_loss}")
        print(f"  - All losses ... {min(abs_min_all_loss,abs_min_all_val_loss,abs_min_all_val2nd_loss)} / {max(abs_max_all_loss,abs_max_all_val_loss,abs_max_all_val2nd_loss)}")
    else:
        print(f"  - All losses ... {min(abs_min_all_loss,abs_min_all_val_loss)} / {max(abs_max_all_loss,abs_max_all_val_loss)}")

    return


def build_test_table_all_models(base_case_to_explore, sub_case_to_explore, models_to_test=None,
                                trained_with_all=False, sample_model=None,
                                set_prefix_to_test='test', 
                                data_in_dir=None, data_out_dir=None,
                                source_dirname='data_source_pl',
                                load_best_val=False, load_best_val2nd=False, force_write=False,
                                verbose=False,
                                ) :
    import os
    import numpy as np
    import pandas as pd
    import pickle

    import generic_tools_pl as gt   # like hexcolor(), ...

    #from specific_nn_tools_pl import Net
    
    # Repertoire des donnees
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        #data_out_dir = './data_out'
        data_out_dir = "/usr/home/habbar/Bureau/data_nc/data_out"

    # Retrieving parameters from base_case_to_explore label:
    # (example: 'out_v5_nn6-TTDGM2-S0-GAN85pc-N132_v4_12mod_NewNet')
    #  - 'n_nnets',
    #  - 'data_gener_method',
    #  - 'seed_value',
    #  - 'gan_train_percent_value',
    #  - 'gan_test_percent_value',
    #  - 'data_and_training_label',
    #  - 'associated_test_label',
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = gt.retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    n_nnets = base_case_dic['n_nnets']
    data_and_training_label = base_case_dic['data_and_training_label']
    test_set_label = base_case_dic['associated_test_label']
    if verbose:
        print(f" - Number of CNN trained: {n_nnets}")
        print(f" - Data and Training Label: {data_and_training_label}")
        print(f" - Associated Test Label: {test_set_label}")

    if set_prefix_to_test.lower() == 'test' :
        set_label_to_use = test_set_label
    elif set_prefix_to_test.lower() == 'train' :
        set_label_to_use = data_and_training_label
    else :
        set_label_to_use = '<UNKNOWN>'
    test_set_case_label = f'{set_prefix_to_test.upper()}_data_set'

    print(f"\n{test_set_case_label}: Testing on {set_prefix_to_test.upper()} data set:")
    print(f" - Test set Label: {set_label_to_use}")

    test_combi_dic = gt.read_data_set_characteristics(data_in_dir,
                                                      file_prefix=set_prefix_to_test,
                                                      set_label=set_label_to_use,
                                                      verbose=verbose)
    #print("#DBG# test_combi_dic.keys():",test_combi_dic.keys())
    #print("#DBG# test_combi_dic['models']:",test_combi_dic['models'])

    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore

    if models_to_test is None :
        models_to_test = test_combi_dic['models']
        if verbose:
            print(f"\nList of models found in Test set:\n  {models_to_test}")
        
        # list of folders os cases inside sub_case_to_explore (normally one folder is a training for a Climat model)
        #glob.glob(os.path.join(data_out_dir,base_case_to_explore,sub_case_to_explore,f"Training-for-mod_*"))
        #all_subcases_trained = np.sort(glob.glob(os.path.join(data_out_dir,base_case_to_explore,sub_case_to_explore,f"Training-for-mod_*"))).tolist()
        #models_to_test = [ s.split('_')[-1] for s in all_subcases_trained]
        #if verbose:
        #    print(f"\nList of models trained found in sub-case folder '{sub_case_to_explore}/':\n  {models_to_test}")
    #print("#DBG# models_to_test:",models_to_test)

    if load_best_val2nd :
        net_label = 'best-val2nd'
        #net_filename = 'Net_best-val2nd.pt'
    elif load_best_val :
        net_label = 'best-val'
        #net_filename = 'Net_best-val.pt'
    else:
        net_label = 'last'
        #net_filename = 'Net.pt'

    case_allmod_out_dir = os.path.join(data_out_dir, base_case_to_explore, f'{cnn_name_base}')

    if trained_with_all :
        # When trained_with_all the training is made for one climat model only
        dic_tables_file = f"dic_test-tables_{1}-models_{n_nnets}-nnets_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST.p"
        #print("#DBG# TRAINED_WITH_ALL, dic_tables_file:",dic_tables_file)
    else:
        #dic_tables_file = f"dic_test-tables_{len(models_to_test)}-models_{n_nnets}-nnets_{net_label}_ST.p"
        dic_tables_file = f"dic_test-tables_{len(models_to_test)}-models_{n_nnets}-nnets_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST.p"
        #print("#DBG# Not TRAINED_WITH_ALL, dic_tables_file:",dic_tables_file)
    dic_tables_filename = os.path.join(case_allmod_out_dir,dic_tables_file)

    if os.path.isfile(dic_tables_filename) :
        print(f" - loading dic of all models/nnets test results file: '{dic_tables_filename}'")
        all_models_test_dic = pickle.load(open(dic_tables_filename, "rb" ) )
    else:
        print(f" ** dic of all models/nnets test results file '{dic_tables_filename}' not found **")
        raise
    
    #print("#DBG# all_models_test_dic.keys():",all_models_test_dic.keys())
    
    # ajoute les autres nnets en renommant aussi les colones MSE et RMSE pour leur ajouter le numero de nnet (innet + 1)
    if trained_with_all :
        # initialise la DataFrame de resultats tout modele pour le premier nnet en renommant les colones MSE et RMSE en leur ajoutant le numero de nnet, commencant a 1
        innet = 0 # pour initialisation
        test_results_all_mod_df = pd.concat(
            [all_models_test_dic[sample_model]['innet_df'][innet].loc[lambda df: df.index == m].rename(columns = { 'MSE': f'MSE-n{innet+1:02d}', 'RMSE': f'RMSE-n{innet+1:02d}' }) \
             for m in models_to_test],
            axis=0)
        #print("#DBG# TRAINED_WITH_ALL, test_results_all_mod_df:",test_results_all_mod_df)
        # Training unique: concatenne les resultats (loss, mse, rmse) pour chaque modele climatique de l'entrainement commun
        for innet in np.arange(1,n_nnets) :
            test_results_all_mod_df = pd.concat(
                [test_results_all_mod_df, pd.concat(
                    [all_models_test_dic[sample_model]['innet_df'][innet].loc[lambda df: df.index == m,['MSE','RMSE']].rename(columns = { 'MSE': f'MSE-n{innet+1:02d}', 'RMSE': f'RMSE-n{innet+1:02d}' }) \
                     for m in models_to_test],
                    axis=0)],
                axis=1)
    else:
        # initialise la DataFrame de resultats tout modele pour le premier nnet en renommant les colones MSE et RMSE en leur ajoutant le numero de nnet, commencant a 1
        innet = 0 # pour initialisation
        test_results_all_mod_df = pd.concat(
            [all_models_test_dic[m]['innet_df'][innet].loc[lambda df: df.index == m].rename(columns = { 'MSE': f'MSE-n{innet+1:02d}', 'RMSE': f'RMSE-n{innet+1:02d}' }) \
             for m in models_to_test],
            axis=0)
        #print("#DBG# Not TRAINED_WITH_ALL, test_results_all_mod_df:",test_results_all_mod_df)
        # Training pa modele: concatenne les resultats (loss, mse, rmse) pour chaque modele climatique de son propre entrainement
        for innet in np.arange(1,n_nnets) :
            test_results_all_mod_df = pd.concat(
                [test_results_all_mod_df, pd.concat(
                    [all_models_test_dic[m]['innet_df'][innet].loc[lambda df: df.index == m,['MSE','RMSE']].rename(columns = { 'MSE': f'MSE-n{innet+1:02d}', 'RMSE': f'RMSE-n{innet+1:02d}' }) \
                     for m in models_to_test],
                    axis=0)],
                axis=1)
    #print("#DBG# test_results_all_mod_df:",test_results_all_mod_df)

    # noms des colonnes de resultats uniquement (toutes sauf ['N','loss']) triées par type de resultat (d'abord les MSE* puis les RMSE*)
    res_col_name_array = test_results_all_mod_df.drop(columns=['N','loss']).columns.values.reshape((n_nnets,2)).T.flatten().tolist()

    # reordre des colonnes (N, MSE, ... RMSE, ...] et des index (des modeles) (on ignore la colonne loss qui est egale a MSE)
    test_results_all_mod_df = test_results_all_mod_df[['N']+res_col_name_array].reindex(index=models_to_test) #.reset_index(drop=False)
    #print("#DBG# test_results_all_mod_df",test_results_all_mod_df)

    df_result_table_file = f"df_test-final-result-table_{len(models_to_test)}-models_{n_nnets}-nnets_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST"

    df_result_raw_table_file = df_result_table_file
    if trained_with_all:
        df_result_raw_table_file += "_TwALL"
    df_result_raw_table_file += ".p"
    df_result_raw_table_filename = os.path.join(case_allmod_out_dir,df_result_raw_table_file)

    df_result_table_csv_file = df_result_table_file
    if trained_with_all:
        df_result_table_csv_file += "_TwALL"
    df_result_table_csv_file += ".csv"
    df_result_table_csv_filename = os.path.join(case_allmod_out_dir,df_result_table_csv_file)

    df_result_table_pkl_file = df_result_table_file
    if trained_with_all:
        df_result_table_pkl_file += "_TwALL"
    df_result_table_pkl_file += ".pkl"
    df_result_table_pkl_filename = os.path.join(case_allmod_out_dir,df_result_table_pkl_file)

    if force_write or not os.path.isfile(df_result_raw_table_filename) :
        print(f" - saving df of test final result in file: '{df_result_raw_table_filename}'")
        pickle.dump(test_results_all_mod_df, open(df_result_raw_table_filename, "wb" ) )
    else:
        print(f" ** df of test final result file '{df_result_raw_table_filename}' already exists, not saved **")

    if force_write or not os.path.isfile(df_result_table_csv_filename) :
        print(f" - saving df of test final result in a CSV file: '{df_result_table_csv_filename}'")
        test_results_all_mod_df.to_csv(df_result_table_csv_filename, sep=',', float_format='%.5f')
    else:
        print(f" ** df of test final result CSV file '{df_result_table_csv_filename}' already exists, not saved **")

    if force_write or not os.path.isfile(df_result_table_pkl_filename) :
        print(f" - saving df of test final result in a Pickle file: '{df_result_table_pkl_filename}'")
        test_results_all_mod_df.to_pickle(df_result_table_pkl_filename)
    else:
        print(f" ** df of test final result Pickle file '{df_result_table_pkl_filename}' already exists, not saved **")

    #print("#DBG# test_results_all_mod_df:",test_results_all_mod_df)
        
    return test_results_all_mod_df


def plot_mse_all_models_and_nets(base_case_to_explore, sub_case_to_explore, col_to_plot='MSE', models_to_plot=None,
                                 trained_with_all=False, sample_model=None,
                                 set_prefix_to_test='test', 
                                 data_in_dir=None, data_out_dir=None, figs_dir=None, save_figs=True,
                                 source_dirname='data_source_pl',
                                 load_best_val=False, load_best_val2nd=False,
                                 limits=None,
                                 local_nb_label="PlotMSECurves", fig_ext='png',
                                 figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w'},
                                 figsize=(8.5,7), force_write=False,
                                 verbose=False,
                                ) :
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import pickle

    import generic_tools_pl as gt   # like hexcolor(), ...

    # Repertoire des donnees
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        #data_out_dir = './data_out'
        data_out_dir = "/usr/home/habbar/Bureau/data_nc/data_out"

    if figs_dir is None :
        figs_dir = '.'

    if load_best_val2nd :
        net_label = 'best-val2nd'
        #net_filename = 'Net_best-val2nd.pt'
    elif load_best_val :
        net_label = 'best-val'
        #net_filename = 'Net_best-val.pt'
    else:
        net_label = 'last'
        #net_filename = 'Net.pt'

    # Retrieving parameters from base_case_to_explore label:
    # (example: 'CNN_Ks7-7-7_nCh24_Reg0.0005_XtrNO-EXTRAP_e200_bs100_Lr0.001-VfT15')
    #  - 'n_nnets',
    #  - 'data_gener_method',
    #  - 'seed_value',
    #  - 'gan_train_percent_value',
    #  - 'gan_test_percent_value',
    #  - 'data_and_training_label',
    #  - 'associated_test_label',
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = gt.retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    n_nnets = base_case_dic['n_nnets']
    data_and_training_label = base_case_dic['data_and_training_label']
    test_set_label = base_case_dic['associated_test_label']
    if verbose:
        print(f" - Number of CNN trained: {n_nnets}")
        print(f" - Data and Training Label: {data_and_training_label}")
        print(f" - Associated Test Label: {test_set_label}")

    if set_prefix_to_test.lower() == 'test' :
        set_label_to_use = test_set_label
    elif set_prefix_to_test.lower() == 'train' :
        set_label_to_use = data_and_training_label
    else :
        set_label_to_use = '<UNKNOWN>'
    test_set_case_label = f'{set_prefix_to_test.upper()}_data_set'

    print(f"\n{test_set_case_label}: Testing on {set_prefix_to_test.upper()} data set:")
    print(f" - Test set Label: {set_label_to_use}")

    test_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix=set_prefix_to_test,
                                                      set_label=set_label_to_use,
                                                      verbose=verbose)

    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore

    if models_to_plot is None :
        models_to_plot = test_combi_dic['models']
        if verbose:
            print(f"\nList of models found in Test set:\n  {models_to_plot}")

        # list of folders os cases inside sub_case_to_explore (normally one folder is a training for a Climat model)
        #glob.glob(os.path.join(data_out_dir,base_case_to_explore,sub_case_to_explore,f"Training-for-mod_*"))
        #all_subcases_trained = np.sort(glob.glob(os.path.join(data_out_dir,base_case_to_explore,sub_case_to_explore,f"Training-for-mod_*"))).tolist()
        #models_to_plot = [ s.split('_')[-1] for s in all_subcases_trained]
        #if verbose:
        #    print(f"\nList of models trained found in sub-case folder '{sub_case_to_explore}/':\n  {models_to_plot}")

    n_models_to_plot = len(models_to_plot)

    models_label, models_label_prnt = gt.models_title_labels(models_to_plot)
    
    case_allmod_out_dir = os.path.join(data_out_dir, base_case_to_explore, f'{cnn_name_base}')
    fig_allmod_out_dir = os.path.join(figs_dir, base_case_to_explore, f'{cnn_name_base}')

    df_result_table_file = f"df_test-final-result-table_{n_models_to_plot}-models_{n_nnets}-nnets_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST"

    if trained_with_all :
        df_result_table_file += "_TwALL"

    df_result_table_filename = os.path.join(case_allmod_out_dir,df_result_table_file+".pkl")
    if os.path.isfile(df_result_table_filename):
        print(f" - loading df of test final result from Pickle file: '{df_result_table_filename}'")
        test_results_all_mod_df = pd.read_pickle(df_result_table_filename)
    else:
        df_result_table_filename = os.path.join(case_allmod_out_dir,df_result_table_file+".csv")
        if os.path.isfile(df_result_table_filename):
            print(f" - loading df of test final result from CSV file: '{df_result_table_filename}'")
            test_results_all_mod_df = pd.read_csv(df_result_table_filename)
        else:
            df_result_table_filename = os.path.join(case_allmod_out_dir,df_result_table_file+".p")
            if os.path.isfile(df_result_table_filename):
                print(f" - loading df of test final result from Raw file: '{df_result_table_filename}'")
                test_results_all_mod_df = pickle.load(open(df_result_table_filename, "rb" ) )
            else:
                print(f" ** df of test final result file '{df_result_table_filename}' not found **")
                raise

    test_results_to_plot_df = test_results_all_mod_df[[f'{col_to_plot}-n{n+1:02d}' for n in np.arange(n_nnets)]]
    
    if verbose:
        display(test_results_to_plot_df)
    
    fig,ax = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=True, figsize=figsize,
                          gridspec_kw={'hspace': 0.05, 'wspace': 0.05, 
                                       'left': 0.07, 'right': 0.76,
                                       'top' : 0.92, 'bottom' : 0.04 })
    
    title_label = f"Mean {col_to_plot} on {set_prefix_to_test.upper()} set - {models_label}, {n_nnets} NNets"+(f" [Train w/All Models]" if trained_with_all else "")
    current_title_label = f"{title_label} [{net_label}]\n{base_case_to_explore}\n{cnn_name_base}"

    test_results_to_plot_df.transpose().plot(ax=ax, legend=False, ls='--', lw=1, marker='o', markersize=3)
    ax.set_title(current_title_label, color='black', size="medium")
    ax.legend(bbox_to_anchor=(1.0, 1.0))
    #ax.plot()

    ax.set_xticks(np.arange(n_nnets), labels=test_results_to_plot_df.columns)

    print(ax.get_xticks())
    print(ax.get_xlim())
    
    last_err = test_results_to_plot_df[f'{col_to_plot}-n{n_nnets:02d}']
    for imod,(mod,y_err) in enumerate(zip(test_results_to_plot_df.index,last_err)) :
        #print(imod,mod)
        ax.text(n_nnets - 1 + 0.1, y_err, mod, size="x-small", va='center', ha='left')
    ax.set_xlim([0-0.2,n_nnets+0.1])
    
    if limits is not None :
        ax.set_ylim(limits)
    

    title_prnt = f"Mean {col_to_plot} {models_label_prnt}, {n_nnets} NNets"
    figfile_label = title_prnt.replace(' ','_').replace('(','').replace(')','').replace(',','').replace("'",'').replace('_-_','_')
    figs_file = f"Fig{local_nb_label}_{figfile_label}_{net_label}_test-on-{set_prefix_to_test.upper()}-set{'_Fix' if limits else ''}"
    if trained_with_all :
        figs_file += "_TwALL"
    figs_file += f".{fig_ext}"
    figs_filename = os.path.join(fig_allmod_out_dir,figs_file)

    if save_figs :
        if force_write or not os.path.isfile(f"{figs_filename}"):
            print(f"-- saving figure in file ... '{figs_filename}'")
            plt.savefig(f"{figs_filename}", **figs_defaults)
        else:
            print(f" ** Mean {col_to_plot} figure already exists. Not saved ... '{figs_filename}'")
    else:
        print(' ** figure not saved. Saving not active **')

    plt.show()
    
    return


def plot_output_uniq_HIST_profils(base_case_to_explore, sub_case_to_explore,
                                  trained_with_all=False, sample_model=None,
                                  set_prefix_to_test='test', 
                                  models_to_plot=None, data_in_dir=None, data_out_dir=None, figs_dir=None, save_figs=True,
                                  source_dirname='data_source_pl',
                                  errorlimits_percent=None, errorlimits_n_rms=1,
                                  force_plot=False, force_write=False,
                                  legend_at_last=True, legend_in_last_page=True,
                                  load_best_val=False, load_best_val2nd=False, t_limits=None,
                                  train_years=np.arange(1900,2015),
                                  HIST_is_array4unique=True,
                                  show_x_forcings=False,
                                  local_nb_label="PLotUniqOutput", fig_ext='png',
                                  figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w'},
                                  figsize=None,
                                  lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
                                  verbose=False,
                                 ) :
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    from scipy.stats import norm

    import generic_tools_pl as gt   # like hexcolor(), ...

    prop_cycle = plt.rcParams['axes.prop_cycle']
    cycle_colors = prop_cycle.by_key()['color']

    fixed_t_limits_ok = False
    if t_limits is not None:
        fixed_t_limits_ok = True
        t_min_limit,t_max_limit = t_limits
    
    if load_best_val2nd :
        net_label = 'best-val2nd'
        #net_filename = 'Net_best-val2nd.pt'
    elif load_best_val :
        net_label = 'best-val'
        #net_filename = 'Net_best-val.pt'
    else:
        net_label = 'last'
        #net_filename = 'Net.pt'

    # Repertoire des donnees et figures
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        #data_out_dir = './data_out'
        data_out_dir = "/usr/home/habbar/Bureau/data_nc/data_out"

    if figs_dir is None :
        figs_dir = '.'

    local_std_coeff = 1
    interval_label = f'{local_std_coeff}xSTD'
    if errorlimits_percent is not None :
        if errorlimits_percent < 1:
            local_ppf = (1 + errorlimits_percent)/2   # Percent point function (inverse of cdf — percentiles).
                                                    # Pour definir un intervalle contenant 90% des valeurs, par exemple,
                                                    # prenant alors entre 5% et 95% de la distribution de probabilités. 
                                                    # Le ppf serait alors 0.95, valeur à passer à la fonction norm.ppf()
                                                    # de SCIPY pour obtenir le coefficient multiplicatif de la std por le 
                                                    # calcul de la taille des barres d'erreur ou largeur de la zone "shaded" (la moitié).
            local_std_coeff = norm.ppf(local_ppf)
            interval_label = f'CI{errorlimits_percent*100:.0f}%'
        else:
            local_std_coeff = errorlimits_percent
            interval_label = f'{errorlimits_percent}xSTD'
        apply_std_coeff = True
        
    elif errorlimits_n_rms != 1 :
        local_std_coeff = errorlimits_n_rms
        interval_label = f'{errorlimits_n_rms}xSTD'
        apply_std_coeff = True

    # Retrieving parameters from base_case_to_explore label:
    # (example: 'CNN_Ks7-7-7_nCh24_Reg0.0005_XtrNO-EXTRAP_e200_bs100_Lr0.001-VfT15')
    #  - 'n_nnets',
    #  - 'data_gener_method',
    #  - 'seed_value',
    #  - 'gan_train_percent_value',
    #  - 'gan_test_percent_value',
    #  - 'data_and_training_label',
    #  - 'associated_test_label',
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = gt.retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    n_nnets = base_case_dic['n_nnets']
    data_and_training_label = base_case_dic['data_and_training_label']
    test_set_label = base_case_dic['associated_test_label']
    lp_nathist_filtering = base_case_dic['lp_nathist_filtering']
    if verbose:
        print(f" - Number of CNN trained: {n_nnets}")
        print(f" - Data and Training Label: {data_and_training_label}")
        print(f" - Associated Test Label: {test_set_label}")
        print(f" - Low-pass NAT and HIST filtering: {lp_nathist_filtering}")

    # Retrieving parameters from specified sub_case_to_explore label:
    # (example: 'out_v5_nn6-TTDGM2-S0-GAN85pc-N132_v4_12mod_NewNet')
    #  - 'kern_size_list',
    #  - 'channel_size',
    #  - 'regularization_weight_decay',
    #  - 'extrapolation_label',
    #  - 'epochs',
    #  - 'batch_size',
    #  - 'learning_rate',
    #  - 'val_part_of_train_fraction'
    if verbose:
        print(f"\nRetrieving parameters from specified sub case to explore '{sub_case_to_explore}':\n(a subdirectory of '{base_case_to_explore}')") # decomposing the sub case to explore 
    # kern_size_list, for instance
    sub_case_dic = gt.retrieve_param_from_sub_case(sub_case_to_explore, verbose=verbose)
    kern_size_list = sub_case_dic['kern_size_list']
    if verbose:
        print(f" - List of sizes of CNN hidden layers or (kernel sizes): {kern_size_list}")

    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    load_data_and_gener_params = {'data_dir':data_in_dir, 'verbose':verbose}
    if lp_nathist_filtering :
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)
        
        if not os.path.isfile(filtering_dic_filename):
            print(f"\n ** FILTERING FILE NOT FOUD '{filtering_dic_filename}'\n ** and 'lp_nathist_filtering' is active !! IT'S AN ERROR ? **/n")
        else:
            print(f"Loading filtering parameters from file '{filtering_dic_filename}'")
            lp_nathist_filtering_dictionary = pickle.load(open( filtering_dic_filename, "rb" ), encoding="latin1")

        load_data_and_gener_params['to_filter'] = ['nat', 'hist']
        load_data_and_gener_params['filtering_dic_filename'] = filtering_dic_filename
    load_data_and_gener_params['verbose'] = verbose
    

    data_dic = gt.load_basic_data_and_gener_df(**load_data_and_gener_params)
    # data dic keys:
    #   'label', 'models', 'forcings', 'forcing_color_dic', 
    #   'forcing_color_names_dic', 'forcing_inv_color_dic', 'forcing_inv_color_names_dic',
    #   'years', 'list_of_df'
    
    #data_label       = data_dic['label']
    all_models_src   = data_dic['models']
    all_forcings_src = data_dic['forcings']
    all_forcing_color_dic     = data_dic['forcing_color_dic']
    all_forcing_inv_color_dic = data_dic['forcing_inv_color_dic']
    all_forcing_light_color_dic = data_dic['forcing_light_color_names_dic']
    #all_forcing_color_names_dic    = data_dic['forcing_color_names_dic']
    #all_forcing_inv_color_names_dic = data_dic['forcing_inv_color_names_dic']
    #all_forcing_light_color_names_dic = data_dic['forcing_light_color_names_dic']
    #all_years        = data_dic['years']
    T_ghg_df,T_aer_df,T_nat_df,T_hist_df = data_dic['list_of_df']

    gan_forcing_names = all_forcings_src[:3]
    gan_forcing_colors = [all_forcing_color_dic[f.lower()] for f in gan_forcing_names]
    gan_forcing_mean_colors = [gt.darker_color(all_forcing_inv_color_dic[f.lower()]) for f in gan_forcing_names]
    #gan_forcing_inv_colors = [all_forcing_inv_color_dic[f.lower()] for f in gan_forcing_names]
    
    # Compute intermodel mean and transpose to have models as columns ...
    GHG_ens_df  = T_ghg_df.groupby('model').mean().transpose()[all_models_src]
    AER_ens_df  = T_aer_df.groupby('model').mean().transpose()[all_models_src]
    NAT_ens_df  = T_nat_df.groupby('model').mean().transpose()[all_models_src]
    HIST_ens_df = T_hist_df.groupby('model').mean().transpose()[all_models_src]

    # Build "all_but" DataFrames
    GHG_ens_all_but_df, AER_ens_all_but_df, NAT_ens_all_but_df, \
        HIST_ens_all_but_df = gt.build_all_but_df(all_models_src,
                                               GHG_ens_df, AER_ens_df,
                                               NAT_ens_df, HIST_ens_df)

    print("\nDATA_DIC KEYS:",data_dic.keys())

    if set_prefix_to_test.lower() == 'test' :
        set_label_to_use = test_set_label
    elif set_prefix_to_test.lower() == 'train' :
        set_label_to_use = data_and_training_label
    else :
        set_label_to_use = '<UNKNOWN>'
    test_set_case_label = f'{set_prefix_to_test.upper()}_data_set'

    print(f"\n{test_set_case_label}: Testing on {set_prefix_to_test.upper()} data set:")
    print(f" - Test set Label: {set_label_to_use}")

    test_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix=set_prefix_to_test,
                                                      set_label=set_label_to_use,
                                                      verbose=verbose)
    print("\nTEST_COMBI_DIC KEYS:",test_combi_dic.keys())

    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore

    if models_to_plot is None :
        models_to_plot = test_combi_dic['models']
        if verbose:
            print(f"\nList of models found in Test set:\n  {models_to_plot}")

    forcing_names = test_combi_dic['forcings']

    lenDS = len(train_years)

    #print(f"\n case models: {model_names}")
    if verbose :
        print(f" case forcings: {forcing_names}")
        print(f" case train years: [{lenDS} values from {train_years[0]} to {train_years[-1]}]")

    model_names, all_years, test_mod_df, \
        data_test_dic = gt.load_forcing_data(data_in_dir, file_prefix=set_prefix_to_test,
                                             set_label=set_label_to_use,
                                             forcing_names=forcing_names, verbose=verbose)

    if verbose:
        print(f"\nModel names ... {model_names}")
        print(f"All years ..... {all_years[0]} to {all_years[-1]}")
        print(f"data_test_dic keys: {data_test_dic.keys()}")
        print("Test mof DF:")
        display(test_mod_df)
        
    if models_to_plot is None :
        models_to_plot = test_combi_dic['models']
        if verbose:
            print(f"\nList of models found in {set_prefix_to_test.upper()} data set:\n  {models_to_plot}")

    test_NAT = data_test_dic['nat']; test_GHG = data_test_dic['ghg']; test_AER = data_test_dic['aer']; test_HIST = data_test_dic['hist']

    if lp_nathist_filtering :
        from scipy import signal

        b_lp_filter, a_lp_filter = gt.filtering_forcing_signal_f (lp_nathist_filtering_dictionary,
                                                                  verbose=False )
        if verbose :
            print("Filtering HIST & NAT Train data having shapes: {test_NAT.shape} and {test_HIST.shape}")
        
        test_NAT = signal.filtfilt(b_lp_filter, a_lp_filter, test_NAT)
        test_HIST = signal.filtfilt(b_lp_filter, a_lp_filter, test_HIST)

    # On lit le dataset de test
    if verbose :
        print("Test Forcings data set .... ",end='')
    
    #n_to_add = np.sum([k//2 for k in kern_size_list])

    #NAT2_T  = test_NAT[:,-(lenDS+n_to_add*2):]
    NAT2_T  = test_NAT[:,-lenDS:]
    GHG2_T  = test_GHG[:,-lenDS:]
    AER2_T  = test_AER[:,-lenDS:]
    HIST2_T = test_HIST[:,-lenDS:]

    # test channels
    #for tai in taille:
    #tai = 16
    
    all_models_test_dic = {}
    for i_mod_tested,tested_model in enumerate(models_to_plot) : 
        
        if trained_with_all :
            mod_from_training = sample_model
        else :
            mod_from_training = tested_model

        # index list of data for current model
        model_indices = test_mod_df[lambda DF: DF['model'] == tested_model].index.values.tolist()
        if verbose:
            print(f"model: {tested_model}, index= [{model_indices[0]} .. {model_indices[-1]}], i.e: {len(model_indices)} patterns from {test_mod_df.shape[0]}")
        
        experiment_name_in_sdir = f'Test-trained-on_{mod_from_training}'
        experiment_name_out_sdir = f'Test-trained-on_{tested_model}'

        case_in_dir = os.path.join(data_out_dir, base_case_to_explore, f'{cnn_name_base}', experiment_name_in_sdir)
        case_out_dir = os.path.join(data_out_dir, base_case_to_explore, f'{cnn_name_base}', experiment_name_out_sdir)

        if verbose:
            print(f"Repertoire de Test du cas (doit exister): {case_in_dir}")
        if not os.path.exists(case_in_dir):
            print(f"\n *** Case Test directory '{case_in_dir}/' not found. Skiping model case ...")
            continue
        else:
            if trained_with_all :
                print(f'\n ** Repertoire du cas: {case_in_dir}/ Ok. Training was realized using all forcings (and a single model for 2nd Val.)\n'+\
                      '    Inversion of Model {mod_to_invert} will be realized using {experiment_name_in_sdir} saved Net ...\n')

        yhat_file = f"hist-hat_{n_nnets}-nnets_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST.p"
        yhat_filename = os.path.join(case_in_dir,yhat_file)

        yHat = pickle.load( open(yhat_filename, "rb" ) )

        GHG_ens_ab_arr_m = GHG_ens_all_but_df[[tested_model]].transpose().iloc[:,-lenDS:].values
        AER_ens_ab_arr_m = AER_ens_all_but_df[[tested_model]].transpose().iloc[:,-lenDS:].values
        NAT_ens_ab_arr_m = NAT_ens_all_but_df[[tested_model]].transpose().iloc[:,-lenDS:].values

        if verbose: 
            print(f"yHat file found for model {tested_model}")
            print(f"Test yHat size: {yHat.shape}")

        #for innet in range(n_nnets):
        #    # On nomme le réseau
        #    cnn_name = f'{cnn_name_base}_ST_N{innet}'
        #    net_dir = os.path.join(case_out_dir,f'CNN_N{innet}')
        #    print(net_dir)

        case_figs_dir = os.path.join(figs_dir, base_case_to_explore, f'{cnn_name_base}', experiment_name_out_sdir)
        if save_figs and not os.path.exists(case_figs_dir):
            os.makedirs(case_figs_dir)

        if HIST_is_array4unique :
            tmp_hyper_array = HIST2_T[model_indices,:].copy()
            uniq_forc_name = 'HIST'
        else:
            tmp_hyper_array = np.concatenate((GHG2_T[model_indices,:],
                                              AER2_T[model_indices,:],
                                              NAT2_T[model_indices,:]),
                                             axis=1)
            uniq_forc_name = 'GHG+AER+NAT'
        HIST_mod = HIST2_T[model_indices,:]
        yHat_mod = yHat[:,model_indices,:]
        XFORC_mod = np.concatenate((GHG2_T[model_indices,:],
                                    AER2_T[model_indices,:],
                                    NAT2_T[model_indices,:]),
                                   axis=0).reshape([3,len(model_indices),GHG2_T.shape[1]]) # dimensions: [NbForc, Nb.Sim, Sz.Sim(years)]

        if verbose:
            print("Data shape for model:")
            print(f"  - tmp_hyper_array shape .. {tmp_hyper_array.shape}")
            print(f"  - HIST_mod shape ......... {HIST_mod.shape}")
            print(f"  - yHat_mod shape ......... {yHat_mod.shape}")
            print(f"  - XFORC_mod shape ........ {XFORC_mod.shape}")
        
        tmp_arrU_T,unique_indices = np.unique(tmp_hyper_array, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices).tolist()

        #tmp_arrU_T,unique_indices,unique_inverse,unique_counts = np.unique(tmp_hyper_array, axis=0, return_index=True, return_inverse=True, return_counts=True)

        #sorted_iunique_indices = np.argsort(unique_indices).tolist()

        #n_patt_to_show = np.min((36,len(unique_indices)))
        n_patt_to_show = len(unique_indices)
        suptitle_label = f"{n_patt_to_show} / {tmp_hyper_array.shape[0]} unique {uniq_forc_name} patterns"

        print_filename_label = suptitle_label.replace(' ','-').replace('-/-','-over-').replace('(','').replace(')','')
        if verbose: 
            print(f"suptitle_label: {suptitle_label}")
            print(f"print_filename_label: {print_filename_label}")
        
        figs_file = f"Fig{local_nb_label}_output-profile-patt-{print_filename_label}_{n_nnets}-nnets_test-on-{set_prefix_to_test.upper()}-set_{net_label}"
        if trained_with_all :
            figs_file += "_TwALL"
        if show_x_forcings :
            figs_file += "_XForc"
        if fixed_t_limits_ok :
            figs_file += "_FIX-T"
        figs_file += f"_ErrBar-{interval_label}"
        figs_file += f".{fig_ext}"
        figs_filename = os.path.join(case_figs_dir,figs_file)

        if not force_plot and save_figs and os.path.isfile(figs_filename):
            print(f" ** {local_nb_label} figure already exists '{figs_filename}'. Figure not prepared")
        else:
            if False:
                top = 0.91;    bottom = 0.04
                left = 0.06;   right = 0.98
                wspace = 0.05; hspace = 0.10
                if n_patt_to_show < 5:
                    ncols = 1
                    left = 0.06+0.26-wspace/2; right = 0.98-0.26+wspace/2
                elif n_patt_to_show < 9:
                    ncols = 2
                elif n_patt_to_show < 19:
                    ncols = 3
                elif n_patt_to_show < 33:
                    ncols = 4
                else:
                    ncols = 5
            else:
                top = 0.92;    bottom = 0.04
                left = 0.06;   right = 0.98
                wspace = 0.05; hspace = 0.10
                if n_patt_to_show < 5:
                    ncols = 1
                    left = 0.06+0.26-wspace/2; right = 0.98-0.26+wspace/2
                    suptitle_fontsize = "medium"
                elif n_patt_to_show < 9:
                    ncols = 2
                    suptitle_fontsize = "large"
                elif n_patt_to_show < 19:
                    ncols = 3
                    #suptitle_fontsize = "x-large"
                    suptitle_fontsize = 18
                elif n_patt_to_show < 33:
                    ncols = 4
                    #suptitle_fontsize = "xx-large"
                    suptitle_fontsize = 24
                else:
                    ncols = 5
                    #suptitle_fontsize = "xx-large"
                    suptitle_fontsize = 32

            nrows = int(np.ceil(n_patt_to_show/ncols))

            alpha   = 0.8
            alpha_h = 0.9
            alpha_f = 0.3
            
            if figsize is None :
                local_figsize = (8*max(2,ncols),5*nrows)
            else:
                local_figsize = figsize
            
            fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=local_figsize,
                                    gridspec_kw={'hspace': hspace, 'wspace': wspace, 
                                                 'left': left,     'right': right,
                                                 'top' : top,      'bottom' : bottom })

            for iax,ax in enumerate(axes.flatten()) :
                if iax < n_patt_to_show :
                    iu = unique_indices[iax]
                    HIST_unique = HIST_mod[iu,:]

                    iu,Hu = unique_indices[iax],HIST_unique
                    iu_all = [ ik for ik,H in enumerate(HIST_mod) if np.alltrue(H == Hu) ]
                    
                    if show_x_forcings :
                        XFORC_to_plot = XFORC_mod[:,iu_all,:]

                        #print(gan_forcing_names,gan_forcing_colors)
                        for iforc,(forc,c_forc) in enumerate(zip(gan_forcing_names,gan_forcing_colors)) :
                            tmp_XFORC = XFORC_to_plot[iforc,:]
                            tmp_label = [f'${forc}$' if i==0 else None for i in np.arange(tmp_XFORC.shape[0])]
                            ax.plot(train_years, tmp_XFORC.T, lw=1, ls='-', label=tmp_label, c=c_forc, alpha=alpha_f)
                    
                        # -------------------------------------------------------------------------
                        # X reference (or Background for inversion)
                        # -------------------------------------------------------------------------
                        ax.plot(train_years, GHG_ens_ab_arr_m.T, c=gt.darker_color(gan_forcing_mean_colors[0]), ls='--', lw=1.5,
                                label="$\\widebar{FORC}_{other}}$")
                        ax.plot(train_years, AER_ens_ab_arr_m.T, c=gt.darker_color(gan_forcing_mean_colors[1]), ls='--', lw=1.5)
                        ax.plot(train_years, NAT_ens_ab_arr_m.T, c=gt.darker_color(gan_forcing_mean_colors[2]), ls='--', lw=1.5)

                    ax.plot(train_years, HIST_unique, lw=1, ls='-', label=f'${uniq_forc_name}$', c='k', alpha=alpha_h)

                    for innet in np.arange(n_nnets) :
                        yHat_to_plot = yHat_mod[innet,iu_all,:].squeeze()
                        current_yhat_color = cycle_colors[innet%len(cycle_colors)]
                        if len(yHat_to_plot.shape) == 1 :
                            x_to_plot = train_years
                            gan_patterns = 1
                            plot_label = f'N{innet}'
                            
                            # plot unique Yhat outputs corresponding to the current unique profile
                            ax.plot(x_to_plot.T, yHat_to_plot.T, lw=1.5, ls='-', color=current_yhat_color, label=plot_label, alpha=alpha)

                        else:
                            x_to_plot = train_years.reshape((1,len(train_years))).repeat(yHat_to_plot.shape[0],axis=0)
                            gan_patterns = yHat_to_plot.shape[0]
                            plot_label = f'N{innet}'
                            #plot_label_multiple = [plot_label if i == 0 else '' for i in np.arange(gan_patterns)]
                            
                            # plot the multiple Yhat outputs corresponding to the current unique profile, No label for legend
                            ax.plot(x_to_plot.T, yHat_to_plot.T, lw=0.75, ls='-', color=gt.lighter_color(current_yhat_color), alpha=alpha/1.5)
                            #print(yHat_to_plot.shape,yHat_to_plot.mean(axis=0).shape)
                            
                            # plot the mean of the multiple Yhat outputs corresponding to the current unique profile (in normal color)
                            # ... first, errorbars (with no label)
                            ax.errorbar(train_years, yHat_to_plot.mean(axis=0), yerr=local_std_coeff * yHat_to_plot.std(axis=0,ddof=1),
                                        lw=1, ls='-', color=current_yhat_color, alpha=alpha)
                            # t... then, the mean (havi nh a label for the legend)
                            ax.plot(train_years, yHat_to_plot.mean(axis=0), lw=1.5, ls='-',
                                    color=current_yhat_color, label=plot_label, alpha=alpha)

                    ax.set_title(f"{uniq_forc_name} pattern {model_indices[unique_indices[iax]]} and {gan_patterns} Output patterns"+("+ X-Forcings" if show_x_forcings else ""),
                                 size="medium")

                    if iax == 0 :
                        ax.legend(loc='upper left', ncol=2)

                    ax.grid(True, lw=0.5, ls=':')

                    xmin,xmax = ax.get_xlim()
                    ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
                    ax.set_xlim([xmin,xmax])
                    
                    if fixed_t_limits_ok :
                        ax.set_ylim([t_min_limit,t_max_limit])
                else:
                    ax.set_visible(False)

            suptitle_first_line = f"Output profiles for {suptitle_label} on {tested_model}, {n_nnets} NNets - {set_prefix_to_test.upper()} set"+\
                f"[ErrBar {interval_label}]"+\
                (f" [Train w/All Models]" if trained_with_all else "")
            current_suptitle_label = f"{suptitle_first_line} [{net_label}]\n{base_case_to_explore}\n{cnn_name_base}"

            plt.suptitle(current_suptitle_label,size=suptitle_fontsize,y=0.99)

            if save_figs :
                if force_write or not os.path.isfile(figs_filename):
                    print(f"-- saving figure in file ... '{figs_filename}'")
                    plt.savefig(figs_filename, **figs_defaults)
                else:
                    print(f" ** {local_nb_label} figure already exists '{figs_filename}'. Figure prepared but not saved")
            else:
                print(' ** figure not saved. Saving not active **')

            plt.show()

    return


def plot_mean_output_by_mod_on_HIST_profils(base_case_to_explore, sub_case_to_explore,
                                            trained_with_all=False, sample_model=None,
                                            set_prefix_to_test='test',
                                            models_to_plot=None, data_in_dir=None, data_out_dir=None, figs_dir=None, save_figs=True, save_figs_stat = True,
                                            do_stat_test = True,
                                            source_dirname='data_source_pl',
                                            hatch_for_shaded_forc='//', plot_forc_shaded_region=False, alpha_for_shaded=0.3, lw_for_shaded=0.5,
                                            errorlimits_percent=None, errorlimits_n_rms=1,
                                            hatch_for_shaded_pred='||', plot_pred_shaded_region=False,
                                            force_plot=False, force_write=False,
                                            legend_at_last=True, legend_in_last_page=True,
                                            load_best_val=False, load_best_val2nd=False, t_limits=None,
                                            train_years=np.arange(1900,2015),
                                            HIST_is_array4unique=True,
                                            show_x_all_forcings=False,
                                            show_x_forcings=False,
                                            forcings_as_df=False,
                                            local_nb_label="PLotUniqOutput", fig_ext='png',
                                            figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w'},
                                            figsize=None,
                                            lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
                                            show_unique=False,
                                            errorbars_on_pred=False,
                                            verbose=False,
                                           ) :
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle
    from scipy.stats import norm

    import generic_tools_pl as gt   # like hexcolor(), ...

    prop_cycle = plt.rcParams['axes.prop_cycle']
    cycle_colors = prop_cycle.by_key()['color']

    fixed_t_limits_ok = False
    if t_limits is not None:
        fixed_t_limits_ok = True
        t_min_limit,t_max_limit = t_limits
    
    if load_best_val2nd :
        net_label = 'best-val2nd'
        #net_filename = 'Net_best-val2nd.pt'
    elif load_best_val :
        net_label = 'best-val'
        #net_filename = 'Net_best-val.pt'
    else:
        net_label = 'last'
        #net_filename = 'Net.pt'

    # Repertoire des donnees et figures
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        #data_out_dir = './data_out'
        data_out_dir = "/usr/home/habbar/Bureau/data_nc/data_out"

    if figs_dir is None :
        figs_dir = '.'

    local_std_coeff = 1
    interval_label = f'{local_std_coeff}xSTD'
    if errorlimits_percent is not None :
        if errorlimits_percent < 1:
            local_ppf = (1 + errorlimits_percent)/2   # Percent point function (inverse of cdf — percentiles).
                                                    # Pour definir un intervalle contenant 90% des valeurs, par exemple,
                                                    # prenant alors entre 5% et 95% de la distribution de probabilités. 
                                                    # Le ppf serait alors 0.95, valeur à passer à la fonction norm.ppf()
                                                    # de SCIPY pour obtenir le coefficient multiplicatif de la std por le 
                                                    # calcul de la taille des barres d'erreur ou largeur de la zone "shaded" (la moitié).
            local_std_coeff = norm.ppf(local_ppf)
            interval_label = f'CI{errorlimits_percent*100:.0f}%'
        else:
            local_std_coeff = errorlimits_percent
            interval_label = f'{errorlimits_percent}xSTD'
        apply_std_coeff = True
        
    elif errorlimits_n_rms != 1 :
        local_std_coeff = errorlimits_n_rms
        interval_label = f'{errorlimits_n_rms}xSTD'
        apply_std_coeff = True

    # Retrieving parameters from base_case_to_explore label:
    # (example: 'CNN_Ks7-7-7_nCh24_Reg0.0005_XtrNO-EXTRAP_e200_bs100_Lr0.001-VfT15')
    #  - 'n_nnets',
    #  - 'data_gener_method',
    #  - 'seed_value',
    #  - 'gan_train_percent_value',
    #  - 'gan_test_percent_value',
    #  - 'data_and_training_label',
    #  - 'associated_test_label',
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = gt.retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    n_nnets = base_case_dic['n_nnets']
    data_and_training_label = base_case_dic['data_and_training_label']
    test_set_label = base_case_dic['associated_test_label']
    lp_nathist_filtering = base_case_dic['lp_nathist_filtering']
    if verbose:
        print(f" - Number of CNN trained: {n_nnets}")
        print(f" - Data and Training Label: {data_and_training_label}")
        print(f" - Associated Test Label: {test_set_label}")
        print(f" - Low-pass NAT and HIST filtering: {lp_nathist_filtering}")

    # Retrieving parameters from specified sub_case_to_explore label:
    # (example: 'out_v5_nn6-TTDGM2-S0-GAN85pc-N132_v4_12mod_NewNet')
    #  - 'kern_size_list',
    #  - 'channel_size',
    #  - 'regularization_weight_decay',
    #  - 'extrapolation_label',
    #  - 'epochs',
    #  - 'batch_size',
    #  - 'learning_rate',
    #  - 'val_part_of_train_fraction'
    if verbose:
        print(f"\nRetrieving parameters from specified sub case to explore '{sub_case_to_explore}':\n(a subdirectory of '{base_case_to_explore}')") # decomposing the sub case to explore 
    # kern_size_list, for instance
    sub_case_dic = gt.retrieve_param_from_sub_case(sub_case_to_explore, verbose=verbose)
    kern_size_list = sub_case_dic['kern_size_list']
    if verbose:
        print(f" - List of sizes of CNN hidden layers or (kernel sizes): {kern_size_list}")

    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    load_data_and_gener_params = {'data_dir':data_in_dir, 'verbose':verbose}
    if lp_nathist_filtering:
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)
        
        if not os.path.isfile(filtering_dic_filename):
            print(f"\n ** FILTERING FILE NOT FOUD '{filtering_dic_filename}'\n ** and 'lp_nathist_filtering' is active !! IT'S AN ERROR ? **/n")
        else:
            print(f"Loading filtering parameters from file '{filtering_dic_filename}'")
            lp_nathist_filtering_dictionary = pickle.load(open( filtering_dic_filename, "rb" ), encoding="latin1")

        load_data_and_gener_params['to_filter'] = ['nat', 'hist']
        load_data_and_gener_params['filtering_dic_filename'] = filtering_dic_filename
    load_data_and_gener_params['verbose'] = verbose

    data_dic = gt.load_basic_data_and_gener_df(**load_data_and_gener_params)
    #print('data_dic keys:',data_dic.keys())
    #print("data_dic['forcing_inv_color_dic']:",data_dic['forcing_inv_color_dic'])
    
    # data dic keys:
    #   'label', 'models', 'forcings', 'forcing_color_dic', 
    #   'forcing_color_names_dic', 'forcing_inv_color_dic', 'forcing_inv_color_names_dic',
    #   'years', 'list_of_df'
    
    #data_label       = data_dic['label']
    all_models_src   = data_dic['models']
    all_forcings_src = data_dic['forcings']
    all_forcing_color_dic     = data_dic['forcing_color_dic']
    all_forcing_inv_color_dic = data_dic['forcing_inv_color_dic']
    all_forcing_light_color_dic = data_dic['forcing_light_color_names_dic']
    #all_forcing_color_names_dic    = data_dic['forcing_color_names_dic']
    #all_forcing_inv_color_names_dic = data_dic['forcing_inv_color_names_dic']
    #all_forcing_light_color_names_dic = data_dic['forcing_light_color_names_dic']
    #all_years        = data_dic['years']
    T_ghg_df,T_aer_df,T_nat_df,T_hist_df = data_dic['list_of_df']

    forcing_names = all_forcings_src[:4]
    forcing_colors = [all_forcing_color_dic[f.lower()] for f in forcing_names]
    forcing_mean_colors = [gt.darker_color(all_forcing_inv_color_dic[f.lower()]) for f in forcing_names]
    #forcing_inv_colors = [all_forcing_inv_color_dic[f.lower()] for f in forcing_names]
    forcing_light_colors = [all_forcing_light_color_dic[f.lower()] for f in forcing_names]
    
    gan_forcing_names = all_forcings_src[:3]
    #gan_forcing_colors = [all_forcing_color_dic[f.lower()] for f in gan_forcing_names]
    #gan_forcing_inv_colors = [all_forcing_inv_color_dic[f.lower()] for f in gan_forcing_names]
    
    gan_forcing_colors = [gt.lighter_color(all_forcing_color_dic[f.lower()],factor=0.33) for f in gan_forcing_names]
    #gan_forcing_colors = [gt.lighter_color(all_forcing_color_dic[f.lower()]) for f in gan_forcing_names]
    gan_forcing_bg_colors = [gt.lighter_color(all_forcing_color_dic[f.lower()],factor=0.80) for f in gan_forcing_names]

    # Compute intermodel mean and transpose to have models as columns ...
    GHG_ens_df  = T_ghg_df.groupby('model').mean().transpose()[all_models_src]
    AER_ens_df  = T_aer_df.groupby('model').mean().transpose()[all_models_src]
    NAT_ens_df  = T_nat_df.groupby('model').mean().transpose()[all_models_src]
    HIST_ens_df = T_hist_df.groupby('model').mean().transpose()[all_models_src]

    # Build "all_but" DataFrames
    GHG_ens_all_but_df, AER_ens_all_but_df, NAT_ens_all_but_df, \
        HIST_ens_all_but_df = gt.build_all_but_df(all_models_src,
                                               GHG_ens_df, AER_ens_df,
                                               NAT_ens_df, HIST_ens_df)

    print("\nDATA_DIC KEYS:",data_dic.keys())
    
    if set_prefix_to_test.lower() == 'test' :
        set_label_to_use = test_set_label
    elif set_prefix_to_test.lower() == 'train' :
        set_label_to_use = data_and_training_label
    else :
        set_label_to_use = '<UNKNOWN>'
    test_set_case_label = f'{set_prefix_to_test.upper()}_data_set'

    print(f"\n{test_set_case_label}: Testing on {set_prefix_to_test.upper()} data set:")
    print(f" - Test set Label: {set_label_to_use}")

    test_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix=set_prefix_to_test,
                                                      set_label=set_label_to_use,
                                                      verbose=verbose)
    print("\nTEST_COMBI_DIC KEYS:",test_combi_dic.keys())
    
    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore

    forcing_names = test_combi_dic['forcings']

    lenDS = len(train_years)

    #print(f"\n case models: {model_names}")
    if verbose :
        print(f" case forcings: {forcing_names}")
        print(f" case train years: [{lenDS} values from {train_years[0]} to {train_years[-1]}]")

    model_names, all_years, test_mod_df, \
        data_test_dic = gt.load_forcing_data(data_in_dir, file_prefix=set_prefix_to_test,
                                             dataframe=forcings_as_df,
                                             set_label=set_label_to_use,
                                             forcing_names=forcing_names, verbose=verbose)

    if verbose:
        print(f"\nModel names ... {model_names}")
        print(f"All years ..... {all_years[0]} to {all_years[-1]}")
        print(f"data_test_dic keys: {data_test_dic.keys()}")
        print("Test mof DF:")
        display(test_mod_df)
        
    if models_to_plot is None :
        models_to_plot = test_combi_dic['models']
        if verbose:
            print(f"\nList of models found in {set_prefix_to_test.upper()} data set:\n  {models_to_plot}")

    n_models_to_plot = len(models_to_plot)
    
    models_label, models_label_prnt = gt.models_title_labels(models_to_plot)

    if forcings_as_df :
        test_NAT_df = data_test_dic['nat']; test_GHG_df = data_test_dic['ghg']; test_AER_df = data_test_dic['aer']; test_HIST_df = data_test_dic['hist']
    else:
        test_NAT = data_test_dic['nat']; test_GHG = data_test_dic['ghg']; test_AER = data_test_dic['aer']; test_HIST = data_test_dic['hist']

    if lp_nathist_filtering :
        if forcings_as_df :
            if verbose :
                print("Filtering HIST & NAT df ...")
                print("BEFORE Filtering:")
                display(test_NAT_df)
                display(test_HIST_df)
            test_NAT_df = gt.filter_forcing_df (test_NAT_df, filt_dic=lp_nathist_filtering_dictionary, 
                                                verbose=verbose)
            test_HIST_df = gt.filter_forcing_df (test_HIST_df, filt_dic=lp_nathist_filtering_dictionary, 
                                                verbose=verbose)
            if verbose :
                print("AFTER Filtering:")
                display(test_NAT_df)
                display(test_HIST_df)
        else:
            from scipy import signal
    
            b_lp_filter, a_lp_filter = gt.filtering_forcing_signal_f (lp_nathist_filtering_dictionary,
                                                                      verbose=verbose )    
            if verbose :
                print("Filtering HIST & NAT Train data having shapes: {tmp_test_NAT.shape} and {tmp_test_HIST.shape}")
                print("BEFORE Filtering:")
                print(test_NAT[:5,:])
                print(test_HIST[:5,:])
                
            test_NAT = signal.filtfilt(b_lp_filter, a_lp_filter, test_NAT)
            test_HIST = signal.filtfilt(b_lp_filter, a_lp_filter, test_HIST)

            if verbose :
                print("AFTER Filtering:")
                print(test_NAT[:5,:])
                print(test_HIST[:5,:])

    # On lit le dataset de test
    if verbose :
        print("Test Forcings data set .... ",end='')
        
    tiny_data_label = 'uniq' if show_unique else 'all'

    hist_mod_color = gt.lighter_color(forcing_colors[-1],factor=0.6)
    hist_mean_mod_color = forcing_colors[-1]
    hist_mod_alpha = 0.6
    hist_mod_lw = 2.0
    
    yhat_alpha = 1.0
    yhat_lw = 2.5
    yhat_elw = 1.5

    hist_bg_color = [0.85,0.85,.85,1]
    hist_bg_alpha = 0.05
    hist_bg_lw = 1.0

    forc_alpha = 0.1
    forc_lw = 1.0
    forc_bg_alpha = 0.05
    forc_bg_lw = 1.0
    
    case_figs_dir = os.path.join(figs_dir, base_case_to_explore, f'{cnn_name_base}')

    if save_figs and not os.path.exists(case_figs_dir):
        os.makedirs(case_figs_dir)

    figs_file = f"Fig{local_nb_label}"+("_NHLpFilt" if lp_nathist_filtering else "")+\
        f"_predicted_hist_{models_label_prnt}_on_{set_prefix_to_test.upper()}_set_{tiny_data_label}-hist-bg-{net_label}-net"
    if errorbars_on_pred:                         figs_file += "-errbar"
    if trained_with_all :                         figs_file += "_TwALL"
    if show_x_all_forcings and show_x_forcings:   figs_file += "_XAllAndModForc"
    elif show_x_all_forcings :                    figs_file += "_XAllForc"
    elif show_x_forcings :                        figs_file += "_XModForc"
    if fixed_t_limits_ok :                        figs_file += "_FIX-T"
    if plot_forc_shaded_region:                   figs_file += f"_Shaded-{interval_label}"
    else:                                         figs_file += f"_ErrBar-{interval_label}"
    if plot_pred_shaded_region:                   figs_file += f"_YhShaded"
    figs_file += f".{fig_ext}"
    figs_filename = os.path.join(case_figs_dir,figs_file)
    
    if do_stat_test:
        figs_file_stat = figs_file + f"_StatTest.{fig_ext}"
        figs_filename_stat = os.path.join(case_figs_dir,figs_file_stat)

    if not force_plot and save_figs and os.path.isfile(figs_filename):
        print(f" ** {local_nb_label} figure already exists '{figs_filename}'. Figure not prepared")
    else:
        suptitle_fontsize = 'xx-large'
        top = 0.93
        left = 0.03
        if n_models_to_plot > 9 :
            ncols = 3
        elif n_models_to_plot > 6 :
            ncols = 3
        elif n_models_to_plot > 4 :
            ncols = 2
            suptitle_fontsize = 'x-large'
            left = 0.04
        else:
            ncols = 1
            suptitle_fontsize = 'large'
            top = 0.92
            left = 0.07

        nrows = int(np.ceil(n_models_to_plot/ncols))
        
        fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(8*ncols,1+5*nrows),
             gridspec_kw={ 'hspace': 0.10, 'wspace': 0.02, 
                           'left': left, 'right': 0.98,
                           'top' : top, 'bottom' : 0.03 })

        fig_stat,axes_stat = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(8*ncols,1+5*nrows),
            gridspec_kw={ 'hspace': 0.10, 'wspace': 0.02, 
                        'left': left, 'right': 0.98,
                        'top' : top, 'bottom' : 0.03 })
    
        #for i_mod_tested,(ax,tested_model) in enumerate(zip(axes.flatten()[:n_models_to_plot],models_to_plot)) : 
        for i_mod_tested, (ax, ax_stat, tested_model) in enumerate(zip(np.ravel(axes)[:n_models_to_plot], np.ravel(axes_stat)[:n_models_to_plot], models_to_plot)):

            
            if trained_with_all :
                mod_from_training = sample_model
            else :
                mod_from_training = tested_model
    
            # index list of data for current model
            model_indices = test_mod_df[lambda DF: DF['model'] == tested_model].index.values.tolist()
            if verbose:
                print(f"model: {tested_model}, index= [{model_indices[0]} .. {model_indices[-1]}], i.e: {len(model_indices)} patterns from {test_mod_df.shape[0]}")
            
            experiment_name_in_sdir = f'Test-trained-on_{mod_from_training}'
            experiment_name_out_sdir = f'Test-trained-on_{tested_model}'
    
            case_in_dir = os.path.join(data_out_dir, base_case_to_explore, f'{cnn_name_base}', experiment_name_in_sdir)
    
            if verbose:
                print(f"Repertoire de Test du cas (doit exister): {case_in_dir}")
            if not os.path.exists(case_in_dir):
                print(f"\n *** Case Test directory '{case_in_dir}/' not found. Skiping model case ...")
                continue
            else:
                if trained_with_all :
                    print(f'\n ** Repertoire du cas: {case_in_dir}/ Ok. Training was realized using all forcings (and a single model for 2nd Val.)\n'+\
                          '    Inversion of Model {mod_to_invert} will be realized using {experiment_name_in_sdir} saved Net ...\n')
    
            yhat_file = f"hist-hat_{n_nnets}-nnets_{net_label}_test-on-{set_prefix_to_test.upper()}-set_ST.p"
            yhat_filename = os.path.join(case_in_dir,yhat_file)
    
            yHat = pickle.load( open(yhat_filename, "rb" ) )
    
            GHG_ens_ab_arr_m = GHG_ens_all_but_df[[tested_model]].transpose().iloc[:,-lenDS:].values
            AER_ens_ab_arr_m = AER_ens_all_but_df[[tested_model]].transpose().iloc[:,-lenDS:].values
            NAT_ens_ab_arr_m = NAT_ens_all_but_df[[tested_model]].transpose().iloc[:,-lenDS:].values

            #if verbose: 
            print(f"yHat file '{yhat_file}' found for model {tested_model}")
            print(f"Test yHat size: {yHat.shape}")
    
            if forcings_as_df :
                n_for_mod = test_HIST_df[lambda DF: DF['model'] == tested_model].shape[0]
                
                GHG_mod = test_GHG_df[lambda DF: DF['model'] == tested_model].drop(columns='model').values[:,-lenDS:]
                AER_mod = test_AER_df[lambda DF: DF['model'] == tested_model].drop(columns='model').values[:,-lenDS:]
                NAT_mod = test_NAT_df[lambda DF: DF['model'] == tested_model].drop(columns='model').values[:,-lenDS:]
                HIST_mod = test_HIST_df[lambda DF: DF['model'] == tested_model].drop(columns='model').values[:,-lenDS:]

                GHGu_mod = np.unique(GHG_mod,axis=0)
                AERu_mod = np.unique(AER_mod,axis=0)
                NATu_mod = np.unique(NAT_mod,axis=0)
                            
                if show_unique:
                    GHGu = np.unique(test_GHG_df.drop(columns='model').values[:,-lenDS:],axis=0)  # reduit les profils GHG aux profils uniques
                    AERu = np.unique(test_AER_df.drop(columns='model').values[:,-lenDS:],axis=0)  # reduit les profils AER aux profils uniques
                    NATu = np.unique(test_NAT_df.drop(columns='model').values[:,-lenDS:],axis=0)  # reduit les profils NAT aux profils uniques
                    HISTu = np.unique(test_HIST_df.drop(columns='model').values[:,-lenDS:],axis=0)  # reduit les profils historiques aux profils uniques
                else:
                    GHGu = test_GHG_df.drop(columns='model').values[:,-lenDS:]        # tous les profils GHG, meme dupliques
                    AERu = test_AER_df.drop(columns='model').values[:,-lenDS:]        # tous les profils AER, meme dupliques
                    NATu = test_NAT_df.drop(columns='model').values[:,-lenDS:]        # tous les profils NAT, meme dupliques
                    HISTu = test_HIST_df.drop(columns='model').values[:,-lenDS:]      # tous les profils HIST, meme dupliques

                if verbose:
                    print(f"HIST_mod size: {HIST_mod.shape}")
                    print(f"HISTu size:      {HISTu.shape}")

            else:
                # index list of data for current model
                model_indices = test_mod_df[lambda DF: DF['model'] == tested_model].index.values.tolist()
                if verbose:
                    print(f"model: {tested_model}, index= [{model_indices[0]} .. {model_indices[-1]}], i.e: {len(model_indices)} patterns from {test_mod_df.shape[0]}")
        
                NAT2_T  = test_NAT[:,-lenDS:]
                GHG2_T  = test_GHG[:,-lenDS:]
                AER2_T  = test_AER[:,-lenDS:]
                HIST2_T = test_HIST[:,-lenDS:]
                
                GHG_mod = GHG2_T[model_indices,:]            
                AER_mod = AER2_T[model_indices,:]            
                NAT_mod = NAT2_T[model_indices,:]            
                HIST_mod = HIST2_T[model_indices,:]
                
                GHGu_mod = np.unique(GHG_mod,axis=0)
                AERu_mod = np.unique(AER_mod,axis=0)
                NATu_mod = np.unique(NAT_mod,axis=0)

                if show_unique:
                    GHGu = np.unique(GHG2_T,axis=0)  # reduit les profils GHG aux profils uniques
                    AERu = np.unique(AER2_T,axis=0)  # reduit les profils AER aux profils uniques
                    NATu = np.unique(NAT2_T,axis=0)  # reduit les profils NAT aux profils uniques
                    HISTu = np.unique(HIST2_T,axis=0)  # reduit les profils historiques aux profils uniques
                else:
                    GHGu = GHG2_T      # tous les profils, meme dupliques
                    AERu = AER2_T      # tous les profils, meme dupliques
                    NATu = NAT2_T      # tous les profils, meme dupliques
                    HISTu = HIST2_T      # tous les profils, meme dupliques

                if verbose:
                    print(f"HIST2_T size:    {HIST2_T.shape}")
                    print(f"HIST_mod size:   {HIST_mod.shape}")
                    print(f"HISTu size:      {HISTu.shape}")

            HIST2mu_T,unique_indices = np.unique(HIST_mod, axis=0, return_index=True)
            unique_indices = np.sort(unique_indices).tolist()
            if verbose:
                print(f"HIST2mu_T size: {HIST2mu_T.shape}")            
                    
            local_hist_mod_alpha = gt.reduce_alpha(HIST2mu_T.shape[0], hist_mod_alpha)
            print(f"local hist_mod_alpha (initially at: {hist_mod_alpha}) for n={HIST2mu_T.shape[0]}={local_hist_mod_alpha}")
    
    
            YHATm = np.mean(yHat[:,model_indices,:],axis=1)
            YHATs = local_std_coeff * np.std(yHat[:,model_indices,:],axis=1,ddof=1)
    
            HIST2mum = HIST2mu_T.mean(axis=0)
            HIST2mus = local_std_coeff * HIST2mu_T.std(axis=0,ddof=1)
            

            if show_x_all_forcings :
                # TOUS les profils GHG+AER+NAT en background (de tous les modeles) 
                for iforc,(forc, c_forc) in enumerate(zip(gan_forcing_names, gan_forcing_bg_colors)) :
                    if forc == 'ghg': FORCu = GHGu
                    elif forc == 'aer': FORCu = AERu
                    elif forc == 'nat': FORCu = NATu
                    
                    #h0gan = ax.plot(train_years,FORCu.T, color=c_forc, lw=forc_bg_lw, alpha=1.0)
                    h0gan = ax.plot(train_years,FORCu.T, color=c_forc, ls='--', lw=forc_bg_lw, alpha=forc_bg_alpha)

            x_forc_label = ''
            if show_x_forcings :
                # TOUS les profils GHG+AER+NAT (du modele en cours)
                h1f = []
                x_forc_label = ''
                for iforc,(forc, c_gan_forc, c_forc) in enumerate(zip(gan_forcing_names, gan_forcing_colors, forcing_colors[:3])) :
                    if forc == 'ghg': FORC_mod = GHGu_mod
                    elif forc == 'aer': FORC_mod = AERu_mod
                    elif forc == 'nat': FORC_mod = NATu_mod
            
                    #h0gan = ax.plot(train_years,FORCu.T, color=c_forc, lw=forc_bg_lw, alpha=1.0)
                    h1gan = ax.plot(train_years,FORC_mod.T, color=c_gan_forc, ls='-', lw=forc_lw, alpha=forc_alpha)
                    if i_mod_tested == 0 : print('h1gan',forc,':',h1gan)
                    
                    if iforc > 0:
                        x_forc_label += ', '
                    x_forc_label += f"{FORC_mod.shape[0]} {forc.upper()}"
                    h1f.append(h1gan if np.isscalar(h1gan) else h1gan[0] )
            
            # TOUS les profils HIST en background (de tous les modeles)
            h0 = ax.plot(train_years,HISTu.T, color=hist_bg_color, ls='--', lw=hist_bg_lw, alpha=hist_bg_alpha)

            # profils du modele en cours
            h1 = ax.plot(train_years,HIST2mu_T.T, lw=hist_mod_lw, ls='-', color=hist_mod_color, alpha=local_hist_mod_alpha);
            
            if show_x_forcings :
                # TOUS les profils GHG+AER+NAT (du modele en cours)
                h3f = []
                for iforc,(forc, c_gan_forc, c_forc) in enumerate(zip(gan_forcing_names, gan_forcing_colors, forcing_colors[:3])) :
                    if forc == 'ghg': FORC_mod = GHG_mod
                    elif forc == 'aer': FORC_mod = AER_mod
                    elif forc == 'nat': FORC_mod = NAT_mod
                    
                    FORC_mod_m = FORC_mod.mean(axis=0)
                    FORC_mod_s = local_std_coeff * FORC_mod.std(axis=0,ddof=1)
            
                    # profil moyen du modele en cours
                    if plot_forc_shaded_region :
                        ax.fill_between(train_years, FORC_mod_m - FORC_mod_s, FORC_mod_m + FORC_mod_s,
                                        ec=c_forc, fc=gt.lighter_color(c_forc), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                        hatch=hatch_for_shaded_forc, zorder=2)
                        h3gan = ax.fill(np.NaN, np.NaN, 
                                     ec=c_forc, fc=gt.lighter_color(c_forc), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                     hatch=hatch_for_shaded_forc)
                        h3ganp = ax.plot(train_years, FORC_mod_m, color=c_forc, lw=1.5, ls='-')
                        
                        h3f.append(h3gan if np.isscalar(h3gan) else h3gan[0])
                    else:
                        h3gan,hc,hb = ax.errorbar(train_years, FORC_mod_m, yerr=FORC_mod_s, color=c_forc, lw=1.5, elinewidth=1.0, ls='-')

                        h3f.append(h3gan)

                # -------------------------------------------------------------------------
                # X reference (or Background for inversion)
                # -------------------------------------------------------------------------
                hfro = ax.plot(train_years, GHG_ens_ab_arr_m.T, c=gt.darker_color(forcing_mean_colors[0]), ls='--', lw=1.5)
                #               label="$FORC_{ref.(other)}$")
                ax.plot(train_years, AER_ens_ab_arr_m.T, c=gt.darker_color(forcing_mean_colors[1]), ls='--', lw=1.5)
                ax.plot(train_years, NAT_ens_ab_arr_m.T, c=gt.darker_color(forcing_mean_colors[2]), ls='--', lw=1.5)


            # profil moyen du modele en cours
            if plot_forc_shaded_region :
                ax.fill_between(train_years, HIST2mum - HIST2mus, HIST2mum + HIST2mus,
                                ec=hist_mean_mod_color, fc=gt.lighter_color(hist_mean_mod_color), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                hatch=hatch_for_shaded_forc, zorder=2)
                h3 = ax.fill(np.NaN, np.NaN, 
                              ec=hist_mean_mod_color, fc=gt.lighter_color(hist_mean_mod_color), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                              hatch=hatch_for_shaded_forc)
                h3p = ax.plot(train_years, HIST2mum, color=hist_mean_mod_color, lw=1.5, ls='-')

                if not np.isscalar(h3) :
                    h3 = h3[0]

            else:
                h3,hc,hb = ax.errorbar(train_years, HIST2mum, yerr=HIST2mus, color=hist_mean_mod_color, lw=1.5, elinewidth=1.0, ls='-')
    
            # reset matplotlib color cycle
            plt.gca().set_prop_cycle(None)
            if not errorbars_on_pred :
                # plot mean 
                h2 = ax.plot(train_years,YHATm.T, alpha=yhat_alpha, lw=yhat_lw, ls='-')
            else:
                hx = []
                for inet in np.arange(YHATm.shape[0]):
                    if plot_pred_shaded_region :
                        hp0, = ax.plot(train_years, FORC_mod_m, lw=0.5, ls='-')
                        cc = hp0.get_color()

                        ax.fill_between(train_years, YHATm[inet,:] - YHATs[inet,:], YHATm[inet,:] + YHATs[inet,:],
                                        ec=cc, fc=gt.lighter_color(cc), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                        hatch=hatch_for_shaded_pred, zorder=2)
                        ax.fill(np.NaN, np.NaN, 
                                     ec=cc, fc=gt.lighter_color(cc), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                     hatch=hatch_for_shaded_pred)
                        hp, = ax.plot(train_years, YHATm[inet,:], color=cc, lw=1.5, ls='-')
                    else:
                        hp,hc,hb = ax.errorbar(train_years+(inet-YHATm.shape[0]/2)/8, YHATm[inet,:], yerr=YHATs[inet,:], alpha=yhat_alpha, lw=0, elinewidth=yhat_elw, ls='-')

                    hx.append(hp)
                    
                h2 = []
                for inet,hh in enumerate(hx):
                    cc = hh.get_color()
                    hy, = ax.plot(train_years,YHATm[inet,:], color=cc, alpha=yhat_alpha, lw=yhat_lw, ls='-')
                    h2.append(hy)
            
            if do_stat_test :
                hhist_stat = ax_stat.plot(train_years, HIST2mum, color= 'black', lw=1.5, ls='-')
                hcnn, = ax_stat.plot(train_years, YHATm[inet,:], color=cc, lw=1.5, ls='-')
                
                # label
                ax_stat.set_title(f"{tested_model} ({len(model_indices)} profils - Uniques {x_forc_label}, {HIST2mu_T.shape[0]} HIST)")
                ax_stat.legend([hhist_stat[0],hcnn],['HIST','CNN'], loc='upper left')
                ax_stat.grid(True, lw=0.5, ls=':')
                ax_stat.set_xlim([train_years[0],train_years[-1]])
                ax_stat.hlines(0,train_years[0],train_years[-1], lw=0.5, ls='-', color='k')

            if int(np.ceil((i_mod_tested+1)/ncols)) == nrows :
                ax.set_xlabel('years')
            
            ax.set_title(f"{tested_model} ({len(model_indices)} profils - Uniques {x_forc_label}, {HIST2mu_T.shape[0]} HIST)")
    
            xlim = ax.get_xlim()
            ax.hlines(0,xlim[0],xlim[1], lw=0.5, ls='-', color='k')
            ax.set_xlim(xlim)

            if fixed_t_limits_ok :
                ax.set_ylim([t_min_limit,t_max_limit])
            
            ax.grid(True, lw=0.5, ls=':')
    
            if i_mod_tested == 0 :
                if show_x_forcings :
                    print('h1f:',h1f)
                    print('h1: ',h1[0])
                    print('h3f:',h3f)
                    print('hfro:',hfro)
                    print('h3: ',h3)
                    print('h2: ',h2)
                    list_of_handlers = [*h1f, h1[0], *h3f, hfro[0], h3, *h2]
                    list_of_labels = ['$GHG_{mod}$', '$AER_{mod}$', '$NAT_{mod}$', '$HIST_{mod}$', 
                                      '$mean$ $GHG_{mod}$', '$mean$ $AER_{mod}$', '$mean$ $NAT_{mod}$',
                                      "$\\widebar{FORC}_{other}}$", '$mean$ $HIST_{mod}$']+[f'N{i}' for i in np.arange(YHATm.shape[0])]

                    print(f"{len(list_of_handlers)} Legend handlers, {len(list_of_labels)} labels.")
                    ax.legend(list_of_handlers,list_of_labels,
                              ncol=3,
                              loc='upper left')
                else:
                    ax.legend([h0[0], h1[0], h3, *h2], [f'{tiny_data_label} HIST', 'Model HIST', 'Model mean HIST']+[f'N{i}' for i in np.arange(YHATm.shape[0])],
                              ncol=2,
                              loc='upper left')
        
        # efface axes non utilisees
        if i_mod_tested < n_models_to_plot - 1 :
            for ax in axes.flatten()[-((ncols*nrows)-n_models_to_plot):] : 
                ax.set_visible(False)
                

        suptitle_label = f"Mean CNN output by Model on Model and all ({tiny_data_label}) on {set_prefix_to_test.upper()} set"
        if plot_forc_shaded_region:  suptitle_label += f" [Shaded {interval_label}]"
        else:                        suptitle_label += f" [ErrBar {interval_label}]"
        suptitle_label_bis = "- HIST profiles on background"+(" [NHLpFilt]" if lp_nathist_filtering else "")+f" [{n_nnets} nets]"+\
            (" [Train w/All Models]" if trained_with_all else "")+f" [{net_label.upper()}]"

        if ncols == 1:
            suptitle_label = f"{suptitle_label}\n{suptitle_label_bis} -"
        else:
            suptitle_label = f"{suptitle_label} {suptitle_label_bis}"
        #plt.suptitle(f"{suptitle_label}\n{base_case_to_explore}\n{cnn_name_base}", size=suptitle_fontsize,y=0.99)
    
        if save_figs :
            if force_write or not os.path.isfile(figs_filename):
                print(f"-- saving figure in file ... '{figs_filename}'")
                fig.savefig(figs_filename, **figs_defaults)
            else:
                print(f" ** {local_nb_label} figure already exists '{figs_filename}'. Figure prepared but not saved")
        else:
            print(' ** figure not saved. Saving not active **')
            
        if save_figs_stat :
            if force_write or not os.path.isfile(figs_filename_stat):
                print(f"-- saving stat figure in file ... '{figs_filename_stat}'")
                fig.savefig(figs_filename_stat, **figs_defaults)
            else:
                print(f" ** {local_nb_label} figure already exists '{figs_filename_stat}'. Figure prepared but not saved")
        else:
            print(' ** Stat figure not saved. Saving not active **')
    
        fig.show()