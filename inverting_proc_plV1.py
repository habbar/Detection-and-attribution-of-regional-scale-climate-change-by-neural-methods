"""
Inverting models function for PL
Contains inverting functions:
    
    # MODEL_CNN_INVERSE:
    #
    # Usage:
    #    Y_hat,X,i = model_cnn_inverse(X_i, X_reg, Y, CNN, dev, ...)
    #    Y_hat,X,i,loss_list = model_cnn_inverse (X_i, X_reg, Y, CNN, dev, ..., , ret_loss=True)
    # DO_COMPUTE_RMSE_WITH_X_INV:
    #
    # Compute RMSE and STD between variables ...
    #
    # Usage:
    #    std_FORC_Xinv, global_std_FORC_Xinv, rmse_FORC_X_Xinv,
    #        rmse_FORC_Xref_Xinv, rmse_HIST_Yinv = do_compute_rmse_with_X_inv(data_X_arr, X_ENS_M, mltlist_HIST_m,
    #                                                                         mltlist_Xinv, mltlist_Yinv, mltlist_nbe,
    #                                                                         verbose=False, debug=False)
    # INVERSION_PROCEDURE:
    #
    # Usage:
    #    inv_label = def inversion_procedure (base_case_to_explore, sub_case_to_explore, inversion_suffix, do_inversion=False,
    #                          take_xo_for_cost=False, models_for_inversion=None,
    #                          trained_with_all=False, sample_model=None,
    #                          data_in_dir=None, data_out_dir=None, load_best_val=False, load_best_val2nd=False,
    #                          n_inv_by_page = None, add_mean_of_all = False,
    #                          inv_n_iter=2000, inv_lr_reg=0.1, inv_lr_opt=0.01, inv_alpha=0.1, inv_delta_loss_limit=1e-6, inv_delta_limit_patience=10,
    #                          train_years=np.arange(1900,2015), 
    #                          lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
    #                          verbose_inv=False,  verbose=False,
    #                          buid_individual_model_x_ens=True,
    #                          #obs_filename = 'obs.npy', obs_name = 'OBS',
    #                          obs_set=['OBS','HadCRUT'],
    #                          lp_obs_filtering=False, lp_obs_filtering_dictionary=None,
    #                          force_inverse=False,
    #                         )
    # SET_INV_LABL_FROM_PATH:
    #
    # Usage:
    #    inv_label = set_inv_labl_from_path (case_out_dir, verbose=False)
    # PLOT_INVERSION_FORCINGS:
    #
    # Base function for plotting one inverted pattern as well as inverted inputs and all data related.
    #
    # Usage:
    #    X_RMSE, Y_RMSE = plot_inversion_forcings(X_arr, Y_hat, X_Ens_m_arr, HIST_m_arr, X_inv, Y_inv, X_mod_arr,
    #                             train_years=np.arange(1900,2015),
    #                             forcing_names=['ghg', 'aer', 'nat'],
    #                             plot_x_inv_shaded_region=False, plot_x_mod_shaded_region=False, plot_y_inv_shaded_region=False,
    #                             plot_mean_x_inv=False, plot_mean_y_inv=False, plot_mean_data_xmod=False,
    #                             errorlimits_percent=None, errorlimits_n_rms=1,
    #                             forcings_t_limits=None,
    #                             alpha_forc_inv=0.3,  ls_forc_inv='-',    lw_forc_inv=0.75,   c_ghg_inv=None,    c_aer_inv=None, c_nat_inv=None,
    #                             alpha_forc_ini=0.5,  ls_forc_ini=':',    lw_forc_ini=1.0,    c_ghg_ini=None,    c_aer_ini=None, c_nat_ini=None,
    #                             alpha_forc_ref=0.3,  ls_forc_ref='--',   lw_forc_ref=1.0,    c_ghg_ref=None,    c_aer_ref=None, c_nat_ref=None,
    #                             alpha_forc_mod=0.3,  ls_forc_mod='-',    lw_forc_mod=1.0,    c_ghg_mod=None,    c_aer_mod=None, c_nat_mod=None,
    #                             alpha_hist_to_inv=1, ls_hist_to_inv='-', lw_hist_to_inv=1.0, c_hist_to_inv=None,
    #                             alpha_hist_inv=0.3,  ls_hist_inv='-',    lw_hist_inv=0.75,   c_hist_inv=None, 
    #                             alpha_hist_ini=0.3,  ls_hist_ini=':',    lw_hist_ini=1.0,    c_hist_ini=None,
    #                             hist_obs_label=None,
    #                             title_label=None,
    #                             ninv_short_label=None,
    #                             ax=None,
    #                             return_pc_error_dic=False,
    #                             verbose =False)
    
    # PLOT_INVERTED_HIST_PROFILES_BY_NET:
    #
    # Usage
    #    plot_inverted_hist_profiles_by_net (base_case_to_explore, sub_case_to_explore, inversion_suffix, models_to_plot=None, ...)
    # PLOT_AVERAGED_INVERTED_PROFILES_BY_NET:
    #
    # Usage
    #    plot_averaged_inverted_profiles_by_net (base_case_to_explore, sub_case_to_explore, inversion_suffix, models_to_plot=None, ...)
                                        
    # PLOT_AVERAGED_INV_ALL_FORCINGS_BY_MODEL:
    #
    # Usage
    #    plot_averaged_inv_all_forcings_by_model (base_case_to_explore, sub_case_to_explore, inversion_suffix, models_to_plot=None, ...)
    # PLOT_AVERAGED_INV_ALL_FORCINGS_BY_NET:
    #
    # Usage
    #    plot_averaged_inv_all_forcings_by_net (base_case_to_explore, sub_case_to_explore, inversion_suffix, models_to_plot=None, ...)
    # GIVE_N_INSIDE_CONFIDENCE:
    #
    # Function preparing the number of inverted data by year and by forcing
    # that fits in a interval arround the mean of the data.
    #
    # Usage:
    #    dic_n_in_interval = give_n_inside_confidence(X_data, X_inv,
    #                               std_coeff=1,
    #                               forcing_names=['ghg', 'aer', 'nat'],
    #                               get_for_mean=False,
    #                               verbose=False)
    # PLOT_N_INSIDE_CONFIDENCE
    #
    # Plot the figure about the Percent of inverted forcings profiles inside an
    # interval arround the data mean, plotting by forcing. Possibility to plot
    # for the mean by forcing of all inverted profiles.
    #
    # Usage:
    #    plot_n_inside_confidence(X_data, X_inv,
    #                            train_years=np.arange(1900,2015),
    #                            forcing_names=['ghg', 'aer', 'nat'],
    #                            errorlimits_percent=None, errorlimits_n_rms=1,
    #                            forcings_t_limits=None,
    #                            alpha_forc_inv=0.3,  ls_forc_inv='-',    lw_forc_inv=0.75,   c_ghg_inv=None,    c_aer_inv=None, c_nat_inv=None,
    #                            alpha_forc_mod=0.3,  ls_forc_mod='-',    lw_forc_mod=1.0,    c_ghg_mod=None,    c_aer_mod=None, c_nat_mod=None,
    #                            current_lw=1, current_ls='-', current_alpha=0.8,
    #                            use_step=False,
    #                            title_label=None,
    #                            ninv_short_label=None,
    #                            axes=None,
    #                            get_for_mean=False, plot_for_mean=None,
    #                            return_pc_error_dic=False,
    #                            legend_loc='lower left',
    #                            verbose =False,
    #                           ) :
Created on Wed Aug 24 17:05:27 2022
@author: carlos.mejia@locean.ipsl.fr
"""
import numpy as np
from IPython.display import display


def model_cnn_inverse(Xo, X_reg, Y, CNN, dev, lr_reg=0.1, alpha=0.1, n_iter=200, lr_opt=0.01,
                      ret_loss=True, loss_trace=False, delta_loss_limit=1e-7,
                      patience=0, ret_loss0et1=False,
                      # patience_delete=True,    # option not implemeted yet
                      verbose=False,
                      ):
    
    #, take_xo_for_cost=False
    
    import numpy as np
    import torch.nn as nn
    from torch.autograd import Variable
    from torch.optim import Adam

    criterion = nn.MSELoss()

    X = Variable(Xo.to(dev).clone().detach(),requires_grad=True).to(dev)
    
    optimizer = Adam([X], lr=lr_opt)

    #if take_xo_for_cost :
    #    X_reference = Xo
    #    if verbose:
    #        print("Taking Xo as X_reference for inversion!")
    #else:
    X_reference = X_reg
    #    if verbose:
    #        print("Taking X_ref as X_reference for inversion!")

    if ret_loss:
        loss_list = []
    
    if ret_loss0et1:
        loss0_list = []
        loss1_list = []

    loss_back = 1e4
    patience_count = 0
    for i in range(n_iter):
        
        if verbose and i==0:
            print(" -- Inversion internal loop first iter ... ")
        
        Y_hat = CNN(X)
        
        loss0 = criterion(Y_hat, Y)
        loss1 = criterion(X, X_reference)
        
        #loss = (1 - lr_reg)*loss0 + lr_reg * loss1
        loss = loss0 + lr_reg * loss1

        
        if loss_trace :
            print('inversion:',i,loss0,loss1,loss)
        
        #loss = criterion(Y_hat,Y) + lr_reg * criterion(X,X_reference)

        if ret_loss:
            detached_loss = loss.cpu().detach().clone().numpy().tolist()
            #print(detached_loss)
            loss_list.append(detached_loss)
        
        if ret_loss0et1:
            loss0_list.append(loss0.cpu().detach().clone().numpy().tolist())
            loss1_list.append(loss1.cpu().detach().clone().numpy().tolist())

        delta_loss = np.abs(loss.cpu().detach().clone().numpy() - loss_back)
        if delta_loss < delta_loss_limit :
            patience_count += 1
        
        if loss < alpha*alpha or (delta_loss_limit is not None and delta_loss < delta_loss_limit and patience_count > patience) :
            if verbose :
                _tmp_loss = loss.cpu().detach().clone().numpy().tolist()
                print(f" ** Stopping at iteration i={i}/{n_iter}, last loss={_tmp_loss}, delta loss={delta_loss}")

                if loss < alpha*alpha :
                    print(f" ** STOPPING CONDITION OBSERVED:\n **   loss < alpha*alpha: {_tmp_loss} < {alpha*alpha}")
                elif delta_loss_limit is not None and delta_loss < delta_loss_limit and patience_count > patience:
                    print(f" ** STOPPING CONDITION OBSERVED:\n **   delta_loss < delta_loss_limit: {delta_loss} < {delta_loss_limit} (patience: {patience})")
                else:
                    print(f" ** STOPPING REASON NOT CLEAR BUT STOP ANY WAY (!?) ... BIZARRE :\n **   Loss: {_tmp_loss}, alpha*alpha: {alpha*alpha}, delta_loss: {delta_loss}, delta_loss_limit: {delta_loss_limit}, patience: {patience}")

            break
        
        loss_back = loss.cpu().detach().clone().numpy()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    
    if ret_loss0et1 :
        return Y_hat,X,i,loss_list,loss0_list,loss1_list
    elif ret_loss :
        return Y_hat,X,i,loss_list
    else:
        return Y_hat,X,i


def do_compute_nbe_rmse_3forc(i_patt, X_first, mltlist_Xinv, mltlist_nbe, rmse_FORC_X_Xinv, verbose=False, debug=False) :
    import numpy as np
    
    n_forc = X_first.shape[1]
    
    #rmse_GHG_X_Xinv = np.zeros([n_nbe]); rmse_AER_X_Xinv = np.zeros([n_nbe]); rmse_NAT_X_Xinv = np.zeros([n_nbe]); 
    #X_first = X_guess.cpu().detach().clone().numpy()
    for i_nbe,(X_inv,k_nbe) in enumerate(zip(mltlist_Xinv,mltlist_nbe)):
        #if type(X_first) is list :
            
        if debug and not i_patt:
            print('Method2(inside):',i_patt,i_nbe,k_nbe,':',end=' ')
        for i_forc in np.arange(n_forc):
            rmse_FORC_X_Xinv[i_forc,i_patt,i_nbe] = np.sqrt(((X_first[0,i_forc,:] - X_inv[i_patt,0,i_forc,:])**2).mean())
            if debug and not i_patt:
                print(rmse_FORC_X_Xinv[i_forc,i_patt,i_nbe],end=', ' if (i_forc+1) < n_forc else '\n')

    return  # computed values are assigned directly to arrays (present in input arguments by reference)

# do_compute_nbe_rmse_4hist(i_patt, mltlist_HIST_m, mltlist_Yinv, mltlist_nbe, rmse_HIST_Yinv)
def do_compute_nbe_rmse_4hist(i_patt, mltlist_HIST_m, mltlist_Yinv, mltlist_nbe, rmse_HIST_Yinv, verbose=False, debug=False) :
    import numpy as np
        
    #rmse_GHG_X_Yinv = np.zeros([n_nbe]); rmse_AER_X_Yinv = np.zeros([n_nbe]); rmse_NAT_X_Yinv = np.zeros([n_nbe]); 
    #X_first = X_guess.cpu().detach().clone().numpy()
    for i_nbe,(HIST_m,Y_inv,k_nbe) in enumerate(zip(mltlist_HIST_m,mltlist_Yinv,mltlist_nbe)):
        #if type(X_first) is list :
        if debug and not i_patt:
            print('HIST_m,Y_inv shapes:',HIST_m.shape,Y_inv.shape)
            
        if debug and not i_patt:
            print('Method2(inside):',i_patt,i_nbe,k_nbe,':',end=' ')
        rmse_HIST_Yinv[i_patt,i_nbe] = np.sqrt(((HIST_m - Y_inv[i_patt,0,:])**2).mean())
        if debug and not i_patt:
            print(rmse_HIST_Yinv[i_patt,i_nbe])

    return  # computed values are assigned directly to arrays (present in input arguments by reference)


def do_compute_rmse_with_X_inv(data_X_arr, X_ENS_M, mltlist_HIST_m, mltlist_Xinv, mltlist_Yinv, mltlist_nbe, verbose=False, debug=False) :
    import numpy as np

    n_patt,n_forc,n_time_x = data_X_arr.shape
    n_nbe = len(mltlist_nbe)
    n_time_y = mltlist_Yinv[0].shape[2]
    
    mltXinv_arr = np.array(mltlist_Xinv).squeeze(axis=2)
    mltXinv_shape = mltXinv_arr.shape
    mltXinv_arr =  mltXinv_arr.reshape([mltXinv_shape[0]*mltXinv_shape[1], *mltXinv_shape[2:]])
    
    mltYinv_arr = np.array(mltlist_Yinv).squeeze(axis=2)
    mltYinv_shape = mltYinv_arr.shape
    mltYinv_arr =  mltYinv_arr.reshape([mltYinv_shape[0]*mltYinv_shape[1], *mltYinv_shape[2:]])

    global_mean_FORC_Xinv = mltXinv_arr.mean(axis=0)
    global_std_FORC_Xinv = mltXinv_arr.std(axis=0,ddof=1)

    global_mean_HIST_Yinv = mltYinv_arr.mean(axis=0)
    global_std_HIST_Yinv = mltYinv_arr.std(axis=0,ddof=1)
    
    mean_FORC_Xinv = np.zeros([n_nbe,n_forc,n_time_x])
    std_FORC_Xinv = np.zeros([n_nbe,n_forc,n_time_x])
    
    rmse_FORC_X_Xinv = np.zeros([n_nbe,n_forc,n_time_x])
    rmse_FORC_Xref_Xinv = np.zeros([n_nbe,n_forc,n_time_x])
    rmse_HIST_Yinv = np.zeros([n_nbe,n_time_y])

    for i_nbe,(k_nbe,X_inv,HIST_m,Y_inv) in enumerate(zip(mltlist_nbe, mltlist_Xinv, mltlist_HIST_m, mltlist_Yinv)):
        
        mean_FORC_Xinv[i_nbe,:,:] = X_inv[:,0,:].mean(axis=0)
        std_FORC_Xinv[i_nbe,:,:] = X_inv[:,0,:].std(axis=0,ddof=1)
        
        rmse_FORC_X_Xinv[i_nbe,:,:] = np.sqrt(((data_X_arr - X_inv[:,0,:])**2).mean(axis=0))
        rmse_FORC_Xref_Xinv[i_nbe,:,:] = np.sqrt(((X_ENS_M - X_inv[:,0,:])**2).mean(axis=0))
        rmse_HIST_Yinv[i_nbe,:] = np.sqrt(((HIST_m - Y_inv[:,0,:])**2).mean(axis=0))

        if debug and not i_nbe:
            print('computing rmse for i_nbe=',i_nbe,', pattern number nbe=:',k_nbe,':')
            for i_forc in np.arange(n_forc):
                print(f"Forc({i_forc}):",
                      '\n - mean_FORC_Xinv ........',mean_FORC_Xinv[i_nbe,i_forc,:],
                      '\n - std_FORC_Xinv ........',std_FORC_Xinv[i_nbe,i_forc,:],
                      '\n - rmse_FORC_X_Xinv .....',rmse_FORC_X_Xinv[i_nbe,i_forc,:],
                      '\n - rmse_FORC_Xref_Xinv ..',rmse_FORC_Xref_Xinv[i_nbe,i_forc,:])
            print(' - rmse_FORC_Xref_Xinv ..',rmse_HIST_Yinv[i_nbe,:])
            print('mean_FORC_Xinv shape:       ',mean_FORC_Xinv.shape)
            print('std_FORC_Xinv shape:       ',std_FORC_Xinv.shape)
            print('global_mean_FORC_Xinv shape:',global_mean_FORC_Xinv.shape)
            print('global_std_FORC_Xinv shape:',global_std_FORC_Xinv.shape)
            print('global_mean_HIST_Yinv shape:',global_mean_HIST_Yinv.shape)
            print('global_std_HIST_Yinv shape:',global_std_HIST_Yinv.shape)
            print('rmse_FORC_X_Xinv shape:    ',rmse_FORC_X_Xinv.shape)
            print('rmse_FORC_Xref_Xinv shape: ',rmse_FORC_Xref_Xinv.shape)
            print('rmse_HIST_Yinv shape:      ',rmse_HIST_Yinv.shape)
    
    return  std_FORC_Xinv, mean_FORC_Xinv, \
            global_mean_FORC_Xinv, global_std_FORC_Xinv, \
            global_mean_HIST_Yinv, global_std_HIST_Yinv, \
            rmse_FORC_X_Xinv, rmse_FORC_Xref_Xinv, rmse_HIST_Yinv


def do_compute_and_save_inv_stats(data_X_array, X_ENS_M_array, mltlist_HIST_m, mltlist_Xinv, mltlist_Yinv, mltlist_nbe, mltlist_nbe_label, 
                                  years=np.arange(1850,2015), forcings=['ghg','aer','nat'], 
                                  verbose=False, debug=False) :
    
    std_FORC_Xinv, mean_FORC_Xinv, \
        global_mean_FORC_Xinv, global_std_FORC_Xinv, \
            global_mean_HIST_Yinv, global_std_HIST_Yinv, \
                rmse_FORC_X_Xinv, rmse_FORC_Xref_Xinv, \
                    rmse_HIST_Yinv = do_compute_rmse_with_X_inv(data_X_array,
                                                    X_ENS_M_array, mltlist_HIST_m,
                                                    mltlist_Xinv, mltlist_Yinv, mltlist_nbe,
                                                    verbose=verbose, debug=debug)
    if debug :
        print('std_FORC_Xinv shape:       ',std_FORC_Xinv.shape)
        print('global_std_FORC_Xinv shape:',global_std_FORC_Xinv.shape)
        print('global_std_HIST_Yinv shape:',global_std_HIST_Yinv.shape)
        print('rmse_FORC_X_Xinv shape:    ',rmse_FORC_X_Xinv.shape)
        print('rmse_FORC_Xref_Xinv shape: ',rmse_FORC_Xref_Xinv.shape)
        print('rmse_HIST_Yinv shape:      ',rmse_HIST_Yinv.shape)

        i_patt = 0
        for i_nbe,k_nbe in enumerate(mltlist_nbe):
            print('Method1 et Method2:',i_nbe,k_nbe,':')
            print(rmse_FORC_X_Xinv[0,i_patt,i_nbe],rmse_FORC_Xref_Xinv[0,i_patt,i_nbe])
            print(rmse_FORC_X_Xinv[1,i_patt,i_nbe],rmse_FORC_Xref_Xinv[1,i_patt,i_nbe])
            print(rmse_FORC_X_Xinv[2,i_patt,i_nbe],rmse_FORC_Xref_Xinv[2,i_patt,i_nbe])
            print(rmse_HIST_Yinv[i_patt,i_nbe])

    stats_FORC_Xinv_dic = {'title' : 'mean and std of Xinv by forcing by Y inversion',
                         'name' : 'mean_and_std_FORC_Xinv', 
                         'mean' : mean_FORC_Xinv,
                         'std' : std_FORC_Xinv,
                         'years': years[-std_FORC_Xinv.shape[-1]:],
                         'forc' : forcings, 
                         'nbe' : mltlist_nbe, 'nbe_label' : mltlist_nbe_label }

    stats_global_FORC_Xinv_dic = {'title' : 'mean and std of Xinv by forcing for all Y inversions',
                                'name' : 'global_mean_and_std_FORC_Xinv',
                                'mean' : global_mean_FORC_Xinv,
                                'std' : global_std_FORC_Xinv,
                                'years': years[-global_std_FORC_Xinv.shape[-1]:],
                                'forc' : forcings }

    stats_global_HIST_Yinv_dic = {'title' : 'mean and std of Yinv for all Y inversions',
                                'name' : 'global_mean_and_std_HIST_Yinv',
                                'mean' : global_mean_HIST_Yinv,
                                'std' : global_std_HIST_Yinv,
                                'years': years[-global_std_HIST_Yinv.shape[-1]:] }

    rmse_FORC_X_Xinv_dic = {'title' : 'rmse X - Xinv by forcing by Y inversion',
                            'name' : 'rmse_FORC_X_Xinv',
                            'rmse' : rmse_FORC_X_Xinv,
                            'years': years[-rmse_FORC_X_Xinv.shape[-1]:],
                            'forc' : forcings, 
                            'nbe' : mltlist_nbe, 'nbe_label' : mltlist_nbe_label }

    rmse_FORC_Xref_Xinv_dic = {'title' : 'rmse of Xref - Xinv by forcing by Y inversion',
                               'name' : 'rmse_FORC_Xref_Xinv',
                               'rmse' : rmse_FORC_Xref_Xinv,
                               'years': years[-rmse_FORC_Xref_Xinv.shape[-1]:],
                               'forc' : forcings, 
                               'nbe' : mltlist_nbe, 'nbe_label' : mltlist_nbe_label }

    rmse_HIST_Yinv_dic = {'title' : 'rmse of Yobs - Yinv by Y inversion',
                          'name' : 'rmse_HIST_Yinv',
                          'rmse' : rmse_HIST_Yinv,
                          'years': years[-rmse_HIST_Yinv.shape[-1]:],
                          'nbe' : mltlist_nbe, 'nbe_label' : mltlist_nbe_label }

    return stats_FORC_Xinv_dic, stats_global_FORC_Xinv_dic, stats_global_HIST_Yinv_dic, rmse_FORC_X_Xinv_dic, rmse_FORC_Xref_Xinv_dic, rmse_HIST_Yinv_dic


def do_get_ninv_to_invert(cnn, x_arr=None, hist_arr=None, train_years=np.arange(1850,2015),
                          dtype=None, device='cpu',
                          nbest_to_choose=None, choose_profiles_by_proximity_criterion='rmse', period_to_choose=None, 
                          verbose=False):
    import torch
    from torch.utils.data import DataLoader

    import generic_tools_pl as gt   # like hexcolor(), ...
    import specific_nn_tools_pl as nnt  # like CustomDataset(), Net(), ...

    if x_arr is None or hist_arr is None:
        raise Exception("*** Both: 'x_arr' and 'hist_arr' arguments must be filled ***\n")

    if dtype is None:
        dtype = torch.float
    
    choose_profiles_by_proximity = nbest_to_choose is not None

    if period_to_choose is not None :
        choose_in_a_period = True
        period_init_year,period_last_year = period_to_choose    # -1 means last year
        if period_last_year is None: period_last_year = -1  # Last year
    else:
        choose_in_a_period = False
        period_to_choose_label = f"Yall"
        period_to_choose_short_label = f"yall"

    n_patt = x_arr.shape[0]
    
    if choose_profiles_by_proximity:
        # Execute a preavious forward pass with all X profiles in order to evalaute the 
        # Y_hat obtained nearest to the HIST to invert.
        # This, in order to select only a set of "best profiles to invert" from all the
        # X patterns available.
        take_all_sims = False

        if verbose :
            print(f" Choosing profiles by proximity is {choose_profiles_by_proximity} ... taking only {nbest_to_choose} X profiles, from nearest Y ones, in place of all {n_patt} existing, for inversion.")
        # nbest_to_choose

        # setting the atchsize as huge as the nomber of patterns, there will be only one big block of 
        # patterns later using the for loop with the XForForward DataLoader tensor array.
        XForForward = DataLoader(nnt.CustomDatasetInv(torch.tensor(x_arr[:,0,:].copy(), dtype=dtype).to(device),
                                                      torch.tensor(x_arr[:,1,:].copy(), dtype=dtype).to(device),
                                                      torch.tensor(x_arr[:,2,:].copy(), dtype=dtype).to(device)),
                                  batch_size=n_patt)

        for i,x_for_forward in enumerate(XForForward) :
            if i > 0:
                print(f"\n *** do_get_ninv_to_invert: unexpected more than one iteration. Stopping ***\n")
                raise

            Y_hat_ini = cnn(x_for_forward).cpu().clone().detach().numpy()
        

        # there is normally only one element in the FOR loop, no need to initialize and append ...
        Y_hat_ini_arr = Y_hat_ini

        if verbose :
            print(f"Output Y_hat_ini_arr shape: {Y_hat_ini_arr.shape}")
        
        if choose_in_a_period  :
            if verbose :
                print(f"Choosing in a period having year limits: {period_init_year} and {period_last_year}\n")

            start_iperiod = period_init_year - train_years[0]
            if period_last_year >=  train_years[0] :
                stop_iperiod = period_last_year - train_years[0]
            elif period_last_year < 0:
                stop_iperiod = period_last_year
            else:
                print(f"\n *** period_last_year, {period_last_year}, must be greater than train_years[0], {train_years[0]}, or negatif (relatif) ***\n")
                raise

            period_to_choose_label = f"Y{period_init_year}-{period_last_year if period_last_year > 0 else train_years[period_last_year]}"
            period_to_choose_short_label = f"y{start_iperiod}-{stop_iperiod if stop_iperiod > 0 else 'end'}"

            if len(hist_arr.shape) == 1:
                tmp_HIST_m_arr = hist_arr[start_iperiod:stop_iperiod]
            elif len(hist_arr.shape) == 2:
                tmp_HIST_m_arr = hist_arr[:,start_iperiod:stop_iperiod]
            else:
                print(f"\n *** choosing in a period impossible: unexpected HIST_m_arr dimensions ({Y_hat_ini_arr.shape}) ***\n")
                raise

            if len(Y_hat_ini_arr.shape) == 2:
                tmp_Y_hat_ini_arr = Y_hat_ini_arr[:,start_iperiod:stop_iperiod]
            elif len(Y_hat_ini_arr.shape) == 3:
                tmp_Y_hat_ini_arr = Y_hat_ini_arr[:,:,start_iperiod:stop_iperiod]
            else:
                print(f"\n *** choosing in a period impossible: unexpected Y_hat_ini_arr dimensions ({Y_hat_ini_arr.shape}) ***\n")
                raise
        else:
            tmp_HIST_m_arr = hist_arr
            tmp_Y_hat_ini_arr = Y_hat_ini_arr
            period_to_choose_label = f"yAll"
            period_to_choose_short_label = f"yall"

        if choose_profiles_by_proximity :

            if choose_profiles_by_proximity_criterion.lower() == 'rmse' :
                # compute the RMSE between the HIST or Obs to invert and the Y_hat predicted profiles. Sorted RMSE, the lower RMSE the nearest
                tmp_Y_MSE_ini = np.square(np.subtract(tmp_HIST_m_arr,tmp_Y_hat_ini_arr)).mean(axis=1)
                evaluated_criterion = np.sqrt(tmp_Y_MSE_ini)   # the evaluated_criterion is the RMSE between HIST to invert and Y_hat
                sorted_criterion_index = np.argsort(evaluated_criterion)
                sorted_criterion = evaluated_criterion[sorted_criterion_index]
            elif choose_profiles_by_proximity_criterion.lower() == 'dist' :
                sorted_criterion,sorted_criterion_index = gt.sorted_distance_vectors(tmp_HIST_m_arr, tmp_Y_hat_ini_arr, axis=1, sort=True, ret_b=False)
                print(sorted_criterion_index)
                xxxsorted_criterion,xxxsorted_criterion_index = gt.sorted_distance_vectors(hist_arr, Y_hat_ini_arr, axis=1, sort=True, ret_b=False)
                print(xxxsorted_criterion_index)
            
        else:
            # particular case, no sorting on any criterion
            sorted_criterion_index = np.arange(n_patt)
            sorted_criterion = sorted_criterion_index
    
        if verbose :
            print(" The first in the list (index/criterion): ",sorted_criterion_index[:15],sorted_criterion[:15])
            print(" and the last of the list: ",sorted_criterion_index[-15:],sorted_criterion[-15:])

        if nbest_to_choose > 0 :
            ninv_subset_label = 'nearest'
        elif nbest_to_choose < 0:
            ninv_subset_label = 'farthest'
        else:
            print(f"\ *** error unexpected  [nbest_to_choose == 0]  ({nbest_to_choose}) ***\n")
            raise

        extended_ninv_subset_label = ninv_subset_label
        if choose_profiles_by_proximity_criterion.lower() != 'dist':
            extended_ninv_subset_label += f"-by-{choose_profiles_by_proximity_criterion.lower()}"

        if nbest_to_choose == 1 :
            ninv = sorted_criterion_index[:1]  # uniquement le premier
            ninv_filename_label = f"patt-{ninv[0]}-{extended_ninv_subset_label}"
        elif nbest_to_choose == -1 :
            ninv = sorted_criterion_index[-1:]  # uniquement le dernier
            ninv_filename_label = f"patt-{ninv[0]}-{extended_ninv_subset_label}"
        elif nbest_to_choose > 0 :
            ninv = sorted_criterion_index[:nbest_to_choose] # les N plus proches HIST (les plus bases RMSE or DIST entre la valeur HIST a inverser et les Yhat (HIST calcules par la fonction directe)
            ninv_filename_label = f"the-{len(ninv)}-{extended_ninv_subset_label}"+(f"-[{'-'.join([str(n) for n in ninv])}]" if len(ninv) < 6 else "")
        else :
            ninv = sorted_criterion_index[nbest_to_choose:] # les N plus eloignes HIST ...
            ninv_filename_label = f"the-{len(ninv)}-{extended_ninv_subset_label}"+(f"-[{'-'.join([str(n) for n in ninv])}]" if len(ninv) < 6 else "")

        ninv_short_label = f"x-patt-[{','.join([str(n) for n in ninv])}]" if len(ninv) < 6 else f"{len(ninv)}-patt"

        if choose_in_a_period :
            ninv_filename_label += f'-In-{period_to_choose_label}'
            ninv_short_label += f'-{period_to_choose_short_label}'
            
    else:
        if verbose :
            print(f" Choosing profiles by proximity is {choose_profiles_by_proximity} ... Taking all {n_patt} X profiles for inversion.")
        take_all_sims = True

        ninv = np.arange(n_patt) # tous les combinaisons de X disponibles ...
        ninv_filename_label = f"the-all-{len(ninv)}-sims"

        ninv_short_label = f"all-x-patt"
    
    if verbose :
        print(f" do_get_ninv_to_invert: Take_all_sims is {take_all_sims}:\n  ninv: {ninv},\n  nb ninv: {len(ninv)},")
        print(f"  ninv_filename_label: {ninv_filename_label},\n  ninv_short_label: {ninv_short_label}")
        print(f"  period_to_choose_label: {period_to_choose_label},\n  period_to_choose_short_label: {period_to_choose_short_label}")

    return ninv, ninv_filename_label, ninv_short_label, period_to_choose_label, period_to_choose_short_label



def inversion_procedure (base_case_to_explore, sub_case_to_explore, inversion_suffix, do_inversion=False,
                         take_xo_for_cost=False, models_for_inversion=None,
                         list_of_trained_models=None,           # valid only in case of "experiences jumelles" and Obs inversion
                         trained_with_all=False, sample_model=None,
                         data_in_dir=None, data_out_dir=None, load_best_val=False, load_best_val2nd=False,
                         source_dirname='data_source_pl',
                         nbest_to_choose=None, choose_profiles_by_proximity_criterion='rmse', period_to_choose=None, 
                         n_inv_by_page = None, add_mean_of_all = False,
                         inv_n_iter=2000, inv_lr_reg=0.1, inv_lr_opt=0.01, inv_alpha=0.1, inv_delta_loss_limit=1e-6, inv_delta_limit_patience=10,
                         train_years=np.arange(1900,2015), 
                         lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
                         verbose_inv=False,  verbose=False, debug=False,
                         buid_individual_model_x_ens=True,
                         #obs_filename = 'obs.npy', obs_name = 'OBS',
                         obs_set=['OBS','HadCRUT','HadCRUT200','HadCRUT+AR1'],
                         number_of_multiobs=None, multiobs_choice_method='random', multiobs_random_seed=0,
                         lp_obs_filtering=False, lp_obs_filtering_dictionary=None,
                         force_inverse=False,
                         default_device='gpu', ngpu=1,   # ngpu value is discarded if default_device is not 'gpu'
                        ) :
    import os
    import numpy as np
    import pickle
    import time

    import torch
    from torch.utils.data import DataLoader

    import generic_tools_pl as gt   # like hexcolor(), ...
    import specific_nn_tools_pl as nnt  # like CustomDataset(), Net(), ...
    
    # Repertoire des donnees
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=None, verbose=verbose)

    if data_out_dir is None :
        data_out_dir = './data_out'

    choose_profiles_by_proximity = nbest_to_choose is not None

    if nbest_to_choose is not None :
        nbest_to_choose_label = f"X-{nbest_to_choose}-profiles-from-Nearest-Y"
        nbest_to_choose_short_label = f"XNearest{nbest_to_choose}"
        
        if period_to_choose is not None  :
            period_init_year,period_last_year = period_to_choose    # -1 at end year means last year
            if period_last_year is None: period_last_year = -1  # Last year
            if period_init_year < train_years[0]:
                raise Exception("*** inversion_procedure: period_to_choose[0] < train_years[0]. Must be equal or greater ***\n")
            start_iperiod = period_init_year - train_years[0]
            if period_last_year >=  train_years[0] :
                stop_iperiod = period_last_year - train_years[0]
            elif period_last_year < 0:
                stop_iperiod = period_last_year
            else:
                print(f"\n *** period_last_year, {period_last_year}, must be greater than train_years[0], {train_years[0]}, or negatif (relatif) ***\n")
                raise
    
            period_to_choose_label = f"year{train_years[start_iperiod]}-{train_years[stop_iperiod]}"
            period_to_choose_short_label = f"y{train_years[start_iperiod]}-{train_years[stop_iperiod]}"
        else:
            period_to_choose_label = f"yAll"
            period_to_choose_short_label = f"yall"

    # Retrieving parameters from base_case_to_explore label:
    # (example: 'out_v5_nn6-TTDGM2-S0-GAN85pc-N132_v4_12mod_NewNet')
    #  - 'n_nnets',
    #  - 'data_gener_method',
    #  - 'seed_value',
    #  - 'gan_train_percent_value',
    #  - 'gan_test_percent_value',
    #  - 'data_and_training_label',
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = gt.retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    n_nnets = base_case_dic['n_nnets']
    data_and_training_label = base_case_dic['data_and_training_label']
    lp_nathist_filtering = base_case_dic['lp_nathist_filtering']
    if verbose:
        print(f" - Number of NNets: {n_nnets}")
        print(f" - Data and Training set Llabel: {data_and_training_label}")
        print(f" - Low-pass NAT and HIST filtering: {lp_nathist_filtering}")

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
    extrap_label = sub_case_dic['extrapolation_label']
    if verbose:
        print(f" - List of sizes of CNN hidden layers or (kernel sizes): {kern_size_list}")
        print(f" - Data Extrapolation Label: {extrap_label}")

    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore
    
    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    load_data_and_gener_params = {'data_dir':data_in_dir, 'verbose':verbose}
    if lp_nathist_filtering:
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)

        load_data_and_gener_params['to_filter'] = ['nat', 'hist']
        load_data_and_gener_params['filtering_dic_filename'] = filtering_dic_filename
    load_data_and_gener_params['verbose'] = verbose
 
    data_dic = gt.load_basic_data_and_gener_df(**load_data_and_gener_params)

    #data_label       = data_dic['label']
    all_models_src   = data_dic['models']
    #all_forcings_src = data_dic['forcings']
    #all_forcing_color_dic = data_dic['forcing_color_dic']
    #all_forcing_inv_color_dic = data_dic['forcing_inv_color_dic']
    #all_forcing_color_names_dic = data_dic['forcing_color_names_dic']
    #all_forcing_inv_color_names_dic = data_dic['forcing_inv_color_names_dic']
    #all_years        = data_dic['years']
    T_ghg_df,T_aer_df,T_nat_df,T_hist_df = data_dic['list_of_df']
    
    #forcing_names = all_forcings_src[:4]
    #forcing_colors = [all_forcing_color_dic[f.lower()] for f in forcing_names]

    if verbose:
        print('\nDF data sizes (GHG,AER,NAT,HIST):',
              T_ghg_df.drop('model', axis=1).values.shape,
              T_aer_df.drop('model', axis=1).values.shape,
              T_nat_df.drop('model', axis=1).values.shape,
              T_hist_df.drop('model', axis=1).values.shape)    
    
    # Compute intermodel mean and transpose to have models as columns ...
    GHG_ens_df  = T_ghg_df.groupby('model').mean().transpose()[all_models_src]
    AER_ens_df  = T_aer_df.groupby('model').mean().transpose()[all_models_src]
    NAT_ens_df  = T_nat_df.groupby('model').mean().transpose()[all_models_src]
    HIST_ens_df = T_hist_df.groupby('model').mean().transpose()[all_models_src]

    if verbose:
        print("\nDefinitive GHG DataFrame (models are arranged in the order in 'all_models_src' variable):")
        print(GHG_ens_df.head())


    inversion_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix='inversion', 
                                                           set_label=inversion_suffix,
                                                           verbose=verbose)

    # Load forcing data as DataFrames
    model_names, all_years, inversion_mod_df, \
        data_inversion_dic = gt.load_forcing_data(data_in_dir, file_prefix='inversion',
                                                  dataframe=True,
                                                  set_label=inversion_suffix, forcing_names=inversion_combi_dic['forcings'],
                                                  verbose=verbose)
    if list_of_trained_models is None :
        list_of_trained_models = model_names
        
    if verbose :
        print(data_inversion_dic.keys())
        print(type(data_inversion_dic['ghg']),data_inversion_dic['ghg'].shape)
    inversion_GHG_df = data_inversion_dic['ghg']
    inversion_AER_df = data_inversion_dic['aer']
    inversion_NAT_df = data_inversion_dic['nat']

    if verbose:
        print(f"\ninversion_combi_dic models .....  {inversion_combi_dic['models']}")
        print(  f"load_forcing_data model_names ..  {model_names}")

    if models_for_inversion is None :
        models_for_inversion = inversion_combi_dic['models']
        #models_for_inversion = model_names
        #models_for_inversion = all_models_src
        #models_for_inversion = GHG_ens_all_but_df.transpose().index.values.tolist()
        #models_for_inversion = ['IPSL-CM6A-LR']
        #models_for_inversion = ['FGOALS-g3']
    if verbose:
        print(f"\nList of models found in Inversion data set:\n  {models_for_inversion}")

    # Build "all_but" DataFrames
    add_obs_options = {}
    for curent_obs_name in obs_set :
        if curent_obs_name in models_for_inversion :
            if 'obsname' in add_obs_options.keys() :
                if type(add_obs_options['obsname']) is list:
                    list_of_obs_names = add_obs_options['obsname']+[curent_obs_name]
                else :
                    list_of_obs_names = [add_obs_options['obsname'], curent_obs_name]
                add_obs_options['obsname'] = list_of_obs_names
            else:
                add_obs_options = { 'add_for_obs':True, 'obsname':curent_obs_name } 
            if verbose:
                print(f"adding {curent_obs_name} column to model list for 'all_but' data - add_obs_options:",add_obs_options)
        # if obs_name in models_for_inversion :
        #     add_obs_options = { 'add_for_obs':True, 'obsname':obs_name } 
        #     if verbose:
        #         print(f"adding {obs_name} column to model list for 'all_but' data")
    #
    #display(GHG_ens_df)
    print('add_obs_options:',add_obs_options)
    GHG_ens_all_but_df, AER_ens_all_but_df, NAT_ens_all_but_df, \
        HIST_ens_all_but_df = gt.build_all_but_df(all_models_src,
                                                  GHG_ens_df, AER_ens_df,
                                                  NAT_ens_df, HIST_ens_df,
                                                  **add_obs_options,
                                                  verbose=verbose)
    #display(GHG_ens_all_but_df)
    print('\n all_models_src .................. ',len(all_models_src),all_models_src)
    print(' model_names ..................... ',len(model_names),model_names)
    print(' current models_for_inversion .... ',len(models_for_inversion),models_for_inversion)
    print(' current list_of_trained_models .. ',len(list_of_trained_models),list_of_trained_models)
    print(' GHG_ens_df.columns .............. ',len(GHG_ens_df.columns),GHG_ens_df.columns.values)
    print(' GHG_ens_all_but_df.columns ...... ',len(GHG_ens_all_but_df.columns),GHG_ens_all_but_df.columns.values)
    
    #all_years = train_combi_dic['years']
    #train_years = np.arange(1900,2015)

    lenDS = len(train_years)

    cnn_name_base = sub_case_to_explore

    n_to_add = np.sum([k//2 for k in kern_size_list])

    inv_label = f'Lr{inv_lr_reg}-Opt{inv_lr_opt}-A{inv_alpha}_{inversion_suffix}'
    
    if not do_inversion :
        print(f"\n {'*'*16}\n *** NO INVERSION AUTHORIZED ***")
        print(f" *** Case output base folder .. '{base_case_to_explore}'")
        print(f" *** CNN case base subfolder .. '{cnn_name_base}'")
        print(f" *** Inversion label .......... '{inv_label}'")
        print(' *** QUITING INVERSION PROCEDURE ...')
        
    else:
        print(f"\nStart Global Inversion process for {len(models_for_inversion)} models, {n_nnets} nnets ...:\n")
        globinvprocess_start = time.time()
        
        if load_best_val2nd :
            net_label = 'best-val2nd'
            net_filename = 'Net_best-val2nd.pt'
        elif load_best_val :
            net_label = 'best-val'
            net_filename = 'Net_best-val.pt'
        else:
            net_label = 'last'
            net_filename = 'Net.pt'
    
        if verbose:
            display(GHG_ens_all_but_df)
        
        models_in_invert_data = GHG_ens_all_but_df.transpose().index.values.tolist()
    
        if not verbose:
            print(f"\nList of Models for Inversion:\n   {models_for_inversion}")
        else:
            print(f"List of Models found loading data (alphabetical order) ... {model_names}")
            print(f"List of Models for Inversion ............................. {models_for_inversion}")
            print(f"List of Models found in Inversion data ................... {models_in_invert_data}")
        
        if trained_with_all :
            print(f"Single training with all models ACTIVE using '{sample_model}' as sample model/")
    
        try:
    
            # On charge dans le device: gpu/mps/cpu
            dtype = torch.float
            device = nnt.get_device_auto(device=default_device, ngpu=ngpu)
            if verbose :
                print('Currently used device is :', device)    
    
            print(f"\nCurrent case:\n   '{cnn_name_base}' in folder {base_case_to_explore}/")
    
            if not trained_with_all :
                # Looking for already trained models
                single_models_already_trained = []
                for i_mod,xmod_from_training in enumerate(list_of_trained_models) :
                    xexperiment_name_in_sdir = f'Training-for-mod_{xmod_from_training}'
                    xcase_in_dir = os.path.join(data_out_dir, base_case_to_explore, cnn_name_base, xexperiment_name_in_sdir)
                    xnet_dir = os.path.join(xcase_in_dir,f'CNN_N{0}')
                    if os.path.exists(xnet_dir):
                        single_models_already_trained.append(xmod_from_training)
            
                print('single_models_already_trained:',single_models_already_trained)
                
            subsettings_inv_sdir_dic = {}

            for i_mod,mod_to_invert in enumerate(models_for_inversion) :
                #print('i_mod,mod_to_invert:',i_mod,mod_to_invert)

                if mod_to_invert not in models_in_invert_data :
                    print(f"\n ** error: model name '{mod_to_invert}' not expected. Not in models for invert data list **\n"+\
                          f" ** {models_in_invert_data}\n"+\
                          " ** Going next ...\n")
                    continue

                if mod_to_invert in obs_set:
                    print(f"\nLoading {mod_to_invert} as Obs, for inversion:")
                    
                    obs_df = gt.load_obs_data(obs_label=mod_to_invert, data_dir=data_in_dir,
                                              #mobs_nb=number_of_multiobs, mobs_method=multiobs_choice_method, mobs_rnd_seed=multiobs_random_seed,
                                              return_as_df=True, verbose=verbose)
                    
                    if verbose :
                        display(obs_df)

                    if lp_obs_filtering :
                        print("LP filtering ... ")
                        obs_df = gt.filter_forcing_df(obs_df, filt_dic=lp_obs_filtering_dictionary, verbose=verbose)
                        if verbose :
                            print("AFTER Filtering:")
                            display(obs_df)

                # Prepare really trained model to test with current mod_to_invert source data
                # ---------------------------------------------------------------------------
                # Its normally the same. The trained model is the same than mod_to_invert unless
                # it could represents un Observations profile (OBS, HadCRUT), not a model !
                # In this case and if trained_with_all flag is True this Obs profile will be inverted using with the uniq trained, trained for all models.
                # But if the training was realised for each model separately (is the case of twin experiences or "experiences jumelles") this Obs
                # profile will be inverted several time one with each separately trained model.
                if trained_with_all :
                    models_from_training = sample_model
                elif mod_to_invert in single_models_already_trained :  # OBS or other not in Training
                     models_from_training = mod_to_invert
                else :
                    #models_from_training = list_of_trained_models
                    models_from_training = single_models_already_trained
                
                if np.isscalar(models_from_training) :
                    models_from_training = [ models_from_training ]

                #print('models_from_training:',models_from_training)
                
                print(f"\n {'-'*132}\n -\n - Inversion case {i_mod+1}/{len(models_for_inversion)}) for modele {mod_to_invert}"+
                      f" (inversion case: '{inversion_suffix}') - [{net_label}]"+\
                      f"{' - lp_nathist_filtering' if lp_nathist_filtering else ''}\n -\n {'-'*132}\n")
                
                for i_trmod,mod_from_training in enumerate(models_from_training) :
                                        
                    experiment_name_in_sdir = f'Training-for-mod_{mod_from_training}'
                    if trained_with_all :
                        experiment_name_in_label = 'Training for mod UNIQUE'
                    else:
                        experiment_name_in_label = experiment_name_in_sdir.replace('-', ' ')
                    
                    experiment_name_out_sdir = f'Inversion-on_{mod_to_invert}'
                    if not trained_with_all and mod_to_invert != mod_from_training :
                        experiment_name_out_sdir += f"-{mod_from_training}"
                    
                    # Repertoires du cas (sorties de l'etape d'entrainement des CNN)
                    case_in_dir = os.path.join(data_out_dir, base_case_to_explore, cnn_name_base, experiment_name_in_sdir)

                    print(f"\n {'*'*132}")
                    if mod_to_invert == mod_from_training :
                        print(f" Inversion case {i_mod+1}/{len(models_for_inversion)}) for model '{mod_to_invert}' using the trained Net for the model ...")
                    else:
                        print(f" Inversion case {i_mod+1}/{len(models_for_inversion)} (sub-case {i_trmod+1}/{len(models_from_training)}) for model '{mod_to_invert}'", end='')
                        print(f" using a trained Net from model '{mod_from_training}' ...")
                    print(f" and trained Net in dir: '{case_in_dir}")
                    print(f" {'*'*132}\n")
                    
                    # if mod_to_invert not in models_in_invert_data :
                    #     print(f"\n ** error: model name '{mod_to_invert}' not expected. Not in models for invert data list **\n"+\
                    #           f" ** {models_in_invert_data}\n"+\
                    #           f" ** Going next ...\n")
                    #     continue
    
                    if mod_to_invert in model_names and not trained_with_all and not os.path.isdir(case_in_dir):
                        print(f'\n *** Repertoire du cas: {case_in_dir}/ innexistant !!!'+\
                              '\n     Avez-vous execute au prealable le Notebook'+\
                              ' Train_CNN_MultiModDF_working_version_ST.ipynb ?'+\
                              '\nSkiping model ...\n')
                        
                        #raise Exception('case_in_dir not found')
                    else:
                        if trained_with_all :
                            print(f"\n ** Repertoire du cas: '{case_in_dir}/' Ok. Training was realized using a single model but all forcings\n"+\
                                  f"    Inversion of Model '{mod_to_invert}' will be realized using '{experiment_name_in_sdir}' saved Net ...\n")
        
                        case_out_dir = os.path.join(data_out_dir, base_case_to_explore, cnn_name_base, experiment_name_out_sdir)
                        if verbose:
                            print(f'Repertoire de sortie du cas: {case_out_dir}')
                        if not os.path.exists(case_out_dir):
                            os.makedirs(case_out_dir)
        
                        suptitlelabel = f"{cnn_name_base} [{experiment_name_in_label}] [{data_and_training_label}] ({n_nnets} Nets)"
                        if verbose:
                            print(suptitlelabel)
        
                        if buid_individual_model_x_ens :    # NEW way, stacking torch tensors for individual model G.A.N. forcings data
                        
                            print(f" Inverse model '{mod_to_invert}' extracted from profiles and form the 'other_arr' matrix for inversion ...")
                            GHG_ens_ab_arr_m = GHG_ens_all_but_df[[mod_to_invert]].transpose().iloc[:,-(lenDS+n_to_add*2):].values
                            AER_ens_ab_arr_m = AER_ens_all_but_df[[mod_to_invert]].transpose().iloc[:,-(lenDS+n_to_add*2):].values
                            NAT_ens_ab_arr_m = NAT_ens_all_but_df[[mod_to_invert]].transpose().iloc[:,-(lenDS+n_to_add*2):].values
                            
                            X_ENS_M_array = np.concatenate((np.expand_dims(GHG_ens_ab_arr_m,axis=0),
                                                            np.expand_dims(AER_ens_ab_arr_m,axis=0),
                                                            np.expand_dims(NAT_ens_ab_arr_m,axis=0)),
                                                           axis=1)
                            
                            if verbose:
                                print("X_ENS_M_arr.shape=",X_ENS_M_array.shape)
                            
                            X_ENS_M = torch.tensor(X_ENS_M_array, dtype=dtype).to(device)
                        
                            if verbose:
                                print('X_ENS_M Tensor shape:',X_ENS_M.shape)
                                
                        # if Model to Inverse is not one from those used for training, because Obs or another Model !
                        if mod_to_invert not in inversion_GHG_df['model'].unique() :
                            inv_GHG_other_arr = inversion_GHG_df.drop('model', axis=1).values.copy()
                            inv_AER_other_arr = inversion_AER_df.drop('model', axis=1).values.copy()
                            inv_NAT_other_arr = inversion_NAT_df.drop('model', axis=1).values.copy()
                            inv_FORC_other_index = inversion_NAT_df.drop('model', axis=1).index.values.copy()
                        else:
                            inv_GHG_other_arr = inversion_GHG_df.loc[lambda df: df['model'] != mod_to_invert, :].drop('model', axis=1).values.copy()
                            inv_AER_other_arr = inversion_AER_df.loc[lambda df: df['model'] != mod_to_invert, :].drop('model', axis=1).values.copy()
                            inv_NAT_other_arr = inversion_NAT_df.loc[lambda df: df['model'] != mod_to_invert, :].drop('model', axis=1).values.copy()
                            inv_FORC_other_index = inversion_NAT_df.loc[lambda df: df['model'] != mod_to_invert, :].drop('model', axis=1).index.values.copy()

                        if lp_nathist_filtering :
                            from scipy import signal
                            b_lp_filter, a_lp_filter = signal.butter(4, [1./10.], btype='lowpass')
                            if verbose:
                                print("Low-Pass NAT filtering having shape: {inv_NAT_other_arr.shape}")
                                print('AVANT:',inv_NAT_other_arr.mean(axis=1).mean(),inv_NAT_other_arr.std(axis=1,ddof=1).mean())
        
                            inv_NAT_other_arr = signal.filtfilt(b_lp_filter, a_lp_filter, inv_NAT_other_arr)
                            
                            if verbose:
                                print('APRES:',inv_NAT_other_arr.mean(axis=1).mean(),inv_NAT_other_arr.std(axis=1,ddof=1).mean())
        
                        data_X_array = np.concatenate((np.expand_dims(inv_GHG_other_arr[:,-(lenDS+n_to_add*2):],axis=1),
                                                       np.expand_dims(inv_AER_other_arr[:,-(lenDS+n_to_add*2):],axis=1),
                                                       np.expand_dims(inv_NAT_other_arr[:,-(lenDS+n_to_add*2):],axis=1)),
                                                      axis=1)
                        
                        # data_X tensor will be formed later
                        
                        if verbose:
                            print("data_X_array.shape=",data_X_array.shape)
        
                        
                        # data_X = DataLoader(nnt.CustomDatasetInv(torch.tensor(data_X_array[:,0,:].copy(), dtype=dtype).to(device),
                        #                                           torch.tensor(data_X_array[:,1,:].copy(), dtype=dtype).to(device),
                        #                                           torch.tensor(data_X_array[:,2,:].copy(), dtype=dtype).to(device)),
                        #                     batch_size=1)
                        
                        if nbest_to_choose is None :
                            nb_data_X_array = data_X_array.shape[0]
                        else:
                            nb_data_X_array = nbest_to_choose

                        ## ##########################################################
                        ## Données d'Inversion: Sortie $Y_{mod}$
                        ## ##########################################################
                        # Profils HIST du modele en cours .. selection des lignes du modele ... retire la colone 'model' ... selectionne les colones (selon lenDS) ... values (array)
                        if mod_to_invert in obs_set :
                            print(f"\n Profils HIST to invert is {mod_to_invert}, obs_df.shape: {obs_df.shape}!!")
                            #display(obs_df)
                            #if obs_df.shape[0] == 1 :
                            #    # normalement un sel profil avec comme index le nom des obs
                            #    HIST_mod_for_inv_arr = obs_df.loc[mod_to_invert, :].values.copy()
                            #else:
                            # normalement tous les profils dans obs_df sont a inverser
                            HIST_mod_for_inv_arr = obs_df.values.copy()
            
                        else:
                            # Profils HIST du modele en cours .. selection des lignes du modele ... retire la colone 'model' ... selectionne les colones (selon lenDS) ... values (array)
                            HIST_mod_for_inv_arr = T_hist_df.loc[lambda df: df['model'] == mod_to_invert, :].drop('model', axis=1).iloc[:,-lenDS:].values.copy()
                            # print(HIST_mod_for_inv_arr)
                        
                                                
                        if len(HIST_mod_for_inv_arr.shape) == 1 :
                            HIST_mod_for_inv_arr = HIST_mod_for_inv_arr.reshape((1,len(HIST_mod_for_inv_arr)))

                        if verbose:
                            print("\nHIST_mod_for_inv_arr shape:",HIST_mod_for_inv_arr.shape)
                            #print(HIST_mod_for_inv_arr[:5,:])
                            
                        #HIST_mod_for_inv_tensor = torch.tensor(HIST_mod_for_inv_arr).float()
        
                        nb_patt4inv,patt_size4inv = HIST_mod_for_inv_arr.shape
                        if verbose:
                            print(f"nb_patt4inv,patt_size4inv: {nb_patt4inv,patt_size4inv}")
                
                        ############### innet ###########
                        for innet in np.arange(n_nnets):
                            #innet = 0
        
                            # On nomme le réseau
                            net_dir = os.path.join(case_in_dir,f'CNN_N{innet}')
                            if not os.path.exists(net_dir):
                                print(f"\n *** Repertoire du reseau innexistant !!!\n ***    '{net_dir}/'"+\
                                      "\n     OBS ou nouvelles donnees ? Ou bien, avez-vous execute au prealable le Notebook d'entrainement pour ce modele ('{mod_to_invert}') ?")
        
                            inv_nnet_dir = f'Inv_N{innet}'

                            inv_dir = os.path.join(case_out_dir, inv_nnet_dir, f'Settings-{inv_label}')
                            if not os.path.exists(inv_dir):
                                os.makedirs(inv_dir)
        
                            print(f" CNN_N{innet}:",end='')
                            print(f" Net_dir: '{net_dir}'")
                            print(f" Inv_dir: '{inv_dir}'")
            
                            # Modif pour tout les historiques
                            i_nb = 0
                            
                            # liste de nbe pour inverser.
                            list_profiles_dic = gt.do_list_of_profiles(nb_patt4inv, n_by_page=n_inv_by_page, 
                                                                       nb_of_profiles=number_of_multiobs,
                                                                       choice_method=multiobs_choice_method,
                                                                       rand_seed=multiobs_random_seed,
                                                                       add_mean_of_all=add_mean_of_all,
                                                                       verbose=verbose)
        
                            list_of_nbe = list_profiles_dic['list']
                            #old_nbe_invert_label = list_profiles_dic['old_label']
                            nbe_invert_label = list_profiles_dic['label']
                            nbe_invert_title = list_profiles_dic['title']
                            #old_nbe_invert_prnt = old_nbe_invert_label.replace(' ','-')
                            nbe_invert_prnt = nbe_invert_label.replace(' ','-')
                                    
                            print(' nbe invert title:',nbe_invert_title)
                            print(' nbe invert label:',nbe_invert_label)
                            print(' list of nbe:',list_of_nbe)
      
                            #old_inv_postfix = f'{old_nbe_invert_prnt}_N{innet}_{net_label}-net_mod-{mod_to_invert}'
                            sdir_inv_postfix = f'{nbe_invert_prnt}_{net_label}-net'
                            if nbest_to_choose is not None : # If chosen, adds labels of number, method and perion of choice
                                sdir_inv_postfix += f'_{nbest_to_choose_short_label}'
                                sdir_inv_postfix += f'_{period_to_choose_short_label}'
                            
                            if mod_to_invert not in subsettings_inv_sdir_dic.keys() :
                                subsettings_inv_sdir_dic[mod_to_invert] = sdir_inv_postfix
                            
                            if not os.path.exists(os.path.join(inv_dir, sdir_inv_postfix)):
                                os.makedirs(os.path.join(inv_dir, sdir_inv_postfix))

                            if verbose:
                                print('sdir_inv_postfix:',sdir_inv_postfix)
                                
                            # Prepare HIST_m for all profiles to invert
                            mltlist_HIST_m = []
                            for nbe in list_of_nbe:
                                #nbe = ind_all_yinv_ini[3,m]
                                #nbe = 0
    
                                if np.isscalar(nbe) :
                                    current_HIST_m = HIST_mod_for_inv_arr[nbe,:]
                                else:
                                    current_HIST_m = HIST_mod_for_inv_arr[nbe,:].mean(axis=0)
                                mltlist_HIST_m.append(current_HIST_m)
                            
                            # old_fileinv_label = "multi_all-but-one"
                            
                            # old_xinv_inv_filename = os.path.join(inv_dir, f'Xinv_{old_fileinv_label}_{old_inv_postfix}_ST.p')
                            # old_yinv_inv_filename = os.path.join(inv_dir, f'Yinv_{old_fileinv_label}_{old_inv_postfix}_ST.p')
                            # old_nbe_inv_filename = os.path.join(inv_dir, f'Nbe_{old_fileinv_label}_{old_inv_postfix}_ST.p')
                            # old_nbelbl_inv_filename = os.path.join(inv_dir, f'NbeLbl_{old_fileinv_label}_{old_inv_postfix}_ST.p')
                            # old_lossinv_filename = os.path.join(inv_dir, f'lossinv_{old_fileinv_label}_{old_inv_postfix}_ST.p')

                            #xinv_inv_filename = os.path.join(inv_dir, sdir_inv_postfix, f'Xinv_ST.p')
                            #yinv_inv_filename = os.path.join(inv_dir, sdir_inv_postfix, f'Yinv_ST.p')
                            #nbe_inv_filename = os.path.join(inv_dir, sdir_inv_postfix, f'Nbe_ST.p')
                            #nbelbl_inv_filename = os.path.join(inv_dir, sdir_inv_postfix, f'NbeLbl_ST.p')
                            #lossinv_filename = os.path.join(inv_dir, sdir_inv_postfix, f'lossinv_ST.p')
                            
                            inversion_dic_filename = os.path.join(inv_dir, sdir_inv_postfix, f'Inv_dic.p')

                            ## for renaming and rearraging OLD inversion files into new place and new name
                            # for fnew,fold in zip([xinv_inv_filename, yinv_inv_filename, nbe_inv_filename, nbelbl_inv_filename, lossinv_filename],
                            #                      [old_xinv_inv_filename, old_yinv_inv_filename, old_nbe_inv_filename, old_nbelbl_inv_filename, old_lossinv_filename]) :
                            #     if os.path.isfile(fold) :
                            #         print(f"\n {'*'*132}\n * Renaming file:\n * - From: '{fold}'\n *   To:   '{fnew}'")
                            #         os.rename(fold,fnew)
                            
                            # noms des fichiers, sans extension, pour sauver la RMS
                            stats_X_inv_filenoext = os.path.join(inv_dir, sdir_inv_postfix, 'stats_Xinv_dic')
                            stats_global_X_inv_filenoext = os.path.join(inv_dir, sdir_inv_postfix, 'stats_global-Xinv_dic')
                            stats_global_Y_inv_filenoext = os.path.join(inv_dir, sdir_inv_postfix, 'stats_global-Yinv_dic')
                            
                            rmse_X_inv_filenoext = os.path.join(inv_dir, sdir_inv_postfix, 'rmse_X-Xinv_dic')
                            rmse_Xref_inv_filenoext = os.path.join(inv_dir, sdir_inv_postfix, 'rmse_Xref-Xinv_dic')
                            rmse_Y_inv_filenoext = os.path.join(inv_dir, sdir_inv_postfix, 'rmse_HIST-Yinv_dic')
                            
                            stats_X_inv_filename        = stats_X_inv_filenoext+'.p'
                            stats_global_X_inv_filename = stats_global_X_inv_filenoext+'.p'
                            stats_global_Y_inv_filename = stats_global_Y_inv_filenoext+'.p'
                            
                            rmse_X_inv_filename         = rmse_X_inv_filenoext+'.p'
                            rmse_Xref_inv_filename      = rmse_Xref_inv_filenoext+'.p'
                            rmse_Y_inv_filename         = rmse_Y_inv_filenoext+'.p'

                            # if not force_inverse and os.path.isfile(xinv_inv_filename) and os.path.isfile(yinv_inv_filename) :
                            #     print(f"\n ** Old Inversion files found:\n **   '{xinv_inv_filename}' and\n **   '{yinv_inv_filename}'\n"+\
                            #           f" ** Reloading ...")
        
                            #     mltlist_Xinv = pickle.load( open( xinv_inv_filename, "rb" ) )
                            #     mltlist_Yinv = pickle.load( open( yinv_inv_filename, "rb" ) )
                            #     #if len(mltlist_lossinv) > 0:
                            #     #    pickle.dump( mltlist_lossinv, open( os.path.join(inv_dir, f'lossinv_{filename_label}_{old_fileinv_label}_ST.p'), "wb" ) )
                            #     mltlist_nbe = pickle.load( open( nbe_inv_filename, "rb" ) )   
                            #     mltlist_nbe_label = pickle.load( open( nbelbl_inv_filename, "rb" ) )
                            #     mltlist_lossinv = pickle.load( open( lossinv_filename, "rb" ) )

                            #     if debug:
                            #         print('mltlist_Xinv type:      [', type(mltlist_Xinv), ' / ', type(mltlist_Xinv[0]), '], len:', len(mltlist_Xinv), ', array[0].shape:', mltlist_Xinv[0].shape)
                            #         print('mltlist_Yinv type:      [', type(mltlist_Yinv), ' / ', type(mltlist_Yinv[0]), '], len:', len(mltlist_Yinv), ', array[0].shape:', mltlist_Yinv[0].shape)
                            #         print('mltlist_nbe type:       [', type(mltlist_nbe), ' / ', type(mltlist_nbe[0]), '], len:', len(mltlist_nbe))
                            #         print('mltlist_nbe_label type: [', type(mltlist_nbe_label), ' / ', type(mltlist_nbe_label[0]), '], len:', len(mltlist_nbe_label))
                            #         print('X_ENS_M type:           [', type(X_ENS_M), '], X_ENS_M.shape:', X_ENS_M.shape)
                            #         print('mltlist_HIST_m type:    [', type(mltlist_HIST_m), ' / ', type(mltlist_HIST_m[0]), '], len:', len(mltlist_HIST_m), ', array[0].shape:', mltlist_HIST_m[0].shape)

                            #     print(f"\n ** Replacing individial Xinv, Yinv, nbe, ... files with a single Inversion_dic file:\n **   '{inversion_dic_filename}'")
                            #     inversion_dic = {'Xinv':mltlist_Xinv,
                            #                      'Yinv':mltlist_Yinv, 
                            #                      'nbe':mltlist_nbe, 
                            #                      'nbe_label':mltlist_nbe_label,
                            #                      'lossinv':mltlist_lossinv}
                            #     pickle.dump( inversion_dic, open( inversion_dic_filename, "wb" ) )
                                
                            #     for f in [xinv_inv_filename, yinv_inv_filename, nbe_inv_filename, nbelbl_inv_filename, lossinv_filename]:
                            #         print(f" ** removing file {f}")
                            #         os.remove(f)
                            
                            if not force_inverse and os.path.isfile(inversion_dic_filename) :
                                print("\n ** Inversion process skiped, Inversion file (with Xinv, Yinv ...) file already exists:    **"+\
                                      f"\n **   '{inversion_dic_filename}'\n"+\
                                      f" ** Reloading ...")
                                    
                                inversion_dic = pickle.load( open( inversion_dic_filename, "rb" ) )
                                
                                mltlist_Xinv = inversion_dic['Xinv']
                                mltlist_Yinv = inversion_dic['Yinv']
                                mltlist_nbe = inversion_dic['nbe']
                                mltlist_nbe_label = inversion_dic['nbe_label']
                                #if 'lossinv' in inversion_dic.keys() :
                                mltlist_lossinv = inversion_dic['lossinv']

                                if debug:
                                    print('mltlist_Xinv type:      [', type(mltlist_Xinv), ' / ', type(mltlist_Xinv[0]), '], len:', len(mltlist_Xinv), ', array[0].shape:', mltlist_Xinv[0].shape)
                                    print('mltlist_Yinv type:      [', type(mltlist_Yinv), ' / ', type(mltlist_Yinv[0]), '], len:', len(mltlist_Yinv), ', array[0].shape:', mltlist_Yinv[0].shape)
                                    print('mltlist_nbe type:       [', type(mltlist_nbe), ' / ', type(mltlist_nbe[0]), '], len:', len(mltlist_nbe))
                                    print('mltlist_nbe_label type: [', type(mltlist_nbe_label), ' / ', type(mltlist_nbe_label[0]), '], len:', len(mltlist_nbe_label))
                                    print('X_ENS_M type:           [', type(X_ENS_M), '], X_ENS_M.shape:', X_ENS_M.shape)
                                    print('mltlist_HIST_m type:    [', type(mltlist_HIST_m), ' / ', type(mltlist_HIST_m[0]), '], len:', len(mltlist_HIST_m), ', array[0].shape:', mltlist_HIST_m[0].shape)

                                if mltlist_Xinv[0].shape[0] != mltlist_Yinv[0].shape[0] or mltlist_Xinv[0].shape[0] != nb_data_X_array :
                                    print(f" ** Nb.of patterns problem:\n"+\
                                        f"     mltlist_Xinv[0].shape[0] != mltlist_Yinv[0].shape[0]: {mltlist_Xinv[0].shape[0]} != {mltlist_Yinv[0].shape[0]} or\n"+\
                                        f"     mltlist_Xinv[0].shape[0] != nb_data_X_array: {mltlist_Xinv[0].shape[0]} != {nb_data_X_array}\n")
                                    raise Exception(f"Nb.of patterns problem:\n"+\
                                                    f" mltlist_Xinv[0].shape[0] != mltlist_Yinv[0].shape[0]: {mltlist_Xinv[0].shape[0]} != {mltlist_Yinv[0].shape[0]} or\n"+\
                                                    f" mltlist_Xinv[0].shape[0] != nb_data_X_array: {mltlist_Xinv[0].shape[0]} != {nb_data_X_array}\n")
                                else:
                                                                        
                                    if not os.path.isfile(stats_X_inv_filename) or not os.path.isfile(stats_global_X_inv_filename) or \
                                        not os.path.isfile(stats_global_Y_inv_filename) or not os.path.isfile(rmse_Y_inv_filename) or \
                                            not os.path.isfile(rmse_X_inv_filename) or not os.path.isfile(rmse_Xref_inv_filename):
                                        print(" ** Attention: Stats and/or RMSE not preaviously computed ... Doing it now and saving ...")
                                        all_inv_stats = do_compute_and_save_inv_stats(data_X_array, X_ENS_M_array, mltlist_HIST_m, mltlist_Xinv,
                                                                                      mltlist_Yinv, mltlist_nbe, mltlist_nbe_label,
                                                                                      years=all_years, forcings=inversion_combi_dic['forcings'], 
                                                                                      verbose=verbose, debug=debug)
                                        
                                        stats_FORC_Xinv_dic, stats_global_FORC_Xinv_dic, \
                                            stats_global_HIST_Yinv_dic, rmse_FORC_X_Xinv_dic, \
                                                rmse_FORC_Xref_Xinv_dic, rmse_HIST_Yinv_dic = all_inv_stats
                                        
                                        pickle.dump( stats_FORC_Xinv_dic,        open( stats_X_inv_filename,        "wb" ) )
                                        pickle.dump( stats_global_FORC_Xinv_dic, open( stats_global_X_inv_filename, "wb" ) )
                                        pickle.dump( stats_global_HIST_Yinv_dic, open( stats_global_Y_inv_filename, "wb" ) )
                                        pickle.dump( rmse_FORC_X_Xinv_dic,       open( rmse_X_inv_filename,         "wb" ) )
                                        pickle.dump( rmse_FORC_Xref_Xinv_dic,    open( rmse_Xref_inv_filename,      "wb" ) )
                                        pickle.dump( rmse_HIST_Yinv_dic,         open( rmse_Y_inv_filename,         "wb" ) )
                                        
                                    else:
                                        print(" ** All RMSE files are already computed ... Skiping and continue ...")

                            else:
                                if verbose or verbose_inv :
                                    print("\n -- Inversion process not skiped because Inversion file (with Xinv, Yinv ...) don't exists.    **"+\
                                          f"\n --   '{inversion_dic_filename}'\n")
                                ################################
                                # 
                                # On charge le réseau
                                # 
                                cnn = torch.load(os.path.join(net_dir,net_filename), map_location=torch.device('cpu')).to(device)
    
                                # ##############################################################################################
                                # Inversion process
                                # ##############################################################################################
        
                                print(f"\n Start Inversion process of {len(list_of_nbe)} simulations and {nb_data_X_array} initial conditions: ", end='')
                                invprocess_start = time.time()
        
                                mltlist_Xinv= []
                                mltlist_Yinv= []
                                mltlist_Ni= []
                                mltlist_lossinv = []
                                mltlist_nbe = []
                                mltlist_nbe_label = []
                                for i_nbe,nbe in enumerate(list_of_nbe):
        
                                    if np.isscalar(nbe) :
                                        nbe_label = f'{nbe}'
                                        #current_HIST_m = HIST_mod_for_inv_tensor[nbe,:]
                                    else:
                                        nbe_label = f"MEAN-{nbe[0]}-to-{nbe[-1]}"
                                        #current_HIST_m = HIST_mod_for_inv_tensor[nbe,:].mean(axis=0)

                                    HIST_M = torch.reshape(torch.tensor(mltlist_HIST_m[i_nbe].copy(), dtype=dtype).to(device),
                                                           (1,len(mltlist_HIST_m[i_nbe]))).to(device)
        
                                    i_nb=i_nb+1
                                    
                                    if verbose:
                                        print(f" Simulation {nbe_label} / {len(list_of_nbe)}:")
                                    else:
                                        print(f" {i_nbe+1}-Sim.{nbe_label}",end='')
                                    
                                    #ninv, ninv_filename_label, ninv_short_label, period_to_choose_label, \
                                    #    period_to_choose_short_label = ...
                                    ninv, _, _, _, _ = do_get_ninv_to_invert(cnn, x_arr=data_X_array, hist_arr=mltlist_HIST_m[i_nbe],
                                                                             train_years=train_years,
                                                                             dtype=dtype, device=device,
                                                                             nbest_to_choose=nbest_to_choose,
                                                                             choose_profiles_by_proximity_criterion=choose_profiles_by_proximity_criterion,
                                                                             period_to_choose=period_to_choose, 
                                                                             verbose=verbose
                                                                             )

                                    data_X = DataLoader(nnt.CustomDatasetInv(torch.tensor(data_X_array[ninv,0,:].copy(), dtype=dtype).to(device),
                                                                             torch.tensor(data_X_array[ninv,1,:].copy(), dtype=dtype).to(device),
                                                                             torch.tensor(data_X_array[ninv,2,:].copy(), dtype=dtype).to(device)),
                                                        batch_size=1)

                                    list_Xinv= []
                                    list_Yinv= []
                                    list_lossinv = []
                                    list_Ni = []
                                    
                                    inversion_start = time.time()
        
                                    for i_inv,X_guess in enumerate(data_X):
        
                                        if verbose_inv :
                                            print(f'inversion_loop({i_inv}) for X_guess/X_ENS_M/HIST_M sizes: {X_guess.shape}/{X_ENS_M.shape}/{HIST_M.shape}')
                                        ###################
                                        #
                                        # do inversion
                                        mod_inv_return_values = model_cnn_inverse(X_guess,  # chaque pattern de data_X: les triplets inv_GHG_tensor,inv_AER_tensor,inv_NAT_tensor pour les autres modèles (en excluant m, le modele IPSL)
                                                                                  X_ENS_M,  # X_ens pour uniquement le modèle m (IPSL), X_ens etant formé par les moyennes des forcages moyens pour tous les modeles autre que m
                                                                                  HIST_M,   # Une simulation Historique du modele m (ici la premiere de m dans HIST_)
                                                                                  cnn,      # modele NN Torch
                                                                                  device,   # device de calcul en cours ('gpu' ou 'cpu')
                                                                                  ret_loss=True, # retour de la loss pour tous ... sinon, i_inv < 20, autorise le retour de la loss pour les 20 premieres inversions
                                                                                  lr_reg=inv_lr_reg,
                                                                                  alpha=inv_alpha,
                                                                                  lr_opt=inv_lr_opt,
                                                                                  n_iter=inv_n_iter,
                                                                                  delta_loss_limit=inv_delta_loss_limit,
                                                                                  verbose=verbose_inv,
                                                                                 )
                                                                                 #take_xo_for_cost=take_xo_for_cost,
    
                                        if len(mod_inv_return_values) == 3:
                                            Y_inv, X_inv, ni = mod_inv_return_values
                                        else:
                                            Y_inv, X_inv, ni, loss = mod_inv_return_values
                                            list_lossinv.append(loss.copy())
                                            if verbose_inv:
                                                print('Y_inv.shape, X_inv.shape, ni, len(loss):',Y_inv.shape, X_inv.shape, ni, len(loss))
                                                print('append(loss)',len(loss),len(list_lossinv))
        
                                        list_Xinv.append(X_inv.detach().cpu().clone().numpy())
                                        list_Yinv.append(Y_inv.detach().cpu().clone().numpy())
                                        list_Ni.append(ni)
                                        
                                    inversion_end = time.time()
                                    inversion_elapsed = inversion_end - inversion_start
                                    
                                    if verbose :
                                        print(f'Temps d\'exécution : {inversion_elapsed:.2f}s')
                                    if verbose :
                                        print(f"Nombre moyen d'iterations: {np.mean(list_Ni):.1f} sur {len(list_Ni)} inversions.")
                                    else:
                                        print(f" ({np.mean(list_Ni):.0f}mi, {inversion_elapsed:.1f}s)", end=", ")
                                    
                                    if verbose:
                                        print('len(list_Xinv), len(list_Yinv):',len(list_Xinv), len(list_Yinv))
    
                                    # On sauve Xinv, de taille nb de membres, 100
                                    list_Xinv = np.array(list_Xinv)
                                    list_Yinv = np.array(list_Yinv)
                                    if verbose:
                                        print('list_Xinv.shape, list_Yinv.shape:',list_Xinv.shape, list_Yinv.shape)
        
                                    list_Xinv0 = list_Xinv.copy()
                                    list_Yinv0 = list_Yinv.copy()
                                    list_lossinv0 = list_lossinv.copy()
        
                                    mltlist_Xinv.append(list_Xinv0)
                                    mltlist_Yinv.append(list_Yinv0)
                                    mltlist_lossinv.append(list_lossinv0)
                                    mltlist_nbe.append(nbe)
                                    mltlist_nbe_label.append(nbe_label)
                                    mltlist_Ni.append(list_Ni)
        
                                invprocess_end = time.time()
                                invprocess_elapsed = invprocess_end - invprocess_start
                                
                                all_inv_stats = do_compute_and_save_inv_stats(data_X_array[ninv,:], X_ENS_M_array, mltlist_HIST_m, mltlist_Xinv, mltlist_Yinv, mltlist_nbe, mltlist_nbe_label,
                                                                              years=all_years, forcings=inversion_combi_dic['forcings'], 
                                                                              verbose=verbose, debug=debug)
                                
                                stats_FORC_Xinv_dic, stats_global_FORC_Xinv_dic, \
                                    stats_global_HIST_Yinv_dic, rmse_FORC_X_Xinv_dic, \
                                        rmse_FORC_Xref_Xinv_dic, rmse_HIST_Yinv_dic = all_inv_stats

                                pickle.dump( stats_FORC_Xinv_dic,        open( stats_X_inv_filename,        "wb" ) )
                                pickle.dump( stats_global_FORC_Xinv_dic, open( stats_global_X_inv_filename, "wb" ) )
                                pickle.dump( stats_global_HIST_Yinv_dic, open( stats_global_Y_inv_filename, "wb" ) )
                                pickle.dump( rmse_FORC_X_Xinv_dic,       open( rmse_X_inv_filename,         "wb" ) )
                                pickle.dump( rmse_FORC_Xref_Xinv_dic,    open( rmse_Xref_inv_filename,      "wb" ) )
                                pickle.dump( rmse_HIST_Yinv_dic,         open( rmse_Y_inv_filename,         "wb" ) )
                                        
                                print(f" <END> of inversion procedure for net {innet} in {invprocess_elapsed:.1f} s")
        
                                if verbose:
                                    print("Saving Xinv, Yinv, Nbe and RMS:")
                                    print(" - Xinv and Yinv in file '{xinv_inv_filename}' ...'")

                                inversion_dic = {'Xinv':mltlist_Xinv,
                                                 'Yinv':mltlist_Yinv,
                                                 'forc_xinv_index':inv_FORC_other_index,
                                                 'nbe':mltlist_nbe,
                                                 'nbe_label':mltlist_nbe_label
                                                 }

                                if len(mltlist_lossinv) > 0:
                                    inversion_dic['lossinv'] = mltlist_lossinv
                                                                
                                print(f"\n ** Saving a single Inversion_dic:\n **   '{inversion_dic_filename}'")
                                pickle.dump( inversion_dic, open( inversion_dic_filename, "wb" ) )
    
            print('<inversion_procedure done>\n')
    
        except Exception as e:
            print(f'\n *** Exception error "{e}" ***\n')
            raise
            
        globinvprocess_end = time.time()
        globinvprocess_elapsed = globinvprocess_end - globinvprocess_start
                            
        if verbose:
            print(f" Global Inversion procedure finishing in {globinvprocess_elapsed:.1f} s using device {device}.")
        else:
            print(f"\n Global Inversion procedure finishing in {globinvprocess_elapsed:.1f} s using device {device}.\n")

    return inv_label, subsettings_inv_sdir_dic


def set_inv_labl_from_path (case_out_dir, settings_label=None, verbose=False) :
    import os
    import glob
    
    try:
        if os.path.isdir(os.path.join(case_out_dir,'Inv_N0')) :
            if settings_label is None :
                settings_pattern = 'Settings-*'
            else:
                settings_pattern = f'Settings-{settings_label}'
            all_in_inv_case = glob.glob(os.path.join(case_out_dir,'Inv_N0',settings_pattern))
            if len(all_in_inv_case) > 0 :
                tmp_case_path,_ = os.path.split(all_in_inv_case[0])
                if len(all_in_inv_case) > 1 :
                    print(f"\n ** Found several Inversion 'Settings-*' sub-folders in path '{tmp_case_path}/':")
                    for tmp_scase_dir in all_in_inv_case :
                        tmp_case_path,tmp_set_dir = os.path.split(tmp_scase_dir)
                        inv_label = tmp_set_dir[len('Settings-'):]
                        print(f" **  - '{inv_label}'")
                    print("**\n ** Choose one using 'inv_model=' option argument.\n")
                    raise Exception("several inversion settings found. Choose one")
                else:
                    _,tmp_settings_dir = os.path.split(all_in_inv_case[0])
                    inv_label = tmp_settings_dir[len('Settings-'):]
                    if verbose:
                        print(f"inv_label found: '{inv_label}'")
                
            else:
                print(f"\n ** No Inversion 'Settings-*' found in path '{case_out_dir}/Inv_N0'")
                print(" ** Is Inversion already done ?\n")
                raise Exception("Cannot determine 'inv_label' because no inversion settings found. Is inversion already done ?")
            
        else:
            print("\n *** No Inversion folders found (using filename pattern '{settings_pattern}') ***")
            print(f" ***    Look for existence of '{case_out_dir}/' directory")
            print(" ***    or ,'Inv_N0' and 'Settings-*' sub-folders")
            print(" ** Is Inversion already done ?\n")
            raise Exception("Cannot determine 'inv_label' because no inversion settings found. Is inversion already done ?")
        
    except Exception as e:
        print(f'\n *** Exception error "{e}" ***\n')
        raise

    return inv_label


def plot_inversion_forcings(X_arr, Y_hat, X_Ens_m_arr, HIST_m_arr, X_inv, Y_inv, X_mod_arr,
                            train_years=np.arange(1900,2015),
                            forcing_names=['ghg', 'aer', 'nat'],
                            plot_x_inv_shaded_region=False, plot_x_mod_shaded_region=False, plot_y_inv_shaded_region=False,
                            plot_mean_x_inv=False, plot_mean_y_inv=False, plot_mean_data_xmod=False,
                            errorlimits_percent=None, errorlimits_n_rms=1,
                            forcings_t_limits=None,
                            alpha_forc_inv=0.4,  ls_forc_inv='-',    lw_forc_inv=1.0,    lw_fill_forc_inv=0.5, hatch_fill_forc_inv='\\\\',
                            c_ghg_inv=None,    c_aer_inv=None, c_nat_inv=None,
                            alpha_forc_ini=0.5,  ls_forc_ini=':',    lw_forc_ini=0.75,
                            c_ghg_ini=None,    c_aer_ini=None, c_nat_ini=None,
                            alpha_forc_ref=0.4,  ls_forc_ref='--',   lw_forc_ref=1.0,
                            c_ghg_ref=None,    c_aer_ref=None, c_nat_ref=None,
                            alpha_forc_mod=0.4,  ls_forc_mod='-',    lw_forc_mod=0.75,   lw_fill_forc_mod=0.5, hatch_fill_forc_mod='//', c_forc_mod_darker_factor=0.5, 
                            c_ghg_mod=None,    c_aer_mod=None, c_nat_mod=None,
                            alpha_hist_to_inv=1, ls_hist_to_inv='-', lw_hist_to_inv=1.0, lw_fill_hist_inv=0.5, hatch_fill_hist_inv='\\\\\\',
                            c_hist_to_inv=None,
                            alpha_hist_inv=0.4,  ls_hist_inv='-',    lw_hist_inv=1.0,
                            c_hist_inv=None,
                            alpha_hist_ini=0.5,  ls_hist_ini=':',    lw_hist_ini=0.75,
                            c_ci_inv="#3F3F3F", c_ci_mod="#4F4F4F", 
                            c_hist_ini=None,
                            hist_obs_label=None,
                            title_label=None,
                            ninv_short_label=None,
                            ax=None,
                            return_pc_error_dic=False,
                            verbose =False,
                            toto=True) :
    """
    Parameters
    ----------
    X_arr : TYPE
        X initial states for inversion (X = [GHG,AER,NAT]).
    Y_hat : TYPE
        Y from forward model on X_arr.
    X_Ens_m_arr : TYPE
        X background for inversion.
    HIST_m_arr : TYPE
        Y to be inverted.
    X_inv : TYPE
        X inverted final states.
    Y_inv : TYPE
        Y inverted.
    X_mod_arr : TYPE
        DESCRIPTION.
    train_years : TYPE, optional
        Years to take for plotting (x axis). The default is np.arange(1900,2015).
    verbose : TYPE, optional
        Verbose flag. The default is False.
    Returns
    -------
    None.
    """
    import math
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    import generic_tools_pl as gt   # like hexcolor(), ...


    lenDS = len(train_years)

    if forcings_t_limits is not None :
        fixed_t_limits_ok = True
        t_min_limit,t_max_limit = forcings_t_limits
    else:
        fixed_t_limits_ok = False

    if ax is None:
        fig,ax = plt.subplots(nrows=1,ncols=1)

    apply_std_coeff = False
    if errorlimits_percent is not None :
        if errorlimits_percent < 1:
            local_ppf = (1 + errorlimits_percent)/2   # Percent point function (inverse of cdf — percentiles).
                                                    # Pour definir un intervalle contenant 90% des valeurs, par exemple,
                                                    # prenant alors entre 5% et 95% de la distribution de probabilités. 
                                                    # Le ppf serait alors 0.95, valeur à passer à la fonction norm.ppf()
                                                    # de SCIPY pour obtenir le coefficient multiplicatif de la std por le 
                                                    # calcul de la taille des barres d'erreur ou largeur de la zone "shaded" (la moitié).
            local_std_coeff = norm.ppf(local_ppf)
        else:
            local_std_coeff = errorlimits_percent
        apply_std_coeff = True
        
    elif errorlimits_n_rms != 1 :
        local_std_coeff = errorlimits_n_rms
        apply_std_coeff = True
    
    if return_pc_error_dic and apply_std_coeff :
        pc_error_dic = { 'std_coef' : local_std_coeff }
    else :
        pc_error_dic = {}
    
    colors_dic = gt.get_forcing_colors()
    
    c_forc_inv = colors_dic['standard']
    c_forc_ref = colors_dic['darker']
    c_forc_ini = [gt.lighter_color(c,factor=0.75) for c in colors_dic['lighter']]

    for iforc,forc in enumerate(colors_dic['order']):
        if forc.lower() == 'ghg' :
            if c_ghg_ref is None: c_ghg_ref = c_forc_ref[iforc]
            if c_ghg_inv is None: c_ghg_inv = c_forc_inv[iforc]
            if c_ghg_ini is None: c_ghg_ini = c_forc_ini[iforc]
            if c_ghg_mod is None:
                c_ghg_mod = gt.darker_color(c_forc_inv[iforc], factor=c_forc_mod_darker_factor)

        elif forc.lower() == 'aer' :
            if c_aer_ref is None: c_aer_ref = c_forc_ref[iforc]
            if c_aer_inv is None: c_aer_inv = c_forc_inv[iforc]
            if c_aer_ini is None: c_aer_ini = c_forc_ini[iforc]
            if c_aer_mod is None:
                c_aer_mod = gt.darker_color(c_forc_inv[iforc], factor=c_forc_mod_darker_factor)

        elif forc.lower() == 'nat' :
            if c_nat_ref is None: c_nat_ref = c_forc_ref[iforc]
            if c_nat_inv is None: c_nat_inv = c_forc_inv[iforc]
            if c_nat_ini is None: c_nat_ini = c_forc_ini[iforc]
            if c_nat_mod is None:
                c_nat_mod = gt.darker_color(c_forc_inv[iforc], factor=c_forc_mod_darker_factor)

        elif forc.lower() == 'hist' :
            if c_hist_to_inv is None: c_hist_to_inv = c_forc_ref[iforc]
            if c_hist_inv is None: c_hist_inv = c_forc_inv[iforc]
            if c_hist_ini is None: c_hist_ini = c_forc_ini[iforc]


    if verbose:
        print("X_arr shape ...", X_arr.shape)
        print("X_inv shape ...", X_inv.shape)
        if X_mod_arr is not None : print("X_mod_arr .....", X_mod_arr.shape)
        else: print("X_mod_arr is None.")
        print("X_Ens_m_arr ...", X_Ens_m_arr.shape)
        print("HIST_m_arr ....", HIST_m_arr.shape)
        print("Y_inv shape ...", Y_inv.shape)
        print("Y_hat shape ...", Y_hat.shape)
        if apply_std_coeff :
            print(f" - Local STD coefficient: {local_std_coeff}")
        else:
            print(f" - Applying STD coefficient: <NOT ACTIVATED>")

    X_arr       = X_arr[:,:,-lenDS:].squeeze()
    X_inv_arr   = X_inv[:,:,-lenDS:].squeeze()
    X_Ens_m_arr = X_Ens_m_arr[:,-lenDS:]
    Y_inv_arr   = Y_inv
    
    if X_mod_arr is not None:
        X_mod_mean  = X_mod_arr.mean(axis=0)
        X_mod_std   = X_mod_arr.std(axis=0,ddof=1)

    if verbose:
        print("X_arr, X_inv_arr, X_Ens_m_arr, Y_inv_arr shapes .....", X_arr.shape, X_inv_arr.shape, X_Ens_m_arr.shape, Y_inv_arr.shape)

    #X_RMSE = [0,0,0]; Y_RMSE = 0; Y_RMSE_ini=0

    #X_MSE = [ np.square(np.subtract(xr,xi)).mean() for xr,xi in zip(X_Ens_m_arr,X_inv_arr) ]
    #X_RMSE = [ math.sqrt(mse) for mse in X_MSE ]
    if len(X_inv_arr.shape) > 2 :
        X_MSE = np.square(np.subtract(X_Ens_m_arr,X_inv_arr)).mean(axis=2).mean(axis=0)
        if X_mod_arr is not None:
            #Xmod_MSE = np.square(np.subtract(X_mod_mean[:,-(lenDS+n_to_add*2):],X_inv_arr)).mean(axis=2).mean(axis=0)
            Xmod_MSE = np.square(np.subtract(X_mod_mean,X_inv_arr)).mean(axis=2).mean(axis=0)

    else:
        X_MSE = np.square(np.subtract(X_Ens_m_arr,X_inv_arr)).mean(axis=1)
        if X_mod_arr is not None:
            #Xmod_MSE = np.square(np.subtract(X_mod_mean[:,-(lenDS+n_to_add*2):],X_inv_arr)).mean(axis=1)
            Xmod_MSE = np.square(np.subtract(X_mod_mean,X_inv_arr)).mean(axis=1)

    X_RMSE = [ math.sqrt(mse) for mse in X_MSE ]
    if X_mod_arr is not None:
        Xmod_RMSE = [ math.sqrt(mse) for mse in Xmod_MSE ]

    Y_MSE = np.square(np.subtract(HIST_m_arr,Y_inv_arr)).mean()
    Y_RMSE = math.sqrt(Y_MSE)

    Y_MSE_ini = np.square(np.subtract(HIST_m_arr,Y_hat)).mean()
    Y_RMSE_ini = math.sqrt(Y_MSE_ini)


    handles_for_legend = []
    # -------------------------------------------------------------------------
    # X initiaux
    # -------------------------------------------------------------------------
    if len(X_arr.shape) == 3 :
        local_alpha_forc_ini = alpha=alpha_forc_ini / np.log(X_arr.shape[0])

        hfi = ax.plot(train_years, X_arr[:,0,:].T, c=c_ghg_ini, ls=ls_forc_ini, lw=lw_forc_ini,
                      alpha=local_alpha_forc_ini, label="$FORC_{INI}$"+(f" ({ninv_short_label})" if ninv_short_label is not None else None))
        ax.plot(train_years, X_arr[:,1,:].T, c=c_aer_ini, ls=ls_forc_ini, lw=lw_forc_ini, alpha=local_alpha_forc_ini)
        ax.plot(train_years, X_arr[:,2,:].T, c=c_nat_ini, ls=ls_forc_ini, lw=lw_forc_ini, alpha=local_alpha_forc_ini)
    else:
        hfi = ax.plot(train_years, X_arr[0,:].T, c=c_ghg_ini, ls=ls_forc_ini, lw=lw_forc_ini, 
                      alpha=alpha_forc_ini, label="$FORC_{INI}$"+(f" ({ninv_short_label})" if ninv_short_label is not None else None))
        ax.plot(train_years, X_arr[1,:].T, c=c_aer_ini, ls=ls_forc_ini, lw=lw_forc_ini, alpha=alpha_forc_ini)
        ax.plot(train_years, X_arr[2,:].T, c=c_nat_ini, ls=ls_forc_ini, lw=lw_forc_ini, alpha=alpha_forc_ini)



    # -------------------------------------------------------------------------
    # Y ini
    # -------------------------------------------------------------------------
    if Y_inv_arr.shape[0] > 1 :
        local_alpha_hist_ini = alpha_hist_ini / (np.log(X_arr.shape[0]) if X_arr.shape[0] > 1 else 1)
        hhini = ax.plot(train_years, Y_hat.T, c=c_hist_ini, ls=ls_hist_ini, lw=lw_hist_ini,
                        alpha=local_alpha_hist_ini, label="$HIST_{INI}"+f"={Y_RMSE_ini:.4f}$")    
    else:
        hhini = ax.plot(train_years, Y_hat[:].T, c=c_hist_ini, ls=ls_hist_ini, lw=lw_hist_ini,
                        alpha=alpha_hist_ini, label="$HIST_{INI}"+f"={Y_RMSE_ini:.4f}$")
   


    # -------------------------------------------------------------------------
    # X inversés
    # -------------------------------------------------------------------------
    hfginv = None
    if len(X_arr.shape) == 3 :
        if plot_x_inv_shaded_region :
            #print('# X inversés ...')
            alpha4mean_forc_inv = 0.8
            alpha4region_forc_inv = alpha_forc_inv/3
            
            for iforc,(c,forc) in enumerate(zip([c_ghg_inv,c_aer_inv,c_nat_inv],forcing_names)):
                local_forc_mean = X_inv_arr[:,iforc,:].mean(axis=0)
                local_forc_std = X_inv_arr[:,iforc,:].std(axis=0,ddof=1)
                if apply_std_coeff:
                    local_forc_std *= local_std_coeff

                ax.fill_between(train_years, local_forc_mean - local_forc_std,
                                local_forc_mean + local_forc_std,
                                alpha=alpha4region_forc_inv, ec=c, fc=gt.lighter_color(c),
                                hatch=hatch_fill_forc_inv, linewidth=lw_fill_forc_inv, zorder=2)  # label=f"${forc} C.I. 90%")
                hr = ax.fill(np.NaN, np.NaN, alpha=alpha4region_forc_inv, ec=c_ci_mod, fc=c_ci_mod, #ec=c, fc=gt.lighter_color(c)
                             hatch=hatch_fill_forc_inv, linewidth=lw_fill_forc_inv, label="$FORC_{INV}$ C.I. 90%")

                hp = ax.plot(train_years, local_forc_mean, c=c, ls=ls_forc_inv, lw=lw_forc_inv,
                             alpha=alpha4mean_forc_inv, label=f"${forc.upper()}"+"_{INV}"+f"={X_RMSE[iforc]:.3f}"+\
                                 (f"/{Xmod_RMSE[iforc]:.3f}$" if X_mod_arr is not None else "$"))
                if iforc == 0: hfginv=hr; hginv=hp
                if iforc == 1: hfainv=hr; hainv=hp
                if iforc == 2: hfninv=hr; hninv=hp
            plot_mean_x_inv = False
            #return X_RMSE, Y_RMSE
        else:
            local_alpha_forc_inv = alpha_forc_inv / (np.log(X_arr.shape[0]) if X_arr.shape[0] > 1 else 1)
            if plot_mean_x_inv :
                local_alpha_forc_inv /= 2
    
            hginv = ax.plot(train_years, X_inv_arr[:,0].T, c=c_ghg_inv, ls=ls_forc_inv, lw=lw_forc_inv,
                            alpha=local_alpha_forc_inv, label="$GHG_{INV}"+f"={X_RMSE[0]:.3f}"+\
                                (f"/{Xmod_RMSE[0]:.3f}$" if X_mod_arr is not None else "$"))
            hainv = ax.plot(train_years, X_inv_arr[:,1].T, c=c_aer_inv, ls=ls_forc_inv, lw=lw_forc_inv,
                            alpha=local_alpha_forc_inv, label="$AER_{INV}"+f"={X_RMSE[1]:.3f}"+\
                                (f"/{Xmod_RMSE[1]:.3f}$" if X_mod_arr is not None else "$"))
            hninv = ax.plot(train_years, X_inv_arr[:,2].T, c=c_nat_inv, ls=ls_forc_inv, lw=lw_forc_inv,
                            alpha=local_alpha_forc_inv, label="$NAT_{INV}"+f"={X_RMSE[2]:.3f}"+\
                                (f"/{Xmod_RMSE[2]:.3f}$" if X_mod_arr is not None else "$"))
    
            if plot_mean_x_inv :
                local_alpha_forc_inv = 0.8
                local_yerr = X_inv_arr[:,0,:].std(axis=0,ddof=1)
                if apply_std_coeff:
                    local_yerr *= local_std_coeff
                
                hgebinv = ax.errorbar(train_years, X_inv_arr[:,0,:].mean(axis=0), yerr=local_yerr,
                                      c=c_ghg_inv, ls=ls_forc_inv, lw=lw_forc_inv,
                                      alpha=local_alpha_forc_inv) #,label="$GHG_{INV}"+f"={X_RMSE[0]:.3f}$")
                
                local_yerr = X_inv_arr[:,1,:].std(axis=0,ddof=1)
                if apply_std_coeff:
                    local_yerr *= local_std_coeff
                
                haebinv = ax.errorbar(train_years, X_inv_arr[:,1,:].mean(axis=0), yerr=local_yerr,
                                      c=c_aer_inv, ls=ls_forc_inv, lw=lw_forc_inv,
                                      alpha=local_alpha_forc_inv) #,label="$AER_{INV}"+f"={X_RMSE[1]:.3f}$")
                
                local_yerr = X_inv_arr[:,2,:].std(axis=0,ddof=1)
                if apply_std_coeff:
                    local_yerr *= local_std_coeff
                
                hnebinv = ax.errorbar(train_years, X_inv_arr[:,2,:].mean(axis=0), yerr=local_yerr,
                                      c=c_nat_inv, ls=ls_forc_inv, lw=lw_forc_inv,
                                      alpha=local_alpha_forc_inv) #,label="$NAT_{INV}"+f"={X_RMSE[2]:.3f}$")
    else:
        hginv = ax.plot(train_years, X_inv_arr[0,:].T, c=c_ghg_inv, ls=ls_forc_inv, lw=lw_forc_inv,
                        alpha=alpha_forc_inv, label="$GHG_{INV}"+f"={X_RMSE[0]:.3f}$")
        hainv = ax.plot(train_years, X_inv_arr[1,:].T, c=c_aer_inv, ls=ls_forc_inv, lw=lw_forc_inv,
                        alpha=alpha_forc_inv, label="$AER_{INV}"+f"={X_RMSE[1]:.3f}$")
        hninv = ax.plot(train_years, X_inv_arr[2,:].T, c=c_nat_inv, ls=ls_forc_inv, lw=lw_forc_inv,
                        alpha=alpha_forc_inv, label="$NAT_{INV}"+f"={X_RMSE[2]:.3f}$")


    
    # -------------------------------------------------------------------------
    # X modele (mean & std)
    # -------------------------------------------------------------------------
    hfgebinv = None
    if X_mod_arr is not None:
        if plot_x_mod_shaded_region :
            #print('# X modele ...')
            alpha4mean_forc_mod = 0.8
            alpha4region_forc_mod = alpha_forc_mod/2
            
            for iforc,c in enumerate([c_ghg_mod,c_aer_mod,c_nat_mod]):
                local_forc_mean = X_mod_mean[iforc,:]
                local_forc_std = X_mod_std[iforc,:]
                if apply_std_coeff:
                    local_forc_std *= local_std_coeff
    
                ax.fill_between(train_years, local_forc_mean - local_forc_std,
                                local_forc_mean + local_forc_std,
                                alpha=alpha4region_forc_mod, ec=c, fc=gt.lighter_color(c),
                                hatch=hatch_fill_forc_mod, linewidth=lw_fill_forc_mod, zorder=2)
                hr = ax.fill(np.NaN, np.NaN,
                                alpha=alpha4region_forc_mod, ec="#AFAFAF", fc="#AFAFAF", # ec=c, fc=gt.lighter_color(c),
                                hatch=hatch_fill_forc_mod, linewidth=lw_fill_forc_mod, label="$FORC_{MOD}$ C.I. 90%")
                hp = ax.plot(train_years, local_forc_mean, c=c, ls=ls_forc_mod, lw=lw_forc_mod,
                             alpha=alpha4mean_forc_mod) #,label="$GHG_{INV}"+f"={X_RMSE[0]:.3f}$")
                if iforc == 0: hfgebinv=hr; hgebinv=hp
                if iforc == 1: haebinv=hp
                if iforc == 2: hnebinv=hp
            plot_mean_data_xmod = False
            #return X_RMSE, Y_RMSE
        elif plot_mean_data_xmod :
            local_alpha_forc_mod = 0.8  #alpha_forc_mod
            local_yerr = X_mod_std[0,:]
            if apply_std_coeff:
                local_yerr *= local_std_coeff
            
            hgebinv = ax.errorbar(train_years, X_mod_mean[0,:], yerr=local_yerr,
                                  c=c_ghg_mod, ls=ls_forc_mod, lw=lw_forc_mod,
                                  alpha=local_alpha_forc_mod) #,label="$GHG_{INV}"+f"={X_RMSE[0]:.3f}$")
            
            local_yerr = X_mod_std[1,:]
            if apply_std_coeff:
                local_yerr *= local_std_coeff
            
            haebinv = ax.errorbar(train_years, X_mod_mean[1,:], yerr=local_yerr,
                                  c=c_aer_mod, ls=ls_forc_mod, lw=lw_forc_mod,
                                  alpha=local_alpha_forc_mod) #,label="$AER_{INV}"+f"={X_RMSE[1]:.3f}$")
            
            local_yerr = X_mod_std[2,:]
            if apply_std_coeff:
                local_yerr *= local_std_coeff
            
            hnebinv = ax.errorbar(train_years, X_mod_mean[2,:], yerr=local_yerr,
                                  c=c_nat_mod, ls=ls_forc_mod, lw=lw_forc_mod,
                              alpha=local_alpha_forc_mod) #,label="$NAT_{INV}"+f"={X_RMSE[2]:.3f}$")
  


    # -------------------------------------------------------------------------
    # X reference (or Background for inversion)
    # -------------------------------------------------------------------------
    hfo = ax.plot(train_years, X_Ens_m_arr[0,:].T, c=c_ghg_ref, ls=ls_forc_ref, lw=lw_forc_ref,
                  label="$FORC_{ref.(other)}$")
    ax.plot(train_years, X_Ens_m_arr[1,:].T, c=c_aer_ref, ls=ls_forc_ref, lw=lw_forc_ref)
    ax.plot(train_years, X_Ens_m_arr[2,:].T, c=c_nat_ref, ls=ls_forc_ref, lw=lw_forc_ref)



    # -------------------------------------------------------------------------
    # Y observed
    # -------------------------------------------------------------------------
    hhd = ax.plot(train_years, HIST_m_arr.T, c=c_hist_to_inv, ls=ls_hist_to_inv, lw=lw_hist_to_inv,
                  label=hist_obs_label if hist_obs_label is not None else "$HIST_{OBS}$")


    # -------------------------------------------------------------------------
    # Y inversée
    # -------------------------------------------------------------------------
    hrinv = None
    if Y_inv_arr.shape[0] > 1 :
        if plot_y_inv_shaded_region :
            #print('# Y inversée ...')
            alpha4mean_hist_inv = 0.8
            alpha4region_hist_inv = alpha_hist_inv/2
            local_c_hist_inv = gt.darker_color(c_hist_inv,factor=0.2)
            local_hist_inv_mean = Y_inv_arr.mean(axis=0)
            local_hist_inv_std = Y_inv_arr.std(axis=0,ddof=1)
            if apply_std_coeff:
                local_hist_inv_std *= local_std_coeff
            
            ax.fill_between(train_years, local_hist_inv_mean - local_hist_inv_std,
                            local_hist_inv_mean + local_hist_inv_std,
                            alpha=alpha4region_hist_inv, ec=c, fc=gt.lighter_color(local_c_hist_inv),
                            hatch=hatch_fill_hist_inv, linewidth=lw_fill_hist_inv, zorder=2)
            hrinv = ax.fill(np.NaN, np.NaN,
                            alpha=alpha4region_hist_inv, ec=c, fc=gt.lighter_color(local_c_hist_inv),
                            hatch=hatch_fill_hist_inv, linewidth=lw_fill_hist_inv,
                            label="$HIST_{INV}$ C.I. 90%")
            hhinv = ax.plot(train_years, local_hist_inv_mean, c=local_c_hist_inv, ls=ls_hist_inv,
                            lw=lw_hist_inv, alpha=alpha4mean_hist_inv,
                            label="$HIST_{INV}"+f"={Y_RMSE:.4f}$")
            plot_mean_y_inv = False

        else:
            local_alpha_hist_inv = alpha_hist_inv / (np.log(X_arr.shape[0]) if X_arr.shape[0] > 1 else 1)
            if plot_mean_x_inv :
                local_alpha_hist_inv /= 2
    
            hhinv = ax.plot(train_years, Y_inv_arr.T, c=c_hist_inv, ls=ls_hist_inv, lw=lw_hist_inv,
                            alpha=local_alpha_hist_inv, label="$HIST_{INV}"+f"={Y_RMSE:.4f}$")
            if plot_mean_y_inv :
                local_alpha_hist_inv = 0.8
                local_yerr = Y_inv_arr.std(axis=0,ddof=1)
                if apply_std_coeff:
                    local_yerr *= local_std_coeff
                
                hhebinv = ax.errorbar(train_years, Y_inv_arr.mean(axis=0), yerr=local_yerr,
                                      c=gt.darker_color(c_hist_inv,factor=0.2),
                                      ls=ls_hist_inv, lw=lw_hist_inv, alpha=local_alpha_hist_inv) #, label="$HIST_{INV}"+f"={Y_RMSE:.4f}$")
    
    else:
        hhinv = ax.plot(train_years, Y_inv_arr.T, c=c_hist_inv, ls=ls_hist_inv, lw=lw_hist_inv,
                        alpha=alpha_hist_inv, label="$HIST_{INV}"+f"={Y_RMSE:.4f}$")

    # all handlers for Legend
    handles_for_legend.append(hfi[0])
    handles_for_legend.append(hfo[0])

    if hfginv is not None :
        handles_for_legend.append(hfginv[0])
        #handles_for_legend.append(hfainv[0])
        #handles_for_legend.append(hfninv[0])

    if hfgebinv is not None :
        handles_for_legend.append(hfgebinv[0])
    
    handles_for_legend.append(hginv[0])
    handles_for_legend.append(hainv[0])
    handles_for_legend.append(hninv[0])

    handles_for_legend.append(hhini[0])
    if hrinv is not None :
        handles_for_legend.append(hrinv[0])
    handles_for_legend.append(hhinv[0])

    handles_for_legend.append(hhd[0])


    if title_label is not None :
        ax.set_title(title_label,size='medium')

    if verbose:
        print("Length of handles:",len(hginv), len(hainv), len(hninv), len(hfi), len(hfo), len(hhinv), len(hhini))
    #if len(hginv) > 1 :
    #list_of_handles = [hginv[0], hainv[0], hninv[0], hfi[0], hfo[0], hhd[0], hhinv[0]]
    #else:
    #    list_of_handles = [hginv, hainv, hninv, hfi, hfo, hhinv, hhini]
    ax.legend(handles=handles_for_legend, loc='upper left', ncol=3, fontsize='medium')

    xmin,xmax = ax.get_xlim()
    ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
    ax.set_xlim([xmin,xmax])

    ax.grid(True,lw=0.75,ls=':')

    if fixed_t_limits_ok :
        ax.set_ylim([t_min_limit,t_max_limit])

    return X_RMSE, Y_RMSE


def plot_inverted_hist_profiles_by_net(base_case_to_explore, sub_case_to_explore, inversion_suffix, settings_label=None, 
                                       models_to_plot=None, plot_mean_inverted_profils=True, plot_also_individual_profils=False,
                                       plot_std_limits_curves=False,
                                       load_best_val=False, load_best_val2nd=False,
                                       data_in_dir=None, data_out_dir=None, figs_dir=None, save_figs=True,
                                       source_dirname='data_source_pl',
                                       force_plot=False, force_write=False,
                                       inv_label=None, n_sim_to_plot=None,
                                       t_limits=None,
                                       train_years=np.arange(1900,2015),
                                       local_nb_label="PlotInvProfByNet", fig_ext='png',
                                       figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w'},
                                       lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
                                       verbose=False,
                                      ) :
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    import generic_tools_pl as gt   # like hexcolor(), ...

    fixed_t_limits_ok = False
    if t_limits is not None:
        fixed_t_limits_ok = True
        t_min_limit,t_max_limit = t_limits

    # Repertoire des donnees
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        data_out_dir = './data_out'

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
    lp_nathist_filtering = base_case_dic['lp_nathist_filtering']
    if verbose:
        print(f" - Number of NNets: {n_nnets}")
        print(f" - Data and Training set Llabel: {data_and_training_label}")
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
    extrap_label = sub_case_dic['extrapolation_label']
    if verbose:
        print(f" - List of sizes of CNN hidden layers or (kernel sizes): {kern_size_list}")
        print(f" - Data Extrapolation Label: {extrap_label}")

    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore
    
    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    load_data_and_gener_params = {'data_dir':data_in_dir, 'verbose':verbose}
    if lp_nathist_filtering:
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)

        load_data_and_gener_params['to_filter'] = ['nat', 'hist']
        load_data_and_gener_params['filtering_dic_filename'] = filtering_dic_filename
    load_data_and_gener_params['verbose'] = verbose
    
    data_dic = gt.load_basic_data_and_gener_df(**load_data_and_gener_params)
    
    #data_label       = data_dic['label']
    #all_models_src   = data_dic['models']
    all_forcings_src = data_dic['forcings']
    all_forcing_color_dic     = data_dic['forcing_color_dic']
    all_forcing_inv_color_dic = data_dic['forcing_inv_color_dic']
    #all_forcing_color_names_dic    = data_dic['forcing_color_names_dic']
    #all_forcing_inv_color_names_dic = data_dic['forcing_inv_color_names_dic']
    #all_years        = data_dic['years']
    _,_,_,T_hist_df = data_dic['list_of_df']

    forcing_names = all_forcings_src[:4]
    forcing_colors = [all_forcing_color_dic[f.lower()] for f in forcing_names]
    forcing_inv_colors = [all_forcing_inv_color_dic[f.lower()] for f in forcing_names]

    #hist_inv_color = gt.lighter_color('#7f7f7f')   # gris (lighter)
    #hist_inv_color = gt.darker_color('#ff7f0e')  # orange (darker)
    
    hist_inv_color = forcing_inv_colors[-1]
    hist_inv_alpha = 1.0
    hist_indiv_inv_alpha = 0.3
    
    hist_color = forcing_colors[-1]
    hist_alpha = 0.8

    if load_best_val2nd :
        net_label = 'best-val2nd'
        #net_filename = 'Net_best-val2nd.pt'
    elif load_best_val :
        net_label = 'best-val'
        #net_filename = 'Net_best-val.pt'
    else:
        net_label = 'last'
        #net_filename = 'Net.pt'

    inversion_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix='inversion', 
                                                           set_label=inversion_suffix,
                                                           verbose=verbose)
    if models_to_plot is None :
        models_to_plot = inversion_combi_dic['models']
        if verbose:
            print(f"\nList of models found in Inversion data set:\n  {models_to_plot}")
        
        #models_to_plot = all_models_src
        #models_to_plot = models_in_invert_data
        #models_to_plot = ['BCC-CSM2-MR', 'FGOALS-g3', 'CanESM5', 'CNRM-CM6-1', 'ACCESS-ESM1-5', 'IPSL-CM6A-LR', 'MIROC6', 'HadGEM3-GC31-LL', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'NorESM2-LM'] #, 'GFDL-ESM4']
        #models_to_plot = ['IPSL-CM6A-LR']
        #models_to_plot = ['BCC-CSM2-MR']
        #models_to_plot = ['FGOALS-g3']
        #models_to_plot = ['NorESM2-LM']

    ################# TEST MODEL 5 ONLY ################    
    #m = 5
    ################################
    #print(f"\nModele {m}- {model_names[m]} en inversion [{inv_label}]:")

    cnn_name_base = sub_case_to_explore

    #all_years = train_combi_dic['years']
    #train_years = np.arange(1900,2015)
    lenDS = len(train_years)

    inversion_years = train_years
    

    #for i_trained,(trained_model) in enumerate(models_to_test) :
    for i_mod,mod_to_invert in enumerate(models_to_plot) : 

        # Historic simulations for current model (from source data)
        HIST_mod_for_inv_array = T_hist_df.loc[lambda df: df['model'] == mod_to_invert, :].drop('model', axis=1).iloc[:,-lenDS:].values

        experiment_name_in_sdir = f'Training-for-mod_{mod_to_invert}'
        experiment_name_out_sdir = f'Inversion-on_{mod_to_invert}'

        print(f"\n{'-'*132}\nInversion case {i_mod+1}/{len(models_to_plot)}) for modele {mod_to_invert})"+
              f"(inversion case: '{inversion_suffix}')")

        case_out_dir = os.path.join(data_out_dir, base_case_to_explore, cnn_name_base, experiment_name_out_sdir)
        print(f'Repertoire de sortie du cas: {case_out_dir}')
        if not os.path.exists(case_out_dir):
            print(f"\n *** Case inversion directory '{case_out_dir}/' not found. Skiping model case ...")
            continue
        
        if inv_label is None :
            inv_label = set_inv_labl_from_path (case_out_dir, settings_label=settings_label, verbose=verbose)
        
        suptitlelabel = f"{cnn_name_base} [{experiment_name_in_sdir}] [{data_and_training_label}] ({n_nnets} Nets)"
        print(suptitlelabel)

        case_figs_dir = os.path.join(figs_dir, base_case_to_explore, cnn_name_base, f'Settings-{inv_label}', experiment_name_out_sdir)
        print(f'Repertoire des figures du cas: {case_figs_dir}')
        if save_figs and not os.path.exists(case_figs_dir):
            os.makedirs(case_figs_dir)

        ## test channels
        #for tai in taille:
        #tai = 16
        #tai=4

        for innet in np.arange(n_nnets):
            #innet = 0

            inv_dir = os.path.join(case_out_dir,f'Inv_N{innet}', f'Settings-{inv_label}')

            inv_net_figs_dir = os.path.join(case_figs_dir,f'Inv_N{innet}')
            if save_figs and not os.path.exists(inv_net_figs_dir):
                os.makedirs(inv_net_figs_dir)

            print(f'Inv_N{innet} - Inv_dir: {inv_dir}')

            xinv_file_prefix = 'Xinv_multi_all-but-one'

            nbe_invert_prnt = gt.look_for_inversion_files(file_prefix=xinv_file_prefix, inv_dir=inv_dir, verbose=verbose)
            nbe_invert_label = nbe_invert_prnt.replace('-',' ')

            print('nbe_invert_label:',nbe_invert_label)

            inv_postfix = f'{nbe_invert_prnt}_N{innet}_{net_label}-net_mod-{mod_to_invert}'

            #mlt_list_Xinv = pickle.load( open( os.path.join(inv_dir, f'Xinv_multi_all-but-one_{inv_postfix}_ST.p'), "rb" ) )
            mlt_list_Yinv = pickle.load( open( os.path.join(inv_dir, f'Yinv_multi_all-but-one_{inv_postfix}_ST.p'), "rb" ) )
            mlt_list_nbe = pickle.load( open( os.path.join(inv_dir, f'Nbe_multi_all-but-one_{inv_postfix}_ST.p'), "rb" ) )
            #print(f"#DBG1# len(mlt_list_Xinv),mlt_list_Xinv[0].shape",len(mlt_list_Xinv),mlt_list_Xinv[0].shape)
            #print(f"#DBG1# len(mlt_list_Yinv),mlt_list_Yinv[0].shape",len(mlt_list_Yinv),mlt_list_Yinv[0].shape)
            #print(f"#DBG1# len(mlt_list_nbe),mlt_list_nbe",len(mlt_list_nbe),mlt_list_nbe)

            added_mean_of_all = type(mlt_list_nbe[-1]) is list
            nb_inverted_patt = len(mlt_list_nbe)
            #n_sim_by_page = n_sim_to_plot if n_sim_to_plot is not None else nb_inverted_patt+1
            if added_mean_of_all :
                # si le ensemble-moyenné a ete inversé aussi, alors on reduit de 1 le nombre de patterns inverses
                nb_inverted_patt -= 1
                #n_sim_by_page -= 1
                print(f"nb_inverted_patt: {nb_inverted_patt}+1, mlt_list_nbe: {mlt_list_nbe[:-1]}, [added_mean_of_all: {added_mean_of_all}]")
                #nbe_inverted_label = f"all {nb_inverted_patt}+ALL Hist profiles"
                list_profiles_dic = gt.do_list_of_profiles(nb_inverted_patt, n_by_page=n_sim_to_plot,
                                                           add_mean_of_all=added_mean_of_all,
                                                           verbose=verbose)
                nbe_inverted_label = list_profiles_dic['label']
                ilist_of_nbe = list_profiles_dic['list']
                ilist_of_nbe = ilist_of_nbe[:-1]  # retire le dernier (la liste d'indices des profils moynennes)
                ilist_of_nbe.append(nb_inverted_patt) # ajoute le derinier comme un seul indice
                print('ilist_of_nbe:',ilist_of_nbe,len(ilist_of_nbe),nbe_inverted_label)
            else:
                print(f"nb_inverted_patt: {nb_inverted_patt}, mlt_list_nbe: {mlt_list_nbe}, [added_mean_of_all: {added_mean_of_all}]")
                #nbe_inverted_label = f"all {nb_inverted_patt} Hist profiles"
                list_profiles_dic = gt.do_list_of_profiles(nb_inverted_patt, n_by_page=n_sim_to_plot,
                                                           verbose=verbose)
                nbe_inverted_label = list_profiles_dic['label']
                ilist_of_nbe = list_profiles_dic['list']
                print('ilist_of_nbe:',ilist_of_nbe,len(ilist_of_nbe),nbe_inverted_label)

            nbe_inverted_prnt = nbe_inverted_label.replace(' ','-')
            
            figs_file = f"Fig{local_nb_label}_{nbe_inverted_prnt}-inversion_multi-patt-N{innet+1}-{net_label}-net"
            if not plot_mean_inverted_profils and plot_also_individual_profils :
                figs_file += "_NoMeanAndIndivProfils"
            elif plot_also_individual_profils :
                figs_file += "_IndivProfils"
            if fixed_t_limits_ok :
                figs_file += "_FIX-T"
            figs_file += f".{fig_ext}"
            figs_filename = os.path.join(inv_net_figs_dir,figs_file)

            if not force_plot and save_figs and os.path.isfile(figs_filename):
                print(f" ** {local_nb_label} figure already exists '{figs_filename}'. Figure not prepared")
                
            else:
                n_nbe = len(ilist_of_nbe)
    
                top = 0.92;    bottom = 0.04
                left = 0.06;   right = 0.98
                wspace = 0.05; hspace = 0.10
                if n_nbe < 5:
                    if n_nnets < 2 :
                        top = 0.72
                    if n_nnets < 4 :
                        top = 0.82
                    ncols = 1
                    left = 0.06+0.26-wspace/2; right = 0.98-0.26+wspace/2
                    suptitle_fontsize = "medium"
                elif n_nbe < 9:
                    ncols = 2
                    suptitle_fontsize = "large"
                elif n_nbe < 19:
                    ncols = 3
                    #suptitle_fontsize = "x-large"
                    suptitle_fontsize = 18
                elif n_nbe < 33:
                    ncols = 4
                    #suptitle_fontsize = "xx-large"
                    suptitle_fontsize = 24
                else:
                    ncols = 5
                    #suptitle_fontsize = "xx-large"
                    suptitle_fontsize = 32
    
                nrows = int(np.ceil(n_nbe/ncols))
        
                fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(8*max(2,ncols),1.5+5*nrows),
                                        gridspec_kw={'hspace': hspace, 'wspace': wspace, 
                                                     'left': left,     'right': right,
                                                     'top' : top,      'bottom' : bottom })
    
                for iax,inbe in enumerate(ilist_of_nbe) :
                    Yinvmean_arr,nbe = mlt_list_Yinv[inbe],mlt_list_nbe[inbe]
    
                    ax = axes.flatten()[iax]  #iax//ncols,iax%ncols]
                    tmp_arr = Yinvmean_arr.squeeze(axis=1)
                    n_tmp = tmp_arr.shape[0]
                    #for ii,y_inv in enumerate(tmp_arr) :
                    #    ax.plot(inversion_years,y_inv,alpha=0.01)
                    if plot_also_individual_profils:
                        nb_inv_label = f"ALL-{n_tmp}"
                        print(f"plotting individual profiles: {nb_inv_label} ...")
                        local_hist_indiv_inv_alpha = gt.reduce_alpha(n_tmp, hist_indiv_inv_alpha)
                        print(f"local hist_indiv_inv_alpha (initially at: {hist_indiv_inv_alpha}) for n={n_tmp}={local_hist_indiv_inv_alpha}")

                        #print(f"#DBG1# PRE STD HIST_mod_for_inv_array[{nbe},:].std(axis=0,ddof=1) ... HIST_mod_for_inv_array[{nbe},:].shape",HIST_mod_for_inv_array[nbe,:].shape)
                        ax.plot(inversion_years,tmp_arr.T,color=gt.lighter_color(hist_inv_color),lw=1,alpha=local_hist_indiv_inv_alpha)
                    else:
                        ax.plot(inversion_years,tmp_arr.T,color=gt.lighter_color(hist_inv_color,factor=0.5),alpha=0.01)

                    if plot_mean_inverted_profils:
                        tmp_y = tmp_arr.mean(axis=0)
                        #print(f"#DBG1# PRE STD tmp_arr.std(axis=0,ddof=1) ... tmp_arr.shape",tmp_arr.shape)
                        tmp_y_err = tmp_arr.std(axis=0,ddof=1)
                        #print(f"#DBG1# ,inversion_years.shape,tmp_y.shape,tmp_y_err.shape):",inversion_years.shape,tmp_y.shape,tmp_y_err.shape)
    
                        ax.errorbar(inversion_years, tmp_y,yerr=tmp_y_err, color=hist_inv_color, lw=2, alpha=hist_inv_alpha)
                        #print(f"#DBG1# POST  ERRORBAR")

                    if plot_std_limits_curves:
                        ax.plot(inversion_years, tmp_y+2*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=hist_inv_alpha)
                        ax.plot(inversion_years, tmp_y-2*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=hist_inv_alpha)
        
                        ax.plot(inversion_years, tmp_y+3*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=hist_inv_alpha)
                        ax.plot(inversion_years, tmp_y-3*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=hist_inv_alpha)
    
                    nbe_label = None
                    if np.isscalar(nbe) :
                        nbe_label = f'{nbe}'
                        print(f"plotting only profile: {nbe_label} ...")
                        current_HIST_m = HIST_mod_for_inv_array[nbe,:]
                        ax.plot(inversion_years,current_HIST_m,color=hist_color,lw=2,alpha=hist_alpha)
                    else:
                        nbe_label = f"MEAN-{nbe[0]}-to-{nbe[-1]}"
                        print(f"plotting profiles with errorbars: {nbe_label} ...")
                        current_HIST_m = HIST_mod_for_inv_array[nbe,:].mean(axis=0)
                        #print(f"#DBG1# PRE STD HIST_mod_for_inv_array[{nbe},:].std(axis=0,ddof=1) ... HIST_mod_for_inv_array[{nbe},:].shape",HIST_mod_for_inv_array[nbe,:].shape)
                        current_HIST_s = HIST_mod_for_inv_array[nbe,:].std(axis=0,ddof=1)
                        ax.errorbar(inversion_years,current_HIST_m,yerr=current_HIST_s,color=hist_color,lw=1,alpha=hist_alpha)
                        ax.plot(inversion_years,current_HIST_m,color=hist_color,lw=2,alpha=hist_alpha)
                        
                    xmin,xmax = ax.get_xlim()
                    ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
                    ax.set_xlim([xmin,xmax])
    
                    ax.grid(True,lw=0.75,ls=':')
                    ax.set_title(f"Inverted {mod_to_invert} - Hist profile ({nbe_label})")
                    
                    if fixed_t_limits_ok :
                        ax.set_ylim([t_min_limit,t_max_limit])

                # efface de la figure les axes non utilises
                if iax+1 < nrows*ncols :
                    for jax in np.arange(iax+1,nrows*ncols):
                        ax = axes.flatten()[jax]  #iax//ncols,iax%ncols]
                        ax.set_visible(False)
    
                #members_label = f"{n_nbe} / {nb_inverted_patt} Hist profiles" if n_nbe < nb_inverted_patt else f"all {nb_inverted_patt} Hist profiles"+(" - NAT & HIST LP filtered" if lp_nathist_filtering else "")
                members_label = nbe_inverted_label+(" - NAT & HIST LP filtered" if lp_nathist_filtering else "")
                if plot_mean_inverted_profils and plot_also_individual_profils :
                    members_label += "- Mean and Indiv. inv. prof."
                elif plot_mean_inverted_profils:
                    members_label += "- Mean inv. prof."
                elif plot_also_individual_profils:
                    members_label += "- Indiv inv. prof."
                members_label += "- FIX-T" if fixed_t_limits_ok else ""
                
                plt.suptitle(f"Inversion for Model {mod_to_invert} [data: {inversion_suffix}] [{members_label}] [Net {innet}/{n_nnets}] [{net_label.upper()}]\n"+\
                             f"{base_case_to_explore} / {cnn_name_base}\n"+\
                             f"[Data: {data_and_training_label} - Inversion Settings: {inv_label}]",
                             size=suptitle_fontsize, y=0.99)
                        
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


def plot_averaged_inverted_profiles_by_net (base_case_to_explore, sub_case_to_explore, inversion_suffix, settings_label=None,
                                            models_to_plot=None, load_best_val=False, load_best_val2nd=False,
                                            data_in_dir=None, data_out_dir=None, figs_dir=None, save_figs=True,
                                            source_dirname='data_source_pl',
                                            force_plot=False, force_write=False,
                                            inv_label=None, n_sim_to_plot=None,
                                            train_years=np.arange(1900,2015),
                                            local_nb_label="PlotAveInvProfByNet", fig_ext='png',
                                            figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w'},
                                            lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
                                            verbose=False,
                                           ) :    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    import generic_tools_pl as gt   # like hexcolor(), ...

    # Repertoire des donnees
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        data_out_dir = './data_out'

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
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = gt.retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    n_nnets = base_case_dic['n_nnets']
    data_and_training_label = base_case_dic['data_and_training_label']
    lp_nathist_filtering = base_case_dic['lp_nathist_filtering']
    if verbose:
        print(f" - Number of NNets: {n_nnets}")
        print(f" - Data and Training set Llabel: {data_and_training_label}")
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
    extrap_label = sub_case_dic['extrapolation_label']
    if verbose:
        print(f" - List of sizes of CNN hidden layers or (kernel sizes): {kern_size_list}")
        print(f" - Data Extrapolation Label: {extrap_label}")

    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore
    
    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    load_data_and_gener_params = {'data_dir':data_in_dir, 'verbose':verbose}
    if lp_nathist_filtering:
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)

        load_data_and_gener_params['to_filter'] = ['nat', 'hist']
        load_data_and_gener_params['filtering_dic_filename'] = filtering_dic_filename
    load_data_and_gener_params['verbose'] = verbose
    
    data_dic = gt.load_basic_data_and_gener_df(**load_data_and_gener_params)
    
    #data_label       = data_dic['label']
    #all_models_src   = data_dic['models']
    all_forcings_src = data_dic['forcings']
    all_forcing_color_dic     = data_dic['forcing_color_dic']
    all_forcing_inv_color_dic = data_dic['forcing_inv_color_dic']
    #all_forcing_color_names_dic    = data_dic['forcing_color_names_dic']
    #all_forcing_inv_color_names_dic = data_dic['forcing_inv_color_names_dic']
    #all_years        = data_dic['years']
    _,_,_,T_hist_df = data_dic['list_of_df']

    forcing_names = all_forcings_src[:4]
    forcing_colors = [all_forcing_color_dic[f.lower()] for f in forcing_names]
    forcing_inv_colors = [all_forcing_inv_color_dic[f.lower()] for f in forcing_names]

    #hist_inv_color = gt.lighter_color('#7f7f7f')   # gris (lighter)
    #hist_inv_color = gt.darker_color('#ff7f0e')  # orange (darker)
    hist_color = forcing_colors[-1]
    hist_inv_color = forcing_inv_colors[-1]

    if load_best_val2nd :
        net_label = 'best-val2nd'
        #net_filename = 'Net_best-val2nd.pt'
    elif load_best_val :
        net_label = 'best-val'
        #net_filename = 'Net_best-val.pt'
    else:
        net_label = 'last'
        #net_filename = 'Net.pt'

    inversion_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix='inversion', 
                                                           set_label=inversion_suffix,
                                                           verbose=verbose)
    if models_to_plot is None :
        models_to_plot = inversion_combi_dic['models']
        if verbose:
            print(f"\nList of models found in Test set:\n  {models_to_plot}")
        
        #models_to_plot = all_models_src
        #models_to_plot = models_in_invert_data
        #models_to_plot = ['BCC-CSM2-MR', 'FGOALS-g3', 'CanESM5', 'CNRM-CM6-1', 'ACCESS-ESM1-5', 'IPSL-CM6A-LR', 'MIROC6', 'HadGEM3-GC31-LL', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'NorESM2-LM'] #, 'GFDL-ESM4']
        #models_to_plot = ['IPSL-CM6A-LR']
        #models_to_plot = ['BCC-CSM2-MR']
        #models_to_plot = ['FGOALS-g3']
        #models_to_plot = ['NorESM2-LM']

    ################# TEST MODEL 5 ONLY ################    
    #m = 5
    ################################
    #print(f"\nModele {m}- {model_names[m]} en inversion [{inv_label}]:")

    cnn_name_base = sub_case_to_explore

    #all_years = train_combi_dic['years']
    #train_years = np.arange(1900,2015)
    lenDS = len(train_years)

    inversion_years = train_years
    
    ################# TEST MODEL 5 ONLY ################    
    #m = 5
    ################################
    #print(f"\nModele {m}- {model_names[m]} en inversion [{inv_label}]:")

    cnn_name_base = sub_case_to_explore

    #for i_trained,(trained_model) in enumerate(models_to_test) :
    for i_mod,mod_to_invert in enumerate(models_to_plot) : 

        # Historic simulations for current model (from source data)
        HIST_mod_for_inv_array = T_hist_df.loc[lambda df: df['model'] == mod_to_invert, :].drop('model', axis=1).iloc[:,-lenDS:].values

        experiment_name_in_sdir = f'Training-for-mod_{mod_to_invert}'
        experiment_name_out_sdir = f'Inversion-on_{mod_to_invert}'

        print(f"\n{'-'*132}\nInversion case {i_mod+1}/{len(models_to_plot)}) for modele {mod_to_invert})"+
              f"(inversion case: '{inversion_suffix}')")

        case_out_dir = os.path.join(data_out_dir, base_case_to_explore, cnn_name_base, experiment_name_out_sdir)
        print(f'Repertoire de sortie du cas: {case_out_dir}')
        if not os.path.exists(case_out_dir):
            print(f"\n *** Case inversion directory '{case_out_dir}/' not found. Skiping model case ...")
            continue

        if inv_label is None :
            inv_label = set_inv_labl_from_path (case_out_dir, settings_label=settings_label, verbose=verbose)

        suptitlelabel = f"{cnn_name_base} [{experiment_name_in_sdir}] [{data_and_training_label}] ({n_nnets} Nets)"
        print(suptitlelabel)

        case_figs_dir = os.path.join(figs_dir, base_case_to_explore, cnn_name_base, f'Settings-{inv_label}', experiment_name_out_sdir)
        print(f'Repertoire des figures du cas: {case_figs_dir}')
        if save_figs and not os.path.exists(case_figs_dir):
            os.makedirs(case_figs_dir)
        
        list_for_average_of_mean_Xinv = []
        list_for_average_of_mean_Yinv = []
        for innet in np.arange(n_nnets):
            #innet = 0

            inv_dir = os.path.join(case_out_dir,f'Inv_N{innet}',f'Settings-{inv_label}')

            if innet == 0:
                print(f'Inv_N{innet} - Inv_dir: {inv_dir}',end='' if verbose else '\n')
            else:
                print(f' ... Inv_N{innet}',end='' if verbose else '\n')
            
            # inverted data Xinv file prefix
            xinv_file_prefix = 'Xinv_multi_all-but-one'

            nbe_invert_prnt = gt.look_for_inversion_files(file_prefix=xinv_file_prefix, inv_dir=inv_dir, verbose=verbose)
            nbe_invert_label = nbe_invert_prnt.replace('-',' ')

            print('nbe_invert_label:',nbe_invert_label)

            # inverted data filename commun postfix
            inv_postfix = f'{nbe_invert_prnt}_N{innet}_{net_label}-net_mod-{mod_to_invert}'

            mlt_list_Xinv = pickle.load( open( os.path.join(inv_dir, f'Xinv_multi_all-but-one_{inv_postfix}_ST.p'), "rb" ) )
            mlt_list_Yinv = pickle.load( open( os.path.join(inv_dir, f'Yinv_multi_all-but-one_{inv_postfix}_ST.p'), "rb" ) )
            mlt_list_nbe = pickle.load( open( os.path.join(inv_dir, f'Nbe_multi_all-but-one_{inv_postfix}_ST.p'), "rb" ) ) 
            #print("#DBG2# len(mlt_list_Xinv),mlt_list_Xinv[0].shape",len(mlt_list_Xinv),mlt_list_Xinv[0].shape)
            #print("#DBG2# len(mlt_list_Yinv),mlt_list_Yinv[0].shape",len(mlt_list_Yinv),mlt_list_Yinv[0].shape)
            #print("#DBG2# len(mlt_list_nbe),mlt_list_nbe",len(mlt_list_nbe),mlt_list_nbe)

            # Averaging Xinv and Yinv paterns
            mlt_arr_mean_Xinv = np.array([Xinv.mean(axis=0) for Xinv in mlt_list_Xinv])
            mlt_arr_mean_Yinv = np.array([Yinv.mean(axis=0) for Yinv in mlt_list_Yinv])

            #print(f"#DBG2# mlt_arr_mean_Xinv.shape,mlt_arr_mean_Yinv.shape",mlt_arr_mean_Xinv.shape,mlt_arr_mean_Yinv.shape)

            list_for_average_of_mean_Xinv.append(mlt_arr_mean_Xinv)
            list_for_average_of_mean_Yinv.append(mlt_arr_mean_Yinv)

        if verbose:
            print()
        
        #mlt_for_average_of_mean_Xinv = np.array(list_for_average_of_mean_Xinv) #.mean(axis=0)  # size for IPSL ((34, 1, 3, 133), (34, 1, 115))
        mlt_for_average_of_mean_Yinv = np.array(list_for_average_of_mean_Yinv) #.mean(axis=0)
        #print(f"#DBG2# mlt_for_average_of_mean_Yinv.shape",mlt_for_average_of_mean_Yinv.shape)

        added_mean_of_all = type(mlt_list_nbe[-1]) is list
        nb_inverted_patt = len(mlt_list_nbe)
        #n_sim_by_page = n_sim_to_plot if n_sim_to_plot is not None else nb_inverted_patt+1
        if added_mean_of_all :
            # si le ensemble-moyenné a ete inversé aussi, alors on reduit de 1 le nombre de patterns inverses
            nb_inverted_patt -= 1
            #n_sim_by_page -= 1
            print(f"nb_inverted_patt: {nb_inverted_patt}+1, mlt_list_nbe: {mlt_list_nbe[:-1]}, [added_mean_of_all: {added_mean_of_all}]")
            #nbe_inverted_label = f"all {nb_inverted_patt}+ALL Hist profiles"
            list_profiles_dic = gt.do_list_of_profiles(nb_inverted_patt, n_by_page=n_sim_to_plot,
                                                       add_mean_of_all=added_mean_of_all,
                                                       verbose=verbose)
            nbe_inverted_label = list_profiles_dic['label']
            ilist_of_nbe = list_profiles_dic['list']
            ilist_of_nbe = ilist_of_nbe[:-1]  # retire le dernier (la liste d'indices des profils moynennes)
            ilist_of_nbe.append(nb_inverted_patt) # ajoute le derinier comme un seul indice
            print('ilist_of_nbe:',ilist_of_nbe,len(ilist_of_nbe),nbe_inverted_label)
        else:
            print(f"nb_inverted_patt: {nb_inverted_patt}, mlt_list_nbe: {mlt_list_nbe}, [added_mean_of_all: {added_mean_of_all}]")
            #nbe_inverted_label = f"all {nb_inverted_patt} Hist profiles"
            list_profiles_dic = gt.do_list_of_profiles(nb_inverted_patt, n_by_page=n_sim_to_plot,
                                                       verbose=verbose)
            nbe_inverted_label = list_profiles_dic['label']
            ilist_of_nbe = list_profiles_dic['list']
            print('ilist_of_nbe:',ilist_of_nbe,len(ilist_of_nbe),nbe_inverted_label)

        nbe_inverted_prnt = nbe_inverted_label.replace(' ','-')
        
        n_nbe = len(ilist_of_nbe)

        figs_file = f"Fig{local_nb_label}_{nbe_inverted_prnt}-profiles-inversion_multi-patt_all-{n_nnets}-nets-averaged_{net_label}-net.{fig_ext}"
        figs_filename = os.path.join(case_figs_dir,figs_file)
        
        if not force_plot and save_figs and os.path.isfile(figs_filename):
            print(f" ** {local_nb_label} figure already exists '{figs_filename}'. Figure not prepared")
            
        else:
                        
            top = 0.92;    bottom = 0.04
            left = 0.06;   right = 0.98
            wspace = 0.05; hspace = 0.10
            if n_nbe < 5:
                if n_nnets < 2 :
                    top = 0.72
                if n_nnets < 4 :
                    top = 0.82
                ncols = 1
                left = 0.06+0.26-wspace/2; right = 0.98-0.26+wspace/2
                suptitle_fontsize = "large"     # 12
            elif n_nbe < 9:
                ncols = 2
                suptitle_fontsize = "x-large"   # 14.4
            elif n_nbe < 19:
                ncols = 3
                suptitle_fontsize = "xx-large"  # 17.28
                suptitle_fontsize = 22
            elif n_nbe < 33:
                ncols = 4
                #suptitle_fontsize = "xx-large"
                suptitle_fontsize = 32
            else:
                ncols = 5
                #suptitle_fontsize = "xx-large"
                suptitle_fontsize = 44

            nrows = int(np.ceil(n_nbe/ncols))
    
            fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(8*max(2,ncols),1.5+5*nrows),
                                    gridspec_kw={'hspace': hspace, 'wspace': wspace, 
                                                 'left': left,     'right': right,
                                                 'top' : top,      'bottom' : bottom })
    
            for iax,inbe in enumerate(ilist_of_nbe) :
                Yinvmean_arr,nbe = mlt_for_average_of_mean_Yinv[:,inbe],mlt_list_nbe[inbe]
                print(f"#DBG2# Yinvmean_arr.shape,nbe",Yinvmean_arr.shape,nbe)

                ax = axes.flatten()[iax]  #iax//ncols,iax%ncols]
                tmp_arr = Yinvmean_arr.squeeze(axis=1)   # squeeze ciblé sur l'axe 1
                if False:
                    for ii,y_inv in enumerate(tmp_arr) :
                        ax.plot(inversion_years,y_inv,alpha=0.1)
                if True:
                    tmp_y = tmp_arr.mean(axis=0)
                    
                    if n_nnets == 1 :
                        ax.plot(inversion_years, tmp_y, color=hist_inv_color,lw=2, alpha=1.0)
                        
                    else:
                        #print(f"#DBG2# PRE STD tmp_arr.std(axis=0,ddof=1) ... tmp_arr.shape",tmp_arr.shape)
                        tmp_y_err = tmp_arr.std(axis=0,ddof=1)
                        #print(f"#DBG2# ,inversion_years.shape,tmp_y.shape,tmp_y_err.shape):",inversion_years.shape,tmp_y.shape,tmp_y_err.shape)
                        ax.errorbar(inversion_years, tmp_y, yerr=tmp_y_err, color=hist_inv_color,lw=2, alpha=0.8)
                        #print(f"#DBG2# POST  ERRORBAR")
                        ax.plot(inversion_years, tmp_y+2*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
                        ax.plot(inversion_years, tmp_y-2*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
                        ax.plot(inversion_years, tmp_y+3*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
                        ax.plot(inversion_years, tmp_y-3*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
    
                    if np.isscalar(nbe) :
                        nbe_label = f'{nbe}'
                        #current_HIST_m = HIST_[nbe,:]
                        current_HIST_m = HIST_mod_for_inv_array[nbe,:]
                        ax.plot(inversion_years,current_HIST_m,color=hist_color,lw=2,alpha=0.8)
    
                    else:
                        nbe_label = f"MEAN-{nbe[0]}-to-{nbe[-1]}"
                        #current_HIST_m = HIST_[nbe,:].mean(axis=0)
                        #current_HIST_s= HIST_[nbe,:].std(axis=0,ddof=1)
                        current_HIST_m = HIST_mod_for_inv_array[nbe,:].mean(axis=0)
                        #print(f"#DBG2# PRE STD HIST_mod_for_inv_array[{nbe},:].std(axis=0,ddof=1) ... HIST_mod_for_inv_array[{nbe},:].shape",HIST_mod_for_inv_array[nbe,:].shape)
                        current_HIST_s = HIST_mod_for_inv_array[nbe,:].std(axis=0,ddof=1)
                        ax.errorbar(inversion_years,current_HIST_m,yerr=current_HIST_s,color=hist_color,lw=1,alpha=0.8)
                        ax.plot(inversion_years,current_HIST_m,color=hist_color,lw=2,alpha=0.8)
    
                for ii,y_inv in enumerate(tmp_arr) :
                    ax.plot(inversion_years,y_inv,alpha=0.5)
    
                xmin,xmax = ax.get_xlim()
                ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
                ax.set_xlim([xmin,xmax])
    
                ax.grid(True,lw=0.75,ls=':')
                ax.set_title(f"Inverted {mod_to_invert} - Hist profile ({nbe_label})")
    
            # efface de la figure les axes non utilises
            if iax+1 < nrows*ncols :
                for jax in np.arange(iax+1,nrows*ncols):
                    ax = axes.flatten()[jax]  #iax//ncols,iax%ncols]
                    ax.set_visible(False)
    
            #members_label = f"{n_nbe} / {nb_inverted_patt} Hist profiles" if n_nbe < nb_inverted_patt else f"all {nb_inverted_patt} Hist profiles"+(" - NAT & HIST LP filtered" if lp_nathist_filtering else "")
            members_label = nbe_inverted_label+(" - NAT & HIST LP filtered" if lp_nathist_filtering else "")
    
            plt.suptitle(f"Inversion for Model {mod_to_invert} [data: {inversion_suffix}] [{members_label}] [{n_nnets} Nets Av.] [{net_label.upper()}]\n"+\
                         f"{base_case_to_explore} / {cnn_name_base}\n"+\
                         f"[Data: {data_and_training_label} - Inversion Settings: {inv_label}]",
                         size=suptitle_fontsize, y=0.99)

            #figs_file = f"Fig{local_nb_label}_{nbe_inverted_prnt}-profiles-inversion_multi-patt_all-{n_nnets}-nets-averaged_{net_label}-net.{fig_ext}"
            #figs_filename = os.path.join(case_figs_dir,figs_file)

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


def plot_averaged_inv_all_forcings_by_model (base_case_to_explore, sub_case_to_explore, inversion_suffix, settings_label=None,
                                             models_to_plot=None, load_best_val=False, load_best_val2nd=False,
                                             data_in_dir=None, data_out_dir=None, figs_dir=None, save_figs=True,
                                             source_dirname='data_source_pl',
                                             force_plot=False, force_write=False,
                                             inv_label=None, t_limits=None,
                                             train_years=np.arange(1900,2015),
                                             local_nb_label="PlotAveInvAllForcfByNet", fig_ext='png',
                                             figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w'},
                                             lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
                                             #obs_filename = 'obs.npy', lp_obs_filtering=False, lp_obs_filtering_dictionary=None,
                                             obs_name = 'OBS',
                                             verbose=False,
                                            ) :    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    import generic_tools_pl as gt   # like hexcolor(), ...

    fixed_t_limits_ok = False
    if t_limits is not None:
        fixed_t_limits_ok = True
        t_min_limit,t_max_limit = t_limits

    # Repertoire des donnees
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        data_out_dir = './data_out'

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
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = gt.retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    n_nnets = base_case_dic['n_nnets']
    data_and_training_label = base_case_dic['data_and_training_label']
    lp_nathist_filtering = base_case_dic['lp_nathist_filtering']
    if verbose:
        print(f" - Number of NNets: {n_nnets}")
        print(f" - Data and Training set Llabel: {data_and_training_label}")
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
    extrap_label = sub_case_dic['extrapolation_label']
    if verbose:
        print(f" - List of sizes of CNN hidden layers or (kernel sizes): {kern_size_list}")
        print(f" - Data Extrapolation Label: {extrap_label}")

    if n_nnets == 1:
        n_nnets_label = "Only 1 nnet"
    else:
        n_nnets_label = "{n_nnets} nets"
    
    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore
    
    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    load_data_and_gener_params = {'data_dir':data_in_dir, 'verbose':verbose}
    if lp_nathist_filtering:
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)

        load_data_and_gener_params['to_filter'] = ['nat', 'hist']
        load_data_and_gener_params['filtering_dic_filename'] = filtering_dic_filename
    load_data_and_gener_params['verbose'] = verbose

    data_dic = gt.load_basic_data_and_gener_df(**load_data_and_gener_params)
    
    #data_label       = data_dic['label']
    #all_models_src   = data_dic['models']
    #all_forcings_src = data_dic['forcings']
    #all_forcing_color_dic     = data_dic['forcing_color_dic']
    #all_forcing_inv_color_dic = data_dic['forcing_inv_color_dic']
    #all_forcing_color_names_dic    = data_dic['forcing_color_names_dic']
    #all_forcing_inv_color_names_dic = data_dic['forcing_inv_color_names_dic']
    #all_years        = data_dic['years']
    _,_,_,T_hist_df = data_dic['list_of_df']

    if load_best_val2nd :
        net_label = 'best-val2nd'
        #net_filename = 'Net_best-val2nd.pt'
    elif load_best_val :
        net_label = 'best-val'
        #net_filename = 'Net_best-val.pt'
    else:
        net_label = 'last'
        #net_filename = 'Net.pt'

    inversion_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix='inversion', 
                                                           set_label=inversion_suffix,
                                                           verbose=verbose)
    if models_to_plot is None :
        models_to_plot = inversion_combi_dic['models']
        if verbose:
            print(f"\nList of models found in Inversion data set:\n  {models_to_plot}")
        
        #models_to_plot = all_models_src
        #models_to_plot = models_in_invert_data
        #models_to_plot = ['BCC-CSM2-MR', 'FGOALS-g3', 'CanESM5', 'CNRM-CM6-1', 'ACCESS-ESM1-5', 'IPSL-CM6A-LR', 'MIROC6', 'HadGEM3-GC31-LL', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'NorESM2-LM'] #, 'GFDL-ESM4']
        #models_to_plot = ['IPSL-CM6A-LR']
        #models_to_plot = ['BCC-CSM2-MR']
        #models_to_plot = ['FGOALS-g3']
        #models_to_plot = ['NorESM2-LM']

    # if obs_name in models_to_plot :
    #     print(f"\nLoading {obs_name} for inversion:")
    #     obs_df = gt.load_obs_data(obs_filename, data_dir=data_in_dir, last_year=train_years[-1], return_as_df=True, verbose=verbose)
    
    #     if lp_obs_filtering :
    #         print(f"Filtering {obs_name} ... ")
    #         obs_df = gt.filter_forcing_df(obs_df, filt_dic=lp_obs_filtering_dictionary, verbose=verbose)

    ################# TEST MODEL 5 ONLY ################    
    #m = 5
    ################################
    #print(f"\nModele {m}- {model_names[m]} en inversion [{inv_label}]:")

    cnn_name_base = sub_case_to_explore

    #all_years = train_combi_dic['years']
    #train_years = np.arange(1900,2015)
    lenDS = len(train_years)

    inversion_years = train_years
    
    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    load_data_and_gener_params = {'data_dir':data_in_dir, 'verbose':verbose}
    if lp_nathist_filtering:
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)

        load_data_and_gener_params['to_filter'] = ['nat', 'hist']
        load_data_and_gener_params['filtering_dic_filename'] = filtering_dic_filename
    load_data_and_gener_params['verbose'] = verbose

    data_dic = gt.load_basic_data_and_gener_df(**load_data_and_gener_params)
    
    #data_label       = data_dic['label']
    #all_models_src   = data_dic['models']
    all_forcings_src = data_dic['forcings']
    all_forcing_color_dic = data_dic['forcing_color_dic']
    all_forcing_inv_color_dic = data_dic['forcing_inv_color_dic']
    #all_forcing_color_names_dic    = data_dic['forcing_color_names_dic']
    #all_forcing_inv_color_names_dic = data_dic['forcing_inv_color_names_dic']
    #all_years        = data_dic['years']
    T_ghg_df,T_aer_df,T_nat_df,T_hist_df = data_dic['list_of_df']

    forcing_names = all_forcings_src[:4]
    forcing_colors = [all_forcing_color_dic[f.lower()] for f in forcing_names]
    forcing_inv_colors = [all_forcing_inv_color_dic[f.lower()] for f in forcing_names]

    #hist_inv_color = gt.lighter_color('#7f7f7f')   # gris (lighter)
    #hist_inv_color = gt.darker_color('#ff7f0e')  # orange (darker)
    hist_color = forcing_colors[-1]
    hist_inv_color = forcing_inv_colors[-1]

    #for i_trained,(trained_model) in enumerate(models_to_test) :
    for i_mod,mod_to_invert in enumerate(models_to_plot) : 

        # if Model to Inverse is not one from those from those used for training, because Obs or another Model !
        if mod_to_invert not in T_ghg_df['model'].unique() :
            # Source simulations for current model (from source data)
            GHG_mod_for_inv_array = T_ghg_df.drop('model', axis=1).iloc[:,-lenDS:].values
            AER_mod_for_inv_array = T_aer_df.drop('model', axis=1).iloc[:,-lenDS:].values
            NAT_mod_for_inv_array = T_nat_df.drop('model', axis=1).iloc[:,-lenDS:].values
            HIST_mod_for_inv_array = T_hist_df.drop('model', axis=1).iloc[:,-lenDS:].values
        else:
            # Source simulations for current model (from source data)
            GHG_mod_for_inv_array = T_ghg_df.loc[lambda df: df['model'] == mod_to_invert, :].drop('model', axis=1).iloc[:,-lenDS:].values
            AER_mod_for_inv_array = T_aer_df.loc[lambda df: df['model'] == mod_to_invert, :].drop('model', axis=1).iloc[:,-lenDS:].values
            NAT_mod_for_inv_array = T_nat_df.loc[lambda df: df['model'] == mod_to_invert, :].drop('model', axis=1).iloc[:,-lenDS:].values
            HIST_mod_for_inv_array = T_hist_df.loc[lambda df: df['model'] == mod_to_invert, :].drop('model', axis=1).iloc[:,-lenDS:].values

        experiment_name_in_sdir = f'Training-for-mod_{mod_to_invert}'
        experiment_name_out_sdir = f'Inversion-on_{mod_to_invert}'

        print(f"\n{'-'*132}\nInversion case {i_mod+1}/{len(models_to_plot)}) for modele {mod_to_invert})"+
              f"(inversion case: '{inversion_suffix}')")

        case_out_dir = os.path.join(data_out_dir, base_case_to_explore, cnn_name_base, experiment_name_out_sdir)
        print(f'Repertoire de sortie du cas: {case_out_dir}')
        if not os.path.exists(case_out_dir):
            print(f"\n *** Case inversion directory '{case_out_dir}/' not found. Skiping model case ...")
            continue

        if inv_label is None :
            inv_label = set_inv_labl_from_path (case_out_dir, settings_label=settings_label, verbose=verbose)

        suptitlelabel = f"{cnn_name_base} [{experiment_name_in_sdir}] [{data_and_training_label}] ({n_nnets} Nets)"
        print(suptitlelabel)

        case_figs_dir = os.path.join(figs_dir, base_case_to_explore, cnn_name_base, f'Settings-{inv_label}', experiment_name_out_sdir)
        print(f'Repertoire des figures du cas: {case_figs_dir}')
        if save_figs and not os.path.exists(case_figs_dir):
            os.makedirs(case_figs_dir)

        list_for_average_of_mean_Xinv = []
        list_for_average_of_mean_Yinv = []
        list_for_average_of_std_Xinv = []
        list_for_average_of_std_Yinv = []
        for innet in np.arange(n_nnets):
            #innet = 0

            inv_dir = os.path.join(case_out_dir,f'Inv_N{innet}',f'Settings-{inv_label}')

            if innet == 0:
                print(f'Inv_N{innet} - Inv_dir: {inv_dir}',end='' if verbose else '\n')
            else:
                print(f' ... Inv_N{innet}',end='' if verbose else '\n')
            
            # inverted data Xinv file prefix
            xinv_file_prefix = 'Xinv_multi_all-but-one'

            nnet_invert_prnt = gt.look_for_inversion_files( file_prefix=xinv_file_prefix, inv_dir=inv_dir, verbose=True)
            #nnet_invert_label = nnet_invert_prnt.replace('-',' ')
            #print('nnet_invert_label:',nnet_invert_label)

            # inverted data filename commun postfix
            #inv_postfix = f'{nnet_invert_prnt}_N{innet}_{net_label}-net_mod-{mod_to_invert}'
            #mlt_list_Xinv = pickle.load( open( os.path.join(inv_dir, f'Xinv_multi_all-but-one_{inv_postfix}_ST.p'), "rb" ) )
            #mlt_list_Yinv = pickle.load( open( os.path.join(inv_dir, f'Yinv_multi_all-but-one_{inv_postfix}_ST.p'), "rb" ) )
            #mlt_list_nbe = pickle.load( open( os.path.join(inv_dir, f'Nbe_multi_all-but-one_{inv_postfix}_ST.p'), "rb" ) ) 
            
            inversion_dic = pickle.load( open( os.path.join(inv_dir, f'Inv_dic.p'), "rb" ) )
            
            mlt_list_Xinv = inversion_dic['Xinv']
            mlt_list_Yinv = inversion_dic['Yinv']
            mlt_list_nbe = inversion_dic['nbe']
            mmlt_list_nbe_label = inversion_dic['nbe_label']

            # Averaging Xinv and Yinv paterns
            mlt_arr_mean_Xinv = np.array([Xinv.mean(axis=0) for Xinv in mlt_list_Xinv])
            mlt_arr_mean_Yinv = np.array([Yinv.mean(axis=0) for Yinv in mlt_list_Yinv])
            mlt_arr_std_Xinv = np.array([Xinv.std(axis=0,ddof=1) for Xinv in mlt_list_Xinv])
            mlt_arr_std_Yinv = np.array([Yinv.std(axis=0,ddof=1) for Yinv in mlt_list_Yinv])

            list_for_average_of_mean_Xinv.append(mlt_arr_mean_Xinv)
            list_for_average_of_mean_Yinv.append(mlt_arr_mean_Yinv)
            list_for_average_of_std_Xinv.append(mlt_arr_std_Xinv)
            list_for_average_of_std_Yinv.append(mlt_arr_std_Yinv)

            if innet == 0:
                added_mean_of_all = type(mlt_list_nbe[-1]) is list
                nb_inverted_single_patt = len(mlt_list_nbe) - 1 if added_mean_of_all else len(mlt_list_nbe)
                print(f"nb_inverted_single_patt: {nb_inverted_single_patt}, mlt_list_nbe for average: {mlt_list_nbe} not including last if composed, [added_mean_of_all: {added_mean_of_all}]")

        if verbose:
            print()

        figs_file = f"Fig{local_nb_label}_inverted-averaged-all-forcings-by-model-{nb_inverted_single_patt}-profiles_for-{n_nnets}-nets_{net_label}"
        if fixed_t_limits_ok :
            figs_file += "_FIX-T"
        figs_file += f".{fig_ext}"
        figs_filename = os.path.join(case_figs_dir,figs_file)

        if not force_plot and save_figs and os.path.isfile(figs_filename):
            print(f" ** {local_nb_label} figure already exists '{figs_filename}'. Figure not prepared")
            
        else:

            mlt_for_average_of_mean_Xinv = np.array(list_for_average_of_mean_Xinv) #.mean(axis=0)  # size for IPSL ((34, 1, 3, 133), (34, 1, 115))
            mlt_for_average_of_mean_Yinv = np.array(list_for_average_of_mean_Yinv) #.mean(axis=0)
            mlt_for_average_of_std_Xinv = np.array(list_for_average_of_std_Xinv) #.mean(axis=0)  # size for IPSL ((34, 1, 3, 133), (34, 1, 115))
            mlt_for_average_of_std_Yinv = np.array(list_for_average_of_std_Yinv) #.mean(axis=0)
            print(f'mlt_for_average_of_mean_Xinv/mlt_for_average_of_mean_Yinv shape: {mlt_for_average_of_mean_Xinv.shape} / {mlt_for_average_of_mean_Yinv.shape}')

            top = 0.92;    bottom = 0.04
            left = 0.06;   right = 0.98
            wspace = 0.05; hspace = 0.10
            if n_nnets < 5 :
                if n_nnets < 2 :
                    top = 0.72
                if n_nnets < 4 :
                    top = 0.82
                ncols = 1
                left = 0.06+0.16-wspace/2; right = 0.98-0.16+wspace/2
                suptitle_fontsize = "large"     # 12
            elif n_nnets < 9 :
                ncols = 2
                suptitle_fontsize = "x-large"   # 14.4
            else :
                ncols = 3
                suptitle_fontsize = "xx-large"  # 17.28
            
            nrows = int(np.ceil(n_nnets/ncols))
            
            fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(8*max(1.5,ncols),1.5+5*nrows),
                                    gridspec_kw={'hspace': hspace, 'wspace': wspace, 
                                                 'left': left,     'right': right,
                                                 'top' : top,      'bottom' : bottom })
    
            for innet in np.arange(n_nnets) :
                nnet_label = f'{innet}'

                if added_mean_of_all:
                    Xinvmean_arr = mlt_for_average_of_mean_Xinv[innet,0:-1]
                    Yinvmean_arr = mlt_for_average_of_mean_Yinv[innet,0:-1]
                    Xinvstd_arr = mlt_for_average_of_std_Xinv[innet,0:-1]
                    Yinvstd_arr = mlt_for_average_of_std_Yinv[innet,0:-1]
                else:
                    Xinvmean_arr = mlt_for_average_of_mean_Xinv[innet]
                    Yinvmean_arr = mlt_for_average_of_mean_Yinv[innet]
                    Xinvstd_arr = mlt_for_average_of_std_Xinv[innet]
                    Yinvstd_arr = mlt_for_average_of_std_Yinv[innet]
                
                if nrows == 1 and ncols == 1 :
                    ax = axes
                else:
                    ax = axes.flatten()[innet]  #iax//ncols,iax%ncols]
                
                # -------------------------------------------------------------
                # Plotting Inverted forcings
                print(f'Xinvmean_arr/Yinvmean_arr shape: {Xinvmean_arr.shape} / {Yinvmean_arr.shape}')
                print(f'Xinvstd_arr/Yinvstd_arr shape: {Xinvstd_arr.shape} / {Yinvstd_arr.shape}')
                tmp_xinvmean_arr = Xinvmean_arr.squeeze(axis=1)
                tmp_yinvmean_arr = Yinvmean_arr.squeeze(axis=1)
                tmp_xinvstd_arr = Xinvstd_arr.squeeze(axis=1)
                tmp_yinvstd_arr = Yinvstd_arr.squeeze(axis=1)
                print(f'tmp_xinvmean_arr/tmp_yinvmean_arr shape: {tmp_xinvmean_arr.shape} / {tmp_yinvmean_arr.shape}')
                print(f'tmp_xinvstd_arr/tmp_yinvstd_arr shape: {tmp_xinvstd_arr.shape} / {tmp_yinvstd_arr.shape}')
                # plotting inversed X: GHG, AER and NAT

                for x_inv in tmp_xinvmean_arr :
                    #print('x_inv shape:',x_inv.shape)
                    for iforc,(c,x_forc) in enumerate(zip(forcing_inv_colors[:3],x_inv)) :
                        #print('x_forc shape:',x_forc.shape, c)
                        ax.plot(inversion_years, x_forc[-lenDS:], color=c, alpha=0.05)
                # plotting inversed Y: HIST
                for y_inv in tmp_yinvmean_arr :
                    #print('y_inv shape:',y_inv.shape)
                    ax.plot(inversion_years, y_inv, color=forcing_inv_colors[-1], alpha=0.05)
               
                # -------------------------------------------------------------
                # Plotting the means of source data forcings X: GHG, AER and NAT
                for iforc,(forc,cs) in enumerate(zip(forcing_names[:3],forcing_colors[:3])) :
                    if forc == 'ghg':
                        FORC_src_from_model = GHG_mod_for_inv_array
                    elif forc == 'aer':
                        FORC_src_from_model = AER_mod_for_inv_array
                    elif forc == 'nat':
                        FORC_src_from_model = NAT_mod_for_inv_array
                    else:
                        FORC_src_from_model = None
                    #current_color_brighter = [[np.min((1,c*1.4)) for c in gt.hexcolor(hexcol)] for hexcol in cycle_color]  # for VAL
                    #c_darker = [ch*0.7 for ch in gt.hexcolor(c)]

                    current_meanFORC_m = FORC_src_from_model.mean(axis=0)
                    current_std_FORC_m = FORC_src_from_model.std(axis=0,ddof=1)
                    
                    ax.errorbar(inversion_years, current_meanFORC_m, yerr=current_std_FORC_m, color=cs,
                                lw=2, elinewidth=1, alpha=0.8, label=f"src {forc.upper()}")
                    #hdf[iforc], = ax.plot(inversion_years, current_meanFORC_m, color=c, lw=2, alpha=0.8)

                # -------------------------------------------------------------
                # Plotting Y: HIST source data forcings (plus errorbars)
                #current_HIST_m = HIST_[nbe,:]
                current_mean_HIST_m = HIST_mod_for_inv_array.mean(axis=0)
                current_std_HIST_m = HIST_mod_for_inv_array.std(axis=0,ddof=1)
                
                ax.errorbar(inversion_years, current_mean_HIST_m, yerr=current_std_HIST_m, color=hist_color,
                            lw=1.5, elinewidth=1, capsize=2, alpha=0.8, label=f"src {forcing_names[3].upper()}")
                hid, = ax.plot(inversion_years, current_mean_HIST_m, color=hist_color, lw=1.5, alpha=0.8)

                # -------------------------------------------------------------
                # Plotting the means of Inverted forcings X: GHG, AER and NAT
                tmp_xarr_mean = tmp_xinvmean_arr.mean(axis=0)
                tmp_xarr_std = tmp_xinvmean_arr.std(axis=0,ddof=1)
                #print('tmp_xarr_mean shape:',tmp_xarr_mean.shape, 'tmp_xarr_std shape:',tmp_xarr_std.shape)
                for iforc,(forc,ci,cs,x_mean,x_std) in enumerate(zip(forcing_names[:3],forcing_inv_colors[:3],forcing_colors[:3],tmp_xarr_mean,tmp_xarr_std)) :
                    #print('x_mean shape:',x_mean.shape, 'x_std shape:',x_std.shape, c)
                    ax.errorbar(inversion_years, x_mean[-lenDS:], yerr=x_std[-lenDS:], color=ci,
                                lw=2, alpha=0.8, label=f"inv {forc.upper()}")
    
                # -------------------------------------------------------------
                # Plotting Y: HIST Inverted forcings (plus errorbars)
                tmp_y = tmp_yinvmean_arr.mean(axis=0)
                #print("#DBG3# PRE STD tmp_yinvmean_arr.std(axis=0,ddof=1) ... tmp_arr.shape",tmp_yinvmean_arr.shape)
                tmp_y_err = tmp_yinvmean_arr.std(axis=0,ddof=1)
                #print("#DBG3# ,inversion_years.shape,tmp_y.shape,tmp_y_err.shape):",inversion_years.shape,tmp_y.shape,tmp_y_err.shape)

                ax.errorbar(inversion_years, tmp_y, yerr=tmp_y_err, color=hist_inv_color,
                            lw=2.5, elinewidth=2, alpha=0.8, label=f"inv {forcing_names[3].upper()}")
                #hih, = ax.plot(inversion_years, tmp_y, color=hist_inv_color, lw=2.5, alpha=0.8)

                if False:
                    ax.plot(inversion_years, tmp_y+2*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
                    ax.plot(inversion_years, tmp_y-2*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)

                    ax.plot(inversion_years, tmp_y+3*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
                    ax.plot(inversion_years, tmp_y-3*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
                #
                # -------------------------------------------------------------

                xmin,xmax = ax.get_xlim()
                ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
                ax.set_xlim([xmin,xmax])
    
                ax.grid(True,lw=0.75,ls=':')
                ax.set_title(f"Inverted {mod_to_invert} Averaged profiles [Net {nnet_label}]")
    
                if innet == 0:
                    plt.legend(loc='upper left', ncol=2)
                                    
                if fixed_t_limits_ok :
                    ax.set_ylim([t_min_limit,t_max_limit])

            # efface de la figure les axes non utilises
            if innet+1 < nrows*ncols :
                for jax in np.arange(innet+1,nrows*ncols):
                    ax = axes.flatten()[jax]  #iax//ncols,iax%ncols]
                    ax.set_visible(False)
    
            members_label = f"{nb_inverted_single_patt} HIST source profiles inverted"+(" - NAT and HIST LP filtered" if lp_nathist_filtering else "")+("- FIX-T" if fixed_t_limits_ok else "")
            #members_label = nbe_inverted_label+(" - NAT & HIST LP filtered" if lp_nathist_filtering else "")
    
            plt.suptitle(f"Inversion of profiles averaged for Model {mod_to_invert} [{members_label}] [{n_nnets_label}] [{net_label.upper()}]\n"+\
                         f"{base_case_to_explore} / {cnn_name_base}\n"+\
                         f"[Data: {data_and_training_label} - Inversion Settings: {inv_label}]",
                         #"[Dark orange: inversed HIST | Normal RGB: inversed GHG-AER-NAT | Darker RGB: GHG-AER-NAT source | Black: HIST source]",
                         size=suptitle_fontsize, y=0.99)
    
            #figs_file = f"Fig{local_nb_label}_inverted-averaged-all-forcings-by-model-{nb_inverted_single_patt}-profiles_by-net.{fig_ext}"
            #figs_filename = os.path.join(case_figs_dir,figs_file)

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


def plot_averaged_inv_all_forcings_by_net (base_case_to_explore, sub_case_to_explore, inversion_suffix,
                                           trained_with_all=False, settings_label=None, sset_sdir_dic=None, 
                                           models_to_plot=None, load_best_val=False, load_best_val2nd=False,
                                           plot_src4inv=False, plot_inv_x=True,
                                           list_of_trained_models=None,           # valid only in case of "experiences jumelles" and Obs inversion
                                           data_in_dir=None, data_out_dir=None, figs_dir=None, save_figs=True,
                                           source_dirname='data_source_pl',
                                           plot_forc_shaded_region=False, alpha_for_shaded=0.4, lw_for_shaded=0.5,
                                           hatch_src_for_shaded='||', hatch_for_shaded='//', hatch_for_shaded_inv='\\\\\\', errorlimits_percent=None, errorlimits_n_rms=1,
                                           alpha_for_mean_shaded=0.4, lw_for_mean_shaded=0.5, hatch_for_mean_shaded_inv='\\\\',
                                           force_plot=False, force_write=False, plot_mean_err=False, mean_err_lightness_factor=0.25,
                                           inv_label=None, t_limits=None,
                                           train_years=np.arange(1900,2015),
                                           local_nb_label="PlotAveInvAllForcfByNet", fig_ext='png',
                                           figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w'},
                                           #obs_filename = 'obs.npy', obs_name = 'OBS',
                                           obs_set=['OBS','HadCRUT','HadCRUT200','HadCRUT+AR1'],
                                           lp_obs_filtering=False, lp_obs_filtering_dictionary=None,
                                           lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
                                           n_models_to_plot=None, figsize=None,
                                           verbose=False,
                                          ) :
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    from scipy.stats import norm

    import generic_tools_pl as gt   # like hexcolor(), ...

    fixed_t_limits_ok = False
    if t_limits is not None:
        fixed_t_limits_ok = True
        t_min_limit,t_max_limit = t_limits

    # Repertoire des donnees
    if data_in_dir is None :
        data_in_dir = gt.get_source_data_dir(dirname=source_dirname, verbose=verbose)

    if data_out_dir is None :
        data_out_dir = './data_out'

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
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = gt.retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    n_nnets = base_case_dic['n_nnets']
    data_and_training_label = base_case_dic['data_and_training_label']
    lp_nathist_filtering = base_case_dic['lp_nathist_filtering']
    if verbose:
        print(f" - Number of NNets: {n_nnets}")
        print(f" - Data and Training set Llabel: {data_and_training_label}")
        print(f" - Low-pass NAT and HIST filtering: {lp_nathist_filtering}")
        if apply_std_coeff :
            print(f" - Local STD coefficient: {local_std_coeff}")
        else:
            print(f" - Applying STD coefficient: <NOT ACTIVATED>")

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
    extrap_label = sub_case_dic['extrapolation_label']
    if verbose:
        print(f" - List of sizes of CNN hidden layers or (kernel sizes): {kern_size_list}")
        print(f" - Data Extrapolation Label: {extrap_label}")

    if n_nnets == 1:
        n_nnets_label = "Only 1 nnet"
    else:
        n_nnets_label = "{n_nnets} nets"
    
    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore
    
    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    load_data_and_gener_params = {'data_dir':data_in_dir, 'verbose':verbose}
    if lp_nathist_filtering:
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)

        load_data_and_gener_params['to_filter'] = ['nat', 'hist']
        load_data_and_gener_params['filtering_dic_filename'] = filtering_dic_filename
    load_data_and_gener_params['verbose'] = verbose

    data_dic = gt.load_basic_data_and_gener_df(**load_data_and_gener_params)
    
    #data_label       = data_dic['label']
    #all_models_src   = data_dic['models']
    #all_forcings_src = data_dic['forcings']
    #all_forcing_color_dic     = data_dic['forcing_color_dic']
    #all_forcing_inv_color_dic = data_dic['forcing_inv_color_dic']
    #all_forcing_color_names_dic    = data_dic['forcing_color_names_dic']
    #all_forcing_inv_color_names_dic = data_dic['forcing_inv_color_names_dic']
    #all_years        = data_dic['years']
    _,_,_,T_hist_df = data_dic['list_of_df']

    if load_best_val2nd :
        net_label = 'best-val2nd'
        #net_filename = 'Net_best-val2nd.pt'
    elif load_best_val :
        net_label = 'best-val'
        #net_filename = 'Net_best-val.pt'
    else:
        net_label = 'last'
        #net_filename = 'Net.pt'

    inversion_combi_dic = gt.read_data_set_characteristics(data_in_dir, file_prefix='inversion', 
                                                           set_label=inversion_suffix,
                                                           verbose=verbose)
    
    # Load INV forcing data as DataFrames
    model_names, all_years, inversion_mod_df, \
        data_inversion_dic = gt.load_forcing_data(data_in_dir, file_prefix='inversion',
                                                  dataframe=True,
                                                  set_label=inversion_suffix, forcing_names=inversion_combi_dic['forcings'],
                                                  verbose=verbose)
    if list_of_trained_models is None :
        list_of_trained_models = model_names
        
    if verbose :
        print(data_inversion_dic.keys())
        print(type(data_inversion_dic['ghg']),data_inversion_dic['ghg'].shape)
    inversion_GHG_df = data_inversion_dic['ghg']
    inversion_AER_df = data_inversion_dic['aer']
    inversion_NAT_df = data_inversion_dic['nat']

    if verbose:
        print(f"\ninversion_combi_dic models .....  {inversion_combi_dic['models']}")
        print(  f"load_forcing_data model_names ..  {model_names}")

    if models_to_plot is None :
        models_to_plot = inversion_combi_dic['models']
        if verbose:
            print(f"\nList of models found in Inversion data set:\n  {models_to_plot}")
        
        #models_to_plot = all_models_src
        #models_to_plot = models_in_invert_data
        #models_to_plot = ['BCC-CSM2-MR', 'FGOALS-g3', 'CanESM5', 'CNRM-CM6-1', 'ACCESS-ESM1-5', 'IPSL-CM6A-LR', 'MIROC6', 'HadGEM3-GC31-LL', 'MRI-ESM2-0', 'GISS-E2-1-G', 'CESM2', 'NorESM2-LM'] #, 'GFDL-ESM4']
        #models_to_plot = ['IPSL-CM6A-LR']
        #models_to_plot = ['BCC-CSM2-MR']
        #models_to_plot = ['FGOALS-g3']
        #models_to_plot = ['NorESM2-LM']

    models_label, models_label_prnt = gt.models_title_labels(models_to_plot)

    ################# TEST MODEL 5 ONLY ################    
    #m = 5
    ################################
    #print(f"\nModele {m}- {model_names[m]} en inversion [{inv_label}]:")

    cnn_name_base = sub_case_to_explore

    #all_years = train_combi_dic['years']
    #train_years = np.arange(1900,2015)
    lenDS = len(train_years)

    inversion_years = train_years
    
    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    load_data_and_gener_params = {'data_dir':data_in_dir, 'verbose':verbose}
    if lp_nathist_filtering:
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)
        load_data_and_gener_params['to_filter'] = ['nat', 'hist']
        load_data_and_gener_params['filtering_dic_filename'] = filtering_dic_filename
    load_data_and_gener_params['verbose'] = verbose

    data_dic = gt.load_basic_data_and_gener_df(**load_data_and_gener_params)
    
    #data_label       = data_dic['label']
    all_models_src   = data_dic['models']
    all_forcings_src = data_dic['forcings']
    all_forcing_color_dic = data_dic['forcing_color_dic']
    all_forcing_inv_color_dic = data_dic['forcing_inv_color_dic']
    #all_forcing_color_names_dic    = data_dic['forcing_color_names_dic']
    #all_forcing_inv_color_names_dic = data_dic['forcing_inv_color_names_dic']
    #all_years        = data_dic['years']
    T_ghg_df,T_aer_df,T_nat_df,T_hist_df = data_dic['list_of_df']

    forcing_names = all_forcings_src[:4]
    forcing_colors = [all_forcing_color_dic[f.lower()] for f in forcing_names]
    forcing_inv_colors = [all_forcing_inv_color_dic[f.lower()] for f in forcing_names]

    forcing_src_colors = [gt.lighter_color(cs) for cs in forcing_colors]

    #hist_inv_color = gt.lighter_color('#7f7f7f')   # gris (lighter)
    #hist_inv_color = gt.darker_color('#ff7f0e')  # orange (darker)
    hist_color = forcing_colors[-1]
    hist_inv_color = forcing_inv_colors[-1]

    if list_of_trained_models is None :
        list_of_trained_models = all_models_src

    # Compute intermodel mean and transpose to have models as columns ...
    GHG_ens_df  = T_ghg_df.groupby('model').mean().transpose()[all_models_src]
    AER_ens_df  = T_aer_df.groupby('model').mean().transpose()[all_models_src]
    NAT_ens_df  = T_nat_df.groupby('model').mean().transpose()[all_models_src]
    HIST_ens_df = T_hist_df.groupby('model').mean().transpose()[all_models_src]

    # Build "all_but" DataFrames
    # add_obs_options = {}
    # if obs_name in models_to_plot :
    #     add_obs_options = { 'add_for_obs':True, 'obsname':obs_name } 
    #     if verbose:
    #         print(f"adding {obs_name} column to model list for 'all_but' data")
    add_obs_options = {}
    for curent_obs_name in obs_set :
        if curent_obs_name in models_to_plot :
            if 'obsname' in add_obs_options.keys() :
                if type(add_obs_options['obsname']) is list:
                    list_of_obs_names = add_obs_options['obsname']+[curent_obs_name]
                else :
                    list_of_obs_names = [add_obs_options['obsname'], curent_obs_name]
                add_obs_options['obsname'] = list_of_obs_names
            else:
                add_obs_options = { 'add_for_obs':True, 'obsname':curent_obs_name } 
            if verbose:
                print(f"adding {curent_obs_name} column to model list for 'all_but' data - add_obs_options:",add_obs_options)
    #
    GHG_ens_all_but_df, AER_ens_all_but_df, NAT_ens_all_but_df, \
        HIST_ens_all_but_df = gt.build_all_but_df(all_models_src,
                                               GHG_ens_df, AER_ens_df,
                                               NAT_ens_df, HIST_ens_df,
                                               **add_obs_options,
                                               verbose=verbose)
    
    #GHG_ens_all_but_all_df, AER_ens_all_but_all_df, NAT_ens_all_but_all_df, \
    #    HIST_ens_all_but_all_df = gt.build_all_but_df(None,
    #                                           GHG_ens_df, AER_ens_df,
    #                                           NAT_ens_df, HIST_ens_df)

    # if not trained_with_all :
    #     # Looking fot Net0 which models for inversions where trained
    #     single_models_already_trained = []
    #     for i_mod,xmod_from_training in enumerate(all_models_src) :
    #         xexperiment_name_in_sdir = f'Training-for-mod_{xmod_from_training}'
    #         xcase_in_dir = os.path.join(data_out_dir, base_case_to_explore, cnn_name_base, xexperiment_name_in_sdir)
    #         xnet_dir = os.path.join(xcase_in_dir,f'CNN_N{0}')
    #         if os.path.exists(xnet_dir):
    #             single_models_already_trained.append(xmod_from_training)
    if not trained_with_all :
        # Looking for already trained models
        single_models_already_trained = []
        for i_mod,xmod_from_training in enumerate(list_of_trained_models) :
            xexperiment_name_in_sdir = f'Training-for-mod_{xmod_from_training}'
            xcase_in_dir = os.path.join(data_out_dir, base_case_to_explore, cnn_name_base, xexperiment_name_in_sdir)
            xnet_dir = os.path.join(xcase_in_dir,f'CNN_N{0}')
            if os.path.exists(xnet_dir):
                single_models_already_trained.append(xmod_from_training)

        print('single_models_already_trained:',single_models_already_trained)

    # # obs_filename = 'obs.npy', obs_name = 'OBS'
    # if obs_name in models_to_plot :
    #     print(f"\nLoading {obs_name} to plot:")
    #     obs_df = gt.load_obs_data(obs_filename, data_dir=data_in_dir, last_year=train_years[-1], return_as_df=True, verbose=verbose)
    
    #     if lp_obs_filtering :
    #         print("Filtering {obs_name} ... ")
    #         obs_df = gt.filter_forcing_df(obs_df, filt_dic=lp_obs_filtering_dictionary, verbose=verbose)
    
    n_models = len(models_to_plot)
    
    inverted_model_to_plot = models_to_plot
    source_model_data_to_plot = models_to_plot
    models_sdirs_to_plot = models_to_plot

    case_figs_dir = None
    figs_filename = None
    
    for innet in np.arange(n_nnets):
        #innet = 0
        
        if inv_label is not None :
            case_figs_dir = os.path.join(figs_dir, base_case_to_explore, cnn_name_base, f'Settings-{inv_label}')
            print(f'Repertoire des figures du cas: {case_figs_dir}')
            if save_figs and not os.path.exists(case_figs_dir):
                os.makedirs(case_figs_dir)
            
            figs_file = f"Fig{local_nb_label}_inverted-averaged-all-forcings-{n_models}-models-{models_label_prnt}_net-{innet}_{net_label}"
            if fixed_t_limits_ok :                figs_file += "_FIX-T"
            if plot_forc_shaded_region:           figs_file += f"_Shaded-{interval_label}"
            else:                                 figs_file += f"_ErrBar-{interval_label}"
            figs_file += f".{fig_ext}"
            figs_filename = os.path.join(case_figs_dir,figs_file)

        if case_figs_dir is not None and not force_plot and save_figs and os.path.isfile(figs_filename):
            print(f" ** {local_nb_label} figure already exists '{figs_filename}'. Figure not prepared")
        
        else:
            print(f"Model to invert:   {inverted_model_to_plot}")
            print(f"Model data_source: {source_model_data_to_plot}")
            print(f" ... and sdirs:    {models_sdirs_to_plot}")
            
            #for i_trained,(trained_model) in enumerate(models_to_test) :
            #for i_mod,(mod_to_invert,ax) in enumerate(zip(models_to_plot,axes.flatten()[:len(models_to_plot)])) : 
            k_ax = 0
            for i_mod,(mod_to_invert,mod_data_source,mod_sdir) in enumerate(zip(inverted_model_to_plot,source_model_data_to_plot,models_sdirs_to_plot)) : 
                print(f"model_to_invert: '{mod_to_invert}' using data from model '{mod_data_source}'")

                # Source simulations for current model (from source data)
                if mod_data_source in T_ghg_df.model.values.tolist() :
                    GHG_mod_for_inv_array = T_ghg_df.loc[lambda df: df['model'] == mod_data_source, :].drop('model', axis=1).iloc[:,-lenDS:].values
                    AER_mod_for_inv_array = T_aer_df.loc[lambda df: df['model'] == mod_data_source, :].drop('model', axis=1).iloc[:,-lenDS:].values
                    NAT_mod_for_inv_array = T_nat_df.loc[lambda df: df['model'] == mod_data_source, :].drop('model', axis=1).iloc[:,-lenDS:].values
                else:
                    GHG_mod_for_inv_array = AER_mod_for_inv_array = NAT_mod_for_inv_array = None
                ## ##########################################################
                ## Données d'Inversion: Sortie $Y_{mod}$
                ## ##########################################################
                # Profils HIST du modele en cours .. selection des lignes du modele ... retire la colone 'model' ... selectionne les colones (selon lenDS) ... values (array)
                # if mod_to_invert == obs_name :
                #     print(f"\nProfils HIST to invert is {obs_name} !!")
                #     HIST_mod_for_inv_array = obs_df.iloc[:,-lenDS:].values
                # else:
                # Profils HIST du modele en cours .. selection des lignes du modele ... retire la colone 'model' ... selectionne les colones (selon lenDS) ... values (array)
                HIST_mod_for_inv_array = T_hist_df.loc[lambda df: df['model'] == mod_data_source, :].drop('model', axis=1).iloc[:,-lenDS:].values
                
                GHG_ens_ab_arr_m = GHG_ens_all_but_df[[mod_data_source]].transpose().iloc[:,-lenDS:].values
                AER_ens_ab_arr_m = AER_ens_all_but_df[[mod_data_source]].transpose().iloc[:,-lenDS:].values
                NAT_ens_ab_arr_m = NAT_ens_all_but_df[[mod_data_source]].transpose().iloc[:,-lenDS:].values

                if trained_with_all :
                    models_from_training = mod_to_invert
                elif mod_to_invert in single_models_already_trained :  # OBS or other not in Training
                     models_from_training = mod_to_invert
                else :
                    #models_from_training = list_of_trained_models
                    models_from_training = single_models_already_trained

                if n_models_to_plot is not None :
                    n_models_to_plot_in_one_page = n_models_to_plot
                
                # obs_filename = 'obs.npy', obs_name = 'OBS'
                if mod_to_invert in obs_set:
                    print(f"\nLoading {mod_to_invert} as Obs, for inversion:")

                    obs_df = gt.load_obs_data(obs_label=mod_to_invert, data_dir=data_in_dir,
                                              return_as_df=True, verbose=verbose)

                    if verbose :
                        display(obs_df)

                    if lp_obs_filtering :
                        print("LP filtering ... ")
                        obs_df = gt.filter_forcing_df(obs_df, filt_dic=lp_obs_filtering_dictionary, verbose=verbose)
                        if verbose :
                            print("AFTER Filtering:")
                            display(obs_df)

                    if n_models_to_plot is None :
                        n_models_to_plot_in_one_page = 1 if np.isscalar(models_from_training) else len(models_from_training)
                else:
                    if n_models_to_plot is None :
                        n_models_to_plot_in_one_page = 1
                
                print('n_models_to_plot_in_one_page:',n_models_to_plot_in_one_page)

                if n_models_to_plot_in_one_page < 5 :
                    ncols = 1
                    suptitle_fontsize = "medium"     # 12
                elif n_models_to_plot_in_one_page < 9 :
                    ncols = 2
                    suptitle_fontsize = "x-large"   # 14.4
                else :
                    ncols = 3
                    suptitle_fontsize = "xx-large"  # 17.28
                nrows = int(np.ceil(n_models_to_plot_in_one_page/ncols))
                
                bottom = 0.03
                left   = 0.10; right  = 0.98
                wspace = 0.05; hspace = 0.12
    
                if   nrows == 1 : bottom = 0.08; top = 0.85
                elif nrows == 2 : bottom = 0.05; top = 0.89
                elif nrows == 3 : bottom = 0.03; top = 0.91
                else            :                top = 0.94
    
                if figsize is None :
                    figsize=(9.5*ncols,1+6*nrows)
                
                print('nrows,ncols:',nrows,ncols,', figsize:',figsize)

                #fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(5*max(2,ncols),1+3.5*nrows),
                fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=False,sharey=True,figsize=figsize,
                                        gridspec_kw={'hspace': hspace, 'wspace': wspace,
                                                      'left' : left,   'right' : right,
                                                      'top'  : top,    'bottom': bottom })    
                
                if np.isscalar(models_from_training) :
                    models_from_training = [ models_from_training ]

                for i_trmod,mod_from_training in enumerate(models_from_training) :
                    experiment_name_in_sdir = f'Training-for-mod_{mod_to_invert}'
                    experiment_name_out_sdir = f'Inversion-on_{mod_sdir}'
                    if not trained_with_all and mod_to_invert != mod_from_training :
                        experiment_name_out_sdir += f"-{mod_from_training}"
                        mod_to_invert_for_title = f"{mod_to_invert}/{mod_from_training}"
                    else:
                        mod_to_invert_for_title = mod_to_invert
    
                    print(f"\n{'-'*132}\nInversion case {i_mod+1}/{len(models_sdirs_to_plot)}) for modele {mod_to_invert_for_title}@{mod_sdir})"+
                          f"(inversion case: '{inversion_suffix}')")
            
                    case_out_dir = os.path.join(data_out_dir, base_case_to_explore, cnn_name_base, experiment_name_out_sdir)
                    print(f'Repertoire de sortie du cas: {case_out_dir}')
                    if not os.path.exists(case_out_dir):
                        print(f"\n *** Case inversion directory '{case_out_dir}/' not found. Skiping model case ...")
                        continue    
            
                    if inv_label is None :
                        inv_label = set_inv_labl_from_path (case_out_dir, settings_label=settings_label, verbose=verbose)
            
                    if sset_sdir_dic is not None and mod_to_invert in sset_sdir_dic.keys() :
                        sset_sdir = sset_sdir_dic[mod_to_invert]
                    else:
                        sset_sdir = None

                    if case_figs_dir is None :
                        case_figs_dir = os.path.join(figs_dir, base_case_to_explore, cnn_name_base, f'Settings-{inv_label}',mod_to_invert)
                        if sset_sdir is not None :
                            case_figs_dir = os.path.join(case_figs_dir,sset_sdir)
                        
                        print(f'Repertoire des figures du cas: {case_figs_dir}')
                        if save_figs and not os.path.exists(case_figs_dir):
                            os.makedirs(case_figs_dir)
                
                        figs_file = f"Fig{local_nb_label}_inverted-averaged-all-forcings-{n_models}-models-{models_label_prnt}_net-{innet}_{net_label}"
                        if plot_mean_err :                    figs_file += "_MeanInv"
                        if plot_src4inv :                     figs_file += "_+XsrcMod"
                        if plot_inv_x :                       figs_file += "_+XiniMod"
                        if fixed_t_limits_ok :                figs_file += "_FIX-T"
                        if plot_forc_shaded_region:           figs_file += f"_Shaded-{interval_label}"
                        else:                                 figs_file += f"_ErrBar-{interval_label}"
                        figs_file += f".{fig_ext}"
                        figs_filename = os.path.join(case_figs_dir,figs_file)
                        
                        if not force_plot and save_figs and os.path.isfile(figs_filename):
                            print(f" ** {local_nb_label} figure already exists '{figs_filename}'. Figure not prepared")
                        
                        #fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(8*max(1.5,ncols),1.5+5*nrows),
                        # fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=False,sharey=True,figsize=(9*ncols,1+5*nrows),
                        #                         gridspec_kw={'hspace': hspace, 'wspace': wspace, 
                        #                                       'left': left,     'right': right,
                        #                                       'top' : top,      'bottom' : bottom })
                    
                    if nrows > 1 and ncols > 1 :
                        ax = axes.flatten()[k_ax]
                    elif nrows > 1 or ncols > 1 :
                        ax = axes[k_ax]
                    else:
                        ax = axes
                    k_ax += 1
                    
                    suptitlelabel = f"{cnn_name_base} [{experiment_name_in_sdir}] [{data_and_training_label}] ({n_nnets} Nets)"+(f" [{sset_sdir}]" if sset_sdir is not None else "")
                    print(suptitlelabel)
            
                    inv_nnet_dir = f'Inv_N{innet}'
                        
                    inv_dir = os.path.join(case_out_dir, inv_nnet_dir, f'Settings-{inv_label}')
                    if sset_sdir is not None :
                        inv_dir = os.path.join(inv_dir,sset_sdir)

                    if innet == 0:
                        print(f'Inv_N{innet} - Inv_dir: {inv_dir}',end='' if verbose else '\n')
                    else:
                        print(f' ... Inv_N{innet}',end='' if verbose else '\n')
                    
                    # inverted data Xinv file prefix
                    #xinv_file_prefix = 'Xinv_multi_all-but-one'
        
                    #nnet_invert_prnt = gt.look_for_inversion_files( file_prefix=xinv_file_prefix, inv_dir=inv_dir, verbose=verbose)
                    #nnet_invert_label = nnet_invert_prnt.replace('-',' ')
                    #print('nnet_invert_label:',nnet_invert_label)
        
                    # inverted data filename commun postfix
                    #inv_postfix = f'{nnet_invert_prnt}_N{innet}_{net_label}-net_mod-{mod_to_invert}'
        
                    inversion_dic = pickle.load( open( os.path.join(inv_dir, f'Inv_dic.p'), "rb" ) )
                    
                    mlt_list_Xinv = inversion_dic['Xinv']
                    mlt_list_Yinv = inversion_dic['Yinv']
                    mlt_list_nix = inversion_dic['forc_xinv_index'].tolist()
                    mlt_list_nbe = inversion_dic['nbe']
                    mlt_list_nbe_label = inversion_dic['nbe_label']

                    # mlt_list_Xinv = pickle.load( open( os.path.join(inv_dir, f'Xinv_ST.p'), "rb" ) )
                    # mlt_list_Yinv = pickle.load( open( os.path.join(inv_dir, f'Yinv_ST.p'), "rb" ) )
                    # mlt_list_nbe = pickle.load( open( os.path.join(inv_dir, f'Nbe_ST.p'), "rb" ) ) 
                    print(f'mlt_list_Xinv/mlt_list_Yinv/mlt_list_nbe lists lengths: {len(mlt_list_Xinv)} / {len(mlt_list_Yinv)} / {len(mlt_list_nbe)}')
                    print(f'mlt_list_Xinv[0]/mlt_list_Yinv[0]/mlt_list_nbe[0] lists types: {type(mlt_list_Xinv[0])} / {type(mlt_list_Yinv[0])} / {type(mlt_list_nbe[0])}')
                    print(f'mlt_list_Xinv[0]/mlt_list_Yinv[0] shapes & mlt_list_nbe[0]: {mlt_list_Xinv[0].shape} / {mlt_list_Yinv[0].shape} & {mlt_list_nbe[0]}')
    
                    n_x_forc = mlt_list_Xinv[0].shape[0]
                    n_y_forc = mlt_list_Yinv[0].shape[0]
                    
                    # Averaging Xinv and Yinv paterns
                    Xinv_arr = np.array(mlt_list_Xinv).squeeze(axis=2)
                    Yinv_arr = np.array(mlt_list_Yinv).squeeze(axis=2)
                    Xinvmean_arr = np.array([Xinv.mean(axis=0).squeeze(axis=0) for Xinv in mlt_list_Xinv])
                    Yinvmean_arr = np.array([Yinv.mean(axis=0).squeeze(axis=0) for Yinv in mlt_list_Yinv])
                    Xinvstd_arr = np.array([Xinv.std(axis=0,ddof=1).squeeze(axis=0) for Xinv in mlt_list_Xinv])
                    Yinvstd_arr = np.array([Yinv.std(axis=0,ddof=1).squeeze(axis=0) for Yinv in mlt_list_Yinv])
                    Nix_list = mlt_list_nix
                    Nbe_list = mlt_list_nbe
                    Nbe_label_list = mlt_list_nbe_label
                    print(f'Xinv_arr/Yinv_arr shape ........... {Xinv_arr.shape} / {Yinv_arr.shape}')
                    print(f'Xinvmean_arr/Yinvmean_arr shape ... {Xinvmean_arr.shape} / {Yinvmean_arr.shape}')
                    print(f'Xinvstd_arr/Yinvstd_arr shape ..... {Xinvstd_arr.shape} / {Yinvstd_arr.shape}')
            
                    if innet == 0:
                        added_mean_of_all = type(Nbe_list[-1]) is list
                        
                        if added_mean_of_all:
                            raise Exception(f" *** added_mean_of_all is activated because Nbe_list[-1] is a list !!  ABORTING because it is noy considered yet\n")
                        
                        nb_inverted_single_patt = len(Nbe_list) - 1 if added_mean_of_all else len(Nbe_list)
                        print(f"nb_inverted_single_patt: {nb_inverted_single_patt}, Nbe_list for average: {Nbe_list} not including last if composed, [added_mean_of_all: {added_mean_of_all}]")
        
                    if verbose:
                        print()
            
                    #for innet in np.arange(n_nnets) :
                    #nnet_label = f'{innet}'
    
                    # if added_mean_of_all:
                    #     Xinvmean_arr = mlt_arr_mean_Xinv[0:-1]
                    #     Yinvmean_arr = mlt_arr_mean_Yinv[0:-1]
                    #     Xinvstd_arr  = mlt_arr_std_Xinv[0:-1]
                    #     Yinvstd_arr  = mlt_arr_std_Yinv[0:-1]
                    # else:
                    # Xinv_arr     = mlt_arr_Xinv
                    # Yinv_arr     = mlt_arr_Yinv
                    # Xinvmean_arr = mlt_arr_mean_Xinv
                    # Yinvmean_arr = mlt_arr_mean_Yinv
                    # Xinvstd_arr  = mlt_arr_std_Xinv
                    # Yinvstd_arr  = mlt_arr_std_Yinv
                    print(f'Xinvmean_arr/Yinvmean_arr shape ... {Xinvmean_arr.shape} / {Yinvmean_arr.shape}')
                    print(f'Xinvstd_arr/Yinvstd_arr shape ..... {Xinvstd_arr.shape} / {Yinvstd_arr.shape}')
                                    
                    # -------------------------------------------------------------
                    # Plotting Inverted forcings
                    #tmp_xinvmean_arr = Xinvmean_arr.squeeze(axis=1)
                    #tmp_yinvmean_arr = Yinvmean_arr.squeeze(axis=1)
                    #tmp_xinvstd_arr = Xinvstd_arr.squeeze(axis=1)
                    #tmp_yinvstd_arr = Yinvstd_arr.squeeze(axis=1)
                    #tmp_nbe_list = mlt_list_nbe
                    #tmp_nbe_label_list = mlt_list_nbe_label
                    #print(f'tmp_xinvmean_arr/tmp_yinvmean_arr shape: {tmp_xinvmean_arr.shape} / {tmp_yinvmean_arr.shape}')
                    #print(f'tmp_xinvstd_arr/tmp_yinvstd_arr shape: {tmp_xinvstd_arr.shape} / {tmp_yinvstd_arr.shape}')
                    # plotting inversed X: GHG, AER and NAT
    
                    # for x_inv in tmp_xinvmean_arr :
                    #     #print('x_inv shape:',x_inv.shape)
                    #     for iforc,(c,x_forc) in enumerate(zip(forcing_inv_colors[:3],x_inv)) :
                    #         #print('x_forc shape:',x_forc.shape, c)
                    #         ax.plot(inversion_years, x_forc[-lenDS:], color=c, alpha=0.05)
                    # # plotting inversed Y: HIST
                    # for y_inv in tmp_yinvmean_arr :
                    #     #print('y_inv shape:',y_inv.shape)
                    #     ax.plot(inversion_years, y_inv, color=forcing_inv_colors[-1], alpha=0.05)
                   
                    # -------------------------------------------------------------
                    # Plotting the means of source data forcings X: GHG, AER and NAT
                    if plot_src4inv :
                        if GHG_mod_for_inv_array is None :
                            print(f" ** No SRC Forcing profiles for inversion  to plot for model '{mod_data_source}' **")
                        else:
                            #for iforc,(forc,cs) in enumerate(zip(forcing_names[:3],forcing_src_colors[:3])) :
                            for iforc,(forc,cs) in enumerate(zip(forcing_names,forcing_src_colors)) :
                                if forc == 'ghg':
                                    FORC_src_from_model = GHG_mod_for_inv_array
                                elif forc == 'aer':
                                    FORC_src_from_model = AER_mod_for_inv_array
                                elif forc == 'nat':
                                    FORC_src_from_model = NAT_mod_for_inv_array
                                elif forc == 'hist':
                                    FORC_src_from_model = HIST_mod_for_inv_array
                                else:
                                    FORC_src_from_model = None
                                #current_color_brighter = [[np.min((1,c*1.4)) for c in gt.hexcolor(hexcol)] for hexcol in cycle_color]  # for VAL
                                #c_darker = [ch*0.7 for ch in gt.hexcolor(c)]
        
                                print(f'{forc} FORC_src_from_model shape:',FORC_src_from_model.shape)
            
                                if FORC_src_from_model.shape[0] == 1 :
                                    current_meanFORC_m = FORC_src_from_model.squeeze(axis=0)
                                    current_std_FORC_m = None
                                    print('MONO FORC_src_from_model: current_meanFORC_m shape:',current_meanFORC_m.shape)
        
                                else:
                                    current_meanFORC_m = FORC_src_from_model.mean(axis=0)
                                    current_std_FORC_m = local_std_coeff * FORC_src_from_model.std(axis=0,ddof=1)
                                    print('MULTI FORC_src_from_model: current_meanFORC_m shape:',current_meanFORC_m.shape)
        
                                if plot_forc_shaded_region :
                                    if current_std_FORC_m is not None:
                                        ax.fill_between(inversion_years, current_meanFORC_m - current_std_FORC_m, current_meanFORC_m + current_std_FORC_m,
                                                        ec=cs, fc=gt.lighter_color(cs), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                                        hatch=hatch_src_for_shaded, zorder=2)
                                        ax.fill(np.NaN, np.NaN, 
                                                ec=cs, fc=gt.lighter_color(cs), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                                hatch=hatch_src_for_shaded, label=f"src {forc.upper()}")
                                    ax.plot(inversion_years, current_meanFORC_m, color=cs, lw=1, ls='-', label=f"src {forc.upper()}")
                                    
                                else:
                                    ax.errorbar(inversion_years, current_meanFORC_m, yerr=current_std_FORC_m, color=cs,
                                                lw=2, elinewidth=1, alpha=0.8, label=f"src {forc.upper()}")
                                    #hdf[iforc], = ax.plot(inversion_years, current_meanFORC_m, color=c, lw=2, alpha=0.8)
                    
                    # -------------------------------------------------------------
                    # Plotting the X-inv ??? of source data forcings X: GHG, AER and NAT
                    if plot_inv_x :
                        for iforc,(forc,cs) in enumerate(zip(forcing_names[:3],forcing_colors[:3])) :
                            if forc == 'ghg':
                                FORC_ini_for_inv_array = inversion_GHG_df.drop('model', axis=1).loc[Nix_list,inversion_years].values
                            elif forc == 'aer':
                                FORC_ini_for_inv_array = inversion_AER_df.drop('model', axis=1).loc[Nix_list,inversion_years].values
                            elif forc == 'nat':
                                FORC_ini_for_inv_array = inversion_NAT_df.drop('model', axis=1).loc[Nix_list,inversion_years].values
                            else:
                                FORC_ini_for_inv_array = None
                            #current_color_brighter = [[np.min((1,c*1.4)) for c in gt.hexcolor(hexcol)] for hexcol in cycle_color]  # for VAL
                            #c_darker = [ch*0.7 for ch in gt.hexcolor(c)]
                            
                            if FORC_ini_for_inv_array is not None:
                                print(f'{forc} FORC_ini_for_inv_array shape:',FORC_ini_for_inv_array.shape)
            
                                if FORC_ini_for_inv_array.shape[0] == 1 :
                                    current_meanFORC_m = FORC_ini_for_inv_array.squeeze(axis=0)
                                    current_std_FORC_m = None
                                    print('MONO FORC_ini_for_inv_array: current_meanFORC_m shape:',current_meanFORC_m.shape)
        
                                else:
                                    current_meanFORC_m = FORC_ini_for_inv_array.mean(axis=0)
                                    current_std_FORC_m = local_std_coeff * FORC_ini_for_inv_array.std(axis=0,ddof=1)
                                    print('MULTI FORC_ini_for_inv_array: current_meanFORC_m shape:',current_meanFORC_m.shape)
        
                                if plot_forc_shaded_region :
                                    if current_std_FORC_m is not None:
                                        ax.fill_between(inversion_years, current_meanFORC_m - current_std_FORC_m, current_meanFORC_m + current_std_FORC_m,
                                                        ec=cs, fc=gt.lighter_color(cs), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                                        hatch=hatch_for_shaded, zorder=2)
                                        ax.fill(np.NaN, np.NaN, 
                                                ec=cs, fc=gt.lighter_color(cs), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                                hatch=hatch_for_shaded, label=f"ini {forc.upper()}")
                                    ax.plot(inversion_years, current_meanFORC_m, color=cs, lw=1, ls='-', label=f"ini {forc.upper()}")
                                    
                                else:
                                    ax.errorbar(inversion_years, current_meanFORC_m, yerr=current_std_FORC_m, color=cs,
                                                lw=2, elinewidth=1, alpha=0.8, label=f"ini {forc.upper()}")
                                    #hdf[iforc], = ax.plot(inversion_years, current_meanFORC_m, color=c, lw=2, alpha=0.8)

                    # -------------------------------------------------------------
                    # Plotting Y: HIST source data forcings (plus errorbars)
                    #current_HIST_m = HIST_[nbe,:]
                    # current_mean_HIST_m = HIST_mod_for_inv_array.mean(axis=0)
                    # current_std_HIST_m = local_std_coeff * HIST_mod_for_inv_array.std(axis=0,ddof=1)
                    #
                    # if plot_forc_shaded_region :
                    #     ax.fill_between(inversion_years, current_mean_HIST_m - current_std_HIST_m, current_mean_HIST_m + current_std_HIST_m,
                    #                     ec=hist_color, fc=gt.lighter_color(hist_color), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                    #                     hatch=hatch_for_shaded, zorder=2)
                    #     hid = ax.fill(np.NaN, np.NaN, 
                    #                   ec=hist_color, fc=gt.lighter_color(hist_color), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                    #                   hatch=hatch_for_shaded, label=f"src {forcing_names[3].upper()}")
                    #     ax.plot(inversion_years, current_mean_HIST_m, color=hist_color, lw=1.5, ls='-')
                    #
                    #     if not np.isscalar(hid) :
                    #         hid = hid[0]
                    # else:
                    #     ax.errorbar(inversion_years, current_mean_HIST_m, yerr=current_std_HIST_m, color=hist_color,
                    #                 lw=1.5, elinewidth=1, capsize=2, alpha=0.8, label=f"src {forcing_names[3].upper()}")
                    #     hid, = ax.plot(inversion_years, current_mean_HIST_m, color=hist_color, lw=1.5, alpha=0.8)
    
                    # -------------------------------------------------------------------------
                    # X reference (or Background for inversion)
                    # -------------------------------------------------------------------------
                    hfo = ax.plot(inversion_years, GHG_ens_ab_arr_m.T, c=gt.darker_color(forcing_inv_colors[0]), ls='--', lw=1.5,
                                  label="$FORC_{ref.(other)}$")
                    ax.plot(inversion_years, AER_ens_ab_arr_m.T, c=gt.darker_color(forcing_inv_colors[1]), ls='--', lw=1.5)
                    ax.plot(inversion_years, NAT_ens_ab_arr_m.T, c=gt.darker_color(forcing_inv_colors[2]), ls='--', lw=1.5)
    
                    # -------------------------------------------------------------
                    # Plotting the means of Inverted forcings X: GHG, AER and NAT
                    # if Xinvmean_arr.shape[0] == 1 :
                    #     print(f"\n ¨** Only a single profile inverted, taken the mean of STD as STD")
                    #     tmp_xarr_mean = Xinvmean_arr.squeeze(axis=0)
                    #     tmp_xarr_std = local_std_coeff * Xinvstd_arr.squeeze(axis=0)
                    # else:
                    if Xinvmean_arr.shape[0] == 1 :
                        tmp_xarr_mean = Xinvmean_arr.squeeze(axis=0)
                        tmp_xarr_std = [None]*tmp_xarr_mean.shape[0]
                        print(' MONO tmp_xarr_mean shape:',tmp_xarr_mean.shape, end='')
                        if Xinvstd_arr.shape[0] == 1 :
                            tmp_xarr_meanstd = local_std_coeff * Xinvstd_arr.squeeze(axis=0)
                            print('MONO tmp_xarr_meanstd shape:',tmp_xarr_meanstd.shape)

                        else:
                            tmp_xarr_meanstd = local_std_coeff * Xinvstd_arr.mean(axis=0)  # mean of all STD (each inversion)
                            print('MULTI tmp_xarr_meanstd shape:',tmp_xarr_meanstd.shape)
                    else:
                        tmp_xarr_mean = Xinvmean_arr.mean(axis=0)
                        tmp_xarr_std = local_std_coeff * Xinvmean_arr.std(axis=0,ddof=1)
                        print('MULTI ALL tmp_xarr_mean shape:',tmp_xarr_mean.shape, 'tmp_xarr_std shape:',tmp_xarr_std.shape)
                        tmp_xarr_meanstd = local_std_coeff * Xinvstd_arr.mean(axis=0)  # mean of all STD (each inversion)
                        print('tmp_xarr_meanstd shape:',tmp_xarr_meanstd.shape)
                    #print('tmp_xarr_mean:',tmp_xarr_mean)
                    #print('tmp_xarr_std:',tmp_xarr_std)
                    #print('tmp_xarr_meanstd:',tmp_xarr_meanstd)
                    for iforc,(forc,ci,x_mean,x_std,x_meanstd) in enumerate(zip(forcing_names[:3],forcing_inv_colors[:3],tmp_xarr_mean,tmp_xarr_std,tmp_xarr_meanstd)) :
                        #print('x_mean shape:',x_mean.shape, 'x_std shape:',x_std.shape, c)
                        if plot_forc_shaded_region :
                            forc_xarr_color = gt.lighter_color(ci)
                            if plot_mean_err :
                                forc_meanxarr_color = gt.lighter_color(forc_xarr_color, factor=mean_err_lightness_factor)
                                ax.fill_between(inversion_years, x_mean[-lenDS:] - x_meanstd[-lenDS:], x_mean[-lenDS:] + x_meanstd[-lenDS:],
                                                ec=ci, fc=forc_meanxarr_color, alpha=alpha_for_mean_shaded, linewidth=lw_for_mean_shaded,
                                                hatch=hatch_for_mean_shaded_inv, zorder=2)
                                ax.fill(np.NaN, np.NaN, 
                                        ec=ci, fc=forc_meanxarr_color, alpha=alpha_for_mean_shaded, linewidth=lw_for_mean_shaded,
                                        hatch=hatch_for_mean_shaded_inv, label=f"mean inv {forc.upper()}")
                            if x_std is not None:
                                ax.fill_between(inversion_years, x_mean[-lenDS:] - x_std[-lenDS:], x_mean[-lenDS:] + x_std[-lenDS:],
                                                ec=ci, fc=forc_xarr_color, alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                                hatch=hatch_for_shaded_inv, zorder=2)
                                ax.fill(np.NaN, np.NaN, 
                                        ec=ci, fc=forc_xarr_color, alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                        hatch=hatch_for_shaded_inv, label=f"inv {forc.upper()}")
                            ax.plot(inversion_years, x_mean[-lenDS:], color=ci, lw=1, ls='-')
                            
                        else:
                            if x_std is not None:
                                ax.errorbar(inversion_years, x_mean[-lenDS:], yerr=x_std[-lenDS:], color=ci,
                                            lw=2, alpha=0.8, label=f"inv {forc.upper()}")
                            else:
                                ax.plot(inversion_years, x_mean[-lenDS:], c=ci,
                                            lw=2, alpha=0.8, label=f"inv {forc.upper()}")
                    # -------------------------------------------------------------
                    # Plotting OBS to Invert
                    if mod_to_invert in obs_set:
                        # uniquement les HIST/Obs selectionnes dans l'inversion (dans la liste Nbe_list)
                        HIST_obs_for_inv_array = obs_df.values[Nbe_list,:].copy()

                        #HIST_obs_for_inv_array = obs_df.iloc[:,-lenDS:].values
                        if HIST_obs_for_inv_array.shape[0] == 1:
                            current_mean_ObsHIST_m = HIST_obs_for_inv_array.squeeze(axis=0)
                        else:
                            current_mean_ObsHIST_m = HIST_obs_for_inv_array.mean(axis=0)
                            #current_std_ObsHIST_m = local_std_coeff * HIST_obs_for_inv_array.std(axis=0,ddof=1)
                        print(f"\nProfils OBS/HIST to invert is {mod_to_invert} !!")
                        ax.plot(inversion_years, current_mean_ObsHIST_m, color=hist_color, lw=2, ls='-', label=f"src {mod_to_invert}")
    
                    # -------------------------------------------------------------
                    # Plotting Y: HIST Inverted forcings (plus errorbars)
                    if Yinvmean_arr.shape[0] == 1:
                        tmp_y = Yinvmean_arr.squeeze(axis=0)
                        tmp_y_err = None
                        if plot_mean_err :
                            tmp_y_meanerr = local_std_coeff * Yinvstd_arr.squeeze(axis=0)
                    else:
                        tmp_y = Yinvmean_arr.mean(axis=0)
                        #print("#DBG3# PRE STD tmp_y.std(axis=0,ddof=1) ... tmp_y.shape",tmp_y.shape)
                        #print("#DBG3# tmp_y:",tmp_y)
                        tmp_y_err = local_std_coeff * Yinvmean_arr.std(axis=0,ddof=1)
                        if plot_mean_err :
                            tmp_y_meanerr = local_std_coeff * Yinvstd_arr.mean(axis=0) # mean of STD
                        
                    #print("#DBG3# ,inversion_years.shape,tmp_y.shape,tmp_y_err.shape,tmp_y_meanerr.shape):",inversion_years.shape,tmp_y.shape,tmp_y_err.shape,tmp_y_meanerr.shape)
                    #print("#DBG3# tmp_y_err:",tmp_y_err)
                    #print("#DBG3# tmp_y_err:",tmp_y_meanerr)
    
                    if mod_to_invert in obs_set:
                        tmp_hist_label_name = mod_to_invert
                    else:
                        tmp_hist_label_name = forcing_names[3].upper()
                
                    if plot_forc_shaded_region :
                        if plot_mean_err :
                            hist_meaninv_color = gt.lighter_color(hist_inv_color, factor=mean_err_lightness_factor)
                            ax.fill_between(inversion_years, tmp_y - tmp_y_meanerr, tmp_y + tmp_y_meanerr,
                                            ec=hist_meaninv_color, fc=gt.lighter_color(hist_meaninv_color), alpha=alpha_for_mean_shaded, linewidth=lw_for_mean_shaded,
                                            hatch=hatch_for_mean_shaded_inv, zorder=2)
                            ax.fill(np.NaN, np.NaN, 
                                    ec=hist_meaninv_color, fc=gt.lighter_color(hist_meaninv_color), alpha=alpha_for_mean_shaded, linewidth=lw_for_mean_shaded,
                                    hatch=hatch_for_mean_shaded_inv, label=f"mean inv {tmp_hist_label_name}")
                        if tmp_y_err is not None:
                            ax.fill_between(inversion_years, tmp_y - tmp_y_err, tmp_y + tmp_y_err,
                                            ec=hist_inv_color, fc=gt.lighter_color(hist_inv_color), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                            hatch=hatch_for_shaded_inv, zorder=2)
                            ax.fill(np.NaN, np.NaN, 
                                    ec=hist_inv_color, fc=gt.lighter_color(hist_inv_color), alpha=alpha_for_shaded, linewidth=lw_for_shaded,
                                    hatch=hatch_for_shaded_inv, label=f"inv {tmp_hist_label_name}")
                        ax.plot(inversion_years, tmp_y, color=hist_inv_color, lw=1, ls='-', label=f"inv {tmp_hist_label_name}")
                        
                    else:
                        if tmp_y_err is not None:
                            ax.errorbar(inversion_years, tmp_y, yerr=tmp_y_err, color=hist_inv_color,
                                        lw=2.5, elinewidth=2, alpha=0.8, label=f"inv {tmp_hist_label_name}")
                            #hih, = ax.plot(inversion_years, tmp_y, color=hist_inv_color, lw=2.5, alpha=0.8)
                        else:
                            ax.plot(inversion_years, tmp_y, c=hist_inv_color,
                                    lw=2.5, alpha=0.8, label=f"inv {tmp_hist_label_name}")
    
                    if False:
                        ax.plot(inversion_years, tmp_y+2*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
                        ax.plot(inversion_years, tmp_y-2*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
    
                        ax.plot(inversion_years, tmp_y+3*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
                        ax.plot(inversion_years, tmp_y-3*tmp_y_err, color=hist_inv_color, ls=':', lw=0.5, alpha=0.8)
                    #
                    # -------------------------------------------------------------
    
                    xmin,xmax = ax.get_xlim()
                    ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
                    ax.set_xlim([xmin,xmax])
                    
                    mod_for_title_label = mod_to_invert_for_title+'@'+mod_data_source if mod_to_invert in obs_set and mod_to_invert != mod_data_source else mod_to_invert_for_title
                    ax.grid(True,lw=0.75,ls=':')
                    ax.set_title(f"{mod_for_title_label}: Averaged profiles"+\
                                 f" ({nb_inverted_single_patt} HIST src. profiles inv. w. {n_x_forc} X forc.) each")
        
                    if i_mod == 0:
                        ax.legend(loc='upper left', ncol=5, framealpha=0.4)
                    
                    print('get_ylim:',ax.get_ylim())
                    
                    if fixed_t_limits_ok :
                        ax.set_ylim([t_min_limit,t_max_limit])
                        print('get_ylim Fixed to:',ax.get_ylim())

            # efface de la figure les axes non utilises
            if k_ax < nrows*ncols :
                for jax in np.arange(k_ax,nrows*ncols):
                    ax = axes.flatten()[jax]  #iax//ncols,iax%ncols]
                    ax.set_visible(False)
    
            members_label = f"{n_models} models inverted"+(" - NAT and HIST LP filtered" if lp_nathist_filtering else "")+("- FIX-T" if fixed_t_limits_ok else "")
            #members_label = nbe_inverted_label+(" - NAT & HIST LP filtered" if lp_nathist_filtering else "")
    
            suptitle_label = f"Inversion of profiles averaged [{members_label}] [net {innet}] [{net_label.upper()}]"
            if plot_forc_shaded_region:  suptitle_label += f" [Shaded {interval_label}]"
            else:                        suptitle_label += f" [ErrBar {interval_label}]"
            if plot_src4inv :            suptitle_label += " [+Xsrc]"
            if plot_inv_x :              suptitle_label += " [+Xini]"

            plt.suptitle(f"{suptitle_label}\n"+\
                         f"{base_case_to_explore}\n{cnn_name_base} [Inv. Settings: {inv_label.split('_')[0]}]",
                         #"[Dark orange: inversed HIST | Normal RGB: inversed GHG-AER-NAT | Darker RGB: GHG-AER-NAT source | Black: HIST source]",
                         size=suptitle_fontsize, y=0.99)
    
            #figs_file = f"Fig{local_nb_label}_inverted-averaged-all-forcings-by-model-{nb_inverted_single_patt}-profiles_by-net.{fig_ext}"
            #figs_filename = os.path.join(case_figs_dir,figs_file)

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


def give_n_inside_confidence(X_data, X_inv,
                            std_coeff=1,
                            forcing_names=['ghg', 'aer', 'nat'],
                            get_for_mean=False,
                            verbose=False) :
    """
    
    Parameters
    ----------
    X_data : TYPE
        DESCRIPTION.
        
    X_inv : TYPE
        DESCRIPTION.
        
    std_coeff : TYPE, optional
        DESCRIPTION. The default is 1.
        
    forcing_names : TYPE, optional
        DESCRIPTION. The default is ['ghg', 'aer', 'nat'].
        
    get_for_mean : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    dic_n_in_interval : TYPE
        DESCRIPTION.
    """
    import numpy as np
    
    if verbose:
        print("X_data shape .......", X_data.shape)
        print("X_inv shape ........", X_inv.shape)

    X_data_mean  = X_data.mean(axis=0)
    X_data_std   = X_data.std(axis=0,ddof=1)
    
    local_forc_n_in_data_total = X_data.shape[0]
    if len(X_inv.shape) == 3 :
        X_inv_mean  = X_inv.mean(axis=0)
        #X_inv_std   = X_inv.std(axis=0,ddof=1)
        local_forc_n_inverted_total = X_inv.shape[0]
    else:
        X_inv_mean  = X_inv
        #X_inv_std   = [ None ] * X_inv_mean.shape[0]
        local_forc_n_inverted_total = 1
    
    if verbose:
        print("X_data_mean shape ..", X_data_mean.shape)
        print("X_date_std shape ...", X_data_std.shape)
        print("X_inv_mean shape ...", X_inv_mean.shape)
        #print("X_inv_std shape ....", X_inv_std.shape)

    #print('# X inversés ...')
    
    dic_n_in_interval = { 'n_data' : local_forc_n_in_data_total, 
                          'n_inv' : local_forc_n_inverted_total,
                          'forcings' : forcing_names }
    
    for iforc,(forc,f_data_mean,f_data_std,f_inv_mean) in enumerate(zip(forcing_names,
                                                                        X_data_mean,
                                                                        X_data_std,
                                                                        X_inv_mean)):

        if verbose:
            print('iforc:',iforc,forc,std_coeff,f_data_mean.shape,f_data_std.shape,f_inv_mean.shape)

        all_years_forc_n_in_interval = []
        for i_year in np.arange(len(f_data_mean)) :
            # all inverted data for YEAR(i)
            if len(X_inv.shape) == 3 :
                x_inv_forc_year = X_inv[:,iforc,i_year]
            else:
                x_inv_forc_year = X_inv[iforc,i_year]
            
            local_n = ((np.array(x_inv_forc_year >= f_data_mean[i_year] - std_coeff*f_data_std[i_year],dtype=int) +  \
                       np.array(x_inv_forc_year <= f_data_mean[i_year] + std_coeff*f_data_std[i_year],dtype=int))  == 2).tolist()
            if np.isscalar(local_n) :
                local_n = 1 if local_n else 0
            else:
                local_n = local_n.count(True)

            all_years_forc_n_in_interval.append(local_n)

        dic_n_in_interval[forc] = { 'n_in' : all_years_forc_n_in_interval, 'std_coeff' : std_coeff }
        
        #print(dic_n_in_interval)

    if get_for_mean :
        dic_mean_in_interval = give_n_inside_confidence(X_data, X_inv_mean,
                                                        std_coeff=std_coeff,
                                                        forcing_names=forcing_names,
                                                        verbose=verbose)
        dic_n_in_interval['mean'] = dic_mean_in_interval

    return dic_n_in_interval


def plot_n_inside_confidence(X_data, X_inv,
                            train_years=np.arange(1900,2015),
                            forcing_names=['ghg', 'aer', 'nat'],
                            errorlimits_percent=None, errorlimits_n_rms=1,
                            forcings_t_limits=None,
                            alpha_forc_inv=0.3,  ls_forc_inv='-',    lw_forc_inv=0.75,   c_ghg_inv=None,    c_aer_inv=None, c_nat_inv=None,
                            alpha_forc_mod=0.3,  ls_forc_mod='-',    lw_forc_mod=1.0,    c_ghg_mod=None,    c_aer_mod=None, c_nat_mod=None,
                            current_lw=1.5, current_ls='-', current_alpha=0.8,
                            current_mean_lw=None, current_mean_ls=None, current_mean_alpha=None,
                            plot_average_of_n=False, average_of_n_lw=1.5, average_of_n_ls=None, average_of_n_alpha=0.8,
                            use_step=False, use_mean_step=None,
                            show_xlabel=True, show_xticklabels=True,
                            title_label=None,
                            ninv_short_label=None,
                            axes=None,
                            plot_by_filling=False, plot_mean_by_filling=False,
                            filling_y_level=None, mean_filling_y_level=None,
                            get_for_mean=False, plot_for_mean=None, plot_in_same_ax=True,
                            y_limits=None,
                            return_pc_error_dic=False,
                            legend_loc='lower left', legend_mean_loc='lower right',
                            legend_ncols=None, legend_mean_ncols=None,
                            lighter_inv=None, lighter_mean_inv=None,
                            stitle_label=None,
                            verbose =False,
                           ) :
    
    """
    Parameters
    ----------
    X_data : TYPE
        DESCRIPTION.
        
    X_inv : TYPE
        DESCRIPTION.
        
    train_years : TYPE, optional
        DESCRIPTION. The default is np.arange(1900,2015).
        
    forcing_names : TYPE, optional
        DESCRIPTION. The default is ['ghg', 'aer', 'nat'].
        
    errorlimits_percent : FLOAT, optional
        An ERRORLIMITS_PERCENT < 1, means the Percent point function to define
        a distance from the mean in the probability distribution of data. For
        instance, a value of ERRORLIMITS_PERCENT equals 0.9 means an interval
        of 90% centered in the probability distribution (i.e.: between 0.05
        and 0.95). With this coeficient we use the SCIPY function norm.ppf()
        to compute a coeficient to multiply the STD and determine the interval
        from -STD*Coeff and +STD*Coeff whom inverted data is accepted.
        An ERRORLIMITS_PERCENT >= 1, means a STD multiplier, exactly like
        ERRORLIMITS_N_RMS option.
    errorlimits_n_rms : TYPE, optional
        DESCRIPTION. The default is 1.
        
    forcings_t_limits : TYPE, optional
        DESCRIPTION. The default is None.
        
    alpha_forc_inv : TYPE, optional
        DESCRIPTION. The default is 0.3.
        
    ls_forc_inv : TYPE, optional
        DESCRIPTION. The default is '-'.
        
    lw_forc_inv : TYPE, optional
        DESCRIPTION. The default is 0.75.
        
    c_ghg_inv : TYPE, optional
        DESCRIPTION. The default is None.
        
    c_aer_inv : TYPE, optional
        DESCRIPTION. The default is None.
        
    c_nat_inv : TYPE, optional
        DESCRIPTION. The default is None.
        
    alpha_forc_mod : TYPE, optional
        DESCRIPTION. The default is 0.3.
        
    ls_forc_mod : TYPE, optional
        DESCRIPTION. The default is '-'.
        
    lw_forc_mod : TYPE, optional
        DESCRIPTION. The default is 1.0.
        
    c_ghg_mod : TYPE, optional
        DESCRIPTION. The default is None.
        
    c_aer_mod : TYPE, optional
        DESCRIPTION. The default is None.
        
    c_nat_mod : TYPE, optional
        DESCRIPTION. The default is None.
        
    current_lw : TYPE, optional
        DESCRIPTION. The default is 1.
        
    current_ls : TYPE, optional
        DESCRIPTION. The default is '-'.
        
    current_alpha : TYPE, optional
        DESCRIPTION. The default is 0.8.
        
    use_step : TYPE, optional
        DESCRIPTION. The default is False.
        
    use_mean_step : TYPE, optional
        DESCRIPTION. The default is False.
        
    title_label : TYPE, optional
        DESCRIPTION. The default is None.
        
    ninv_short_label : TYPE, optional
        DESCRIPTION. The default is None.
        
    axes : TYPE, optional
        DESCRIPTION. The default is None.
        
    plot_by_filling : TYPE, optional
        DESCRIPTION. The default is False.
    plot_mean_by_filling : TYPE, optional
        DESCRIPTION. The default is False.
    filling_y_level : TYPE, optional
        DESCRIPTION. The default is None.
    mean_filling_y_level : TYPE, optional
        DESCRIPTION. The default is None.
    get_for_mean : TYPE, optional
        DESCRIPTION. The default is False.
    plot_for_mean : TYPE, optional
        DESCRIPTION. The default is None.
    plot_in_same_ax : TYPE, optional
        DESCRIPTION. The default is True.
    return_pc_error_dic : TYPE, optional
        DESCRIPTION. The default is False.
        
    legend_loc : TYPE, optional
        DESCRIPTION. The default is 'lower left'.
    legend_mean_loc : TYPE, optional
        DESCRIPTION. The default is 'lower right'.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
        
     : TYPE
        DESCRIPTION.
    Returns
    -------
    None.
    """
    import math
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    import generic_tools_pl as gt   # like hexcolor(), ...


    lenDS = len(train_years)

    if forcings_t_limits is not None :
        fixed_t_limits_ok = True
        t_min_limit,t_max_limit = forcings_t_limits
    else:
        fixed_t_limits_ok = False

    if get_for_mean and plot_for_mean is None :
        plot_for_mean = True
    else:
        plot_for_mean = False

    if axes is None:
        nrows = 1
        ncols = 1
        figsize = (10,4.5)
        
        top    = 0.94; bottom = 0.10
        left   = 0.08; right  = 0.98
        hspace = 0.05; wspace = 0.05

        #fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,3))
        if plot_for_mean :
            if plot_in_same_ax:
                top = 0.90; bottom = 0.16; right = 0.92
            else:
                nrows += 1
                figsize = (10,8)
                hspace = 0.14
        fig,axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize,
                                gridspec_kw={'hspace': hspace, 'wspace': wspace,
                                             'left': left, 'right': right,
                                             'top': top, 'bottom': bottom })

    if errorlimits_percent is not None and errorlimits_n_rms != 1 :
        print(f"\n *** both ERRORLIMITS_PERCENT and ERRORLIMITS_N_RMS are specified ***"+\
              f"\n *** Considering only ERRORLIMITS_PERCENT {errorlimits_percent} value. The given ERRORLIMITS_N_RMS value of {errorlimits_n_rms} is thus negleted ***")

    if legend_ncols is None:
        legend_ncols = 1
    
    if use_mean_step is None :
        use_mean_step = use_step
    
    apply_std_coeff = False
    local_std_coeff = 1
    interval_label = f'{local_std_coeff}X'
    if errorlimits_percent is not None :
        if errorlimits_percent < 1:
            local_ppf = (1 + errorlimits_percent)/2     # Percent point function (inverse of cdf — percentiles).
                                                        # Pour, par exemple, definir un intervalle contenant 90% des valeurs en prenant
                                                        # alors entre 5% et 95% de la distribution de probabilités, le ppf serait alors
                                                        # 0.95, valeur à passer à la fonction norm.ppf() de SCIPY pour obtenir le coeff.
                                                        # multiplicatif de la std por le calcul de la (moitié de la) taille des barres
                                                        # d'erreur ou la moitié de la largeur de la zone "shaded".
            local_std_coeff = norm.ppf(local_ppf)
            interval_label = f'{errorlimits_percent*100:.0f}%'
        else:
            local_std_coeff = errorlimits_percent
            interval_label = f'{errorlimits_percent}X'
        apply_std_coeff = True
        
    elif errorlimits_n_rms != 1 :
        local_std_coeff = errorlimits_n_rms
        interval_label = f'{errorlimits_n_rms}X'
        apply_std_coeff = True
    
    if return_pc_error_dic and apply_std_coeff :
        pc_error_dic = { 'std_coeff' : local_std_coeff }
    else :
        pc_error_dic = {}
    
    colors_dic = gt.get_forcing_colors()
    
    c_forc_inv = colors_dic['standard']

    for iforc,forc in enumerate(colors_dic['order']):
        c_forc_inv_color = c_forc_inv[iforc] if lighter_inv is None else gt.lighter_color(c_forc_inv[iforc],factor=lighter_inv)
        c_forc_mod_color = gt.darker_color(c_forc_inv[iforc]) if lighter_mean_inv is None else gt.lighter_color(gt.darker_color(c_forc_inv[iforc]),factor=lighter_inv)
        if forc.lower() == 'ghg' :
            if c_ghg_inv is None: c_ghg_inv = c_forc_inv_color
            if c_ghg_mod is None: c_ghg_mod = c_forc_mod_color

        elif forc.lower() == 'aer' :
            if c_aer_inv is None: c_aer_inv = c_forc_inv_color
            if c_aer_mod is None: c_aer_mod = c_forc_mod_color

        elif forc.lower() == 'nat' :
            if c_nat_inv is None: c_nat_inv = c_forc_inv_color
            if c_nat_mod is None: c_nat_mod = c_forc_mod_color

    local_forcing_colors = [c_ghg_inv, c_aer_inv, c_nat_inv]
    local_forc_mod_colors = [c_ghg_mod, c_aer_mod, c_nat_mod]

    if verbose:
        #print("X_arr shape ...", X_arr.shape)
        print("X_data shape .....", X_data.shape)
        print("X_inv shape ......", X_inv.shape)
        #print("X_Ens_m_arr ...", X_Ens_m_arr.shape)
        #print("HIST_m_arr ....", HIST_m_arr.shape)
        #print("Y_inv shape ...", Y_inv.shape)
        #print("Y_hat shape ...", Y_hat.shape)
        if apply_std_coeff :
            print(f" - Local STD coefficient: {local_std_coeff}")
        else:
            print(f" - Applying STD coefficient: <NOT ACTIVATED>")

    if len(X_data.shape) < 3 :
        print(f"\n *** X data must have 3 dimensions [Nb patterns, Nb. forcings (3 normally, Forcing size (115 or something like that)]\n")
        raise
        
    if len(X_inv.shape) < 2 :
        print(f"\n *** X data must have 2 or 3 dimensions [[Nb patterns,] Nb. forcings (3 normally, Forcing size (115 or something like that)]\n")
        raise

    if X_data.shape[-1] > lenDS :
        X_data = X_data[:,:,-lenDS:]
        
    if len(X_inv.shape) == 3 :
        if X_inv.shape[-1] > lenDS :
            X_inv = X_inv[:,:,-lenDS:]
    else:
        if X_inv.shape[-1] > lenDS :
            X_inv = X_inv[:,-lenDS:]
    
    
    dic_n_in_interval = give_n_inside_confidence(X_data, X_inv,
                                                 std_coeff=local_std_coeff,
                                                 forcing_names=forcing_names,
                                                 get_for_mean=get_for_mean,
                                                 verbose=verbose)

    dic_n_in_interval['colors'] = local_forcing_colors
    dic_n_in_interval['mod_colors'] = local_forc_mod_colors

    if verbose:
        print("\ndic_n_in_interval:",dic_n_in_interval)
    
    if type(axes) is np.ndarray :
        ax = axes[0]
    else:
        ax = axes
    
    if average_of_n_ls is None:
        average_of_n_ls = [(0, (3,2,1,2)),(1, (3,2,1,2)),(2, (3,2,1,2))]

    if plot_by_filling :
        plot_func = ax.fill_between
        plot_options = { 'ls':current_ls, 'lw':current_lw, 'alpha':current_alpha, 'zorder':2}
        if filling_y_level is not None:
            plot_options = { **plot_options, 'y2':filling_y_level }
        if use_step :
            plot_options = { **plot_options, 'step':'mid' }
    elif use_step :
        plot_func = ax.step
        plot_options = { 'where':'mid', 'ls':current_ls, 'lw':current_lw, 'alpha':current_alpha }
    else:
        plot_func = ax.plot
        plot_options = { 'ls':current_ls, 'lw':current_lw, 'alpha':current_alpha }

    n_data_values = dic_n_in_interval['n_data']
    n_inv_values = dic_n_in_interval['n_inv']

    for iforc,(forc,c,av_ls) in enumerate(zip(dic_n_in_interval['forcings'],
                                        dic_n_in_interval['colors'],
                                        average_of_n_ls)) :
        local_forc_dic = dic_n_in_interval[forc]
        list_n_inside_values = local_forc_dic['n_in']
        percent_inside = [n / n_inv_values for n in list_n_inside_values]
        average_of_n_value = np.mean(percent_inside)
        
        if plot_average_of_n :
            plot_options = { **plot_options, 'label':f"${forc}$" }
        else:
            plot_options = { **plot_options, 'label':f"${forc}$ {average_of_n_value:.2f}" }
        if plot_by_filling :
            plot_options = { **plot_options, 'color':c }
        else:
            plot_options = { **plot_options, 'c':c }
            
        plot_func(train_years, percent_inside, **plot_options)
    
        if plot_average_of_n :
            ax.plot(train_years, average_of_n_value*np.ones(len(train_years)),
                    **{'ls':av_ls, 'lw':average_of_n_lw, 'alpha':average_of_n_alpha, 'c':c, 'label':f"{average_of_n_value:.3f}" })
        
    if (plot_in_same_ax or not get_for_mean) and show_xlabel :
        ax.set_xlabel('years')
    ax.set_ylabel('percent inside interval'+(f"\n{stitle_label}" if stitle_label is not None else ""))
    ax.grid(True,lw=0.5,ls=':')

    if not show_xticklabels :
        ax.set_xticklabels([])
    
    lax = list(ax.axis())
    if y_limits is not None:
        print(" lax:",lax, end = '')
        lax[2:] = y_limits
        ax.axis(lax)
    print(" lax:",lax)

    if legend_loc is not None :
        ax.legend(loc=legend_loc, ncol=legend_ncols)
    
    
    ax.set_title(f"Percent of data inside {interval_label} (+/-{local_std_coeff:.3f}*STD) [n_data={n_data_values}, n_inv={n_inv_values}]"+\
                 (f" - '{stitle_label}'" if stitle_label is not None else ""),size="medium")
    
    if get_for_mean :
        dic_mean_in_interval = dic_n_in_interval['mean']
        
        if legend_mean_ncols is None:
            legend_mean_ncols = legend_ncols
            
        if filling_y_level is not None and mean_filling_y_level is None :
            mean_filling_y_level = filling_y_level
            
        if plot_in_same_ax :
            ax = ax.twinx()
        else:
            ax = axes[1]

        if current_mean_ls is None:
            #current_mean_ls = [(0, (1,2)),(1, (1,2)),(2, (1,2))]
            current_mean_ls = [(0, (4,2)),(1, (4,2)),(2, (4,2))]
        if current_mean_lw is None: current_mean_lw = current_lw
        if current_mean_alpha is None: current_mean_alpha = current_alpha
        
        if plot_mean_by_filling :
            plot_func = ax.fill_between
            plot_options = { 'lw':current_mean_lw, 'alpha':current_mean_alpha, 'zorder':2}
            if mean_filling_y_level is not None :
                plot_options = { **plot_options, 'y2':mean_filling_y_level }
            if use_mean_step :
                plot_options = { **plot_options, 'step':'mid' }
        elif use_mean_step :
            plot_func = ax.step
            plot_options = { 'where':'mid', 'lw':current_mean_lw, 'alpha':current_mean_alpha }
        else:
            plot_func = ax.plot
            plot_options = { 'lw':current_mean_lw, 'alpha':current_mean_alpha }
    
        n_mean_inv_values = dic_mean_in_interval['n_inv']
    
        for iforc,(forc,c,ls) in enumerate(zip(dic_n_in_interval['forcings'],
                                               dic_n_in_interval['mod_colors'],
                                               current_mean_ls)) :
            plot_options = { **plot_options, 'label':f"mean {forc}" }
            if plot_mean_by_filling :
                plot_options = { **plot_options, 'color':c }
                if forc.lower() == 'ghg' :
                    plot_options = { **plot_options, 'hatch':'/' }
                elif forc.lower() == 'aer' :
                    plot_options = { **plot_options, 'hatch':'\\' }
                elif forc.lower() == 'nat' :
                    plot_options = { **plot_options, 'hatch':'-' }
            else:
                plot_options = { **plot_options, 'c':c, 'ls':ls }
                
            local_forc_dic = dic_mean_in_interval[forc]
            list_n_inside_values = local_forc_dic['n_in']
            percent_inside = [n / n_mean_inv_values for n in list_n_inside_values]
            
            plot_func(train_years, percent_inside, **plot_options)
            #plot_func(train_years, percent_inside, c=c, ls=current_mean_ls, lw=current_mean_lw,
            #          alpha=current_mean_alpha, label=f"mean {forc}")
                
        if not plot_in_same_ax :
            if show_xlabel :
                ax.set_xlabel('years')
            if not show_xticklabels :
                ax.set_xticklabels([])
            ax.grid(True,lw=0.5,ls=':')
        ax.set_ylabel('percent inside interval')
    
        if legend_mean_loc is not None :
            ax.legend(loc=legend_mean_loc, ncol=legend_mean_ncols)
        
        ax.axis(lax)
        
        if not plot_in_same_ax :
            ax.set_title(f"Percent of data inside {interval_label} (+/- {local_std_coeff:.2f}*STD) [n_data={n_data_values}, n_inv={n_mean_inv_values}]")

    return














































































