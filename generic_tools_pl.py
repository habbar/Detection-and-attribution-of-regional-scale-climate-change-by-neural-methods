# -*- coding: utf-8 -*-
"""
Generic Tools for PL
Contains functions:
    
    # HEXCOLOR: return a three values tuple (R.G.B. values) corresponfing to
    #           the Hex color in argument.
    # Usage:
    #    numcolor = hexcolor ('#ffaa88')
    # DARKER_COLOR:
    # 
    # Return a darker color in the form of a [R- ,G- ,B-] or [R- ,G- ,B-,alpha]
    # list, where 'C-' is a darker version of color component 'C'. Do not touch
    # 'alpha' (the fourth component) if present.
    #
    # Usage:
    #    numcolor = darker_color ('#ffaa88')
    #    numcolor = darker_color ([R, G, B])
    #    numcolor = darker_color ([R, G, B, a])
    # LIGHTER_COLOR:
    #
    # Return a lighter color in the form of a [R+ ,G+ ,B+] or [R+ ,G+ ,B+,alpha]
    # list, where 'C+' is a lighter version of color component 'C'. Do not touch
    # 'alpha' (the fourth component) if present.
    #
    # Usage:
    #    numcolor = lighter_color ('#ffaa88')
    #    numcolor = lighter_color ([R, G, B])
    #    numcolor = lighter_color ([R, G, B, a])
    # GET_FORCING_COLORS:
    #
    # Usage:
    #   colors_dic = get_forcing_colors ()
    #
    # Returns a dictionary having keys:
    #  - 'standard', standard forcing colors list,
    #  - 'standard_names', names of standard colors list,
    #  - 'darker', a darker version of colors for uses specially in inversion,
    #  - 'darker_names': names of darker colors, 
    #  - 'order', color forcing order, for instance: ['ghg', 'aer', 'nat', 'host']
    #    Names taken from from https://www.w3schools.com/colors/colors_names.asp
    
    # SHOWING_FORCING_COLORS_EXEMPLE:
    #
    # Plots random data using forcing colors proposed by GET_FORCING_COLORS()
    # plus a lighther color version of the 'light' set of colors.
    #
    # Usage:
    #    showing_forcing_colors_exemple
    
    # REDUCE_ALPHA:
    #
    # Compute an "adapted" value of alpha (transparency factor in matplotlib 
    # figures) depending of the number 'n' of values to plot.
    #
    # Usage:
    #    new_alpha = reduce_alpha(n,alpha) :
    
    # LOOK_FOR_INVERSION_FILES:
    #
    # Usage:
    #    file_nbe_suffix = look_for_inversion_files (file_prefix='Xinv_multi_all-but-one',
    #                                                inv_dir='.', n=None, verbose=False):
    # DO_LIST_OF_PROFILES:
    #
    # Get a list of indices on forcings to be inverses or plotted in a page.
    # Possibilitry to add the whole also. Used mostly for inversion procedure.
    #
    # Usage:
    #    dic = do_list_of_profiles (nb_patt, n_by_page=None, add_mean_of_all=False):
    # GET_SOURCE_DATA_DIR: 
    #
    # Usage:
    #    data_dir = get_source_data_dir()
    # LOAD_OBS_DATA:
    #
    # Usage:
    #    df = def load_obs_data(obs_filename=None, data_dir=None, obs_label='OBS',
    #                           first_year=1900, last_year=2014, limit_years=True,
    #                           current_obs_last_year=None,  # last_year for actual OBS data (from file 'obs.npy'), if different than default 2020 (see code inside)
    #                           preindustrial_lapse=[1850,1900], preindustrial_mean_shift_flg=None,
    #                           obs_scale=1.06, obs_scale_flg=True,
    #                           nc_varname=None, return_as_df=False,
    #                           rename_shifted_var=False, verbose=False)
    #    obs_arr,obs_years = load_obs_data(obs_label='OBS', return_as_df=True, verbose=False) :
    #    obs_df = load_obs_data(obs_label='OBS', return_as_df=True, verbose=False) :
    #    obs_df = load_obs_data(obs_label='HadCRUT', return_as_df=True, verbose=False) :
    # FILTERING_FORCING_SIGNAL_F:
    #
    # Input dictionary should have fields, corresponding to
    # the arguments for the scipy.signal.butter() function. I.e:
    #  - 'n' 
    #  - 'Wn'
    #  - 'btype', in option, default: 'lowpass'
    #  - 'analog', in option, default: False
    #  - 'output', in option, default: 'ba'
    #  - 'fs', in option, default: None
    #
    # Usage:
    #    b_lp_filter, a_lp_filter = filtering_forcing_signal_f (filt_dic, verbose=False ) :
    #
    # then, we must use signal.filtfilt to filter, for example, NAT and HIST train arrays:
    #       train_NAT = signal.filtfilt(b_lp_filter, a_lp_filter, train_NAT)
    #       train_HIST = signal.filtfilt(b_lp_filter, a_lp_filter, train_HIST)
    # FILTER_FORCING_DF:
    # 
    # Filter DF data. DF shouls contain a column 'models' with cliat odel names
    # and all other columns are the data to filter, normally one column one year.
    #
    # Usage:
    #    filtered_df =  filter_forcing_df (forc_df, filt_dic=filter_dictionary, verbose=False)
    # 
    # Return new DF with filtered data.
    # LOAD_BASIC_DATA_AND_GENER_DF: 
    #
    # RETURNS a dictionary containing:
    #  - 'label',                       # the name or very short description of source data
    #  - 'models',                      # list of climat models in simulations data
    #  - 'forcings',                    # list of forcings to extract ['ghg','aer','nat','hist'], by default.
    #  - 'forcing_color_dic',           # forcing colors proposed for source forcing data (near to those used by Constantin Bone)
    #  - 'forcing_inv_color_dic',       # additional forcing colors proposed for inverted forcing data 
    #  - 'forcing_color_names_dic',     # names of the forcing colors in 'forcing_color_dic'
    #  - 'forcing_inv_color_names_dic', # names of the forcing colors in 'forcing_inv_color_dic' 
    #  - 'years',                       # list of years, normally [1850 .. 2014]
    #  - 'list_of_df'                   # a DataFrame for each forcing specified i 'forcings' variable,
    #                                   # or in the list [ 'ghg', 'aer', 'nat', 'hist' ], by default.
    #
    # Usage:
    #    data_dic = load_basic_data_and_gener_df (models=None, forcings=None, to_filter=None, 
    #                                             data_dir='.', sim_file="ALL_sim_v4.p", models_file="Models_v4.p", 
    #                                             forcings_src=[ 'ghg', 'aer', 'nat', 'hist', 'other' ], verbose=False)
    # READ_DATA_SET_CHARACTERISTICS: reading basic forcing data parameters fom saved
    # dictionary train, test, invert, ... files.
    #
    # Usage:
    #    train_dic = read_data_set_characteristics (data_dir, file_prefix='train', set_label=something)
    # LOAD_FORCING_DATA: 
    #
    # Usage:
    #    load_forcing_data_info_list = load_forcing_data (data_dir, file_prefix, set_label=set_label,
    #                                                     forcing_names=['ghg', 'aer', 'nat', 'hist'],
    #                                                     dataframe=False, verbose=False)
    # ASSOCIATED_TEST_LABEL:
    # Returns the TEST_LABEL associated to the specified TRAIN_LABEL when 
    # data sets where generated.
    #
    # Usage:
    #    test_set_label = associated_test_label (data_and_training_label)
    # MODELS_TITLE_LABELS:
    #
    # Usage:
    #    models_label, models_label_prnt = models_title_labels(list_of_models) :
    # RETRIEVE_PARAM_FROM_BASE_CASE:
    #
    # Returns a dictionnary with keys:
    #  - 'n_nnets',
    #  - 'data_gener_method',
    #  - 'seed_value',
    #  - 'gan_train_percent_value',
    #  - 'gan_test_percent_value',
    #  - 'data_and_training_label',
    #  - 'associated_test_label',
    #  - 'lp_nathist_filtering'
    #
    # Usage:
    #    dic = retrieve_param_from_base_case (base_case_label, verbose=False)
    # RETRIEVE_PARAM_FROM_SUB_CASE:
    #
    # Returns a dictionnary with keys:
    #  - 'kern_size_list',
    #  - 'channel_size',
    #  - 'regularization_weight_decay',
    #  - 'extrapolation_label',
    #  - 'epochs',
    #  - 'batch_size',
    #  - 'learning_rate',
    #  - 'val_part_of_train_fraction'
    #
    # Usage:
    #    dic = retrieve_param_from_sub_case (sub_case_to_explore, verbose=False)
    # SORTED_DISTANCE_VECTORS:
    #
    # Computes distance between a and b, globally, or between vector a and each 
    # line or column of 2D matrix b.
    #
    # Usage:
    #    [...] = sorted_distance_vectors(a, b, ord=None, axis=None, sort=True, ret_b=False)
    # Examples:
    #    dist_ab = sorted_distance_vectors(a, b)
    #    sorted_dist_ab, idist_ab = sorted_distance_vectors(a, b, axis=1, sort=True)
    #    sorted_dist_ab, idist_ab, sorted_b = sorted_distance_vectors(a, b, axis=1, sort=True, ret_b=True)
    # PRINT_LOSS_TABLE:
    #
    # Builds a DataFrame from a loss table a print it as table using 'tabulate' Python module.
    #
    # Usage:
    #    summary_by_mod_df = print_loss_table (loss_test_tab,
    #                                          test_mod_df, y_obs, y_hat, tablefmt='simple',
    #                                          columnorder=['N', 'loss', 'MSE', 'RMSE'],
    #                                          indexname='model', indexorder=None)
Created on Mon Apr 17 10:18:51 2023
@author: hamza.abbar@etudiant.univ-lr.fr
"""

from IPython.display import display

region="50"

def hexcolor(hexcol):
    if hexcol[0] == '#' :
        hexcol = hexcol[1:]
    return tuple(int(hexcol[i:i+2], base=16)/255 for i in (0, 2, 4))


def darker_color(col, factor=0.25):
    if type(col) is str and col[0] == '#' :
        col = hexcolor(col)
    
    if type(col) in [list, tuple] and (len(col) == 3 or len(col) == 4) :
        hist_inv_color = [max((0,c*(1-factor))) if i < 3 else c for i,c in enumerate(col)]
    else:
        print(f" *** color must be a [R,G,V] or [R,G,V,alpha] vector\n *** or a str HEXCOLOR, like '#ffaa00' not this <<{col}>>\n")
        raise
       
    return hist_inv_color


def lighter_color(col, factor=0.25):
    if type(col) is str and col[0] == '#' :
        col = hexcolor(col)
    
    if type(col) in [list, tuple] and (len(col) == 3 or len(col) == 4) :
        hist_inv_color = [min((1,c+(1-c)*factor)) if i < 3 else c for i,c in enumerate(col)]
    else:
        print(f" *** color must be a [R,G,V] or [R,G,V,alpha] vector\n *** or a str HEXCOLOR, like '#ffaa00' not this <<{col}>>\n")
        raise
    
    return hist_inv_color


def get_forcing_colors ():
    """  get_forcing_colors:
        
    Usage:
      colors_dic = get_forcing_colors ()
        
    Returns
    -------
    dict: returns a dictionary having keys:
          - 'standard', standard forcing colors list,
          - 'standard_names', names of standard colors list,
          - 'darker', a darker version of colors for uses specially in inversion,
          - 'darker_names': names of darker colors, 
          - 'order', color forcing order, for instance: ['ghg', 'aer', 'nat', 'host']
          
    Names taken from from https://www.w3schools.com/colors/colors_names.asp
    """
    #local_forcing_colors =              [ '#ff0000', '#0000FF', '#008000', '#8b0000' ]  # couleurs Constantin (names from w3schools): RED, BLUE, GREEN, DARK RED
    #local_forcing_color_names =         [ 'Red',     'Blue',    'Green',   'DarkRed' ]  # couleurs Constantin (names from w3schools):
    local_forcing_colors =              [ '#ff0000', '#0000FF', '#008000', '#d2691e' ]  # couleurs Constantin (names from w3schools): RED, BLUE, GREEN, CHOCOLATE
    local_forcing_color_names =         [ 'Red',     'Blue',    'Green',   'Chocolate' ]  # couleurs Constantin (names from w3schools):

    #local_forcing_darker_color =       [ '#750075', '#000080', '#757500', '#b7653b' ]  # couleurs Constantin: VIOLET, NAVY, VERT CACA D'OIE?, ORANGE
    #local_forcing_darker_colors =       [ '#8b008b',     '#000080', '#808000', '#000000' ]  # proche Constantin (names from w3schools): DARK MAGENTA, NAVY, OLIVE, Black
    #local_forcing_darker_color_names =  [ 'DarkMagenta', 'Navy',    'Olive',   'Black'   ]  # selon w3schools
    local_forcing_darker_colors =       [ '#b22222',   '#000080', '#808000', '#000000' ]  # proche Constantin (names from w3schools): FIRE BRICK, NAVY, OLIVE, Black
    local_forcing_darker_color_names =  [ 'FireBrick', 'Navy',    'Olive',   'Black'   ]  # selon w3schools
    
    local_forcing_lighter_colors =      [ '#f08080',    '#00bfff',     '#9acd32',     '#ff8c00'    ]   # LightCoral, DeepSkyBlue, YellowGreen, DarkOrange
    local_forcing_lighter_color_names = [ 'LightCoral', 'DeepSkyBlue', 'YellowGreen', 'DarkOrange' ]  # selon w3schools
    
    return { 'standard': local_forcing_colors,
            'standard_names': local_forcing_color_names, 
            'darker': local_forcing_darker_colors,
            'darker_names': local_forcing_darker_color_names, 
            'lighter': local_forcing_lighter_colors,
            'lighter_names': local_forcing_lighter_color_names, 
            'order': ['ghg', 'aer', 'nat', 'hist'] }


def showing_forcing_colors_exemple() :
    import matplotlib.pyplot as plt
    import numpy as np
    
    colors_dic = get_forcing_colors()

    top = 0.92;    bottom = 0.04
    left = 0.06;   right = 0.98
    wspace = 0.05; hspace = 0.10
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,7),
                            gridspec_kw={'hspace': hspace, 'wspace': wspace, 
                                         'left': left,     'right': right,
                                         'top' : top,      'bottom' : bottom })
    n=10
    s=0.03
    lw=3;
    ysep = 0; dysep = 0.2
    ybbase = 4; ymbase = 1.0
    
    for iall,(c,name,forc) in enumerate(zip(colors_dic['standard'],colors_dic['standard_names'],colors_dic['order'])) :
        ax.plot(np.arange(n),ybbase - iall*ymbase - ysep - s*np.random.randn(n),c=c,lw=lw,label=f"{forc.upper()} {name} (Inv)")
    ysep += dysep
    for iall,(c,name,forc) in enumerate(zip(colors_dic['darker'],colors_dic['darker_names'],colors_dic['order'])) :
        ax.plot(np.arange(n),ybbase - iall*ymbase - ysep - s*np.random.randn(n),c=c,lw=lw,label=f"{forc.upper()} {name} (Normal)")
    ysep += dysep
    for iall,(c,name,forc) in enumerate(zip(colors_dic['lighter'],colors_dic['lighter_names'],colors_dic['order'])) :
        ax.plot(np.arange(n),ybbase - iall*ymbase - ysep - s*np.random.randn(n),c=c,lw=lw,label=f"{forc.upper()} {name} (Lighther)")
    ysep += dysep
    for iall,(c,name,forc) in enumerate(zip(colors_dic['lighter'],colors_dic['lighter_names'],colors_dic['order'])) :
        ax.plot(np.arange(n),ybbase - iall*ymbase - ysep - s*np.random.randn(n),c=lighter_color(c,factor=0.75),lw=lw,label=f"{forc.upper()} {name} (Light.L75)")
    
    ax.set_ylim(0,6)
    ax.legend(ncol=2,loc='upper center')
    ax.set_title("Examples of colors for forcings using $gt.get\_forcing\_colors()$",size='large');    
    return


def reduce_alpha(n,a) :
    """
    REDUCE_ALPHA:
    Compute an "adapted" value of alpha (transparency factor in matplotlib 
    figures) depending of the number 'n' of values to plot.
    
    Usage
    -----
    new_alpha = reduce_alpha(n,alpha)
    Parameters
    ----------
    n : int
        The number of values to be plotted (not in the function).
    a : float
        The ALPHA value for the 'alpha=' option in matplotlib figure functions.
    Returns
    -------
    new_a : float
        The new ALPHA value reduced depending at the value of N.
    """
    if n < 20 :
        local_a = a
    elif n < 50 :
        local_a = a * 0.3
    elif n < 100 :
        local_a = a * 0.1
    else :
        local_a = a * 0.03
        
    return local_a


def look_for_inversion_files(file_prefix='Xinv', inv_dir='.', n=None, verbose=False):
    import os
    import glob
    import numpy as np

    # get existing filenames for Xinv
    all_xinv_found = glob.glob(os.path.join(inv_dir, f'{file_prefix}_*.p'))
    if verbose:
        print(f"\ninv_dir: {inv_dir}")
        print(f" complete path: {os.path.join(inv_dir, f'{file_prefix}_*.p')}")
        print('all_xinv_found:',all_xinv_found)

    # extract unique suffix corr a
    file_nbe_suffixes = np.unique([os.path.basename(f).replace(f'{file_prefix}_','').split('_')[0] for f in all_xinv_found]).tolist()
    
    if len(file_nbe_suffixes) > 1:
        print('\n ** attention, several file_nbe_suffixes found:',file_nbe_suffixes, '\n ** choosing first!\n')

    if verbose:
        print('file_nbe_suffixes:',file_nbe_suffixes)

    file_nbe_suffix = None
    if n is not None :
        file_nbe_suffix = [s for s in file_nbe_suffixes if f'only-{n}' in s ][0]
    else:
        file_nbe_suffix = [s for s in file_nbe_suffixes if 'all-' in s ]
        
    if file_nbe_suffix is None:
        file_nbe_suffix = file_nbe_suffixes[0]
    
    if verbose:
        print(f"All suffixex found: {file_nbe_suffixes}, Coosing file suffix: '{file_nbe_suffix}'")

    return file_nbe_suffix


# def do_list_of_profiles(nb_patt, n_by_page=None,
#                         nb_of_profiles=None, choice_method='random', rand_seed=0,
#                         add_mean_of_all=False, verbose=False):
#     """ Get a list of indices on forcings to be inverses or plotted in a page.
#     Possibilitry to add the whole also.
#     Used mostly for inversion procedure and the subsequent forcing plot figures.
    
#     Returns a dictionnary having two fields:
#         - 'list', the list of indices to be takent into account
#         - 'label', a label summarysing the selection.
        
#     """
#     import numpy as np
    
#     # nb_of_profiles ....... number of obs to choose in case of multiple data (like 'HadCrut200')
#     # choice_method ... ('sequential','random') method to choose the obs to work if multiple
#     # rand_seed ..... ramdom generation seed in case of 'random' method
#     if nb_of_profiles is not None :
#         if type(nb_of_profiles) is int and choice_method.lower() in ['sequential','random','seq','rand','rnd']:
#             if nb_of_profiles > nb_patt:
#                 nb_of_profiles = nb_patt
#             if verbose:
#                 print(f"Selecting '{choice_method}'-ly {nb_of_profiles} profiles from the {nb_patt} availables")
#             if choice_method.lower() in ['random','rand','rnd'] :
#                 if rand_seed is not None :
#                     np.random.seed(rand_seed)
#                 list_of_patt = np.random.permutation(nb_patt)[np.arange(nb_of_profiles)].tolist()
#                 nbe_invert_label = f"SEL{nb_of_profiles}RND"+(f"{rand_seed}" if rand_seed else "")+f"-FROM{nb_patt}"

#             elif choice_method.lower() in ['sequential' ,'seq'] :
#                 list_of_patt = np.arange(nb_of_profiles).tolist()
#                 nbe_invert_label = f'SEL{nb_of_profiles}SEQ-FROM{nb_patt}'
                
#             if verbose:
#                 print(f"   Selecting rows {nb_of_profiles}/{nb_patt} from obs_arr data array: '{nbe_invert_label}'")

#         elif type(nb_of_profiles) is list :
#             list_of_patt = nb_of_profiles
#             nbe_invert_label = f'SEL{len(list_of_patt)}LST-FROM{nb_patt}'

#             if verbose:
#                 print(f"   Selecting rows by list {len(list_of_patt)}/{nb_patt} from obs_arr data array: '{nbe_invert_label}'")
        
#     else:
#         list_of_patt = np.arange(nb_patt).tolist()
#         nbe_invert_label = f'ALL{nb_patt}'
        
#         if verbose:
#             print(f"   Selecting all {nb_patt} from obs_arr data array: '{nbe_invert_label}'")

#     nb_sel_patt = len(list_of_patt)

#     select_patt_for_inversion_ok = n_by_page is not None

#     if select_patt_for_inversion_ok :
#         one_page_n = min((n_by_page,nb_sel_patt))
#         if one_page_n < n_by_page :
#             list_of_nbe = np.arange(one_page_n).tolist()  # all patterns
#             nbe_invert_title = f"all {len(list_of_nbe)}{'+ALL' if add_mean_of_all else ''} profiles"
#         else:
#             list_of_nbe = np.linspace(0, nb_sel_patt-1, one_page_n-1, endpoint=True, dtype=int).tolist()  # uniquement nb_sel_patt patterns
#             nbe_invert_title = f"only {len(list_of_nbe)}{'+ALL' if add_mean_of_all else ''} profiles"
#         if verbose:
#             print('select_patt_for_inversion:',list_of_nbe,nbe_invert_title)
        
#     else:
#         list_of_nbe = list_of_patt       # tous les patterns de la liste prefixee
#         nbe_invert_title = old_nbe_invert_title = f"all {len(list_of_nbe)}{'+ALL' if add_mean_of_all else ''} profiles"
#         if nb_of_profiles is not None:
#             nbe_invert_title = f"{choice_method} sel {len(list_of_nbe)} from {nb_patt}{'+ALLSEL' if add_mean_of_all else ''} profiles"
#         if verbose:
#             print('All:',list_of_nbe,nbe_invert_title)
    
#     if add_mean_of_all:
#         list_of_nbe.append(np.arange(nb_sel_patt).tolist()) # on ajoute a la liste un element contenant les indices des tous les profils du modele m (pour inverser l'HIST_ moyenne)
#         if verbose:
#             print('End:',list_of_nbe)

#     #print(list_of_nbe,len(list_of_nbe),nbe_invert_title)
#     tmp_dic = { 'list':list_of_nbe, 'label':nbe_invert_label, 'title':nbe_invert_title , 'old_label':old_nbe_invert_title }

#     return tmp_dic

def do_list_of_profiles(nb_patt, n_by_page=None,
                        nb_of_profiles=None, choice_method='random', rand_seed=0,
                        do_mean_of_each_model=False, add_mean_of_all=False,
                        verbose=False):
    """ Get a list of indices on forcings to be inverses or plotted in a page.
    Possibilitry to add the whole also.
    Used mostly for inversion procedure and the subsequent forcing plot figures.
    
    Returns a dictionnary having two fields:
        - 'list', the list of indices to be takent into account
        - 'label', a label summarysing the selection.
        
    """
    import numpy as np
    
    if add_mean_of_all and do_mean_of_each_model:
        print("\n ** unspected simultaneous `True` values of flags 'add_mean_of_all' and 'do_mean_of_each_model'. Only one is spected to be True. Setting 'add_mean_of_all' to `False` **")
        add_mean_of_all = False

    # nb_of_profiles ....... number of obs to choose in case of multiple data (like 'HadCrut200')
    # choice_method ... ('sequential','random') method to choose the obs to work if multiple
    # rand_seed ..... ramdom generation seed in case of 'random' method
    if nb_of_profiles is not None :
        if type(nb_of_profiles) is int and choice_method.lower() in ['sequential','random','seq','rand','rnd']:
            if nb_of_profiles > nb_patt:
                nb_of_profiles = nb_patt
            if verbose:
                print(f"Selecting '{choice_method}'-ly {nb_of_profiles} profiles from the {nb_patt} availables")
            if choice_method.lower() in ['random','rand','rnd'] :
                if rand_seed is not None :
                    np.random.seed(rand_seed)
                list_of_patt = np.random.permutation(nb_patt)[np.arange(nb_of_profiles)].tolist()
                nbe_invert_label = f"SEL{nb_of_profiles}RND"+(f"{rand_seed}" if rand_seed else "")+f"-FROM{nb_patt}"

            elif choice_method.lower() in ['sequential' ,'seq'] :
                list_of_patt = np.arange(nb_of_profiles).tolist()
                nbe_invert_label = f'SEL{nb_of_profiles}SEQ-FROM{nb_patt}'
                
            if verbose:
                print(f"   Selecting rows {nb_of_profiles}/{nb_patt} from obs_arr data array: '{nbe_invert_label}'")

        elif type(nb_of_profiles) is list :
            list_of_patt = nb_of_profiles
            nbe_invert_label = f'SEL{len(list_of_patt)}LST-FROM{nb_patt}'

            if verbose:
                print(f"   Selecting rows by list {len(list_of_patt)}/{nb_patt} from obs_arr data array: '{nbe_invert_label}'")
        
    else:
        list_of_patt = np.arange(nb_patt).tolist()
        nbe_invert_label = f'ALL{nb_patt}'
        
        if verbose:
            print(f"   Selecting all {nb_patt} from obs_arr data array: '{nbe_invert_label}'")

    nb_sel_patt = len(list_of_patt)

    select_patt_for_inversion_ok = n_by_page is not None

    if select_patt_for_inversion_ok :
        one_page_n = min((n_by_page,nb_sel_patt))
        if one_page_n < n_by_page :
            list_of_nbe = np.arange(one_page_n).tolist()  # all patterns
            nbe_invert_title = f"all {len(list_of_nbe)}{'+ALL' if add_mean_of_all else ''} profiles"
        else:
            list_of_nbe = np.linspace(0, nb_sel_patt-1, one_page_n-1, endpoint=True, dtype=int).tolist()  # uniquement nb_sel_patt patterns
            nbe_invert_title = f"only {len(list_of_nbe)}{'+ALL' if add_mean_of_all else ''} profiles"
        if verbose:
            print('select_patt_for_inversion:',list_of_nbe,nbe_invert_title)
        
    else:
        list_of_nbe = list_of_patt       # tous les patterns de la liste prefixee
        nbe_invert_title = old_nbe_invert_title = f"all {len(list_of_nbe)}{'+ALL' if add_mean_of_all else ''} profiles"
        if nb_of_profiles is not None:
            nbe_invert_title = f"{choice_method} sel {len(list_of_nbe)} from {nb_patt}{'+ALLSEL' if add_mean_of_all else ''} profiles"
        if verbose:
            print('All:',list_of_nbe,nbe_invert_title)
    
    if add_mean_of_all and not do_mean_of_each_model:
        list_of_nbe.append(np.arange(nb_sel_patt).tolist()) # on ajoute a la liste un element contenant les indices des tous les profils du modele m (pour inverser l'HIST_ moyenne)
        if verbose:
            print('End:',list_of_nbe)

    if do_mean_of_each_model:
        new_list_of_nbe = [e for e in list_of_nbe if type(e) is not list]
        list_of_nbe = [new_list_of_nbe]  # la liste de nbe est la loiste de la liste ...pour inverser l'HIST_moyenne uniquement
        nbe_invert_label = f"AVERAGED-{nbe_invert_label}"
        nbe_invert_title = f"Averaged {nbe_invert_title}"
        if verbose:
            print('End:',list_of_nbe)

    #print(list_of_nbe,len(list_of_nbe),nbe_invert_title)
    tmp_dic = { 'list':list_of_nbe, 'label':nbe_invert_label, 'title':nbe_invert_title,
               #'old_label':old_nbe_invert_title
               }

    return tmp_dic

def get_source_data_dir(dirname=None, verbose=False):
    import os
    if dirname is None :
        dirname = "/usr/home/habbar/Bureau/data_nc/data_source_dr/Region{}".format(region)
    # Repertoire des donnees
    try:
        # WORK dir carlos projet ryn sur Jean Zay
        data_dir = "/usr/home/habbar/Bureau/data_nc/stagelong/projetlong/data_source_dr/Region{}".format(region)
        if not os.path.isdir(data_dir):
            if verbose:
                print(f" ** data_dir '{data_dir}' not found. Trying next...")
            
            # WORK dir Guillaume sur Jean Zay ** NON **
            #data_dir = '/gpfswork/rech/ryn/rces866/Constantin'
            #if not os.path.isdir(data_dir):
            #print(f" ** data_dir '{data_dir}' not found. Trying next...")
            
            # SSD sur Acratopotes au Locean
            data_dir = f'/net/acratopotes/datatmp/data/constantin_data/{dirname}'
            if not os.path.isdir(data_dir):
                if verbose:
                    print(f" ** data_dir '{data_dir}' not found. Trying next...")
                
                # sur Cloud SU (carlos)
                data_dir = os.path.expanduser(f'~/Clouds/SUnextCloud/Labo/Travaux/Theses-et-stages/These_Constantin/constantin_data/{dirname}')
                if not os.path.isdir(data_dir):
                    if verbose:
                        print(f" ** data_dir '{data_dir}' not found. Trying next...")
                    
                    # sur Cloud SU (thiria)
                    data_dir = os.path.expanduser(f'~/Documents/Carlos/These_Constantin/constantin_data/{dirname}')
                    if not os.path.isdir(data_dir):
                        if verbose:
                            print(f" ** data_dir '{data_dir}' not found. Trying next...")
                        
                        # en dernier recours, en esperant qu'il y a un repertoire 'data' present ...
                        data_dir = os.path.expanduser('data')
                        if not os.path.isdir(data_dir):
                            if verbose:
                                print(f" ** data_dir '{data_dir}' not found at all **\n")
                            raise Exception('data_dir not found')
     
    except Exception as e:
        print(f'\n *** Exception error "{e}" ***\n')
        raise
        
    if verbose :
        print(f"Data directory found: '{data_dir}'")

    return data_dir


def load_obs_data(obs_filename=None, data_dir=None, obs_label='OBS',
                  first_year=1900, last_year=2014, limit_years=True,
                  source_dirname='data_source_pl',
                  current_obs_last_year=None,  # last_year for actual OBS data (from file 'obs.npy'), i doferent than default 2020 (see code lower)
                  preindustrial_lapse=[1850,1900], preindustrial_mean_shift_flg=None, # for shifting HadCRUT mean values (already centered in a more recent period)
                  obs_scale=1.06,obs_scale_flg=True,
                  n_gen_ar1 = 100, lp_obs_filtering_dic = { 'n':4, 'Wn':[1./10.], 'btype':'lowpass' },
                  nc_varname=None, return_as_df=False,
                  rename_shifted_var=False, verbose=False) :
    
    import numpy as np
    import os, sys
    import pandas as pd
    import xarray as xr
    import pickle

    try:
        # To load personal libraries:
        sys.path.append('.')
        
        if verbose:
            print("\nReading Obs:")
        
        if data_dir is None :
            if obs_filename is not None and os.path.isfile(obs_filename) :
                data_dir = '.'
            else:
                data_dir = get_source_data_dir(dirname=source_dirname, verbose=verbose)
    
        # -----------------------------------------------------------------------------------
        # Load Observations data. Three sets are possible:
        # -----------------------------------------------------------------------------------
        # - 'OBS', Observations data set used by Constantin, I think those taken from HadCRUT
        #    version 4 (yearly global mean)
        # -----------------------------------------------------------------------------------
        # - 'HadCRUT', The version 5 of the temperature anomalies in a yearly global mean presentation
        #    relative to the 1961–1990 period and shifted in reference to the mean of the 1850–1900 period mean.
        # -----------------------------------------------------------------------------------
        # - 'HadCRUT200', The version 5 of the temperature anomalies 200-member ensemble, in reference to
        #   1850–1900 period and in a montly global mean. Converted to yearly mean and shifted in
        #   reference to the mean of the 1850–1900 period.
        # -----------------------------------------------------------------------------------
        if obs_label == 'OBS':
            # Load OBS (Observations used by Constantin, I think those are from HadCRUT version 4)
            if obs_filename is None :
                obs_filename = 'obs.npy'
            if current_obs_last_year is None:
                current_obs_last_year = 2020
            
        elif obs_label == 'HadCRUT':
            if obs_filename is None :
                obs_filename = "/usr/home/habbar/Bureau/data_nc/stagelong/projetlong/data_source_dr/Region{}/HadCRUT/HadCRUT.5.0.1.0.analysis.summary_series.global.annual.region.nc".format(region)
            if nc_varname is None :
                nc_varname = 'tas_mean'
            if preindustrial_mean_shift_flg is None:
                hadcrut_preindustrial_mean_shift_flg = True
            else:
                hadcrut_preindustrial_mean_shift_flg = preindustrial_mean_shift_flg
            
        elif obs_label == 'HadCRUT200':
            if obs_filename is None :
                obs_filename = "/usr/home/habbar/Bureau/data_nc/stagelong/projetlong/data_source_dr/Region{}/HadCRUT/HadCRUT.5.0.1.0.analysis.ensemble_series.global.monthly.nc".format(region)
            if nc_varname is None :
                nc_varname = 'tas'
            if preindustrial_mean_shift_flg is None:
                hadcrut_preindustrial_mean_shift_flg = True
            else:
                hadcrut_preindustrial_mean_shift_flg = preindustrial_mean_shift_flg

        elif obs_label == 'HadCRUT+AR1':
            if obs_filename is None :
                # ---------------------------------------------------------------------------------------------
                # default obs_filename is a three-parts string vector composing the fixed parts of the filename.
                # There are two variable values composing the name that should be inserted in between the
                # three-part filename. These are the number of random AR1 vectors generated and the size of
                # those vectors in number of time steps (years).  The time step size is important, if it
                # changes, even for a smaller size, we should generate again in order to let border take into
                # account the frontier effect that could be loosen if we simple cut a larger vector.
                # ---------------------------------------------------------------------------------------------
                obs_filename = ['T_pi/T_AR1.generate_R','xL', '.v2.p']  # The three parts of a file named for instance 'T_pi/T_AR1.generate_R100xL115.v2.p', for n=100, sz=115
            if preindustrial_mean_shift_flg is None:
                hadcrut_preindustrial_mean_shift_flg = True
            else:
                hadcrut_preindustrial_mean_shift_flg = preindustrial_mean_shift_flg

        else:
            raise Exception(f" ** obs label '{obs_label}' not considered. Give one in ['OBS', 'HadCRUT', 'HadCRUT200', 'HadCRUT+AR1'] list **")

        # -----------------------------------------------------------------------------------
        # Reading Obs data
        # -----------------------------------------------------------------------------------
        if verbose:
            print(f"reading '{obs_label}' from '{obs_filename}' filename ...")
        
        if obs_label == 'OBS':
            
            if not os.path.isfile(os.path.join(data_dir,obs_filename)) :
                print(f"\n *** load_obs_data error: OBS file '{obs_filename}' not found ***\n")
                raise Exception(f"\n *** load_obs_data error: OBS file '{obs_filename}' not found ***\n")

            obs_arr = np.load(os.path.join(data_dir,obs_filename))
            if verbose:
                print(f"Obs loaded: shape: {obs_arr.shape}")
                
            if limit_years and last_year < current_obs_last_year :
                if verbose :
                    print(f" ** last_year({last_year}) < current_obs_last_year({current_obs_last_year}). Cutting dates array ...")
                obs_arr = obs_arr[:-(current_obs_last_year - last_year)]
            else:
                last_year = current_obs_last_year
            
            if limit_years and (last_year - first_year + 1) < len(obs_arr) :
                n_to_reject = len(obs_arr) - (last_year - first_year + 1)
                if verbose :
                    print(f" ** last_year is {last_year} and first_year {first_year}: number of expected time steps (years) is {(last_year - first_year + 1)}"+\
                          "\n    but we have {len(obs_arr)} in obs_arr. We must cut first {n_to_reject} years ...")
                obs_arr = obs_arr[n_to_reject:]
            
            print('obs_label obs_arr:',obs_arr.shape)
            obs_years = np.arange(last_year + 1 - len(obs_arr), last_year+1)
            out_label = obs_label

        elif obs_label == 'HadCRUT+AR1' :
            
            if verbose:
                print(f"For {obs_label}:\n 1- first loading HadCRUT as base ...")
            
            obs_df = load_obs_data(obs_label='HadCRUT', return_as_df=True, limit_years=True,
                                   first_year=first_year, last_year=last_year,
                                   preindustrial_mean_shift_flg=hadcrut_preindustrial_mean_shift_flg,
                                   obs_scale_flg=False,
                                   verbose=verbose)
            
            hadcrut_preindustrial_mean_shift_flg = False   # to avoid extra shifting
            
            if verbose:
                print(f" 2- low-pass filtering this base Obs profile ...")
            
            obs_df = filter_forcing_df(obs_df, filt_dic=lp_obs_filtering_dic, verbose=verbose)
            
            n_gen_ar1 = 100
            nyears = last_year - first_year + 1
            ar1_obs_filename = f"{obs_filename[0]}{n_gen_ar1}{obs_filename[1]}{nyears}{obs_filename[2]}"
            
            if verbose:
                print(f" 3- loading the AR1 noisy generated vectors from file {ar1_obs_filename} ...")
            
            if not os.path.isfile(os.path.join(data_dir,ar1_obs_filename)) :
                print(f"\n *** load_obs_data error: AR1 OBS file '{ar1_obs_filename}' not found ***\n")
                raise Exception(f"\n *** load_obs_data error: AR1 OBS file '{ar1_obs_filename}' not found ***\n")

            ar1_obs_arr = pickle.load( open( os.path.join(data_dir, ar1_obs_filename), "rb" ) )
            
            if verbose:
                print(f"    AR1 noisy generated vectors size: {ar1_obs_arr.shape}")

            obs_arr = obs_df.values + ar1_obs_arr
            obs_years = obs_df.columns.values
            out_label = obs_label

            if verbose :
                print(f"'{obs_label}' data time from: {obs_years[0]} to  {obs_years[-1]} - obs_arr shape: {obs_arr.shape}")
                #display(obs_ds)
            
        elif obs_label in ['HadCRUT','HadCRUT200'] :
            
            if not os.path.isfile(os.path.join(data_dir,obs_filename)) :
                print(f"\n *** load_obs_data error: OBS file '{obs_filename}' not found ***\n")
                raise Exception(f"\n *** load_obs_data error: OBS file '{obs_filename}' not found ***\n")

            if obs_label == 'HadCRUT' :
                obs_ds = xr.open_dataset(os.path.join(data_dir,obs_filename))
                                
            else:
                montly_ds = xr.open_dataset(os.path.join(data_dir,obs_filename)) # reading monthly data
                obs_ds = montly_ds.groupby('time.year').mean() # mean by year
                obs_ds = obs_ds.rename({ 'year':'time'})
                
            if verbose :
                print(f"'{obs_label}' data time from: {obs_ds['time'].isel(time=0).values} to  {obs_ds['time'].isel(time=-1).values}")
                #display(obs_ds)
            
            # adding shifted data, shifting based on pre-industrial mean
            if hadcrut_preindustrial_mean_shift_flg :
                hadcrut_preindustrial_shift_value = obs_ds[nc_varname].sel(time=slice(f'{preindustrial_lapse[0]}',f'{preindustrial_lapse[1]}')).mean().values
                if verbose :
                    print(f"Decalage de la serie '{obs_label}' par la moyenne des annees pre-industriales : {hadcrut_preindustrial_shift_value}")
                out_varname = f'{nc_varname}_shifted'
                if rename_shifted_var :
                    out_label = f'{obs_label}-shifted'
                else:
                    out_label = obs_label
                obs_ds[out_varname] = obs_ds[nc_varname] - hadcrut_preindustrial_shift_value
                #if verbose :
                #    display(obs_ds)
                
            else:
                out_varname = nc_varname
                out_label = obs_label
                
            if verbose :
                print(f"\nnc_varname='{nc_varname}':"); display(obs_ds[nc_varname])
                print(f"\nout_varname='{out_varname}':"); display(obs_ds[out_varname])
            
            first_year_in_ncfile = int(pd.to_datetime(str(obs_ds['time'].isel(time=0).values)).strftime('%Y'))
            last_year_in_ncfile = int(pd.to_datetime(str(obs_ds['time'].isel(time=-1).values)).strftime('%Y'))
            if first_year is None :
                first_year = first_year_in_ncfile
            if last_year is None :
                last_year = last_year_in_ncfile
            if not limit_years or (first_year == first_year_in_ncfile and last_year == last_year_in_ncfile) :
                tmp_ds = obs_ds.copy()
            elif limit_years and first_year >= first_year_in_ncfile and last_year <= last_year_in_ncfile :
                tmp_ds = obs_ds.sel(time=slice(f'{first_year}',f'{last_year}')).copy()
            else:
                raise Exception(f"\n *** load_obs_data error: data '{obs_label}' from file '{obs_filename}' has not convenient limit dates."+\
                                f" Expected {first_year} <= dates <= {last_year} but found extremes years {first_year_in_ncfile} and {last_year_in_ncfile} ***\n")
            
            print(f" NetCDF has values from  {last_year_in_ncfile}. Keeping data until: {last_year} (i.e: {tmp_ds['time'].isel(time=-1).values})")
            if obs_label == 'HadCRUT200' :
                obs_arr = tmp_ds[out_varname].values.T  # transpose to set the shape in [nb_obs, sz_obs] order
            else:
                obs_arr = tmp_ds[out_varname].values
            
            print('obs_label obs_arr:',obs_arr.shape)
            obs_years = [int(pd.to_datetime(str(t)).strftime('%Y')) for t in tmp_ds['time'].values]
            
            #obs2_df = pd.DataFrame(data={nc_varname:tmp_ds[nc_varname].values}, index=tmp_ds['time'].values)
            
        else:
            print(f"\n *** load_obs_data error: OBS file from label '{obs_label}' not identified ***\n")
            raise Exception(f"\n *** load_obs_data error: OBS file from label '{obs_label}' not identified ***\n")
        
        if len(obs_arr.shape) == 1 :
            obs_arr = obs_arr.reshape((1,len(obs_arr)))
                
        nb_obs,sz_obs = obs_arr.shape
                
        print(f"Obs {out_label} array shape: {obs_arr.shape}")
        print(f"    {sz_obs} years: from {obs_years[0]} to {obs_years[-1]}")
        
        if obs_scale_flg:
            print(f"Scaling Obs data by the factor of {obs_scale}")
            obs_arr *= obs_scale

        if return_as_df :
            if nb_obs == 1:
                df = pd.DataFrame(obs_arr, columns=obs_years, index=[out_label])
            else:
                df = pd.DataFrame(obs_arr, columns=obs_years)
            
            return df
        else:
            return obs_arr,obs_years

    except Exception as e:
        print(f'\n *** Exception error "{e}" ***\n')
        raise
        
    if verbose :
        print("Obs Data not found")

    if return_as_df :
        return []
    else:
        return [],[]


def filtering_forcing_signal_f (filt_dic, verbose=False ) :
    """
    Butterworth filtering using scipy.signal.butter()
    Parameters
    ----------
    filt_dic : TYPE
        DESCRIPTION.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    b_lp_filter : TYPE
        DESCRIPTION.
    a_lp_filter : TYPE
        DESCRIPTION.
    """
    from scipy import signal

    n = filt_dic['n']
    Wn = filt_dic['Wn']
    
    if 'btype' in filt_dic.keys():
        btype = filt_dic['btype']
    else:
        btype='lowpass'
    
    if 'analog' in filt_dic.keys():
        analog = filt_dic['analog']
    else:
        analog=False
    
    if 'output' in filt_dic.keys():
        output = filt_dic['output']
    else:
        output='ba'
    
    if 'fs' in filt_dic.keys():
        fs = filt_dic['fs']
    else:
        fs=None
    
    b_lp_filter, a_lp_filter = signal.butter(n, Wn, btype=btype, analog=analog, output=output, fs=fs)

    return b_lp_filter, a_lp_filter


def filter_forcing_df (forc_df, filt_dic={ 'n':4, 'Wn':[1./10.], 'btype':'lowpass' }, model_column='model', verbose=False):
    from scipy import signal
    import pandas as pd
    
    if verbose:
        display(forc_df)

    if model_column in forc_df.columns :
        tmp_arr = forc_df.drop(columns=model_column).values
        
    else:
        # IF there is no column 'model', we consider thus that all columns are data to filter ...
        if verbose :
            print(f" ** filter_forcing_df: No '{model_column}' column found.  Consider thus that all columns are data to filter! ")

        tmp_arr = forc_df.values
    
    if verbose :
        print("Filtering Forc DF data having shapes: {tmp_arr.shape} ...")
        print(tmp_arr[:5,:])
    
    b_lp_filter, a_lp_filter = filtering_forcing_signal_f(filt_dic, verbose=verbose)
    tmp_filtered_arr = signal.filtfilt(b_lp_filter, a_lp_filter, tmp_arr)

    if model_column in forc_df.columns :
        # creating a new DF having same index and 'models' column as original DF
        tmp_filtered_df = pd.DataFrame(data={model_column:forc_df[model_column].values}, index=forc_df.index)
    
        # creating a second DF with new filtered data and with columns names (i.e: years) as original DF
        tmp_filtered_years_df = pd.DataFrame(tmp_filtered_arr, columns=forc_df.drop(columns=model_column).columns)
        
        # concatenating first DF with model names abs second one with filtered years data
        tmp_filtered_df = pd.concat([tmp_filtered_df, tmp_filtered_years_df], axis=1)

    else:
        # creating a new DF with no data, having same index as original DF
        tmp_index = forc_df.index
    
        # creating a second DF with new filtered data and with columns names (i.e: years) as original DF
        tmp_filtered_years_df = pd.DataFrame(tmp_filtered_arr, columns=forc_df.columns)
    
        # building new DataFame with filtered data, having same columns and index than original DataFrame
        tmp_filtered_df = pd.DataFrame(tmp_filtered_arr, columns=forc_df.columns, index=tmp_index)

    return tmp_filtered_df

# def load_basic_data_and_gener_df (models=None, forcings=None, to_filter=None, filtering_dic_filename=None, 
#                                   data_dir=None, sim_file="All_sim_z{}v0.p".format(region), models_file="models_z{}v0.p".format(region), 
#                                   source_dirname='data_source_dr',
#                                   forcings_src=[ 'ghg', 'aer', 'nat', 'hist', 'other' ],
#                                   load_forcing_files=True, verbose=False,
#                                   strict_version_identifying=True):
def load_basic_data_and_gener_df (models=None, forcings=None, to_filter=None, filtering_dic_filename=None, 
                                  data_dir=None, sim_file=None, models_file=None, 
                                  source_dirname='data_source_dr',
                                  forcings_src=[ 'ghg', 'aer', 'nat', 'hist', 'other' ],
                                  load_forcing_files=True, verbose=False,
                                  strict_version_identifying=True):
    """
    Loading basic source data from forcings in files such as the list of models in models_file, "Models_v4.p" by default.
    and the complete set of simulations available for all forcings in file sim_file, "ALL_sim_v4.p" by default.
    
    Converts all Forcings data into DataFrames, one by forcing.
    
    In returned DataFrames, the INDEX is a increasing integer and they have the YEARS as columns.
    An aditional column, MODEL, contains the name of the Climat model source of simulation.
    
    As an example, for getting the array of data in one of those DataFrames, for instance GHG,
    you can "drop" the column 'models' and get the values of the DataFrame like this:
    
        ghg_array = T_ghg_df.drop(columns='model').values
    
    Until now, only data from ALL sim (v4) source files are readed.
    
    RETURNS a dictionary containing:
      - 'label',                       # the name or very short description of source data
      - 'models',                      # list of climat models in simulations data
      - 'forcings',                    # list of forcings to extract ['ghg','aer','nat','hist'], by default.
      - 'forcing_color_dic',           # forcing colors proposed for source forcing data (near to those used by Constantin Bone)
      - 'forcing_inv_color_dic',       # additional forcing colors proposed for inverted forcing data 
      - 'forcing_color_names_dic',     # names of the forcing colors in 'forcing_color_dic'
      - 'forcing_inv_color_names_dic', # names of the forcing colors in 'forcing_inv_color_dic' 
      - 'years',                       # list of years, normally [1850 .. 2014]
      - 'list_of_df'                   # a DataFrame for each forcing specified i 'forcings' variable,
                                       # or in the list [ 'ghg', 'aer', 'nat', 'hist' ], by default.
    """
    import numpy as np
    import pickle
    import os
    import pandas as pd

    # Repertoire des donnees
    if data_dir is None :
        data_dir = get_source_data_dir(dirname=source_dirname, verbose=verbose)
    if sim_file is None :
        sim_file="All_sim_z{}v0.p".format(region)
    if models_file is None : 
        models_file="models_z{}v0.p".format(region)
    
    # detect data version in sim filename, normally 'v4' or other
    sim_name_ext = sim_file.split('.')[-1]
    sim_name_splitted = sim_file.split('_')
    last_splitted_part_sim = sim_name_splitted[-1].split('_')[-1].split('.') # split the last part to retire file extension. This part is considered the identification of the data set
    data_indent_version = last_splitted_part_sim[0]
    sim_name_splitted = sim_name_splitted[:-1]
    
    # detect data version in models filename, normally equal to the one on sim filename.
    last_splitted_part_models = models_file.split('_')[-1].split('.')
    last_part_models = last_splitted_part_models[0]
    
    if data_indent_version != last_part_models :
        if strict_version_identifying :
            print(f"\n *** data indentifier version do not correpond between simulations and models files '{data_indent_version} not equal '{last_part_models}' ! ***"+\
                  f"\n *** Sim filename '{sim_file}' don not have same data identifier as models filename '{models_file}'. Please verify ?"+\
                  f"\n *** Flag strict_version_identifying is {strict_version_identifying}. Program will raise.\n")
            raise Exception(f"\n *** data indentifier version do not correpond between simulations and models files '{data_indent_version} not equal '{last_part_models}' ! ***\n")
        else:
            print(f"\n ** data indentifier version do not correpond between simulations and models files '{data_indent_version} not equal '{last_part_models}' ! ***"+\
                  f"\n ** Sim filename '{sim_file}' don not have same data identifier as models filename '{models_file}'. Please verify ?"+\
                  f"\n ** Flag strict_version_identifying is {strict_version_identifying}. Program will continue using '{data_indent_version} as data identifier.\n")

    local_forcings_names = [ 'ghg',    'aer',     'nat',     'hist' ]

    # Color palettes for source data ou predicted/inverted data (same as used by Constantin Bone)
    colors_dic = get_forcing_colors()
    
    local_forcing_src_colors = colors_dic['standard']
    local_forcing_src_color_names = colors_dic['standard_names']

    local_forcing_darker_colors = colors_dic['darker']
    local_forcing_darker_color_names = colors_dic['darker_names']

    local_forcing_lighter_colors = colors_dic['lighter']
    local_forcing_lighter_color_names = colors_dic['lighter_names']

    local_forcing_colors = local_forcing_darker_colors
    local_forcing_color_names = local_forcing_darker_color_names
    
    local_forcing_inv_colors = local_forcing_src_colors
    local_forcing_inv_color_names = local_forcing_src_color_names

    local_forcing_light_colors = local_forcing_lighter_colors
    local_forcing_light_color_names = local_forcing_lighter_color_names


    if to_filter is not None:
        from scipy import signal

        if verbose:
            print("Preparing Low-Pass filter function:")

        if  filtering_dic_filename is None or not os.path.isfile(filtering_dic_filename):
            print(f"\n ** FILTERING FILE NAME IS NONE or NOT FOUD '{filtering_dic_filename}'\n ** and filtering seems active !! IT'S AN ERROR ? **/n")
        else:
            print(f"Loading filtering parameters from file '{filtering_dic_filename}'")
            lp_nathist_filtering_dictionary = pickle.load( open( filtering_dic_filename, "rb" ), encoding="latin1")
            
            b_lp_filter, a_lp_filter = filtering_forcing_signal_f (lp_nathist_filtering_dictionary,
                                                                   verbose=verbose )

    print("\nLecture des meta donnees (ALL_sim, liste de modeles, ...) ...")

    data_label = "{' '.join(sim_name_splitted)} ({data_indent_version})".lower()
    
    # ALL_sim_v4 et Models_v4:
    _all_sim_src  = pickle.load( open( os.path.join(data_dir,sim_file), "rb" ), encoding="latin1")

    # liste de modeles
    _all_models_src  = pickle.load( open( os.path.join(data_dir,models_file), "rb" ), encoding="latin1")
    
    # forcages dans l'ordre des lignes de _all_sim_src.
    # l'ordre des forcages est trouve en comparant le nombre de simulations par forcage (tout modele)
    # avec les dimensions des donnees dans les fichiers "T_xxx_v4.p", pour les differents forcages xxx
    _all_forcings_src = forcings_src

    # list of years in source files 
    years = np.arange(1850,2015)

    if models is None :
        models = _all_models_src

    if forcings is None :
        forcings = forcings_src[:-1] # tous sauf le dernier

    forcing_color_dic       = {f:local_forcing_colors[local_forcings_names.index(f.lower())] for f in forcings}
    forcing_color_names_dic = {f:local_forcing_color_names[local_forcings_names.index(f.lower())] for f in forcings}

    forcing_inv_color_dic       = {f:local_forcing_inv_colors[local_forcings_names.index(f.lower())] for f in forcings}
    forcing_inv_color_names_dic = {f:local_forcing_inv_color_names[local_forcings_names.index(f.lower())] for f in forcings}

    forcing_light_color_dic       = {f:local_forcing_lighter_colors[local_forcings_names.index(f.lower())] for f in forcings}
    forcing_light_color_names_dic = {f:local_forcing_lighter_color_names[local_forcings_names.index(f.lower())] for f in forcings}

    # tableau d'indices de debut et de fin des donnees du modele (colonne) pour chaque forcage (ligne)
    simulation_ini_index_list = np.cumsum(_all_sim_src,axis=1) - _all_sim_src
    simulation_end_index_list = np.cumsum(_all_sim_src,axis=1)
    # print(_all_sim_src)
    # print(simulation_ini_index_list)
    # print(simulation_end_index_list)
    if verbose:
        print(f"\nData label: '{data_label}'")

        print(f'\n List of {len(_all_models_src)} models:\n   {_all_models_src}')
        print(f'\n List of {len(_all_forcings_src)} forcings:\n   {_all_forcings_src}')

        print(f"\n '{sim_file}' contents [forcings x models]:")
        print(_all_sim_src)
        print(" Dim:",_all_sim_src.shape)
        print('\n Sum by column (number of simulations by model):\n',np.sum(_all_sim_src,axis=0))  #,"    (je suppose, c'est le nombre de simulations par modele, pour les 4+1 forcages)")
        print('\n Sum by line (number of simulations by forcing):\n',np.sum(_all_sim_src,axis=1))  #,"    (je suppose, c'est le nombre de simulations par forcace, tout modele confondu)" )
        print(f"\n INI and END indices for data tables in 'T_xxx_{data_indent_version}.p' files (for each forcing 'xxx'):")
        print(f" INI index:\n{simulation_ini_index_list}")
        print(f" END index:\n{simulation_end_index_list}")

    print("\nReading forcing data files ...")

    data_dic = {'label':data_label, 'data_dir':data_dir, 'models':models, 'forcings':forcings,
                'forcing_color_dic':forcing_color_dic, 'forcing_color_names_dic':forcing_color_names_dic,
                'forcing_inv_color_dic':forcing_inv_color_dic, 'forcing_inv_color_names_dic':forcing_inv_color_names_dic, 
                'forcing_light_color_dic':forcing_light_color_dic, 'forcing_light_color_names_dic':forcing_light_color_names_dic, 
                'years':years, }

    if load_forcing_files :
        list_of_df = []
        for iforcing,forcing in enumerate(forcings) :
            if verbose:
                print(f"\n {forcing.upper()}",end='')
            kfor = _all_forcings_src.index(forcing)
            current_t_array_filename = f"T_{forcing.lower()}_{data_indent_version}.{sim_name_ext}"
            T_xxx  = pickle.load( open( os.path.join(data_dir,current_t_array_filename), "rb" ),
                                 encoding="latin1")
            #if verbose:
            #print(T_xxx)
            print(f" Data '{current_t_array_filename}' -> dim: {T_xxx.shape}")
            
            if to_filter is not None:
                if forcing in to_filter:
                    print(f"Filtering {forcing.upper()} data having shape: {T_xxx.shape}")
                    if verbose:
                        print('AVANT:',T_xxx.mean(axis=1).mean(),T_xxx.std(axis=1,ddof=1).mean())
                    T_xxx = signal.filtfilt(b_lp_filter, a_lp_filter, T_xxx.copy())
                    if verbose:
                        print('APRES:',T_xxx.mean(axis=1).mean(),T_xxx.std(axis=1,ddof=1).mean())
            
            #current_t_df_varname = f'T_{forcing.lower()}_df'
            xxx_df = pd.DataFrame(T_xxx, columns=years)
    
            xxx_df['model'] = '<NA>'   # ajoute colonne 'model'
            cols = xxx_df.columns.tolist()
            xxx_df = xxx_df[cols[-1:] + cols[:-1]]    # remet la colonne 'model' en premier
            # print(xxx_df.shape)
            # print(xxx_df)
            # print(_all_models_src)
            for imod,mod in enumerate(models) :
                # print(imod,mod,kfor)
                kmod = _all_models_src.index(mod)
                ii = simulation_ini_index_list[kfor,kmod]
                jj = simulation_end_index_list[kfor,kmod]
                print(ii,jj)
                xxx_df.loc[np.arange(ii,jj),'model'] = mod
                if verbose:
                    print(f"   {imod}) '{mod}' -> {ii}, {jj} ({jj-ii})")
            # retire les lignes non identifiees par un modele
            xxx_df = xxx_df.loc[lambda df: df['model'] != '<NA>', :]
            #globals()[current_t_df_varname] = xxx_df
            #print(f" ... assigned to DataFrame '{current_t_df_varname}'")
            list_of_df.append(xxx_df)
        
        data_dic['list_of_df'] = list_of_df
    
    return data_dic


def read_data_set_characteristics(data_dir, file_prefix='test', set_label=None, verbose=False):
    # ################################################################################################
    # Reading data characteristics for one particulat data set: Train, Test, Invert, ...
    # Don't read data but index of data about combination of forcings in the data set.
    #
    # Lecture d'un dictionnaire contenant les tableaux d'indices, le nombre de simulations
    # par modele, la liste de modeles, liste de forçages et liste d'années. Il y a un
    # dictionnaire pour TRAIN et un autre pour TEST.
    # ################################################################################################

    import os
    import pickle
    
    current_combi_dic_filename = f"{file_prefix}-combi-dic_{set_label}.p"
    if  verbose:
        print(f"\nReading {file_prefix.upper()} combi_dic in filename '{current_combi_dic_filename}' ...")
        print(f"In '{data_dir}/'")
        
    combi_dic = pickle.load( open( os.path.join( data_dir, current_combi_dic_filename), "rb" ),
                            encoding="latin1")
    
    if verbose:
        print(f" having keys: {combi_dic.keys()}")
    
    return combi_dic


def load_forcing_data(data_dir=None, file_prefix=None, set_label=None, forcing_names=['ghg', 'aer', 'nat', 'hist'], dataframe=False, verbose=False):
    import os
    import numpy as np
    import pandas as pd
    if data_dir is None:
        data_dir = "/usr/home/habbar/Bureau/data_nc/stagelong/projetlong/data_source_dr/Region{}".format(region)
    data_dic = {}
    for iforcing,forcing in enumerate(forcing_names) :
        if verbose:
            print(f"\n{forcing.upper()}:")

        current_df_file = f"{file_prefix}-{forcing.lower()}_{set_label}_df.p"
        current_df_filename = os.path.join(data_dir,current_df_file)
        
        if not os.path.exists(current_df_filename):
            print(f" ** '{forcing.upper()}' forcing file not found for {file_prefix.upper()} data set ***\n")
        else:
            if verbose:
                print(f" Loading data from preavious saved {file_prefix.upper()} DataFrame in file:\n - {current_df_filename}")
            
            xxx_df = pd.read_pickle( os.path.join(data_dir,current_df_filename) )
            if dataframe :
                data_dic[forcing] = xxx_df.copy()
            else:
                data_dic[forcing] = xxx = xxx_df.drop('model',axis=1).values
        
        # model_names = None
        # all_years = None
        # mod_df= None 

        if len(data_dic) > 0:  # au moin un fichier lu
            model_names = np.unique(xxx_df['model'].values).tolist()

            all_years = xxx_df.drop(columns={'model'}).columns.values.tolist()

            mod_df = xxx_df[['model']].copy()
            
        if verbose:
            # if dataframe :
            #     # print(f" {forcing.upper()}: TRAIN DF readed -> dim: {xxx_df.shape}", end='')
            # else:
                # print(f" {forcing.upper()}: TRAIN array readed -> dim: {xxx.shape}", end='')
            xxx = None
            if dataframe:
                xxx_df = pd.read_pickle(os.path.join(data_dir,current_df_filename))
                data_dic[forcing] = xxx_df.copy()
            else:
                xxx_df = pd.read_pickle(os.path.join(data_dir,current_df_filename))
                xxx = xxx_df.drop('model', axis=1).values
                data_dic[forcing] = xxx
    if verbose:
        print()

    return model_names, all_years, mod_df, data_dic


def associated_test_label(data_and_training_label) :
    
    splited_train_label = data_and_training_label.split('-')
    data_gener_method   = splited_train_label[0]
    dgen_seed_particle  = splited_train_label[1]
    dgen_GANpc_particle = splited_train_label[2]
    dgen_Npc_particle   = splited_train_label[3]
    
    # identifiant des données en TEST selon le type identifié pour TRAIN
    if dgen_Npc_particle == 'N132_z{}v0'.format(region) :
        test_N_part = 'NMx100_z{}v0'.format(region)
        
    elif dgen_Npc_particle == 'N1000_z{}v0'.format(region) :
        test_N_part = 'NMx1000_z{}v0'.format(region)
        
    else:
        print(f"\n *** unknown or not referenced particle '{dgen_Npc_particle}' in data_and_training_label '{data_and_training_label}' ***\n"+\
              " *** Is an error in label or a lack in code ?  ***\n")
        raise

    scase_particle = 'GAN'
    if dgen_GANpc_particle[:len(scase_particle)] == scase_particle :
        gan_train_percent_value = int(dgen_GANpc_particle[len(scase_particle):-2])  # '-2' pour enlever le 'pc' a la fin

    gan_test_percent_value = 100 - gan_train_percent_value


    test_set_label = f'{data_gener_method}-{dgen_seed_particle}-GAN{gan_test_percent_value}pc-{test_N_part}'
    
    return test_set_label


def models_title_labels(list_of_models) :
    import numpy as np

    n_models = len(list_of_models)

    if n_models >= 12 :
        models_label = f'All models ({n_models})'
    elif n_models < 4 :
        models_label = f"{n_models} models [{', '.join(np.sort(list_of_models))}]"
    else :
        models_label = f"{n_models} models [{', '.join([m.split('-')[0] for m in np.sort(list_of_models)])}]"
    models_label_prnt = models_label.replace(' [','-').replace(']','').replace(' (','-').replace(')','').replace(', ','-').replace(' ','-')

    return models_label,models_label_prnt


def retrieve_param_from_base_case(base_case_label, verbose=False) :
    """
    Retrieving parameters from specified base_case_to_explore label.
    
    Example: 'out_v5_nn6-TTDGM2-S0-GAN85pc-N132_v4_12mod_NewNet'
    Values are returned in a dictionnary with keys:
     - 'n_nnets',
     - 'data_gener_method',
     - 'seed_value',
     - 'gan_train_percent_value',
     - 'gan_test_percent_value',
     - 'data_and_training_label',
     - 'associated_test_label',
     - 'lp_nathist_filtering'
    
    Usage:
        
        dic = retrieve_param_from_base_case(base_case_label)
    """
    splited_base_case = base_case_label.split('-')
    splited_base_case

    bctoex_begining_particle    = splited_base_case[0]
    bctoex_dgenmethod_particle  = splited_base_case[1]
    bctoex_dgenseed_particle    = splited_base_case[2]
    bctoex_dgenGANpc_particle   = splited_base_case[3]

    bctoex_dgenmethod_particle

    # Recuperation et verification du nombre de CNN entrainés n_nnets
    splited_begining_particle = bctoex_begining_particle.split('_')

    try :
        # should begin with 'out_v5_nn' exactly
        if splited_begining_particle[0] != 'out' or \
           splited_begining_particle[1] != 'v5' or \
           splited_begining_particle[2][:2] != 'nn' :
            raise NameError(f"\n  ** Unexpected base directory name. Should begin with 'out_v5_nn',\n  ** but we have: '{bctoex_begining_particle}'. Is a new specification of base case directory or new syntax in name?")

    except NameError:
        print("** Cant found the number of CNN trained alltogether, syntax of base case directory name has changed ?" )
        raise

    tmp_dic = {}
    
    tmp_dic['n_nnets'] = int(splited_begining_particle[2][2:])
    
    if verbose:
        print(f" - Number of CNN trained found in label: {tmp_dic['n_nnets']}")

    # Recuperation et verification du label de la methode de generation des donnees ('TTDGM1', 'TTDGM2', ...). Variable data_gener_method
    accepted_data_gener_methods = ['TTDGM1', 'TTDGM2' ]
    
    try :
        # should begin with 'out_v5_nn' exactly
        if bctoex_dgenmethod_particle not in accepted_data_gener_methods :
            raise NameError(f"\n  ** Unexpected or unknown data generation method, should be one in {accepted_data_gener_methods},\n  ** but we have: '{bctoex_dgenmethod_particle}'. Is a new data generation method or new syntax in name ?")

    except NameError:
        print(" has changed ?" )
        raise

    tmp_dic['data_gener_method'] = bctoex_dgenmethod_particle
    if verbose:
        print(f" - Data generation method: {tmp_dic['data_gener_method']}")

    scase_particle = 'S'
    if bctoex_dgenseed_particle[:len(scase_particle)] == scase_particle :
        tmp_dic['seed_value'] = int(bctoex_dgenseed_particle[len(scase_particle):])
        if verbose:
            print(f" - Seed value for initializing Random sequence : {tmp_dic['seed_value']}")
    else:
        raise NameError(f"\n  ** unexpected information in base_case_label label ('{base_case_label}')\n"+\
                        f"  ** when examine the part '{bctoex_dgenseed_particle}' to find seed_value.\n"+\
                        f"  ** Reason: expected to find the '{scase_particle}' particule at the begining\n"+\
                        "  **  Is a correct case label?")

    # Recuperation et verification du label d'identification des donnees et du cas. Variable data_and_training_label

    # recolle les splited_base_case sans la premiere case puis decoupe par des underscores pour identier la particule 'v4' identifiant la fin du label  ...
    joined_for_data_label = "-".join(splited_base_case[2:])
    splitedunder_for_data_label = joined_for_data_label.split('_')

    try :
        # should begin with 'out_v5_nn' exactly
        if 'z{}v0'.format(region) not in splitedunder_for_data_label :
            print('ERR: z{}v0'.format(region), "not in", splitedunder_for_data_label)
            errmessg = f"\n  ** unspected information in part of base directory name. Expected to find the file particule,\n  ** in the '{joined_for_data_label}' part of base directory. Is a correct case?"
            print(errmessg)
            raise NameError(errmessg)

    except NameError:
        print(" has changed ?" )
        raise

    v_ipos = splitedunder_for_data_label.index('z{}v0'.format(region))
    tmp_dic['data_and_training_label'] = f"{tmp_dic['data_gener_method']}-{'_'.join(splitedunder_for_data_label[:v_ipos+1])}"

    scase_particle = 'GAN'
    if bctoex_dgenGANpc_particle[:len(scase_particle)] == scase_particle :
        tmp_dic['gan_train_percent_value'] = int(bctoex_dgenGANpc_particle[len(scase_particle):-2])  # '-2' pour enlever le 'pc' a la fin
        if verbose:
            print(f" - Franction of G.A.N. (GHG, AER, NAT) data for Training : {tmp_dic['gan_train_percent_value']}")
    else:
        raise NameError(f"\n  ** unexpected information in base_case_label label ('{base_case_label}')\n"+\
                        f"  ** when examine the part '{bctoex_dgenGANpc_particle}' to find GAN -GHG, AER, NAT- Train percent value.\n"+\
                        f"  ** Reason: expected to find the '{scase_particle}' particule at the begining\n"+\
                        "  **  Is a correct case label?")
    if tmp_dic['gan_train_percent_value'] < 100 :
        tmp_dic['gan_test_percent_value'] = 100 - tmp_dic['gan_train_percent_value']


    tmp_dic['associated_test_label'] = associated_test_label(tmp_dic['data_and_training_label'])
    
    if verbose:
        print(f" - Data and Training Label: {tmp_dic['data_and_training_label']}")
        print(f" - Associated Test Label: {tmp_dic['associated_test_label']}")
    
    tmp_dic['lp_nathist_filtering'] = 'NHLpFilt' in splitedunder_for_data_label
    if verbose:
        print(f" - Low-pass NAT anf HIST filtering: {tmp_dic['lp_nathist_filtering']}")

    return tmp_dic


def retrieve_param_from_sub_case(sub_case_to_explore, verbose=False) :
    """
    Retrieving parameters from specified sub_case_to_explore label.
    
    Example: 'CNN_Ks7-7-7_nCh24_Reg0.0005_XtrNO-EXTRAP_e200_bs100_Lr0.001-VfT15'
    Values are returned in a dictionnary with keys:
     - 'kern_size_list',
     - 'channel_size',
     - 'regularization_weight_decay',
     - 'extrapolation_label',
     - 'epochs',
     - 'batch_size',
     - 'learning_rate',
     - 'val_part_of_train_fraction'
     
    Usage:
        
        dic = retrieve_param_from_sub_case(base_case_label)
        
    """

    try :

        splited_sub_case = sub_case_to_explore.split('_')
        kern_size_list = [ int(k) for k in splited_sub_case[1][2:].split('-') ]
        if verbose:
            print(f" - List of sizes of CNN hidden layers or (kernel sizes): {kern_size_list}")
        
        i_scase = 2; scase_particle =  'nCh'
        if splited_sub_case[i_scase][:len(scase_particle)] == scase_particle :
            channel_size = int(splited_sub_case[i_scase][len(scase_particle):])
            if verbose:
                print(f" - CNN sub case channel size: {channel_size}")
        else:
            raise NameError(f"\n  ** unspected information in sub_case_to_explore label ('{sub_case_to_explore}')\n"+\
                            f"  ** when examine third part '{splited_sub_case[i_scase]}' to find channel_size.\n"+\
                            f"  ** Reason: expected to find the '{scase_particle}' particule at the begining\n"+\
                            "  **  Is a correct sub-case label?")

        i_scase = 3; scase_particle = 'Reg'
        if splited_sub_case[i_scase][:len(scase_particle)] == scase_particle :
            regularization_weight_decay = float(splited_sub_case[i_scase][len(scase_particle):])
            if verbose:
                print(f" - CNN sub case regularization weight decay value: {regularization_weight_decay}")
        else:
            raise NameError(f"\n  ** unspected information in sub_case_to_explore label ('{sub_case_to_explore}')\n"+\
                            f"  ** when examine the part '{splited_sub_case[i_scase]}' to find regularization_weight_decay.\n"+\
                            f"  ** Reason: expected to find the '{scase_particle}' particule at the begining\n"+\
                            "  **  Is a correct sub-case label?")

        i_scase = 4; scase_particle = 'Xtr'
        if splited_sub_case[i_scase][:len(scase_particle)] == scase_particle :
            extrapolation_label = splited_sub_case[i_scase][len(scase_particle):]
            if verbose:
                print(f" - CNN sub case extrapolation label: {extrapolation_label}")
        else:
            raise NameError(f"\n  ** unspected information in sub_case_to_explore label ('{sub_case_to_explore}')\n"+\
                            f"  ** when examine the part '{splited_sub_case[i_scase]}' to find extrapolation_label.\n"+\
                            f"  ** Reason: expected to find the '{scase_particle}' particule at the begining\n"+\
                            "  **  Is a correct sub-case label?")

        i_scase = 5; scase_particle = 'e'
        if splited_sub_case[i_scase][:len(scase_particle)] == scase_particle :
            epochs = int(splited_sub_case[i_scase][len(scase_particle):])
            if verbose:
                print(f" - CNN sub case epochs: {epochs}")
        else:
            raise NameError(f"\n  ** unspected information in sub_case_to_explore label ('{sub_case_to_explore}')\n"+\
                            f"  ** when examine the part '{splited_sub_case[i_scase]}' to find epochs.\n"+\
                            f"  ** Reason: expected to find the '{scase_particle}' particule at the begining\n"+\
                            "  **  Is a correct sub-case label?")

        i_scase = 6; scase_particle = 'bs'
        if splited_sub_case[i_scase][:len(scase_particle)] == scase_particle :
            batch_size = int(splited_sub_case[i_scase][len(scase_particle):])
            if verbose:
                print(f" - CNN sub case batch size: {batch_size}")
        else:
            raise NameError(f"\n  ** unspected information in sub_case_to_explore label ('{sub_case_to_explore}')\n"+\
                            f"  ** when examine the part '{splited_sub_case[i_scase]}' to find batch_size.\n"+\
                            f"  ** Reason: expected to find the '{scase_particle}' particule at the begining\n"+\
                            "  **  Is a correct sub-case label?")

        i_scase = 7; splited_sub_sub_case = splited_sub_case[i_scase].split('-')
        
        i_sscase = 0; scase_particle = 'Lr'
        if splited_sub_sub_case[i_sscase][:len(scase_particle)] == scase_particle :
            learning_rate = float(splited_sub_sub_case[i_sscase][len(scase_particle):])
            if verbose:
                print(f" - CNN sub case learning rate: {learning_rate}")

        else:
            raise NameError(f"\n  ** unspected information in sub_case_to_explore label ('{sub_case_to_explore}')\n"+\
                            f"  ** when examine a part of '{splited_sub_case[i_scase]}', ie: '{splited_sub_sub_case[i_sscase]}', to find learning_rate.\n"+\
                            f"  ** Reason: expected to find the '{scase_particle}' particule at the begining\n"+\
                            "  **  Is a correct sub-case label?")
        
        i_sscase = 1; scase_particle = 'VfT'
        if splited_sub_sub_case[i_sscase][:len(scase_particle)] == scase_particle :
            val_part_of_train_fraction = float(splited_sub_sub_case[i_sscase][len(scase_particle):])/100
            if verbose:
                print(f" - CNN sub case fraction part of train set for validation: {val_part_of_train_fraction}")

        else:
            raise NameError(f"\n  ** unspected information in sub_case_to_explore label ('{sub_case_to_explore}')\n"+\
                            f"  ** when examine a part of '{splited_sub_case[i_scase]}', ie: '{splited_sub_sub_case[i_sscase]}', to find val_part_of_train_fraction.\n"+\
                            f"  ** Reason: expected to find the '{scase_particle}' particule at the begining\n"+\
                            "  **  Is a correct sub-case label?")

    except NameError:
        print(" something has changed in sub_case_to_explore label ?" )
        raise

    tmp_dic = {'kern_size_list':kern_size_list,
               'channel_size':channel_size,
               'regularization_weight_decay':regularization_weight_decay,
               'extrapolation_label':extrapolation_label,
               'epochs':epochs,
               'batch_size':batch_size,
               'learning_rate':learning_rate,
               'val_part_of_train_fraction':val_part_of_train_fraction,
               }
    
    return tmp_dic


def sorted_distance_vectors(a, b, ord=None, axis=None, sort=True, ret_b=False) :
    """
    Function based on numpy linalg.norm() to compute the distance between a
    pattern (a 1D array) and a set of patterns (the lines of a 2D array).
    
    Parameters
    ----------
    a : 1D or 2D vector
        DESCRIPTION.
        
    b : vector or 2D matrix
        DESCRIPTION.
        
    ord : TYPE, optional
        Same as in linalg.norm() numpy function.
        The default is None.
        
    axis : int, optional
        Axis 0 or 1 only expected, as the function is used to compute distance
        between vertors or between a vector and a 2D matrix only.
        The default is None.
        
    sort : Boolean, optional
        Valid only in case of 'b' being a 2D Matrix. If True returns the sorted
        distance between a and each line or vector of 'b' and the sorted list on
        indices representing the nearest to the farthest to 'a'.
        The default is True.
        
    ret_b : Boolean, optional
        Valid only in case of 'b' being a 2D Matrix. If True the ordered 'b'
        matrix is also returned.
        The default is False.
    Returns
    -------
    float or tuple
        Returns the distance between 'a' and 'b'.
        
        If sort is True and axis is given and if 'b' is a 2D matrix, the function
        returns at least two arguments:
            
            1- the sorted list of distances between 'a' and each column (if axis=0) or
               line of 'b' (if axis=1), and
            
            2- the list of indices representing the order from nearest to farthest
               distance.
                
        If 'ret_b' is True (and sort is True and axis is given), the sorted
        version of input matrix 'b' is also returned.
    """
    import numpy as np

    if len(a.shape) > 2 or len(b.shape) > 2 or (len(a.shape) == 2 and 1 not in a.shape):
        print(f"\n *** sorted_distance_vectors error: 'a' should be a vector only and 'b' also a vector or a 2D matrix ***"+\
              f"\n ***     Dimensions found:\n ***     a.shape: {a.shape}\n ***     b.shape: {b.shape}\n")
        raise
    
    if len(a.shape) == 1 and len(b.shape) == 2 :
        if len(a) == b.shape[0]:
            a = a.reshape((len(a),1))

        else:
            a = a.reshape((1,len(a)))

    dist_ab = np.linalg.norm(a - b, ord=ord, axis=axis)
    
    if sort and axis is not None:
        idist_ab = dist_ab.argsort()
        
        # returns sorted distance + index of sorted values of B
        return_values = dist_ab[idist_ab],idist_ab
        
        # adds sorted B to return values
        if ret_b :
            if axis == 1:
                return_values = *return_values, b[idist_ab,:]
            else:
                return_values = *return_values, b[:,idist_ab]
        
        return return_values
    else:
        return dist_ab
    
def RMSE(y_obs, y_hat):
    """
    Function to compute the RMSE between two vectors.
    """
    import numpy as np
    return np.sqrt(((y_obs - y_hat) ** 2).mean(axis=1))

def RMSE_mean(y_obs, y_hat, n_end = None):
    """
    Function to compute the RMSE between the mean of two arrays.
    """
    import numpy as np
    
    if len(y_obs.shape) > 1 and y_obs.shape[0] > 1:
        vect_1 = y_obs.mean(axis=0)
    else:
        vect_1 = y_obs
        
    if len(y_hat.shape) > 1 and y_hat.shape[0] > 1:
        vect_2 = y_hat.mean(axis=0)
    else:
        vect_2 = y_hat
        
    if n_end is not None:
        if len(vect_1.shape) > 1:
            vect_1 = vect_1[:,-n_end:]
        else:
            vect_1 = vect_1[-n_end:]
        if len(vect_2.shape) > 1:
            vect_2 = vect_2[:,-n_end:]
        else:
            vect_2 = vect_2[-n_end:]
    return np.sqrt(((vect_1 - vect_2) ** 2).mean())


def print_loss_table(loss_test_tab, test_mod_df, y_obs, y_hat, tablefmt='simple',
                     columnorder=['N', 'loss', 'MSE', 'RMSE'],
                     indexname='model', indexorder=None, no_print=False,
                     verbose=False):

    import numpy as np
    import pandas as pd
    from tabulate import tabulate

    tmp1_loss_mod_df = test_mod_df.copy()
    tmp1_loss_mod_df['loss'] = loss_test_tab

    tmp2_ydiff_mod_df = pd.concat((tmp1_loss_mod_df.copy(),
                                  pd.DataFrame(((y_obs - y_hat) ** 2).mean(axis=1),
                                               columns=['MSE'])), axis=1)
    tmp2_ydiff_mod_df['RMSE'] = np.sqrt(tmp2_ydiff_mod_df['MSE'])

    line_for_all = [tmp1_loss_mod_df['loss'].mean(),((y_obs - y_hat) ** 2).mean(),np.sqrt(((y_obs - y_hat) ** 2).mean()),tmp1_loss_mod_df.shape[0]]

    tmp3_total_for_all_df = pd.DataFrame(np.array(line_for_all).reshape((1,len(line_for_all))),
                                    index=['ALL'],
                                    columns=['loss','MSE','RMSE','N'])

    tmp4_summary_by_mod_df = tmp2_ydiff_mod_df.groupby('model').mean()
    tmp4_summary_by_mod_df['N'] = tmp2_ydiff_mod_df.groupby('model')['loss'].count().values

    #display('AVANT:',tmp4_summary_by_mod_df)
    if indexorder is not None:
        #print("indexorder:", indexorder)
        tmp4_summary_by_mod_df = tmp4_summary_by_mod_df.reindex(index=indexorder)
        #display('APRES:',tmp4_summary_by_mod_df)

    tmp5_summary_by_mod_df = pd.concat((tmp4_summary_by_mod_df,tmp3_total_for_all_df), axis=0)

    if indexname is not None :
        tmp5_summary_by_mod_df.index.name = indexname

    if columnorder is not None:
        tmp5_summary_by_mod_df = tmp5_summary_by_mod_df[columnorder]

    if not no_print :
        print(tabulate(tmp5_summary_by_mod_df, headers='keys', tablefmt=tablefmt))

    return tmp5_summary_by_mod_df


def build_all_but_df (all_models, GHG_df, AER_df, NAT_df, HIST_df,
                      add_for_obs=False, obsname='OBS',
                      verbose=False) :
    
    import numpy as np
    import pandas as pd
    
    if np.isscalar(obsname):
        list_of_obsnames = [obsname]
    else:
        list_of_obsnames = obsname
    
    if verbose:
        print('list_of_obsnames:',list_of_obsnames)
    
    n_df_col = len(GHG_df.columns)
    
    if all_models is not None :
        for imod,mod in enumerate(all_models):
            if imod == 0:
                # Cree les DataFrames
                jmod = GHG_df.columns.values.tolist().index(mod)
                if verbose:
                    print(imod,jmod, end=', ')
                GHG_all_but_df  = GHG_df.iloc[ :,np.r_[0:jmod,(jmod+1):n_df_col]].copy().mean(axis=1).to_frame(name=mod)
    
                jmod = AER_df.columns.values.tolist().index(mod)
                if verbose:
                    print(imod,jmod, end=', ')
                AER_all_but_df  = AER_df.iloc[ :,np.r_[0:jmod,(jmod+1):n_df_col]].copy().mean(axis=1).to_frame(name=mod)
    
                jmod = NAT_df.columns.values.tolist().index(mod)
                if verbose:
                    print(imod,jmod, end=', ')
                NAT_all_but_df  = NAT_df.iloc[ :,np.r_[0:jmod,(jmod+1):n_df_col]].copy().mean(axis=1).to_frame(name=mod)
    
                jmod = HIST_df.columns.values.tolist().index(mod)
                if verbose:
                    print(imod,jmod, end=', ')
                HIST_all_but_df = HIST_df.iloc[:,np.r_[0:jmod,(jmod+1):n_df_col]].copy().mean(axis=1).to_frame(name=mod)
    
            else:
                # Ajoute dans les DataFrames
    
                jmod = GHG_df.columns.values.tolist().index(mod)
                if verbose:
                    print(imod,jmod, end=', ')
                GHG_all_but_df[mod]  = GHG_df.iloc[ :,np.r_[0:jmod,(jmod+1):n_df_col]].mean(axis=1)
    
                jmod = AER_df.columns.values.tolist().index(mod)
                if verbose:
                    print(imod,jmod, end=', ')
                AER_all_but_df[mod]  = AER_df.iloc[ :,np.r_[0:jmod,(jmod+1):n_df_col]].mean(axis=1)
    
                jmod = NAT_df.columns.values.tolist().index(mod)
                if verbose:
                    print(imod,jmod, end=', ')
                NAT_all_but_df[mod]  = NAT_df.iloc[ :,np.r_[0:jmod,(jmod+1):n_df_col]].mean(axis=1)
    
                jmod = HIST_df.columns.values.tolist().index(mod)
                if verbose:
                    print(imod,jmod, end=', ')
                HIST_all_but_df[mod] = HIST_df.iloc[:,np.r_[0:jmod,(jmod+1):n_df_col]].mean(axis=1)
    
            if verbose:
                print(imod,mod,jmod,np.r_[0:jmod,(jmod+1):n_df_col])

    if add_for_obs or all_models is None:
        #display(GHG_df)
        for iobs,obsname in enumerate(list_of_obsnames):
            if all_models is None and iobs == 0 :
                # creation du DataFrame
                if verbose:
                    print(f"Generating new all_but DFs global mean (for Obs or a new model '{obsname}'): adding mean for all FORC as 'all_but'")
                #display(GHG_df)
                GHG_all_but_df = pd.DataFrame(GHG_df.mean(axis=1).values, index=GHG_df.index, columns=[obsname])
                AER_all_but_df = pd.DataFrame(AER_df.mean(axis=1).values, index=AER_df.index, columns=[obsname])
                NAT_all_but_df = pd.DataFrame(NAT_df.mean(axis=1).values, index=NAT_df.index, columns=[obsname])
                HIST_all_but_df = pd.DataFrame(HIST_df.mean(axis=1).values, index=HIST_df.index, columns=[obsname])
            else:
                # ajout dans un DataFrame existant
                if verbose:
                    print(f"Adding column '{obsname}' to the all_but DFs global mean (for Obs or a new model): adding mean for all FORC as 'all_but'")
                #display(GHG_df)
                #display(GHG_df.mean(axis=1))
                #display(GHG_all_but_df)
                GHG_all_but_df[obsname] = GHG_df.mean(axis=1)
                AER_all_but_df[obsname] = AER_df.mean(axis=1)
                NAT_all_but_df[obsname] = NAT_df.mean(axis=1)
                HIST_all_but_df[obsname] = HIST_df.mean(axis=1)
        #display(GHG_all_but_df)

    if verbose:
        print(GHG_all_but_df)

    return GHG_all_but_df, AER_all_but_df, NAT_all_but_df, HIST_all_but_df


def plotting_mean_profiles(base_case_to_explore, sub_case_to_explore,
                           plot_intermodel=False, plot_all_but=False,
                           data_in_dir=None, data_out_dir=None, figs_dir=None, save_figs=True,
                           local_nb_label="MeanProfiles", fig_ext='png',
                           source_dirname='data_source_pl',
                           figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w'},
                           lp_nathist_filtering_dic_file='lp_nat_and_hist_filtering_param_dictionary.p',
                           verbose=False,
                          ):
    import os
    import matplotlib.pyplot as plt

    # Repertoire des donnees
    if data_in_dir is None :
        data_in_dir = get_source_data_dir(dirname=source_dirname, verbose=verbose)

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
    #  - 'lp_nathist_filtering'
    if verbose:
        print(f"\nRetrieving parameters from specified base case to explore '{base_case_to_explore}':")# decomposing base case name to explore 
    base_case_dic = retrieve_param_from_base_case(base_case_to_explore, verbose=verbose)
    lp_nathist_filtering = base_case_dic['lp_nathist_filtering']
    if verbose:
        print(f" - Low-pass NAT and HIST filtering: {lp_nathist_filtering}")

    # Identifiant global des cas et repertoire des sorties (commun a tous les sous-cas de l'ensemble)
    cnn_name_base = sub_case_to_explore
    
    case_figs_dir = os.path.join(figs_dir, base_case_to_explore, f'{cnn_name_base}')

    case_out_base_path = os.path.join(data_out_dir, base_case_to_explore)
    print(f"Repertoire de base de entree-sortie pour tous les Cas: '{case_out_base_path}/'")

    load_data_and_gener_params = {'data_dir':data_in_dir, 'verbose':verbose}
    if lp_nathist_filtering:
        filtering_dic_filename = os.path.join(case_out_base_path,lp_nathist_filtering_dic_file)

        load_data_and_gener_params['to_filter'] = ['nat', 'hist']
        load_data_and_gener_params['filtering_dic_filename'] = filtering_dic_filename
    load_data_and_gener_params['verbose'] = verbose
    
    data_dic = load_basic_data_and_gener_df(**load_data_and_gener_params)

    data_label       = data_dic['label']
    all_models_src   = data_dic['models']
    #all_forcings_src = data_dic['forcings']
    #all_forcing_color_dic = data_dic['forcing_color_dic']
    #all_forcing_inv_color_dic = data_dic['forcing_inv_color_dic']
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
    
    # ##########################################################################
    # Plotting Intermodel Mean Profiles
    # ##########################################################################

    if not plot_intermodel :
        print(" ** Plotting Intermodel Mean Profiles Not activated **")
    else:
        ncols = 2
        nrows = 2

        fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(10*ncols,6*nrows),
                                gridspec_kw={'hspace': 0.10, 'wspace': 0.05, 
                                             'left': 0.06, 'right': 0.98,
                                             'top' : 0.92, 'bottom' : 0.04 })

        ax = axes[0,0]
        ax.set_prop_cycle(None)
        GHG_ens_df.plot(ax=ax)
        ax.set_prop_cycle(None)
        ax.set_title('GHG')
        xmin,xmax = ax.get_xlim()
        ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
        ax.set_xlim([xmin,xmax])
        ax.grid(True,lw=0.75,ls=':')

        ax = axes[0,1]
        ax.set_prop_cycle(None)
        AER_ens_df.plot(ax=ax)
        ax.set_prop_cycle(None)
        ax.set_title('AER')
        xmin,xmax = ax.get_xlim()
        ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
        ax.set_xlim([xmin,xmax])
        ax.grid(True,lw=0.75,ls=':')

        ax = axes[1,0]
        ax.set_prop_cycle(None)
        NAT_ens_df.plot(ax=ax)
        ax.set_prop_cycle(None)
        title_lbl = 'NAT'
        if lp_nathist_filtering:
            title_lbl = f"{title_lbl} LP filtered"
        ax.set_title(title_lbl)
        xmin,xmax = ax.get_xlim()
        ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
        ax.set_xlim([xmin,xmax])
        ax.grid(True,lw=0.75,ls=':')

        ax = axes[1,1]
        ax.set_prop_cycle(None)
        HIST_ens_df.plot(ax=ax)
        ax.set_prop_cycle(None)
        title_lbl = 'HIST'
        if lp_nathist_filtering:
            title_lbl = f"{title_lbl} LP filtered"
        ax.set_title(title_lbl)
        xmin,xmax = ax.get_xlim()
        ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
        ax.set_xlim([xmin,xmax])
        ax.grid(True,lw=0.75,ls=':')

        suptitle_label = f"Mean Intermodel Profiles ('{data_label}' forcings data)"
        if lp_nathist_filtering:
            suptitle_label = f"{suptitle_label} having NAT and HIST LP filtered"
        plt.suptitle(suptitle_label,size="xx-large",y=0.98)

        figfile_label = suptitle_label.replace(' ','_').replace('(','').replace(')','').replace("'",'')
        if lp_nathist_filtering:
            figfile_label = f"{figfile_label}-LP-filtered"
        figs_file = f"Fig{local_nb_label}_{figfile_label}.{fig_ext}"
        figs_filename = os.path.join(case_figs_dir,figs_file)

        if save_figs:
            if os.path.isfile(figs_filename):
                print(f"\n ** Mean Intermodel figure not saved because file already exists **\n **   '{figs_filename}' **'\n")
    
            else:
                print("-- saving figure in file ... '{}'".format(figs_filename))
                plt.savefig(figs_filename, **figs_defaults)
        else:
            print(' ** figure not saved. Saving not active **')

        plt.show()
        
    # ##########################################################################
    # Plotting "ALL BUT ..." Intermodel Mean Profiles
    # ##########################################################################

    if not plot_all_but :
        print(" ** Plotting ALL BUT Intermodel Mean Profiles Not activated **")
    else:

        # Build "all_but" DataFrames
        GHG_ens_all_but_df, AER_ens_all_but_df, NAT_ens_all_but_df, \
            HIST_ens_all_but_df = build_all_but_df(all_models_src,
                                                   GHG_ens_df, AER_ens_df,
                                                   NAT_ens_df, HIST_ens_df)
        
        # for plotting intermodel mean on background, sets it to True
        plot_immean4bg_ok = True

        immean4bg_color = [0.8,0.8,0.8,1]
        immean4bg_alpha = 0.5
        immean4bg_lw = 1
        immean4bg_ls = '-'

        ncols = 2
        nrows = 2

        fig,axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=True,sharey=True,figsize=(10*ncols,6*nrows),
                                gridspec_kw={'hspace': 0.10, 'wspace': 0.05, 
                                             'left': 0.06, 'right': 0.98,
                                             'top' : 0.92, 'bottom' : 0.04 })

        ax = axes[0,0]
        if plot_immean4bg_ok :
            ax.set_prop_cycle(None)
            GHG_ens_df.plot(ax=ax, legend=False, color=immean4bg_color, alpha=immean4bg_alpha, ls=immean4bg_ls, lw=immean4bg_lw)
        ax.set_prop_cycle(None)
        GHG_ens_all_but_df.plot(ax=ax)
        ax.set_prop_cycle(None)
        ax.set_title('GHG')
        xmin,xmax = ax.get_xlim()
        ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
        ax.set_xlim([xmin,xmax])
        ax.grid(True,lw=0.75,ls=':')

        ax = axes[0,1]
        if plot_immean4bg_ok :
            AER_ens_df.plot(ax=ax, legend=False, color=immean4bg_color, alpha=immean4bg_alpha, ls=immean4bg_ls, lw=immean4bg_lw)
        ax.set_prop_cycle(None)
        AER_ens_all_but_df.plot(ax=ax)
        ax.set_prop_cycle(None)
        ax.set_title('AER')
        xmin,xmax = ax.get_xlim()
        ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
        ax.set_xlim([xmin,xmax])
        ax.grid(True,lw=0.75,ls=':')

        ax = axes[1,0]
        if plot_immean4bg_ok :
            NAT_ens_df.plot(ax=ax, legend=False, color=immean4bg_color, alpha=immean4bg_alpha, ls=immean4bg_ls, lw=immean4bg_lw)
        ax.set_prop_cycle(None)
        NAT_ens_all_but_df.plot(ax=ax)
        ax.set_prop_cycle(None)
        title_lbl = 'NAT'
        if lp_nathist_filtering:
            title_lbl = f"{title_lbl} LP filtered"
        ax.set_title(title_lbl)
        xmin,xmax = ax.get_xlim()
        ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
        ax.set_xlim([xmin,xmax])
        ax.grid(True,lw=0.75,ls=':')

        ax = axes[1,1]
        if plot_immean4bg_ok :
            HIST_ens_df.plot(ax=ax, legend=False, color=immean4bg_color, alpha=immean4bg_alpha, ls=immean4bg_ls, lw=immean4bg_lw)
        ax.set_prop_cycle(None)
        HIST_ens_all_but_df.plot(ax=ax)
        ax.set_prop_cycle(None)
        title_lbl = 'HIST'
        if lp_nathist_filtering:
            title_lbl = f"{title_lbl} LP filtered"
        ax.set_title(title_lbl)
        xmin,xmax = ax.get_xlim()
        ax.hlines(0, xmin=xmin, xmax=xmax, lw=0.5, ls='-', color='k')
        ax.set_xlim([xmin,xmax])
        ax.grid(True,lw=0.75,ls=':')

        suptitle_label = "Mean Intermodel \"ALL BUT ...\" Profiles for each Climat model"
        if lp_nathist_filtering:
            suptitle_label = f"{suptitle_label} having NAT and HIST LP filtered"
        if plot_immean4bg_ok:
            suptitle_label =f"{suptitle_label} - intermodel mean on background"
        suptitle_label =f"{suptitle_label} [forcings data: '{data_label}']"
        plt.suptitle(suptitle_label,size="xx-large",y=0.98)

        figfile_label = suptitle_label.replace(' ','_').replace(': ','-').replace(':','-').replace('-_','-').replace('_-','-').replace('(','').replace(')','').replace('[','').replace(']','').replace("'",'').replace('"','').replace('_...','')
        if lp_nathist_filtering:
            figfile_label = f"{figfile_label}-LP-filtered"
        figs_file = f"Fig{local_nb_label}_{figfile_label}.{fig_ext}"
        figs_filename = os.path.join(case_figs_dir,figs_file)

        if save_figs :
            if os.path.isfile(figs_filename):
                print(f"\n ** Mean Intermodel \"ALL BUT ...\" figure not saved because file already exists **\n **   '{figs_filename}' **'\n")
    
            else:
                print("-- saving figure in file ... '{}'".format(figs_filename))
                plt.savefig(figs_filename, **figs_defaults)
        else:
            print(' ** figure not saved. Saving not active **')

        plt.show()
        
    return