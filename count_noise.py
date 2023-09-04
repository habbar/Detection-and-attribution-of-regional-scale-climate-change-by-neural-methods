import numpy as np

def get_data_set_tls(model='IPSL-CM6A-LR', cluster=-1, filtrage=False, mean=True):

    liste_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CESM2', 'CNRM-CM6-1', 'CanESM5', 'FGOALS-g3',
                'GISS-E2-1-G', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6', 'MRI-ESM2-0', 'NorESM2-LM']

    aer = []
    nat = []
    historical = []
    ghg = []
    noise = np.empty((0, 115))
    print('debut tls')

    count = np.array([0., 0., 0., 0.])

    for model_curr in liste_models:
        if model_curr != model:
            print(model_curr)

            # Charger les données correspondantes à chaque forçage à partir des fichiers .npy 
            aer_curr = np.load(f'figures/Europe/aer_{model_curr}_no_mean.npy')
            nat_curr = np.load(f'figures/Europe/nat_{model_curr}_no_mean.npy')
            historical_curr = np.load(f'figures/Europe/hist_{model_curr}_no_mean.npy')
            ghg_curr = np.load(f'figures/Europe/ghg_{model_curr}_no_mean.npy')

            aer_bruit = (aer_curr - np.mean(aer_curr, axis=0)) * np.sqrt((aer_curr.shape[0]) / (aer_curr.shape[0] - 1))
            print(f"aer_bruit shape : {aer_bruit[0:-1].shape}")
            noise = np.concatenate((noise, aer_bruit[0:-1]), axis=0)

            ghg_bruit = (ghg_curr - np.mean(ghg_curr, axis=0)) * np.sqrt((ghg_curr.shape[0]) / (ghg_curr.shape[0] - 1))
            noise = np.concatenate((noise, ghg_bruit[0:-1]), axis=0)

            nat_bruit = (nat_curr - np.mean(nat_curr, axis=0)) * np.sqrt((nat_curr.shape[0]) / (nat_curr.shape[0] - 1))
            noise = np.concatenate((noise, nat_bruit[0:-1]), axis=0)

            hist_bruit = (historical_curr - np.mean(historical_curr, axis=0)) * np.sqrt(
                (historical_curr.shape[0]) / (historical_curr.shape[0] - 1))
            noise = np.concatenate((noise, hist_bruit[0:-1]), axis=0)

            count += np.array(
                [1 / ghg_curr.shape[0], 1 / aer_curr.shape[0], 1 / nat_curr.shape[0], 1 / historical_curr.shape[0]])

            aer.append(np.mean(aer_curr, axis=0))
            ghg.append(np.mean(ghg_curr, axis=0))
            nat.append(np.mean(nat_curr, axis=0))
            historical.append(np.mean(historical_curr, axis=0))

    number_model = np.array(aer).shape[0]
    print(number_model)
    if mean:
        aer = np.mean(np.array(aer), axis=0)
        nat = np.mean(np.array(nat), axis=0)
        historical = np.mean(np.array(historical), axis=0)
        ghg = np.mean(np.array(ghg), axis=0)
    mean = np.array([ghg, aer, nat, historical])

    count = (number_model * number_model) / count

    return mean, count, noise

# appel de la fonction get_data_set_tls
mean, count, noise = get_data_set_tls(model='IPSL-CM6A-LR', cluster=-1, filtrage=False, mean=True)

# affichage des données
print(f"mean shape : {mean.shape}")
print(f"count shape : {count.shape}")
print(f"noise shape : {noise.shape}")
print(f"count : {count}")

# téléchargement de count et noise en .npy
np.save('figures/Europe/count.npy', count)
np.save('figures/Europe/noise.npy', noise)
