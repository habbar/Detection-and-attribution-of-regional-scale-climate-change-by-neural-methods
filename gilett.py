#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:05:18 2022

@author: cbone
"""

#fichier refaisant la procédure de Gilett avec nos données pour comparaison

#import extraction_data as extr
import numpy as np
import detatt_mk as da
import torch
import matplotlib.pyplot as plt
from scipy import stats
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


obs = torch.load('figures/Europe/rand_hist_IPSL-CM6A-LR.pt').detach().numpy()[0:115] * 1.06
print(f"obs shape : {obs.shape}")
#print(f"obs : {obs}")

liste_models = ['ACCESS', 'BCC', 'CESM2', 'CNRM', 'CanESM5', 'FGOALS', 
                'GISS', 'HadGEM3', 'IPSL', 'MIRO', 'ESM2', 'NorESM2']
#liste_models = ['ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CESM2', 
 #               'CNRM-CM6-1', 'CanESM5', 'FGOALS-g3', 'GISS-E2-1-G', 
  #              'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6', 'MRI-ESM2-0',
   #             'NorESM2-LM']

variab = np.array([0.1,0.13,0.15,0.11,0.17,0.10,0.11,0.13,0.1,0.15,0.13,0.15])

Mean_orig = np.load('figures/Europe/concatenated_IPSL-CM6A-LR.npy')
print(f"Mean_orig[3] : {Mean_orig[3]}")  
print(f"Mean_orig shape : {Mean_orig.shape}")
mean_tot = np.mean(np.load('figures/Europe/combined_data.npy')[:,1],axis=0) 
print(f"mean_tot shape : {mean_tot.shape}")
aer = torch.load('figures/Europe/aer.pt')
print(f"aer : {aer}")

for tensor in aer:
    print(tensor.shape)

mean = np.copy(Mean_orig)


#obs = obs - np.mean(obs)
# =============================================================================
# mean[0] -= np.mean(mean,axis=1)[0]
# mean[1] -= np.mean(mean,axis=1)[1]
# mean[2] -= np.mean(mean,axis=1)[2]
# mean[3] -= np.mean(mean,axis=1)[3]
# =============================================================================


#mean = np.subtract(mean - np.mean(mean,axis=1))
count = np.load("figures/Europe/count.npy")   
print(f"count shape : {count.shape}")
print(f"count : {count}")
noise = np.load("figures/Europe/noise.npy")
print(f"noise shape : {noise.shape}")


mean = np.transpose(mean)
anom_index = noise.shape[0]
noise = np.transpose(noise)
model = 'IPSL-CM6A-LR'


(xr,yr,cn1,cn2)=da.reduce_dim(mean,obs[:,None],noise[:,list(range(1,anom_index,2))],noise[:,list(range(0,anom_index,2))])
result=da.tls(xr[:,0:3],yr,cn1,ne=count[0:3],cn2=cn2,rof_flag=1,RCT_flag=0)
name_exp = 'ROF/'+model+'/'
path = '/usr/home/habbar/Bureau/data_nc/'
mkdir_p(path+name_exp)
print(f"result Beta : {result['beta']}")  
print(f"result Beta shape : {result['beta'].shape}")

mean_variab = np.mean(variab)
sum_combined = np.zeros((115))
for forc in range(3):
    ecart = (result['betaCI'][forc,1] - result['betaCI'][forc,0]) / 2
    
    ecart = ecart / 1.64
    print(ecart)
    x_inf=np.zeros((115))
    x_sup=np.zeros((115))
    nn=np.arange(115)
    for i in nn:
        Xn=stats.norm.rvs(size=10000,loc = Mean_orig[forc][i], scale = mean_variab/np.sqrt(count[forc]))
        betan=stats.norm.rvs(size=10000,loc = result['beta'][forc], scale = ecart)
        mult=Xn*betan
        x_inf[i]=np.percentile(mult,5)
        x_sup[i]=np.percentile(mult,95)

    plt.fill_between(nn,x_inf,x_sup,alpha=0.5)
    np.save(path+name_exp+str(forc)+'_result_mean',Mean_orig[forc]*result['beta'][forc])
    np.save(path+name_exp+str(forc)+'_result_min',x_inf)
    np.save(path+name_exp+str(forc)+'_result_max',x_sup)
    plt.plot(nn,Mean_orig[forc]*result['beta'][forc])
    plt.grid()
    plt.savefig(path+name_exp+str(forc)+'_result.png')
    
    sum_product = np.sum(result['beta'][forc] * Mean_orig[forc])

    print(f"Somme des produits pour le forçage {forc}: {sum_product}")

    # Ajouter la courbe calculée à la somme globale
    sum_combined += Mean_orig[forc] * result['beta'][forc]

# Plot de la somme des courbes combinées
plt.plot(nn, sum_combined, label='Somme des courbes combinées', color='black')
plt.legend()

# Plot de l'observation historique cible
plt.plot(nn, obs, label='Observation historique cible', color='red')
plt.legend()
# sauvegarde de la figure
plt.savefig(path+name_exp+'sum_combined.png')

plt.show()