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


obs = torch.load('ROF_review_no_mean/ALL/data_hist_mean.pt').detach().numpy()[0:115] * 1.06
print(f"obs shape : {obs.shape}")

liste_models = ['CanESM5', 'CNRM', 'IPSL', 'ACCESS', 'BCC', 'FGOALS', 'HadGEM3', 'MIRO', 'ESM2',
                    'NorESM2','CESM2','GISS']

variab = np.array([0.1,0.13,0.15,0.11,0.17,0.10,0.11,0.13,0.1,0.15,0.13,0.15])

Mean_orig = np.load('ROF_review_no_mean/ALL/data_all.npy')
print(Mean_orig[3])
print(f"Mean_orig shape : {Mean_orig.shape}")
mean_tot = np.mean(np.load('ROF_review_no_mean/mean_by_model_all.npy')[:,1],axis=0)
print(f"mean_tot : {mean_tot}")
print(f"mean_tot shape : {mean_tot.shape}")
aer = torch.load('/net/ether/data/varclim/cbone/data_article_final/aer.pt')

mean = np.copy(Mean_orig)




#obs = obs - np.mean(obs)
# =============================================================================
# mean[0] -= np.mean(mean,axis=1)[0]
# mean[1] -= np.mean(mean,axis=1)[1]
# mean[2] -= np.mean(mean,axis=1)[2]
# mean[3] -= np.mean(mean,axis=1)[3]
# =============================================================================


#mean = np.subtract(mean - np.mean(mean,axis=1))
count = np.load("/net/ether/data/varclim/cbone/data_article_final/count_tls_ALL.npy")   
noise = np.load("/net/ether/data/varclim/cbone/data_article_final/noise_tls_ALL.npy")



mean = np.transpose(mean)
anom_index = noise.shape[0]
noise = np.transpose(noise)
model = 'ALL'


(xr,yr,cn1,cn2)=da.reduce_dim(mean,obs[:,None],noise[:,list(range(1,anom_index,2))],noise[:,list(range(0,anom_index,2))])
result=da.tls(xr[:,0:3],yr,cn1,ne=count[0:3],cn2=cn2,rof_flag=1,RCT_flag=0)
name_exp = 'ROF_review_no_mean/'+model+'/'
path = '/usr/home/habbar/Bureau/data_nc/'
mkdir_p(path+name_exp)
print(result['beta'])


mean_variab = np.mean(variab)
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
    plt.savefig(path+name_exp+str(forc)+'_result.png')
    
plt.show()


cpt = 0
forc_mean = np.load('/net/ether/data/varclim/cbone/data_article_final/forc_mean.npy') 


for model in liste_models:
    historical = np.array(torch.load('/net/ether/data/varclim/cbone/data_article_final/hist_test.pt')[cpt])
    if cpt==12:break
    cpt+=1
    
    
    for hist in range(1):
    

        
        obs = historical[hist]

        
        Mean_orig = np.load('/net/ether/data/varclim/cbone/data_article_final/mean_tls_'+model+'.npy')
        plt.plot(Mean_orig[1])
        plt.show()
        
        mean = np.copy(Mean_orig)
