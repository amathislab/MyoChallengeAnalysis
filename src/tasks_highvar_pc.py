import pickle
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from definitions import ROOT_DIR

# Cross-projection similarity analysis of hold, cw and ccw kinematic subspaces using last model of phase 2 

# Cross-projection
def cross_project_kin(vel1,vel2,n_comp,n):
    pca = PCA(n_components=n_comp)
    pca_2 = pca.fit(vel2)
    pca_1 = pca.fit(vel1)
    return {'projection 1 on 2':pca_2.transform(vel1),'projection 2 on 1':pca_1.transform(vel2),'V1_1':np.cumsum(pca_1.explained_variance_ratio_)[n-1],'V1_2':np.cumsum(pca_2.explained_variance_ratio_)[n-1]}

# Compute explained variance without using sklearn
def exp_var(X,n):
    cov_matrix = np.cov(X,rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    total_eigenvalues = sum(eigenvalues)
    var_exp = [(i/total_eigenvalues) for i in sorted(eigenvalues,reverse=True)]
    return np.cumsum(var_exp),np.cumsum(var_exp)[n-1]

def mean_ratio(cproj,label1,label2):
    cum_1_2, v_1_2 = exp_var(cproj['projection 1 on 2'],n=n)
    cum_2_1, v_2_1 = exp_var(cproj['projection 2 on 1'],n=n)
    plt.clf()
    plt.step(range(1,n_comp+1),cum_1_2,alpha=0.7,where='mid',label=label1)
    plt.step(range(1,n_comp+1),cum_2_1,alpha=0.7,where='mid',label=label2)
    plt.legend(fontsize=21)
    plt.xlabel('Number of PCs',fontsize=21)
    plt.ylabel('Explained variance',fontsize=21)
    plt.yticks(fontsize=21)
    plt.xticks(fontsize=21)
    plt.title('Absolute joint velocities',fontsize=21)
    plt.subplots_adjust(left=0.15,bottom=0.15)
    plt.show()
    print(label1,v_1_2 / cproj['V1_1'],label2,v_2_1 / cproj['V1_2'])
    return [(v_1_2 / cproj['V1_1'] + v_2_1 / cproj['V1_2']) / 2, plt.figure()]

if __name__=='__main__':

    n_comp = 23
    n = 12

    conds = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/synergies_tasks','rb'))
    hold_velocities = np.concatenate([cond['hand velocity'] for cond in conds if cond['encoding']=='hold'])
    cw_velocities = np.concatenate([cond['hand velocity'] for cond in conds if cond['encoding']=='cw'])
    ccw_velocities = np.concatenate([cond['hand velocity'] for cond in conds if cond['encoding']=='ccw'])

    cproj_hold_cw = cross_project_kin(vel1=hold_velocities,vel2=ccw_velocities,n_comp=n_comp,n=n)
    cproj_hold_ccw = cross_project_kin(vel1=hold_velocities,vel2=ccw_velocities,n_comp=n_comp,n=n)
    cproj_ccw_cw = cross_project_kin(vel1=ccw_velocities,vel2=cw_velocities,n_comp=n_comp,n=n)

    hold_cw = mean_ratio(cproj_hold_cw,label1='Hold on CW',label2='CW on hold')
    hold_ccw = mean_ratio(cproj_hold_ccw,label1='Hold on CCW',label2='CCW on hold')
    ccw_cw = mean_ratio(cproj_ccw_cw,label1='CCW on CW',label2='CW on CCW')

    V = {'hold vs. cw':hold_cw[0], 'hold vs. ccw':hold_ccw[0], 'cw vs. ccw':ccw_cw[0]}
    pd.DataFrame(V,index=[0]).to_csv(os.path.join(ROOT_DIR,'SIL-Results/Motor-synergies/High-level-PCs/CW-vs-CCW-vs-Hold/V1-V2.csv'))




