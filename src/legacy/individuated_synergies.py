import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from definitions import ROOT_DIR

# Todorov, control task

def PCvsVar(df,title,filename,n_comp=23):
    pca = PCA(n_components=n_comp)
    pca.fit_transform(np.copy(df))
    plt.clf()
    plt.bar(range(1,n_comp+1), pca.explained_variance_ratio_, alpha=0.5, align='center',label='Individual variance')
    plt.step(range(1,n_comp+1), np.cumsum(pca.explained_variance_ratio_), where='mid',label='Cumulative variance')
    plt.xlabel('Number of PCs',fontsize=21)
    plt.ylabel('Explained variance',fontsize=21)
    plt.legend(fontsize=21,loc='best')
    plt.title(title,fontsize=21)
    plt.yticks(fontsize=21)
    plt.xticks(fontsize=21)
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.axhline(y=0.85, color='g', linestyle='-')
    plt.text(21, 0.87, '95%', color = 'red', fontsize=21)
    plt.text(21, 0.77, '85%', color = 'green', fontsize=21)
    plt.subplots_adjust(left=0.15,bottom=0.15)
    plt.savefig(os.path.join(ROOT_DIR,"SIL-Results/Motor-synergies/DOF-control/"+filename+".png"))
    return pca.explained_variance_ratio_


fp = os.path.join(ROOT_DIR,'rollouts_hand_pose_lattice.h5')
observations = pd.read_hdf(fp).to_dict(orient='dict')['observation']
hand_positions = np.array([observations[ep][0:23] for ep in observations.keys()])
hand_velocities = np.array([observations[ep][23:46] for ep in observations.keys()])

# Absolute joint angles
abs_pos_var = PCvsVar(df=hand_positions, title='Absolute joint angles, control', filename='abs_position')
print('Abs, position\n',np.cumsum(abs_pos_var))

# Normalized to range (0,1)
norm01_hand_positions = np.array([(hand_pos-np.min(hand_pos))/(np.max(hand_pos)-np.min(hand_pos)) for hand_pos in hand_positions.T]).T
norm01_pos_var = PCvsVar(df=norm01_hand_positions,title='[0,1] normalized joint angles, control',filename='0-1_position')
print('0-1 range, position \n',np.cumsum(norm01_pos_var))

# Normalized to unit variance
scaler = StandardScaler(with_mean=False)
norm_hand_positions = np.array([np.squeeze(scaler.fit_transform(np.reshape(hand_pos,(hand_positions.shape[0],1)))) for hand_pos in hand_positions.T]).T
norm_pos_var = PCvsVar(df=norm_hand_positions,title='Unit variance normalized joint angles, control',filename='unit-var_position')
print('Unit variance, position \n',np.cumsum(norm_pos_var))


# Absolute joint velocities
abs_vel_var = PCvsVar(df=hand_velocities, title='Absolute joint velocities, control', filename='abs_velocity')
print('Abs, velocity\n',np.cumsum(abs_vel_var))

# Normalized to range (0,1)
norm01_hand_velocities = np.array([(hand_vel-np.min(hand_vel))/(np.max(hand_vel)-np.min(hand_vel)) for hand_vel in hand_velocities.T]).T
norm01_vel_var = PCvsVar(df=norm01_hand_velocities,title='[0,1] normalized joint velocities, control',filename='0-1_velocity')
print('0-1 range, velocity\n',np.cumsum(norm01_vel_var))

# Normalized to unit variance
scaler = StandardScaler(with_mean=False)
norm_hand_velocities = np.array([np.squeeze(scaler.fit_transform(np.reshape(hand_vel,(hand_velocities.shape[0],1)))) for hand_vel in hand_velocities.T]).T
norm_vel_var = PCvsVar(df=norm_hand_velocities,title='Unit variance normalized joint velocities, control',filename='unit-var_velocity')
print('Unit variance, velocity\n',np.cumsum(norm_vel_var))
