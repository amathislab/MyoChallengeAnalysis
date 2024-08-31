from definitions import ROOT_DIR
import os
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO
from envs.environment_factory import EnvironmentFactory
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from legacy.functions import make_parallel_envs
import pandas as pd

# Cross-projection similarity analysis of hold and ccw kinematic subspaces

def get_pos_vel(env_fp,net_fp,config,num_ep,env_name,render=False, print_cum=True):
    envs = make_parallel_envs(env_name, config, num_env=1)
    envs = VecNormalize.load(env_fp, envs)
    envs.training = False
    envs.norm_reward = False
    custom_objects = {
        "learning_rate": lambda _: 0,
        "lr_schedule": lambda _: 0,
        "clip_range": lambda _: 0,
    }
    model = RecurrentPPO.load(
            net_fp, env=envs, device="cpu", custom_objects=custom_objects
        )

    eval_model = model
    eval_env = EnvironmentFactory.create(env_name,**config)
    tot_pos = []
    tot_vel = []
    for n in range(num_ep):
        obs_tot = []
        cum_reward = 0
        lstm_states = None
        obs = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        timestep = 0
        while not done: 
            if render :
                eval_env.sim.render(mode="window")
                
            timestep += 1
            action, lstm_states = eval_model.predict(envs.normalize_obs(obs),
                                                    state=lstm_states,
                                                    episode_start=episode_starts,
                                                    deterministic=True,
                                                    )
                                                        
            obs, rewards, done, info = eval_env.step(action)
            episode_starts = done
            cum_reward += rewards
            obs_tot.append(obs)
        if print_cum : 
            print('episode %s : '%n,cum_reward)

        # MEASURE JOINT POSITION AND VELOCITY
        hand_positions = np.array(obs_tot)[:,0:23]    
        hand_velocities = np.array([np.diff(pos)/0.0025 for pos in hand_positions.T]).T
        hand_velocities = np.vstack((np.zeros((1,23)),hand_velocities))
        tot_pos.append(hand_positions)
        tot_vel.append(hand_velocities)
            
    return {'pos':np.concatenate(tot_pos,axis=0),'vel':np.concatenate(tot_vel,axis=0)}

# Project Hold onto Rotation subspace and vice versa
def cross_project_kin(vel_hold,vel_rotation,n_comp,n=10):
    pca = PCA(n_components=n_comp)
    pca_rotation = pca.fit(vel_rotation)
    pca_hold = pca.fit(vel_hold)
    return {'hold projection':pca_rotation.transform(vel_hold),'rotation projection':pca_hold.transform(vel_rotation),'V1_hold':np.cumsum(pca_hold.explained_variance_ratio_)[n-1],'V1_rotation':np.cumsum(pca_rotation.explained_variance_ratio_)[n-1]}

# Compute explained variance without using sklearn
def exp_var(X,n=10):
    cov_matrix = np.cov(X,rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    total_eigenvalues = sum(eigenvalues)
    var_exp = [(i/total_eigenvalues) for i in sorted(eigenvalues,reverse=True)]
    return np.cumsum(var_exp),np.cumsum(var_exp)[n-1]

if __name__=="__main__":

    env_name = 'CustomMyoBaodingBallsP1'
    render = False

    PATH_TO_NORMALIZED_ENV_hold = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/01_rsi_static/env.pkl",
    )
    PATH_TO_PRETRAINED_NET_hold = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/01_rsi_static/model.zip",
    )

    PATH_TO_NORMALIZED_ENV_rotation = os.path.join(
        ROOT_DIR,
        'Last-model-P1/normalized_env_phase1_final',
    )
    PATH_TO_PRETRAINED_NET_rotation = os.path.join(
        ROOT_DIR,
        'Last-model-P1/phase1_final.zip',
    )

    config_hold = {
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "alive": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0
        },
        "task": "ccw", 
        "enable_rsi": True,
        "goal_time_period": [
            1e+100,
            1e+100
        ],
        "goal_xrange": [
            0.025,
            0.025
        ],
        "goal_yrange": [
            0.028,
            0.028
        ]
    }

    config_rotation = {
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "alive": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0
        },
        "task": "ccw",
        "enable_rsi": False,
        "rsi_probability": 0,
        "noise_palm": 0,
        "noise_fingers": 0,
        "noise_balls": 0,
        "goal_time_period": [
            5,
            5
        ],
        "goal_xrange": [
            0.025,
            0.025
        ],
        "goal_yrange": [
            0.028,
            0.028
        ],
        "drop_th": 1.3
        }
    
    num_ep = 10
    scaler = StandardScaler(with_mean=False)
    n_comp = 23

    kin_hold = get_pos_vel(env_name=env_name,env_fp=PATH_TO_NORMALIZED_ENV_hold,net_fp=PATH_TO_PRETRAINED_NET_hold,config=config_hold,num_ep=num_ep,render=False, print_cum=True)
    abs_vel_hold = kin_hold['vel']
    norm01_vel_hold = np.array([(hand_vel-np.min(hand_vel))/(np.max(hand_vel)-np.min(hand_vel)) for hand_vel in abs_vel_hold.T]).T
    var1_vel_hold = np.array([np.squeeze(scaler.fit_transform(np.reshape(hand_vel,(abs_vel_hold.shape[0],1)))) for hand_vel in abs_vel_hold.T]).T

    kin_rot = get_pos_vel(env_name=env_name,env_fp=PATH_TO_NORMALIZED_ENV_rotation,net_fp=PATH_TO_PRETRAINED_NET_rotation,config=config_rotation,num_ep=num_ep,render=False, print_cum=True)
    abs_vel_rot = kin_rot['vel']
    norm01_vel_rot = np.array([(hand_vel-np.min(hand_vel))/(np.max(hand_vel)-np.min(hand_vel)) for hand_vel in abs_vel_rot.T]).T
    var1_vel_rot = np.array([np.squeeze(scaler.fit_transform(np.reshape(hand_vel,(abs_vel_rot.shape[0],1)))) for hand_vel in abs_vel_rot.T]).T

    abs_proj = cross_project_kin(vel_hold=abs_vel_hold,vel_rotation=abs_vel_rot,n_comp=n_comp)
    norm01_proj = cross_project_kin(vel_hold=norm01_vel_hold,vel_rotation=norm01_vel_rot,n_comp=n_comp)
    var1_proj = cross_project_kin(vel_hold=var1_vel_hold,vel_rotation=var1_vel_rot,n_comp=n_comp)

    fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(16,10))
    
    R_abs = {'V2/V1 hold to rot':None,'V2/V1 rot to hold':None,'Mean':None}
    R_norm01 = {'V2/V1 hold to rot':None,'V2/V1 rot to hold':None,'Mean':None}
    R_var1 = {'V2/V1 hold to rot':None,'V2/V1 rot to hold':None,'Mean':None}

    cumabs_hold_rot, vabs_hold_rot = exp_var(abs_proj['hold projection'])
    cumabs_rot_hold, vabs_rot_hold = exp_var(abs_proj['rotation projection'])
    R_abs['V2/V1 hold to rot'] = vabs_hold_rot / abs_proj['V1_hold']
    R_abs['V2/V1 rot to hold'] = vabs_rot_hold / abs_proj['V1_rotation']
    R_abs['Mean'] = (R_abs['V2/V1 hold to rot'] + R_abs['V2/V1 rot to hold']) / 2
    axs[0].step(range(1,n_comp+1),cumabs_hold_rot,alpha=0.7,where='mid',label='Hold kinematics projected onto Rotation subspace')
    axs[0].step(range(1,n_comp+1),cumabs_rot_hold,alpha=0.7,where='mid',label='Rotation kinematics projected onto Hold subspace')
    axs[0].set_title('Absolute joint velocities')


    cum01_hold_rot, v01_hold_rot = exp_var(norm01_proj['hold projection'])
    cum01_rot_hold, v01_rot_hold = exp_var(norm01_proj['rotation projection'])
    R_norm01['V2/V1 hold to rot'] = v01_hold_rot / norm01_proj['V1_hold']
    R_norm01['V2/V1 rot to hold'] = v01_rot_hold / norm01_proj['V1_rotation']
    R_norm01['Mean'] = (R_norm01['V2/V1 hold to rot'] + R_norm01['V2/V1 rot to hold']) / 2
    axs[1].step(range(1,n_comp+1),cum01_hold_rot,alpha=0.7,where='mid',label='Hold kinematics projected onto Rotation subspace')
    axs[1].step(range(1,n_comp+1),cum01_rot_hold,alpha=0.7,where='mid',label='Rotation kinematics projected onto Hold subspace')
    axs[1].set_title('[0-1] normalized joint velocities')

    cumv1_hold_rot, vv1_hold_rot = exp_var(var1_proj['hold projection'])
    cumv1_rot_hold, vv1_rot_hold = exp_var(var1_proj['rotation projection'])
    R_var1['V2/V1 hold to rot'] = vv1_hold_rot / var1_proj['V1_hold']
    R_var1['V2/V1 rot to hold'] = vv1_rot_hold / var1_proj['V1_rotation']
    R_var1['Mean'] = (R_var1['V2/V1 hold to rot'] + R_var1['V2/V1 rot to hold']) / 2
    axs[2].step(range(1,n_comp+1),cum01_hold_rot,alpha=0.7,where='mid',label='Hold kinematics projected onto Rotation subspace')
    axs[2].step(range(1,n_comp+1),cum01_rot_hold,alpha=0.7,where='mid',label='Rotation kinematics projected onto Hold subspace')
    axs[2].set_title('Unit variance normalized joint velocities')

    pd.DataFrame([R_abs,R_norm01,R_var1]).to_csv(os.path.join(ROOT_DIR,'SIL-Results/Motor-synergies/Hold-Rotation_v1-v2_Last-model-P1.csv'))

    for j in range(3):
        axs[j].set_xlabel('Number of principal components')
        axs[j].set_ylabel('Percentage of explained variance')
        axs[j].legend(fontsize=9,loc='lower right')
            
    plt.savefig(os.join.path(ROOT_DIR,'SIL-Results/Motor-synergies/Hold-Rotation_proj_Last-model-P1.png'))