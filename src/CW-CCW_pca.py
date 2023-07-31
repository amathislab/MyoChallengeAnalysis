from definitions import ROOT_DIR
import os
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO
from envs.environment_factory import EnvironmentFactory
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from functions import make_parallel_envs
import pandas as pd

# Cross-projection similarity analysis of CW and CCW kinematic subspaces

if __name__=='__main__':

    env_name = 'CustomMyoBaodingBallsP1'
    render = False

    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        'Last-model-P1/normalized_env_phase1_final',
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        'Last-model-P1/phase1_final.zip',
    )
    
    def set_configs(task):
        return [{
                "weighted_reward_keys": {
                    "pos_dist_1": 0,
                    "pos_dist_2": 0,
                    "act_reg": 0,
                    "alive": 0,
                    "solved": 5,
                    "done": 0,
                    "sparse": 0
                },
                "task": task,
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
                    x,
                    x
                ],
                "goal_yrange": [
                    0.028,
                    0.028
                ],
                "drop_th": 1.3
            } for x in np.linspace(0.022,0.027,12)]+[{
                "weighted_reward_keys": {
                    "pos_dist_1": 0,
                    "pos_dist_2": 0,
                    "act_reg": 0,
                    "alive": 0,
                    "solved": 5,
                    "done": 0,
                    "sparse": 0
                },
                "task": task,
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
                    y,
                    y
                ],
                "drop_th": 1.3
            } for y in np.linspace(0.022,0.027,12)]

    def get_pos_vel(configs,num_ep,render=False, print_cum=True):
        for config in configs :
            envs = make_parallel_envs(env_name, config, num_env=1)
            envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)
            envs.training = False
            envs.norm_reward = False
            custom_objects = {
                "learning_rate": lambda _: 0,
                "lr_schedule": lambda _: 0,
                "clip_range": lambda _: 0,
            }
            model = RecurrentPPO.load(
                    PATH_TO_PRETRAINED_NET, env=envs, device="cpu", custom_objects=custom_objects
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
                print(eval_env.which_task)
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
    
    # Project CCW on CW subspace and vice versa
    def cross_project_kin(vel_ccw,vel_cw,n_comp,n=10):
        pca = PCA(n_components=n_comp)
        pca_cw = pca.fit(vel_cw)
        pca_ccw = pca.fit(vel_ccw)
        return {'ccw projection':pca_cw.transform(vel_ccw),'cw projection':pca_ccw.transform(vel_cw),'V1_cw':np.cumsum(pca_cw.explained_variance_ratio_)[n-1],'V1_ccw':np.cumsum(pca_ccw.explained_variance_ratio_)[n-1]}

    # Compute explained variance without using sklearn
    def exp_var(X,n=10):
        cov_matrix = np.cov(X,rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        total_eigenvalues = sum(eigenvalues)
        var_exp = [(i/total_eigenvalues) for i in sorted(eigenvalues,reverse=True)]
        return np.cumsum(var_exp),np.cumsum(var_exp)[n-1]
    

    num_ep = 1
    n_comp = 23
    scaler = StandardScaler(with_mean=False)

    # CW and CCW kinematics generation

    kin_cw = get_pos_vel(configs=set_configs(task='cw'),num_ep=num_ep,render=False,print_cum=True)
    abs_vel_cw = kin_cw['vel']
    norm01_vel_cw = np.array([(hand_vel-np.min(hand_vel))/(np.max(hand_vel)-np.min(hand_vel)) for hand_vel in abs_vel_cw.T]).T
    normvar_vel_cw = np.array([np.squeeze(scaler.fit_transform(np.reshape(hand_vel,(abs_vel_cw.shape[0],1)))) for hand_vel in abs_vel_cw.T]).T

    kin_ccw = get_pos_vel(configs=set_configs(task='ccw'),num_ep=num_ep,render=False,print_cum=True)
    abs_vel_ccw = kin_ccw['vel']
    norm01_vel_ccw = np.array([(hand_vel-np.min(hand_vel))/(np.max(hand_vel)-np.min(hand_vel)) for hand_vel in abs_vel_ccw.T]).T
    normvar_vel_ccw = np.array([np.squeeze(scaler.fit_transform(np.reshape(hand_vel,(abs_vel_ccw.shape[0],1)))) for hand_vel in abs_vel_ccw.T]).T

    abs_proj = cross_project_kin(vel_ccw=abs_vel_ccw,vel_cw=abs_vel_cw,n_comp=n_comp)
    norm01_proj = cross_project_kin(vel_ccw=norm01_vel_ccw,vel_cw=norm01_vel_cw,n_comp=n_comp)
    normvar_proj = cross_project_kin(vel_ccw=normvar_vel_ccw,vel_cw=normvar_vel_cw,n_comp=n_comp)

    fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(16,10))
    
    R_abs = {'V2/V1 ccw to cw':None,'V2/V1 cw to ccw':None,'Mean':None}
    R_norm01 = {'V2/V1 ccw to cw':None,'V2/V1 cw to ccw':None,'Mean':None}
    R_var1 = {'V2/V1 ccw to cw':None,'V2/V1 cw to ccw':None,'Mean':None}

    cumabs_ccw_cw, vabs_ccw_cw = exp_var(abs_proj['ccw projection'])
    cumabs_cw_ccw, vabs_cw_ccw = exp_var(abs_proj['cw projection'])
    R_abs['V2/V1 ccw to cw'] = vabs_ccw_cw / abs_proj['V1_ccw']
    R_abs['V2/V1 cw to ccw'] = vabs_cw_ccw / abs_proj['V1_cw']
    R_abs['Mean'] = (R_abs['V2/V1 ccw to cw'] + R_abs['V2/V1 cw to ccw']) / 2
    axs[0].step(range(1,n_comp+1),cumabs_ccw_cw,alpha=0.7,where='mid',label='CCW kinematics projected onto CW subspace')
    axs[0].step(range(1,n_comp+1),cumabs_cw_ccw,alpha=0.7,where='mid',label='CW kinematics projected onto CCW subspace')
    axs[0].set_title('Absolute joint velocities')

    cum01_ccw_cw, v01_ccw_cw = exp_var(norm01_proj['ccw projection'])
    cum01_cw_ccw, v01_cw_ccw = exp_var(norm01_proj['cw projection'])
    R_norm01['V2/V1 ccw to cw'] = v01_ccw_cw / norm01_proj['V1_ccw']
    R_norm01['V2/V1 cw to ccw'] = v01_cw_ccw / norm01_proj['V1_cw']
    R_norm01['Mean'] = (R_norm01['V2/V1 ccw to cw'] + R_norm01['V2/V1 cw to ccw']) / 2
    axs[1].step(range(1,n_comp+1),cum01_ccw_cw,alpha=0.7,where='mid',label='CCW kinematics projected onto CW subspace')
    axs[1].step(range(1,n_comp+1),cum01_cw_ccw,alpha=0.7,where='mid',label='CW kinematics projected onto CCW subspace')
    axs[1].set_title('[0-1] normalized joint velocities')

    cumv1_ccw_cw, vv1_ccw_cw = exp_var(normvar_proj['ccw projection'])
    cumv1_cw_ccw, vv1_cw_ccw = exp_var(normvar_proj['cw projection'])
    R_var1['V2/V1 ccw to cw'] = vv1_ccw_cw / normvar_proj['V1_ccw']
    R_var1['V2/V1 cw to ccw'] = vv1_cw_ccw / normvar_proj['V1_cw']
    R_var1['Mean'] = (R_var1['V2/V1 ccw to cw'] + R_var1['V2/V1 cw to ccw']) / 2
    axs[2].step(range(1,n_comp+1),cumv1_ccw_cw,alpha=0.7,where='mid',label='CCW kinematics projected onto CW subspace')
    axs[2].step(range(1,n_comp+1),cumv1_cw_ccw,alpha=0.7,where='mid',label='CW kinematics projected onto CCW subspace')
    axs[2].set_title('Unit variance normalized joint velocities')

    pd.DataFrame([R_abs,R_norm01,R_var1]).to_csv('/home/ingster/Bureau/SIL-myochallenge/SIL-Results/Motor-synergies/CCW-CW_v1-v2_Last-model-P1.csv')

    for j in range(3):
        axs[j].set_xlabel('Number of principal components')
        axs[j].set_ylabel('Percentage of explained variance')
        axs[j].legend(fontsize=9,loc='best')
            
    plt.savefig(os.path.jooin(ROOT_DIR,'SIL-Results/Motor-synergies/CCW-CW_proj_Last-model-P1.png'))