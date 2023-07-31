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

# Todorov, manipulation task

if __name__=='__main__':
    
    env_name = 'CustomMyoBaodingBallsP2' 
    render = False

    '''config = {
        "weighted_reward_keys": {
        "pos_dist_1": 0,
        "pos_dist_2": 0,
        "act_reg": 0,
        "alive": 0,
        "solved": 5,
        "done": 0,
        "sparse": 0
        }
        
    }

    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        'Last-model-P1/normalized_env_phase1_final',
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        'Last-model-P1/phase1_final.zip',
    )'''

    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/env.pkl",
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/model.zip",
    )

    config = {
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "alive": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0
        },
        "enable_rsi": False,
        "rsi_probability": 0,
        "balls_overlap": False,
        "overlap_probability": 0,
        "noise_fingers": 0,
        "limit_init_angle": 3.141592653589793,
        "goal_time_period": [
            5,
            5
        ],
        "goal_xrange": [
            0.02,
            0.03
        ],
        "goal_yrange": [
            0.022,
            0.032
        ],
        "obj_size_range": [
            0.018,
            0.024
        ],
        "obj_mass_range": [
            0.03,
            0.3
        ],
        "obj_friction_change": [
            0.2,
            0.001,
            2e-05
        ],
        "task_choice": "fixed",
        "rotation_direction" : "cw"
    }

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

    num_ep = 25
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
        print(cum_reward)

        # MEASURE JOINT POSITION AND VELOCITY
        hand_positions = np.array(obs_tot)[:,0:23]    
        hand_velocities = np.array([np.diff(pos)/0.0025 for pos in hand_positions.T]).T
        hand_velocities = np.vstack((np.zeros((1,23)),hand_velocities))
        tot_pos.append(hand_positions)
        tot_vel.append(hand_velocities)
        
    tot_pos = np.concatenate(tot_pos,axis=0)
    tot_vel = np.concatenate(tot_vel,axis=0)

    def PCvsVar(df,title,filename,n_comp=23):
        pca = PCA(n_components=n_comp)
        pca.fit_transform(np.copy(df))
        plt.clf()
        plt.bar(range(1,n_comp+1), pca.explained_variance_ratio_, alpha=0.5, align='center',label='Individual explained variance')
        plt.step(range(1,n_comp+1), np.cumsum(pca.explained_variance_ratio_), where='mid',label='Cumulative explained variance')
        plt.xlabel('Number of principal components',fontsize=13)
        plt.ylabel('Percentage of explained variance',fontsize=13)
        plt.legend(fontsize=13,loc='best')
        plt.title(title,fontsize=13)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.axhline(y=0.95, color='r', linestyle='-')
        plt.axhline(y=0.85, color='g', linestyle='-')
        plt.text(15.5, 0.89, '95% cut-off threshold', color = 'red', fontsize=12)
        plt.text(15.5, 0.79, '85% cut-off threshold', color = 'green', fontsize=12)
        plt.savefig(os.path.join(ROOT_DIR,"SIL-Results/Motor-synergies/DOF-baoding/"+filename+".png"))
        return pca.explained_variance_ratio_

    # Absolute joint angles
    abs_pos_var = PCvsVar(df=tot_pos, title='Absolute joint angles', filename='abs_position')
    print('Abs, position\n',np.cumsum(abs_pos_var))

    # Normalized to range (0,1)
    norm01_hand_positions = np.array([(hand_pos-np.min(hand_pos))/(np.max(hand_pos)-np.min(hand_pos)) for hand_pos in tot_pos.T]).T
    norm01_pos_var = PCvsVar(df=norm01_hand_positions,title='[0,1] normalized joint angles',filename='0-1_position')
    print('0-1 range, position \n',np.cumsum(norm01_pos_var))

    # Normalized to unit variance
    scaler = StandardScaler(with_mean=False)
    norm_hand_positions = np.array([np.squeeze(scaler.fit_transform(np.reshape(hand_pos,(tot_pos.shape[0],1)))) for hand_pos in tot_pos.T]).T
    norm_pos_var = PCvsVar(df=norm_hand_positions,title='Unit variance normalized joint angles',filename='unit-var_position')
    print('Unit variance, position \n',np.cumsum(norm_pos_var))


    # Absolute joint velocities
    abs_vel_var = PCvsVar(df=tot_vel, title='Absolute joint velocities', filename='abs_velocity')
    print('Abs, velocity\n',np.cumsum(abs_vel_var))

    # Normalized to range (0,1)
    norm01_hand_velocities = np.array([(hand_vel-np.min(hand_vel))/(np.max(hand_vel)-np.min(hand_vel)) for hand_vel in tot_vel.T]).T
    norm01_vel_var = PCvsVar(df=norm01_hand_velocities,title='[0,1] normalized joint velocities',filename='0-1_velocity')
    print('0-1 range, velocity\n',np.cumsum(norm01_vel_var))

    # Normalized to unit variance
    scaler = StandardScaler(with_mean=False)
    norm_hand_velocities = np.array([np.squeeze(scaler.fit_transform(np.reshape(hand_vel,(tot_vel.shape[0],1)))) for hand_vel in tot_vel.T]).T
    norm_vel_var = PCvsVar(df=norm_hand_velocities,title='Unit variance normalized joint velocities',filename='unit-var_velocity')
    print('Unit variance, velocity\n',np.cumsum(norm_vel_var))
