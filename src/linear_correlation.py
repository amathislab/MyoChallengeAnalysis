import os
import torch
import numpy as np
import pandas as pd
from definitions import ROOT_DIR
import sklearn.linear_model
from envs.environment_factory import EnvironmentFactory
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from functions import make_parallel_envs
import matplotlib.pyplot as plt
import pickle

if __name__=='__main__':

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
        4,
        6
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
        0.021
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
    "task_choice": "fixed"
    }

    env_name = 'CustomMyoBaodingBallsP2'
    render = False

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

    # EVALUATE
    eval_model = model
    eval_env = EnvironmentFactory.create(env_name, **config)   
    
    num_episodes = 50
    M = []; Ra = []; Fr = []; S = []; VEL = []; ACC = []
    LSTM = []; OBS = []

    for n in range(num_episodes) :
        print(n)
        data_tot = []
        lstm_tot = []; obs_tot = []
        lstm_states = None
        cum_rew = 0
        step = 0
        obs = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        while not done:
            if render:
                eval_env.sim.render(mode="window")

            action, lstm_states = eval_model.predict(
                envs.normalize_obs(obs),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, rewards, done, info = eval_env.step(action)
            episode_starts = done
            cum_rew += rewards
            step += 1

            mlp1_output = model.policy.mlp_extractor.policy_net[0](torch.Tensor(np.squeeze(lstm_states[0]))) # the input of the first linear layer is h_n = lstm_states[0]
            mlp2_output = model.policy.mlp_extractor.policy_net[2](model.policy.mlp_extractor.policy_net[1](mlp1_output)) # the output of the first linear transformation is activated by ReLU() before being processed in the second linear layer 
            
            d = {'episode' : n, 'time step': step, 'observation': obs, 'LSTM output' : np.squeeze(lstm_states[0])}
            data_tot.append(d)
            lstm_tot.append(np.squeeze(lstm_states[0]))
            obs_tot.append(obs)

        hand_positions = np.array([d.get('observation')[0:23] for d in data_tot])
        
        hand_velocities = np.array([np.diff(pos)/0.0025 for pos in hand_positions.T]).T
        hand_velocities = np.vstack((np.zeros((1,23)),hand_velocities))
        
        hand_accelerations = np.array([np.diff(vel)/0.0025 for vel in hand_velocities.T]).T
        hand_accelerations = np.vstack((hand_accelerations,np.zeros((1,23))))

        VEL.append(hand_velocities)
        ACC.append(hand_accelerations)

        OBS.append(np.array(obs_tot))
        LSTM.append(np.array(lstm_tot))

        mass = np.full((step,1),eval_env.sim.model.body_mass[eval_env.object1_bid])
        size = np.full((step,3),eval_env.sim.model.geom_size[eval_env.object1_gid])
        friction = np.full((step,3),eval_env.sim.model.geom_friction[eval_env.object1_gid])
        r = np.array([eval_env.x_radius, eval_env.y_radius])
        radius = np.full((step,2),r)

        M.append(mass)
        Ra.append(radius)
        Fr.append(friction)
        S.append(size)

    M=np.concatenate(M,axis=0)
    Ra=np.concatenate(Ra,axis=0)
    Fr=np.concatenate(Fr,axis=0)
    S=np.concatenate(S,axis=0)
    VEL=np.concatenate(VEL,axis=0)
    ACC=np.concatenate(ACC,axis=0)

    OBS=np.concatenate(OBS,axis=0)
    LSTM=np.concatenate(LSTM,axis=0)

    fp_params = open('/home/ingster/Bureau/SIL-BigResults/lin_corr', 'wb')
    pickle.dump({'Mass':M,'Radius':Ra,'Friction':Fr,'Size':S,'Velocity':VEL,'Acceleration':ACC,'LSTM':LSTM,'Observations':OBS},fp_params)
    fp_params.close()

    params = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/lin_corr','rb'))

    M=params['Mass']
    Ra=params['Radius']
    Fr=params['Friction']
    S=params['Size']
    VEL=params['Velocity']
    ACC=params['Acceleration']
    OBS=params['Observations']
    LSTM=params['LSTM']

    regression = sklearn.linear_model.LinearRegression()

    R_lstm = {'hand velocity':None,'hand acceleration':None,'mass':None,'size':None,'friction':None,'radius':None}
    R_obs = {'hand velocity':None,'hand acceleration':None,'mass':None,'size':None,'friction':None,'radius':None}

    layers = [OBS,LSTM]
    R = [R_obs,R_lstm]

    for i in range(len(R)):

        lin_model = regression.fit(y=VEL,X=layers[i])
        R[i]['hand velocity'] = np.round(lin_model.score(y=VEL,X=layers[i]),5)

        lin_model = regression.fit(y=ACC,X=layers[i])
        R[i]['hand acceleration'] = np.round(lin_model.score(y=ACC,X=layers[i]),5)

        lin_model = regression.fit(y=M,X=layers[i])
        R[i]['mass'] = np.round(lin_model.score(X=layers[i],y=M),5)

        lin_model = regression.fit(y=S,X=layers[i])
        R[i]['size'] = np.round(lin_model.score(y=S,X=layers[i]),5)

        lin_model = regression.fit(y=Fr,X=layers[i])
        R[i]['friction'] = np.round(lin_model.score(y=Fr,X=layers[i]),5)

        lin_model = regression.fit(y=Ra,X=layers[i])
        R[i]['radius'] = np.round(lin_model.score(y=Ra,X=layers[i]),5)

    for R_layer in R : 
        print(R_layer)

    pd.DataFrame(R).to_csv(os.path.join(ROOT_DIR,"SIL-Results/Linear-correlation/v2_32_phase_2_smaller_rate_resume.csv"))