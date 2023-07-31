from definitions import ROOT_DIR
import os 
import numpy as np
from functions import make_parallel_envs
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO
from envs.environment_factory import EnvironmentFactory
import matplotlib.pyplot as plt
import pickle

if __name__=='__main__':

    config_yradius = {
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
            0.021,
            0.021
        ],
        "obj_mass_range": [
            0.043,
            0.043
        ],
        "obj_friction_change": [
            0,
            0,
            0
        ],
        "task_choice": "cw"
    }

    env_name = 'CustomMyoBaodingBallsP2' 
    curriculum_step = '32_phase_2_smaller_rate_resume'
    render = False

    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        'trained_models/curriculum_steps_complete_baoding_winner/'+curriculum_step+'/env.pkl',
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        'trained_models/curriculum_steps_complete_baoding_winner/'+curriculum_step+'/model.zip',
    )

    config = config_yradius
    num_episodes = 1

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

    for n in range(num_episodes):
        print(n)
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
                                                    ) # lstm_states is a tuple of length 2 : (h_n, c_n), we are first interested in c_n (we consider h_n as the output), each element being a numpy array of shape (1,1,256)
                                                        
            obs, rewards, done, info = eval_env.step(action)
            episode_starts = done
            cum_reward += rewards
