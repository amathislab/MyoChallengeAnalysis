from definitions import ROOT_DIR
import os 
import numpy as np
from functions import make_parallel_envs
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO
from envs.environment_factory import EnvironmentFactory
import matplotlib.pyplot as plt

# P1 model, analysis of robustness to changes in phyisical parameters

if __name__=='__main__':
    
    # X radius
    def takeParameter(dict):
        return dict['xradius']
    
    env_name = 'CustomMyoBaodingBallsP1' 
    render = False

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
            0.01,
            0.04
        ],
        "goal_yrange": [
            0.028,
            0.028
        ],
        "drop_th": 1.3
    }

    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        'Last-model-P1/normalized_env_phase1_final',
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        'Last-model-P1/phase1_final.zip',
    )

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

    num_ep = 300
    rwd_conf = []

    for n in range(num_ep):
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
                                                    ) 
                                                        
            obs, rewards, done, info = eval_env.step(action)
            episode_starts = done
            cum_reward += rewards
        rwd_conf.append({'xradius': eval_env.x_radius,'reward':cum_reward})
    
    rwd_conf.sort(key=takeParameter)
    params = [d.get('xradius') for d in rwd_conf]
    rwds = [d.get('reward') for d in rwd_conf]
    plt.plot(params,rwds,linewidth=0.6)
    
    window_size = 20
    i=0
    moving_averages = []
    while i < len(rwds) - window_size + 1 :
        moving_averages.append(np.sum(rwds[i:i+window_size])/window_size)
        i += 1

    plt.plot(params[0:len(rwds)-window_size+1],moving_averages,linewidth = 1.2)
    plt.xlabel('x trajectory radius'); plt.ylabel('Cumulative reward')
    plt.title('Performance vs. x trajectory radius, last model phase 1')
    plt.savefig(os.path.join(ROOT_DIR,"SIL-Results/Perf-vs-Physics/Xradius/Last-model-P1_%s-configs_"%num_ep+"window-size-%s"%window_size+".png"))

    # Y radius
    def takeParameter(dict):
        return dict['yradius']

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
            0.022,
            0.032
        ],
        "drop_th": 1.3
    }

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

    rwd_conf = []
    num_ep = 300

    for n in range(num_ep):
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
                                                    ) 
                                                        
            obs, rewards, done, info = eval_env.step(action)
            episode_starts = done
            cum_reward += rewards
        rwd_conf.append({'yradius': eval_env.y_radius,'reward':cum_reward})
    
    rwd_conf.sort(key=takeParameter)
    params = [d.get('yradius') for d in rwd_conf]
    rwds = [d.get('reward') for d in rwd_conf]
    plt.clf()
    plt.plot(params,rwds,linewidth=0.6)
    
    window_size = 20
    i=0
    moving_averages = []
    while i < len(rwds) - window_size + 1 :
        moving_averages.append(np.sum(rwds[i:i+window_size])/window_size)
        i += 1

    plt.plot(params[0:len(rwds)-window_size+1],moving_averages,linewidth = 1.2)
    plt.xlabel('y trajectory radius'); plt.ylabel('Cumulative reward')
    plt.title('Performance vs. y trajectory radius, last model phase 1')
    plt.savefig(os.path.join(ROOT_DIR,"SIL-Results/Perf-vs-Physics/Yradius/Last-model-P1_%s-configs_"%num_ep+"window-size-%s"%window_size+".png"))
    
    # Time period
    def takeParameter(dict):
        return dict['time period']

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
        "task": "ccw",
        "enable_rsi": False,
        "rsi_probability": 0,
        "noise_palm": 0,
        "noise_fingers": 0,
        "noise_balls": 0,
        "goal_time_period": [
            5,
            30
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

    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        'Last-model-P1/normalized_env_phase1_final',
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        'Last-model-P1/phase1_final.zip',
    )

    env_name = 'CustomMyoBaodingBallsP1' 
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
    
    eval_model = model
    eval_env = EnvironmentFactory.create(env_name,**config)

    rwd_conf = []
    num_ep = 200

    for n in range(num_ep):
        print(n)
        cum_reward = 0
        lstm_states = None
        reset_env = eval_env.reset(return_time_period=True)
        obs = reset_env[0]
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
        rwd_conf.append({'time period': reset_env[1],'reward':cum_reward})

    rwd_conf.sort(key=takeParameter)
    params = [d.get('time period') for d in rwd_conf]
    rwds = [d.get('reward') for d in rwd_conf]
    plt.clf()
    plt.plot(params,rwds,linewidth=0.6)
    
    window_size = 10
    i=0
    moving_averages = []
    while i < len(rwds) - window_size + 1 :
        moving_averages.append(np.sum(rwds[i:i+window_size])/window_size)
        i += 1

    plt.plot(params[0:len(rwds)-window_size+1],moving_averages,linewidth = 1.2)
    plt.xlabel('Goal time period'); plt.ylabel('Cumulative reward')
    plt.title('Performance vs. rotation period, last model phase 1')
    plt.savefig(os.path.join(ROOT_DIR,"SIL-Results/Perf-vs-Physics/Period/Last-model-P1_%s-configs_"%num_ep+"window-size-%s"%window_size+".png"))

