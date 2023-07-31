from definitions import ROOT_DIR
import os 
import numpy as np
from functions import make_parallel_envs
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO
from envs.environment_factory import EnvironmentFactory
import matplotlib.pyplot as plt


# P2 model, analysis of robustness to changes in phyisical parameters

if __name__=='__main__':

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

    num_config = 40
    num_episodes = 100
    s_min = 0.015; s_max = 0.027; m_min = 0.01; m_max = 0.4; x_min = 0.01; x_max = 0.04; y_min = 0.012; y_max = 0.042

    # Size

    rwd_mean = []
    rwd_std = []
    for size in np.linspace(s_min,s_max,num_config):
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
            size,
            size
        ],
        "obj_mass_range": [
            0.03,
            0.3
        ],
        "obj_friction_change": [
            0.2,
            0.001,
            2e-5
        ],
        "task_choice": "ccw"
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
        rwd_ep = []

        for n in range(num_episodes):
            print('size : ',size,' ', n)
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
            rwd_ep.append(cum_reward)
        rwd_mean.append(np.mean(np.array(rwd_ep)))
        rwd_std.append(np.std(np.array(rwd_ep)))
        
    plt.clf()
    plt.errorbar(np.linspace(s_min,s_max,num_config), rwd_mean, rwd_std,linestyle='None', marker='o',markersize=3,linewidth=0.5)
    plt.xlabel('Ball size'); plt.ylabel('Cumulative reward')
    plt.title('Performance vs. ball size, last model phase 2')

    plt.savefig(os.path.join(ROOT_DIR,"SIL-Results/Perf-vs-Physics/Size/"+curriculum_step+"_%s-configs_largerange_" %num_config+"_%s-eps"%num_episodes+".png"))

    # Mass

    rwd_mean = []
    rwd_std = []
    for mass in np.linspace(m_min,m_max,num_config):
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
            mass,
            mass
        ],
        "obj_friction_change": [
            0.2,
            0.001,
            2e-5
        ],
        "task_choice": "ccw"
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
        rwd_ep = []

        for n in range(num_episodes):
            print('mass : ',mass,' ', n)
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
            rwd_ep.append(cum_reward)
        rwd_mean.append(np.mean(np.array(rwd_ep)))
        rwd_std.append(np.std(np.array(rwd_ep)))
        
    plt.clf()
    plt.errorbar(np.linspace(m_min,m_max,num_config), rwd_mean, rwd_std,linestyle='None', marker='o',markersize=3,linewidth=0.5)
    plt.xlabel('Ball mass'); plt.ylabel('Cumulative reward')
    plt.title('Performance vs. ball mass, last model phase 2')

    plt.savefig(os.path.join(ROOT_DIR,"SIL-Results/Perf-vs-Physics/Mass/"+curriculum_step+"_%s-configs_largerange_" %num_config+"_%s-eps"%num_episodes+".png"))
    

    # x radius

    num_episodes = 25
    rwd_mean = []
    rwd_std = []
    for xradius in np.linspace(x_min,x_max,num_config):
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
            xradius,
            xradius
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
            2e-5
        ],
        "task_choice": "ccw"
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
        rwd_ep = []

        for n in range(num_episodes):
            print('xradius : ',xradius,' ', n)
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
            rwd_ep.append(cum_reward)
        rwd_mean.append(np.mean(np.array(rwd_ep)))
        rwd_std.append(np.std(np.array(rwd_ep)))
        
    plt.clf()
    plt.errorbar(np.linspace(x_min,x_max,num_config), rwd_mean, rwd_std,linestyle='None', marker='o',markersize=3,linewidth=0.5)
    plt.xlabel('x radius'); plt.ylabel('Cumulative reward')
    plt.title('Performance vs. x trajectory radius, last model phase 2')

    plt.savefig(os.path.join(ROOT_DIR,"SIL-Results/Perf-vs-Physics/Xradius/"+curriculum_step+"_%s-configs_largerange_" %num_config+"_%s-eps"%num_episodes+".png"))

    # y radius
    
    num_episodes = 25
    rwd_mean = []
    rwd_std = []
    for yradius in np.linspace(y_min,y_max,num_config):
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
            yradius,
            yradius
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
            2e-5
        ],
        "task_choice": "ccw"
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
        rwd_ep = []

        for n in range(num_episodes):
            print('yradius : ',yradius,' ', n)
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
            rwd_ep.append(cum_reward)
        rwd_mean.append(np.mean(np.array(rwd_ep)))
        rwd_std.append(np.std(np.array(rwd_ep)))

    plt.clf()
    plt.errorbar(np.linspace(y_min,y_max,num_config), rwd_mean, rwd_std,linestyle='None', marker='o',markersize=3,linewidth=0.5)
    plt.xlabel('y radius'); plt.ylabel('Cumulative reward')
    plt.title('Performance vs. y trajectory radius, last model phase 2')

    plt.savefig(os.path.join(ROOT_DIR,"SIL-Results/Perf-vs-Physics/Yradius/"+curriculum_step+"_%s-configs_largerange_" %num_config+"_%s-eps"%num_episodes+".png"))
