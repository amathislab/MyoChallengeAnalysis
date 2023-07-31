from definitions import ROOT_DIR
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from functions import make_parallel_envs
import pickle
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO
from envs.environment_factory import EnvironmentFactory

def set_config(period,rot_dir):
    return {
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
            period,
            period
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
        "rotation_direction" : rot_dir
    }

if __name__=="__main__":

    num_ep = 20
    n_comp = 39

    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/env.pkl",
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/model.zip",
    )

    env_name = "CustomMyoBaodingBallsP2"
    render = False
    
    config = set_config(period=5,rot_dir="cw")
    rollouts = []

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

    for n in range(num_ep):
        acts = []
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
            acts.append(action)
        print('episode %s : '%n,cum_reward)
        rollouts.append({'reward':cum_reward,'action':np.array(acts)})
        
    fp_rollouts = open('/home/ingster/Bureau/SIL-BigResults/rollouts', 'wb')
    pickle.dump(rollouts,fp_rollouts)
    fp_rollouts.close()

    rollouts = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/rollouts','rb'))

    actions = np.concatenate([rollout['action'] for rollout in rollouts])
    performance = []

    for k in range(n_comp):
        performance_ep = []
        pca = PCA(n_components=n_comp-k).fit(actions)
        for n in range(num_ep):
            acts = []
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
                
                action_proj = pca.inverse_transform(pca.transform(action.reshape(-1,39)))
                obs, rewards, done, info = eval_env.step(action_proj.reshape(39,))
                episode_starts = done
                cum_reward += rewards
            performance_ep.append(cum_reward)
            print(cum_reward)
        performance.append({'components':pca.components_,'reward':np.mean(np.array(performance_ep))})

    fp_acts_pcs = open('/home/ingster/Bureau/SIL-BigResults/performance_actions_components', 'wb')
    pickle.dump(performance,fp_acts_pcs)
    fp_acts_pcs.close()

    performance_components = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/performance_actions_components','rb'))
    perfs = [d['reward'] for d in performance_components]
    comps = [d['components'] for d in performance_components]
    plt.plot([k for k in range(n_comp)],perfs,linewidth=0.8,color='black')
    plt.xlabel('Dimensions removed in the action space')
    plt.ylabel('Average cumulative reward over 20 episodes')
    plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Motor-synergies/Muscle-activations/principal_actions_perf.png'))

