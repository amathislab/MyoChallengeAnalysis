from definitions import ROOT_DIR
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from functions import make_parallel_envs, set_config
import pickle
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO
from envs.environment_factory import EnvironmentFactory



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
    render = True
    
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
    pca = PCA(n_components=n_comp).fit(actions)

    for k in range(n_comp):
        print(k)
        components = pca.components_[:n_comp-k]
        performance_ep = []
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
                
                action_proj = np.dot(action.reshape(-1,39)-pca.mean_,components.T)
                action_backproj = np.dot(action_proj,components)+pca.mean_
                obs, rewards, done, info = eval_env.step(action_backproj.reshape(39,))
                episode_starts = done
                cum_reward += rewards
            performance_ep.append(cum_reward)
        performance.append({'components':components,'reward':np.mean(np.array(performance_ep))})

    fp_acts_pcs = open('/home/ingster/Bureau/SIL-BigResults/performance_actions_components_t', 'wb')
    pickle.dump(performance,fp_acts_pcs)
    fp_acts_pcs.close()

    performance_components = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/performance_actions_components_t','rb'))
    perfs = [d['reward'] for d in performance_components]
    comps = [d['components'] for d in performance_components]
    plt.plot([k for k in range(n_comp)],perfs,linewidth=1)
    plt.xlabel('Number of dimensions \nremoved in the action space',fontsize=21,labelpad=10)
    plt.ylabel('Cumulative reward',fontsize=21,labelpad=10)
    plt.yticks(fontsize=21)
    plt.xticks(fontsize=21)
    plt.subplots_adjust(left=0.2,bottom=0.23)
    plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Motor-synergies/Muscle-activations/principal_actions_perf.png'))

