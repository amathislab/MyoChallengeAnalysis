from sklearn.decomposition import PCA
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs.environment_factory import EnvironmentFactory
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import torch
import matplotlib.pyplot as plt

def make_parallel_envs(env_name, env_config, num_env, start_index=0):
    # pylint: disable=redefined-outer-name
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(env_name, **env_config)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


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

def get_layers(config,envir,net,env_name,num_ep,render=False):

    data_layers = []
    envs = make_parallel_envs(env_name, config, num_env=1)
    envs = VecNormalize.load(envir, envs)
    envs.training = False
    envs.norm_reward = False
    custom_objects = {
        "learning_rate": lambda _: 0,
        "lr_schedule": lambda _: 0,
        "clip_range": lambda _: 0,
    }
    model = RecurrentPPO.load(
            net, env=envs, device="cpu", custom_objects=custom_objects
        )
    
    eval_model = model
    eval_env = EnvironmentFactory.create(env_name,**config)

    for i in range(num_ep):
        print('Episode %s'%i)
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

            mlp1_state = model.policy.mlp_extractor.policy_net[0](torch.Tensor(np.squeeze(lstm_states[0]))) 
            mlp2_state = model.policy.mlp_extractor.policy_net[2](model.policy.mlp_extractor.policy_net[1](mlp1_state)) 

            mlp1_state = mlp1_state.detach().numpy()
            mlp2_state = mlp2_state.detach().numpy()

            d = {'config' : config, 'episode' : i, 'timestep': timestep, 'observation': obs,'actions' : action, 'rewards' : rewards, 'LSTM cell state' : np.squeeze(lstm_states[1]),
                'LSTM hidden state':np.squeeze(lstm_states[0]),'Linear layer 1' : mlp1_state, 'Linear layer 2' : mlp2_state}
            data_layers.append(d)

    return data_layers

def measure_tangling(data):
    derivative = np.gradient(data,axis=0)

    epsilon = 0.1*np.mean(np.linalg.norm(data,axis=1))
    Q_all = []
    for t in range(derivative.shape[0]):
        Q = (np.linalg.norm(derivative[t] - derivative,axis=1)**2) / (epsilon + np.linalg.norm(data[t] - data,axis=1)**2)
        Q = np.max(Q)
        Q_all.append(Q)
    
    return Q_all

def PCvsVar(df,n_comp=23):
    pca = PCA(n_components=n_comp)
    pca.fit_transform(np.copy(df))
    return pca.explained_variance_ratio_

def plot_cumvar(n_comp,exp_var_ratio,title):
    plt.clf()
    plt.bar(range(1,n_comp+1), exp_var_ratio, alpha=0.5, align='center',label='Individual variance')
    plt.step(range(1,n_comp+1), np.cumsum(exp_var_ratio), where='mid',label='Cumulative variance')
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
    plt.show()


def cross_project_kin(vel1,vel2,n_comp,n_highpc):
    pca = PCA(n_components=n_comp)
    pca_2 = pca.fit(vel2)
    pca_1 = pca.fit(vel1)
    return {'projection 1 on 2':pca_2.transform(vel1),'projection 2 on 1':pca_1.transform(vel2),'V1_1':np.cumsum(pca_1.explained_variance_ratio_)[n_highpc-1],'V1_2':np.cumsum(pca_2.explained_variance_ratio_)[n_highpc-1]}

def exp_var(X,n_highpc):
    cov_matrix = np.cov(X,rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    total_eigenvalues = sum(eigenvalues)
    var_exp = [(i/total_eigenvalues) for i in sorted(eigenvalues,reverse=True)]
    return np.cumsum(var_exp),np.cumsum(var_exp)[n_highpc-1]

def mean_ratio(n_highpc,cproj):
    cum_1_2, v_1_2 = exp_var(X=cproj['projection 1 on 2'],n_highpc=n_highpc)
    cum_2_1, v_2_1 = exp_var(X=cproj['projection 2 on 1'],n_highpc=n_highpc)
    return [(v_1_2 / cproj['V1_1'] + v_2_1 / cproj['V1_2']) / 2, cum_1_2, cum_2_1]

def plot_cross_projection(n_comp,cum_1_2,cum_2_1,label1,label2):
    plt.clf()
    plt.step(range(1,n_comp+1),cum_1_2,alpha=0.7,where='mid',label=label1)
    plt.step(range(1,n_comp+1),cum_2_1,alpha=0.7,where='mid',label=label2)
    plt.legend(fontsize=21)
    plt.xlabel('Number of PCs',fontsize=21)
    plt.ylabel('Explained variance',fontsize=21)
    plt.yticks(fontsize=21)
    plt.xticks(fontsize=21)
    plt.title('Absolute joint velocities',fontsize=21)
    plt.subplots_adjust(left=0.15,bottom=0.15)
    plt.show()
