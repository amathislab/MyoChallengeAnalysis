import numpy as np
import torch
import pandas as pd
from envs.environment_factory import EnvironmentFactory
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
#import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_name, env_config, num_env, start_index=0):
    # pylint: disable=redefined-outer-name
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(env_name, **env_config)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

# Mass
def set_configuration_mass(mass):
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
        0.024
    ],
    "obj_mass_range": [
        mass,
        mass
    ],
    "obj_friction_change": [
        0.2,
        0.001,
        2e-05
    ],
    "task_choice": "ccw"
}

# Size
def set_configuration_size(size):
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
        2e-05
    ],
    "task_choice": "ccw"
}

# x radius
def set_configuration_xradius(xradius):
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
        4,
        6
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
        2e-05
    ],
    "task_choice": "ccw"
}

# y radius
def set_configuration_yradius(yradius):
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
        4,
        6
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
        2e-05
    ],
    "task_choice": "ccw"
    }

def get_layers(ball_parameter,envir,net,env_name,num_ep,n_config,render=False):

    global Params

    if ball_parameter == 'mass' :
        Params = np.linspace(0.03,0.3,n_config)
    elif ball_parameter == 'size' :
        Params = np.linspace(0.018,0.024,n_config)
    elif ball_parameter == 'xradius' :
        Params = np.linspace(0.02,0.03,n_config)
    elif ball_parameter == 'yradius' :
        Params == np.linspace(0.022,0.032,n_config)

    data_param = []
    rwd_ep = []

    for p in Params :
        if ball_parameter == 'mass' :
            config = set_configuration_mass(mass=p)
        elif ball_parameter == 'size' :
            config = set_configuration_size(size=p)
        elif ball_parameter == 'xradius' :
            config = set_configuration_xradius(xradius=p)
        elif ball_parameter == 'yradius' :
            config = set_configuration_yradius(yradius=p)   

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
        print(ball_parameter,' ',p)

        for i in range(num_ep):
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

                mlp1_state = model.policy.mlp_extractor.policy_net[0](torch.Tensor(np.squeeze(lstm_states[0]))) # the input of the first linear layer is h_n = lstm_states[0]
                mlp2_state = model.policy.mlp_extractor.policy_net[2](model.policy.mlp_extractor.policy_net[1](mlp1_state)) # the output of the first linear transformation is activated by ReLU() before being processed in the second linear layer 

                mlp1_state = mlp1_state.detach().numpy()
                mlp2_state = mlp2_state.detach().numpy()

                d = {ball_parameter : p, 'episode' : i, 'timestep': timestep, 'last control': eval_env.last_ctrl, 'observation': obs,'actions' : action, 'rewards' : rewards, 'LSTM cell state' : np.squeeze(lstm_states[1]),
                    'LSTM hidden state':np.squeeze(lstm_states[0]),'Linear layer 1' : mlp1_state, 'Linear layer 2' : mlp2_state}
                data_param.append(d)

            d_rwd = {ball_parameter : p, 'episode' : i, 'episode reward' : cum_reward}
            rwd_ep.append(d_rwd) 
                
                # shape of np.squeeze(lstm_states[i]) is (256,) instead of (1,1,256)
                # shape of mlp_state is [256] using np.squeeze instead of [1,1,256]
        
    return [data_param, rwd_ep]

def UMAP_physics(ball_parameter,df,fp,n_config,curriculum_step) :
    umap_apply = umap.UMAP(n_neighbors=30,min_dist=0.01)
    targets = [d.get(ball_parameter) for d in df]

    # LSTM

    lstm_mass = [d.get('LSTM cell state',None) for d in df] # list of length n_environments*n_episodes*n_timesteps of numpy arrays of shape (256,)
    transformed_lstm_mass = umap_apply.fit_transform(lstm_mass) # shape 2 *[number of environments*number of episodes*number of timesteps]

    plt.scatter(transformed_lstm_mass[:,0],transformed_lstm_mass[:,1],c=targets,s=1,alpha=0.4)
    plt.title('UMAP embedding of LSTM states, n_neighbors=30, min_dist=0.01')
    plt.savefig(fp+curriculum_step+'_LSTM_%s-config.png' %n_config)

    # MLP1

    mlp1_mass = [d.get('Linear layer 1',None) for d in df] 
    transformed_mlp1_mass = umap_apply.fit_transform(mlp1_mass)

    plt.clf()
    plt.scatter(transformed_mlp1_mass[:,0],transformed_mlp1_mass[:,1],c=targets,s=1,alpha=0.4)
    plt.title('UMAP embedding of MLP1 states, n_neighbors=30, min_dist=0.01')
    plt.savefig(fp+curriculum_step+'_MLP1_%s-config.png' %n_config)

    # MLP2

    mlp2_mass = [d.get('Linear layer 2',None) for d in df] 
    transformed_mlp2_mass = umap_apply.fit_transform(mlp2_mass)

    plt.clf()
    plt.scatter(transformed_mlp2_mass[:,0],transformed_mlp2_mass[:,1],c=targets,s=1,alpha=0.4)
    plt.title('UMAP embedding of MLP2 states, n_neighbors=30, min_dist=0.01')
    plt.savefig(fp+curriculum_step+'_MLP2_%s-config.png' %n_config)

    # last control (normalized actions)

    activations = [d.get('last control') for d in df]
    transformed_activations = umap_apply.fit_transform(activations)

    plt.clf()
    plt.scatter(transformed_activations[:,0],transformed_activations[:,1],c=targets,s=1,alpha=0.4)
    plt.title('UMAP embedding of activations (last control), n_neighbors=30, min_dist=0.01')
    plt.savefig(fp+curriculum_step+'_activations_%s-config.png' %n_config)

scaler = StandardScaler()

def pca_apply(df,n):
    pc = PCA(n_components=n)
    scaled_df = scaler.fit_transform(np.array(df))
    return pc.fit_transform(scaled_df)

def pca_fit(df,n):
    pc = PCA(n_components=n)
    scaled_df = scaler.fit_transform(np.array(df))
    return pc.fit(scaled_df)

def pc_number_var(df,n,cum=False) :
    pca = PCA(n_components=n)
    pca.fit_transform(scaler.fit_transform(np.array(df)))
    fig = plt.figure()
    
    if cum :
        plt.plot(range(1,n+1),pca.explained_variance_ratio_.cumsum())
    else :
        plt.plot(range(1,n+1),pca.explained_variance_ratio_)

    plt.xlabel('Number of principal components')
    plt.ylabel('Percentage of explained variance')

    return fig

def PCA_physics(ball_parameter,df,fp,n_config,curriculum_step,n_comp=3,n_var=90):
    
    targets = [d.get(ball_parameter) for d in df] # 1 color per mass

    # LSTM

    lstm = [d.get('LSTM cell state',None) for d in df] # list of length n_environments*n_episodes*n_timesteps of numpy arrays of shape (256,)
    transformed_lstm = pca_apply(lstm,n_comp) 

    pca_lstm_df = pd.DataFrame(data=transformed_lstm, columns=['PC1', 'PC2','PC3'])

    fig1_lstm = px.scatter_3d(pca_lstm_df,x='PC1',y='PC2',z='PC3',color=targets,opacity=0.4)
    fig1_lstm.update_traces(marker_size = 3)
    fig1_lstm.write_html(fp+curriculum_step+'_LSTM_%s-config.html' %n_config)

    pc_number_var(lstm,n_var,cum=True).savefig(fp+curriculum_step+'_expvar_LSTM_%s-config.png' %n_config)

    # MLP1

    mlp1 = [d.get('Linear layer 1',None) for d in df]
    transformed_mlp1 = pca_apply(mlp1,n=n_comp) # converts the list into a numpy array of size n_samples*n_features(256)

    pca_mlp1_df = pd.DataFrame(data=transformed_mlp1, columns=['PC1', 'PC2','PC3'])
    
    fig1_mlp1 = px.scatter_3d(pca_mlp1_df,x='PC1',y='PC2',z='PC3',color=targets,opacity=0.4)
    fig1_mlp1.update_traces(marker_size = 3)
    fig1_mlp1.write_html(fp+curriculum_step+'_MLP1_%s-config.html' %n_config)

    pc_number_var(mlp1,n_var,cum=True).savefig(fp+curriculum_step+'_expvar_MLP1_%s-config.png' %n_config)

    # MLP2

    mlp2 = [d.get('Linear layer 2',None) for d in df]
    transformed_mlp2 = pca_apply(mlp2,n=n_comp) # converts the list into a numpy array of size n_samples*n_features(256)

    pca_mlp2_df = pd.DataFrame(data=transformed_mlp2, columns=['PC1', 'PC2','PC3'])

    fig1_mlp2 = px.scatter_3d(pca_mlp2_df,x='PC1',y='PC2',z='PC3',color=targets,opacity=0.4)
    fig1_mlp2.update_traces(marker_size = 3)
    fig1_mlp2.write_html(fp+curriculum_step+'_MLP2_%s-config.html' %n_config)

    pc_number_var(mlp2,n_var,cum=True).savefig(fp+curriculum_step+'_expvar_MLP2_%s-config.png' %n_config)

    # Activations (last control)

    activations = [d.get('last control') for d in df]
    transformed_activations_pca = pca_apply(activations,n=n_comp)
    pca_df = pd.DataFrame(data=transformed_activations_pca, columns=['PC1', 'PC2','PC3'])
    fig = px.scatter_3d(pca_df,x='PC1',y='PC2',z='PC3',color=targets,opacity=0.4)
    fig.update_traces(marker_size = 3)
    fig.write_html(fp+curriculum_step+'_activations_%s-config.html' %n_config)

    pc_number_var(df=activations,n=39,cum=True).savefig(fp+curriculum_step+'_expvar_activations_%s-config.png' %n_config)

def set_configuration_xradius_P1():
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
            0.02,
            0.03
        ],
        "goal_yrange": [
            0.028,
            0.028
        ],
        "drop_th": 1.3
    }

def set_configuration_yradius_P1():
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

def get_layers_P1(ball_parameter,envir,net,env_name,n_config,render=False):

    data_param = []
    rwd_ep = []

    if ball_parameter == 'xradius' :
        config = set_configuration_xradius_P1()
    elif ball_parameter == 'yradius' :
        config = set_configuration_yradius_P1()   

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

    for i in range(n_config):
        cum_reward = 0
        lstm_states = None
        obs = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        timestep = 0
        global p
        if ball_parameter == 'xradius' :
            p = eval_env.x_radius
        elif ball_parameter == 'yradius' :
            p = eval_env.y_radius
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

            mlp1_state = model.policy.mlp_extractor.policy_net[0](torch.Tensor(np.squeeze(lstm_states[0]))) # the input of the first linear layer is h_n = lstm_states[0]
            mlp2_state = model.policy.mlp_extractor.policy_net[2](model.policy.mlp_extractor.policy_net[1](mlp1_state)) # the output of the first linear transformation is activated by ReLU() before being processed in the second linear layer 

            mlp1_state = mlp1_state.detach().numpy()
            mlp2_state = mlp2_state.detach().numpy()

            d = {ball_parameter : p, 'episode' : i, 'timestep': timestep, 'last control': eval_env.last_ctrl, 'observation': obs,'actions' : action, 'rewards' : rewards, 'LSTM cell state' : np.squeeze(lstm_states[1]),
                'LSTM hidden state':np.squeeze(lstm_states[0]),'Linear layer 1' : mlp1_state, 'Linear layer 2' : mlp2_state}
            data_param.append(d)

        d_rwd = {ball_parameter : p, 'configuration reward' : cum_reward}
        rwd_ep.append(d_rwd)
            
            # shape of np.squeeze(lstm_states[i]) is (256,) instead of (1,1,256)
            # shape of mlp_state is [256] using np.squeeze instead of [1,1,256]
        
    return [data_param, rwd_ep]
