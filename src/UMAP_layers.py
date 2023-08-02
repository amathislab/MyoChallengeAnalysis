from functions import make_parallel_envs,VecNormalize,RecurrentPPO,EnvironmentFactory
import numpy as np
import torch
from definitions import ROOT_DIR
import os
import pickle
import umap
import plotly.express as px 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


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

def get_eps(data,num_ep):
    return [[d for d in data if data['episode']==i] for i in range(num_ep)]


if __name__=="__main__":
    
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

    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/env.pkl",
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/model.zip",
    )

    env_name = "CustomMyoBaodingBallsP2"
    num_ep = 3

    '''layers = get_layers(configuration=config,envir=PATH_TO_NORMALIZED_ENV,env_name=env_name,net=PATH_TO_PRETRAINED_NET,num_ep=num_ep)

    fp_rollouts = open('/home/ingster/Bureau/SIL-BigResults/layers_%seps'%num_ep, 'wb')
    pickle.dump(layers,fp_rollouts)
    fp_rollouts.close()'''

    layers = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/layers_%seps'%num_ep,'rb'))

    n_comp = 3
    umap_apply = umap.UMAP(n_components=n_comp,random_state=42)

    obs = np.array([d['observation'] for d in layers])
    obs_trans = umap_apply.fit_transform(obs)
    Q_obs = (np.max(measure_tangling(obs_trans[0:200]))+np.max(measure_tangling(obs_trans[200:400]))+np.max(measure_tangling(obs_trans[400:600])))/3
    Q_obs = np.round(Q_obs,2)

    '''lstm = np.array([d['LSTM hidden state'] for d in layers])
    lstm_trans = umap_apply.fit_transform(lstm)
    Q_lstm = (np.max(measure_tangling(lstm_trans[0:200]))+np.max(measure_tangling(lstm_trans[200:400]))+np.max(measure_tangling(lstm_trans[400:600])))/3

    l1 = np.array([d['Linear layer 1'] for d in layers])
    l1_trans = umap_apply.fit_transform(l1)
    Q_l1 = (np.max(measure_tangling(l1_trans[0:200]))+np.max(measure_tangling(l1_trans[200:400]))+np.max(measure_tangling(l1_trans[400:600])))/3

    l2 = np.array([d['Linear layer 2'] for d in layers])
    l2_trans = umap_apply.fit_transform(l2)
    Q_l2 = (np.max(measure_tangling(l2_trans[0:200]))+np.max(measure_tangling(l2_trans[200:400]))+np.max(measure_tangling(l2_trans[400:600])))/3

    acts = np.array([d['actions'] for d in layers])
    acts_trans = umap_apply.fit_transform(acts)
    Q_acts = (np.max(measure_tangling(acts_trans[0:200]))+np.max(measure_tangling(acts_trans[200:400]))+np.max(measure_tangling(acts_trans[400:600])))/3'''

    cmap = mpl.colormaps['Set1']
    colors = cmap(np.linspace(0,0.5,num_ep))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    projected_obs = [obs_trans[0:200],obs_trans[200:400],obs_trans[400:600]]
    for i in range(num_ep):
        obs_trans = projected_obs[i]
        ax.plot(obs_trans[:,0],obs_trans[:,1],obs_trans[:,2],color=colors[i],linewidth=1,label='Episode %s'%i)
    ax.view_init(elev=30.,azim=45)
    plt.legend(fontsize=12,loc='best')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Proj. 1',fontsize=12)
    plt.ylabel('Proj. 2',fontsize=12)
    plt.title('Observation space',fontsize=16)
    ax.set_zlabel('Proj. 3',fontsize=12)
    ax.text(x=14, y=2, z=13, s='Q = %s'%Q_obs, color='black', 
        bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round',pad=0.5),fontsize=16)
    plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/UMAP_layers/obs.png'))

    targets = [d['episode'] for d in layers]
    cols = ['Proj. 1','Proj. 2','Proj. 3']

    '''obs_df = pd.DataFrame(data=obs_trans,columns=cols)
    lstm_df = pd.DataFrame(data=lstm_trans,columns=cols)
    l1_df = pd.DataFrame(data=l1_trans,columns=cols)
    l2_df = pd.DataFrame(data=l2_trans,columns=cols)
    acts_df = pd.DataFrame(data=acts_trans,columns=cols)

    fig1 = px.line_3d(lstm_df,x=cols[0],y=cols[1],z=cols[2],color=targets)
    fig1.write_html(os.path.join(ROOT_DIR,'SIL-Results/UMAP_layers/lstm.html'))

    fig2 = px.line_3d(obs_df,x=cols[0],y=cols[1],z=cols[2],color=targets)
    fig2.write_html(os.path.join(ROOT_DIR,'SIL-Results/UMAP_layers/obs.html'))

    fig3 = px.line_3d(l1_df,x=cols[0],y=cols[1],z=cols[2],color=targets)
    fig3.write_html(os.path.join(ROOT_DIR,'SIL-Results/UMAP_layers/l1.html'))

    fig4 = px.line_3d(l2_df,x=cols[0],y=cols[1],z=cols[2],color=targets)
    fig4.write_html(os.path.join(ROOT_DIR,'SIL-Results/UMAP_layers/l2.html'))

    fig5 = px.line_3d(acts_df,x=cols[0],y=cols[1],z=cols[2],color=targets)
    fig5.write_html(os.path.join(ROOT_DIR,'SIL-Results/UMAP_layers/acts.html'))'''
    



