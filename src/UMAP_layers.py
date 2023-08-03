from functions import make_parallel_envs,VecNormalize,RecurrentPPO,EnvironmentFactory
import numpy as np
import torch
from definitions import ROOT_DIR
import os
import pickle
import umap
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


if __name__=="__main__":
    
    n_mass = 3
    configs = [{
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
            0.025,
            0.025
        ],
        "goal_yrange": [
            0.027,
            0.027
        ],
        "obj_size_range": [
            0.021,
            0.021
        ],
        "obj_mass_range": [
            mass,
            mass
        ],
        "obj_friction_change": [
            0,
            0,
            0
        ],
        "task_choice": "fixed",
        "rotation_direction" : "cw"
    } for mass in np.linspace(0.03,0.3,n_mass)]

    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/env.pkl",
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/model.zip",
    )

    env_name = "CustomMyoBaodingBallsP2"
    
    layers = []
    for config in configs : 
        num_ep = 1
        layers.append(get_layers(config=config,envir=PATH_TO_NORMALIZED_ENV,env_name=env_name,net=PATH_TO_PRETRAINED_NET,num_ep=num_ep))

    fp_rollouts = open('/home/ingster/Bureau/SIL-BigResults/layers_mass', 'wb')
    pickle.dump(layers,fp_rollouts)
    fp_rollouts.close()

    layers = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/layers_mass','rb'))
    len_ep1 = len(layers[0])
    len_ep2 = len(layers[1])
    len_ep3 = len(layers[2])

    layers = [item for sublist in layers for item in sublist] # concatenation
    n_comp = 3
    umap_apply = umap.UMAP(n_components=n_comp,random_state=42)

    obs = np.array([d['observation'] for d in layers])
    obs_trans = umap_apply.fit_transform(obs)
    Q_obs = np.mean(np.mean(measure_tangling(obs_trans[0:len_ep1]))+np.mean(measure_tangling(obs_trans[len_ep1:len_ep1+len_ep2]))+np.mean(measure_tangling(obs_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3])))

    lstm = np.array([d['LSTM hidden state'] for d in layers])
    lstm_trans = umap_apply.fit_transform(lstm)
    Q_lstm = np.mean(np.mean(measure_tangling(lstm_trans[0:len_ep1]))+np.mean(measure_tangling(lstm_trans[len_ep1:len_ep1+len_ep2]))+np.mean(measure_tangling(lstm_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3])))

    l1 = np.array([d['Linear layer 1'] for d in layers])
    l1_trans = umap_apply.fit_transform(l1)
    Q_l1 = np.mean(np.mean(measure_tangling(l1_trans[0:len_ep1]))+np.mean(measure_tangling(l1_trans[len_ep1:len_ep1+len_ep2]))+np.mean(measure_tangling(l1_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3])))

    l2 = np.array([d['Linear layer 2'] for d in layers])
    l2_trans = umap_apply.fit_transform(l2)
    Q_l2 = np.mean(np.mean(measure_tangling(l2_trans[0:len_ep1]))+np.mean(measure_tangling(l2_trans[len_ep1:len_ep1+len_ep2]))+np.mean(measure_tangling(l2_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3])))

    acts = np.array([d['actions'] for d in layers])
    acts_trans = umap_apply.fit_transform(acts)
    Q_acts = np.mean(np.mean(measure_tangling(acts_trans[0:len_ep1]))+np.mean(measure_tangling(acts_trans[len_ep1:len_ep1+len_ep2]))+np.mean(measure_tangling(acts_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3])))

    projected_all = [obs_trans,lstm_trans,l1_trans,l2_trans,acts_trans]
    filenames = ['obs','lstm','l1','l2','acts']
    titles = ['Observations','LSTM','Linear 1','Linear 2','Actions']
    Qs = [Q_obs,Q_lstm,Q_l1,Q_l2,Q_acts]

    for projected,filename,title,q in zip(projected_all,filenames,titles,Qs):
        eps = [projected[0:len_ep1],projected[len_ep1:len_ep1+len_ep2],projected[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3]]
        cmap = mpl.colormaps['Set1']
        colors = cmap(np.linspace(0,0.5,n_mass))
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(n_mass):
            projected = eps[i]
            ax.plot(projected[:,0],projected[:,1],projected[:,2],color=colors[i],linewidth=1,label='Mass '+str(i+1))

        ax.view_init(elev=30.,azim=45)

        plt.legend(fontsize=21,loc='center left', bbox_to_anchor=(0.7, 0.9))
        plt.xlabel('Proj. 1',fontsize=21,labelpad=10)
        plt.ylabel('Proj. 2',fontsize=21,labelpad=10)
        ax.set_zlabel('Proj. 3',fontsize=21,labelpad=10)
        plt.title(title,fontsize=21)
        ax.text2D(0.01,0.96, ha='left', va='top', transform=ax.transAxes, s='Q = %s'%np.round(q,3), color='black', 
            bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round',pad=0.3),fontsize=21)
        
        ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='y', nbins=5)
        ax.locator_params(axis='z', nbins=5)
        font = {'size': 19}
        plt.xticks(fontsize=font['size'])
        plt.yticks(fontsize=font['size'])
        ax.tick_params('z', labelsize=font['size'])
        plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/UMAP_layers/%s.png'%filename))


    obs_trans_per_mass = [obs_trans[0:len_ep1],obs_trans[len_ep1:len_ep1+len_ep2],obs_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3]]
    lstm_trans_per_mass = [lstm_trans[0:len_ep1],lstm_trans[len_ep1:len_ep1+len_ep2],lstm_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3]]
    acts_trans_per_mass = [acts_trans[0:len_ep1],acts_trans[len_ep1:len_ep1+len_ep2],acts_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3]]

    for trans, title,filename in zip([obs_trans_per_mass,lstm_trans_per_mass],['Q observations','Q LSTM'],['obs-acts','lstm-acts']):
        figure = plt.figure()
        for i in range(n_mass):
            plt.scatter(measure_tangling(acts_trans_per_mass[i]),measure_tangling(trans[i]),s=10,label='Mass %s'%i)
        plt.ylabel(title,fontsize=21)
        plt.xlabel('Q actions',fontsize=21)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        lgnd = plt.legend(fontsize=21)
        lgnd.legendHandles[0]._sizes = [40]
        lgnd.legendHandles[1]._sizes = [40]
        lgnd.legendHandles[2]._sizes = [40]
        plt.subplots_adjust(left=0.15,bottom=0.2)
        pt = (0, 0)
        plt.axline(pt, slope=1, color='black')
        plt.axis('equal')
        plt.xlim(xmin=0)
        plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/UMAP_layers/%s.png'%filename))
        plt.clf()
