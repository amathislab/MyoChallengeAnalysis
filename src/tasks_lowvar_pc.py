from definitions import ROOT_DIR
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from functions import make_parallel_envs
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import RecurrentPPO
from envs.environment_factory import EnvironmentFactory
import plotly.express as px
import pandas as pd


# Analysis of the low-variance PCs of the kinematic subspaces of CW, CCW and Hold tasks

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

    num_ep = 50
    num_cond = 3
    n_comp = 23

    '''PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/env.pkl",
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/model.zip",
    )

    env_name = "CustomMyoBaodingBallsP2"
    render = False

    C_hold = set_config(period=1e100,rot_dir=None)
    C_cw = set_config(period=5,rot_dir="cw")
    C_ccw = set_config(period=5,rot_dir="ccw")
    
    configs = {'hold':C_hold,'cw':C_cw,'ccw':C_ccw}

    conds = []

    for task in configs :
        envs = make_parallel_envs(env_name, configs[task], num_env=1)
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
        eval_env = EnvironmentFactory.create(env_name,**configs[task])
        tot_vel = []
        for n in range(num_ep):
            obs_tot = []
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
                obs_tot.append(obs)
            if len(obs_tot) < 200 :
                print("Stopped before 200, task : %s" %task, " number of steps : ",timestep)
                temp = np.zeros((200,86))
                temp[:len(obs_tot)] += obs_tot
                obs_tot = temp
            print('episode %s : '%n,cum_reward)

            # MEASURE JOINT POSITION AND VELOCITY
            hand_positions = np.array(obs_tot)[:,0:23]    
            hand_velocities = np.array([np.diff(pos)/0.0025 for pos in hand_positions.T]).T
            hand_velocities = np.vstack((np.zeros((1,23)),hand_velocities))                                

            conds.append({'task':configs[task],'encoding':task,'reward':cum_reward,'hand velocity':np.array(hand_velocities)})
        
    
    fp_conditions = open('/home/ingster/Bureau/SIL-BigResults/synergies_tasks', 'wb')
    pickle.dump(conds,fp_conditions)
    fp_conditions.close()'''

    conds = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/synergies_tasks','rb'))
    hand_kinematics = np.concatenate([cond['hand velocity'] for cond in conds])

    pca = PCA(n_components=n_comp).fit(hand_kinematics)

    '''performance = []
    stds = []

    for k in range(n_comp):
        print(k)
        components = pca.components_[k:]
        projected_conds = [{'label':cond['encoding'], 'projected velocity':np.dot(cond['hand velocity']-pca.mean_,components.T)} for cond in conds]
        X = [d['projected velocity'].flatten() for d in projected_conds]
        y = [d['label'] for d in projected_conds]
        class_performance = []
        for i in range(num_ep):
            x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=num_cond*(num_ep-1),test_size=num_cond)
            lda = LDA().fit(x_train,y_train)
            class_performance.append(lda.score(X=x_test,y=y_test))

        performance.append(np.mean(np.array(class_performance)))
        #stds.append(np.std(np.array(class_performance)))
        
    #plt.errorbar([n for n in range(n_comp)],performance,stds,linestyle='None', marker='o',markersize=3,linewidth=0.5)
    fp = open('/home/ingster/Bureau/SIL-BigResults/class_performance_tasks_r', 'wb')
    pickle.dump(performance,fp)
    fp.close()'''
    
    pc_low_variance = next(x[0] for x in enumerate(pca.explained_variance_ratio_) if x[1] < 0.01)
    performance = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/class_performance_tasks_r','rb'))
    plt.plot([n for n in range(n_comp)],performance,'-o',linewidth=1,markersize=2,color='black')
    plt.axvline(x=pc_low_variance, ymax=0.46,color='r', linestyle='-',linewidth=1,label='1% Variance')
    plt.legend(fontsize=16.5)
    plt.xlabel('Number of PCs removed',fontsize=21)
    plt.ylabel('Accuracy',fontsize=21)
    plt.title('Classification performance',fontsize=21)
    plt.yticks(fontsize=21)
    plt.xticks(fontsize=21)
    plt.subplots_adjust(left=0.21,bottom=0.15)
    plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Motor-synergies/Low-level-PCs/tasks_classperf_r.png'))

    '''plt.clf()

    # Take the first episode of each task only
    targets = [conds[0]['encoding'],conds[1]['encoding'],conds[2]['encoding'],conds[50]['encoding'],conds[51]['encoding'],conds[52]['encoding'],conds[100]['encoding'],conds[101]['encoding'],conds[102]['encoding']]
    targets = [target for target in targets for i in range(200)]
    low_projected_velocities = np.concatenate([np.dot(conds[0]['hand velocity'],pca.components_[19:22].T),np.dot(conds[1]['hand velocity'],pca.components_[19:22].T),np.dot(conds[2]['hand velocity'],pca.components_[19:22].T),
                                               np.dot(conds[50]['hand velocity'],pca.components_[19:22].T),np.dot(conds[51]['hand velocity'],pca.components_[19:22].T),np.dot(conds[52]['hand velocity'],pca.components_[19:22].T),
                                               np.dot(conds[100]['hand velocity'],pca.components_[19:22].T),np.dot(conds[101]['hand velocity'],pca.components_[19:22].T),np.dot(conds[102]['hand velocity'],pca.components_[19:22].T)])
    df = pd.DataFrame(data=low_projected_velocities, columns=['PC 21', 'PC 22','PC 23'])
    #fig = px.scatter_3d(df,x='PC 1',y='PC 2',z='PC 3',color=targets,opacity=0.7)
    fig = px.line_3d(df,x='PC 21',y='PC 22',z='PC 23',color=targets)
    fig.update_traces(line=dict(width=5),marker_size = 4, opacity = 0.7)
    fig.write_html(os.path.join(ROOT_DIR,'SIL-Results/Motor-synergies/Low-level-PCs/Low-var-projection.html'))'''
