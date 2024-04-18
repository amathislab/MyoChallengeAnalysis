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

# Analysis of the low-variance PCs under slightly different conditions

if __name__=="__main__":
    num_ep = 6
    num_cond = 32
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

    def get_range(m,M):
        return [[m,m + 0.1*(M-m)],[M - 0.1*(M-m),M]]

    pm = 5; pM = 30
    xm = 0.02; xM = 0.03
    ym = 0.022; yM = 0.032
    sm = 0.018; sM = 0.024
    mm = 0.03; mM = 0.3

    conds = []
    i = 0
    for period_range in get_range(pm,pM) :
        for x_range in get_range(xm,xM) :
            for y_range in get_range(ym,yM) :
                for size_range in get_range(sm,sM) : 
                    for mass_range in get_range(mm,mM) :
                        print("condition ",i)
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
                            "goal_time_period": period_range,
                            "goal_xrange": x_range,
                            "goal_yrange": y_range,
                            "obj_size_range": size_range,
                            "obj_mass_range": mass_range,
                            "obj_friction_change": [
                                0,
                                0,
                                0
                            ],
                            "task_choice": "fixed"
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
                        tot_pos = []
                        tot_vel = []
                        for n in range(num_ep):
                            obs_tot = []
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
                                obs_tot.append(obs)
                            if len(obs_tot) < 200 :
                                print("Stopped before 200, condition %s" %i, " number of steps : ",timestep)
                                temp = np.zeros((200,86))
                                temp[:len(obs_tot)] += obs_tot
                                obs_tot = temp
                            print('episode %s : '%n,cum_reward)

                            # MEASURE JOINT POSITION AND VELOCITY
                            hand_positions = np.array(obs_tot)[:,0:23]    
                            hand_velocities = np.array([np.diff(pos)/0.0025 for pos in hand_positions.T]).T
                            hand_velocities = np.vstack((np.zeros((1,23)),hand_velocities))                                

                            conds.append({'condition':config,'encoding':i,'reward':cum_reward,'hand velocity':np.array(hand_velocities)})
                        i += 1

    '''fp_conditions = open('/home/ingster/Bureau/SIL-BigResults/synergies_conditions', 'wb')
    pickle.dump(conds,fp_conditions)
    fp_conditions.close()

    conds = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/synergies_conditions','rb'))'''

    hand_kinematics = np.concatenate([cond['hand velocity'] for cond in conds])

    n_comp = 23
    pca = PCA(n_components=n_comp).fit(hand_kinematics)
    performance = []
    stds = []

    for k in range(n_comp):
        print(k)
        components = pca.components_[k:]
        projected_conds = [{'label':cond['encoding'], 'projected velocity':np.dot(cond['hand velocity'],components.T)} for cond in conds] # length 6x32=192
        X = [d['projected velocity'].flatten() for d in projected_conds]
        y = [d['label'] for d in projected_conds]
        class_performance = []
        for i in range(num_ep):
            x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=num_cond*(num_ep-1),test_size=num_cond)
            lda = LDA().fit(x_train,y_train)
            class_performance.append(lda.score(X=x_test,y=y_test))

        performance.append(np.mean(np.array(class_performance)))
        stds.append(np.std(np.array(class_performance)))
        
    plt.errorbar([n for n in range(n_comp)],performance,stds,linestyle='None', marker='o',markersize=3,linewidth=0.5)
    plt.xlabel('Number of PC removed')
    plt.ylabel('Accuracy of the linear classifier')
    plt.title('Rotation : classification performance')
    plt.savefig(os.path.join(ROOT_DIR, "SIL-Results/Motor-synergies/Low-level-PCs/rotation_classperf.png"))
