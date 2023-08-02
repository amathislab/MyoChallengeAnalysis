from PA_performance import * 
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd

if __name__=="__main__":

    num_ep = 100
    n_comp = 39

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
    actions = []
    for n in range(num_ep):
        print(n)
        acts_1ep = []
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
            acts_1ep.append(action)
        if len(acts_1ep) < 200 :
            temp = np.zeros((200,39))
            temp[:len(acts_1ep)] += acts_1ep
            acts_1ep = temp
        actions.append(np.array(acts_1ep))
    
    fp_rollouts = open('/home/ingster/Bureau/SIL-BigResults/rollout_100ep', 'wb')
    pickle.dump(actions,fp_rollouts)
    fp_rollouts.close()'''

    actions = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/rollout_100ep','rb'))
    
    pca = PCA(n_components=n_comp)
    mean_actions = sum(actions)/len(actions)
    mean_weights = pca.fit_transform(mean_actions)

    minmax = MinMaxScaler(feature_range=(-1,1))
    weights=[]
    for j in range(15):
        norm_weights = minmax.fit_transform(mean_weights[13:,j].reshape(187,1))
        weights.append(norm_weights)
        '''plt.plot([n for n in range(200)],norm_weights,label='PA'+str(j+1),linewidth=1.2)
    plt.legend(loc='upper right')
    plt.xlabel('Time step')
    plt.ylabel('PC weight')
    plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Motor-synergies/Muscle-activations/principal_actions_phase.png'))'''
    
    plt.clf()
    fig = sns.heatmap(pd.DataFrame(np.squeeze(weights)),cmap="coolwarm").get_figure()
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.xlabel('Time step')
    plt.ylabel('Principal action weight')
    plt.show()


