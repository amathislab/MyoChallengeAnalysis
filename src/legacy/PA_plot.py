from PA_performance import *

# To plot actions in the rotated action space

if __name__=='__main__':

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

    performance_components = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/performance_actions_components','rb'))
    principal_actions = [d['components'] for d in performance_components][0]

    
    num_ep = 1
    acts_proj = []
    for n in range(num_ep):
        cum_reward = 0
        lstm_states = None
        obs = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        timestep = 0
        while not done : 
            if render :
                eval_env.sim.render(mode="window")
                
            timestep += 1
            action, lstm_states = eval_model.predict(envs.normalize_obs(obs),
                                                    state=lstm_states,
                                                    episode_start=episode_starts,
                                                    deterministic=True,
                                                    )
            
            obs, rewards, done, info = eval_env.step(action)
            act_proj = np.dot(action.reshape(-1,39),principal_actions[0:2].T)
            episode_starts = done
            cum_reward += rewards
            acts_proj.append(act_proj)
        
        acts_proj = np.squeeze(acts_proj)
        print(cum_reward)

    fig = plt.figure()
    ax = plt.axes(projection ='3d')    
    ax.plot3D(acts_proj[:,0], acts_proj[:,1], acts_proj[:,2],linewidth=0.7,color='black')
    ax.set_xlabel('Principal action 37')
    ax.set_ylabel('Principal action 38')
    ax.set_zlabel('Principal action 39')
    plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Motor-synergies/Muscle-activations/36-39principal_actions.png'))
