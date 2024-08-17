import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from legacy.functions import make_parallel_envs
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory


# Analysis of the performance for the rotation and the holding tasks through the curriculum steps

if __name__ == '__main__':
    # evaluation parameters:
    render = False
    num_episodes = 20
    env_name = "CustomMyoBaodingBallsP1"

    l_trained_models = sorted(filter(os.path.isdir,glob.glob(os.path.join(ROOT_DIR,"trained_models/curriculum_steps_complete_baoding_winner/")+'*')))
    rwd_mean = []
    for filename in l_trained_models :
        # Path to normalized Vectorized environment and best model (if not first task)
        PATH_TO_NORMALIZED_ENV = os.path.join(
            ROOT_DIR,
            filename+"/env.pkl",
        )
        PATH_TO_PRETRAINED_NET = os.path.join(
            ROOT_DIR,
            filename+"/model.zip",
        )
            
        # ROTATION TASK

        # Reward structure and task parameters:
        config = {
            "weighted_reward_keys": {
                "pos_dist_1": 0,
                "pos_dist_2": 0,
                "act_reg": 0,
                "solved": 5,
                "done": 0,
                "sparse": 0,
            },
            "enable_rsi": False,
            "rsi_probability": 1,
            "drop_th": 1.3,
            # "balls_overlap": True, 
            # "overlap_probability": 1, 
            # "limit_init_angle": 0,
            "goal_time_period": [4.75, 5.25],  # phase 2: (4, 6)
            "goal_xrange": (0.025, 0.025),  # phase 2: (0.020, 0.030)
            "goal_yrange": (0.028, 0.028),  # phase 2: (0.022, 0.032)
            # Randomization in physical properties of the baoding balls
            "obj_size_range": (
                0.022,
                0.022,
            ),  # (0.018, 0.024)   # Object size range. Nominal 0.022
            "obj_mass_range": (
                0.043,
                0.043,
            ),  # (0.030, 0.300)   # Object weight range. Nominal 43 gms
            "obj_friction_change": (0, 0, 0),  # (0.2, 0.001, 0.00002)
            #"task_choice": "fixed",
            "seed":2
        }

        # Create vectorized environments:
        envs = make_parallel_envs(env_name, config, num_env=1)

        # Normalize environment:
        envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)
        envs.training = False
        envs.norm_reward = False

        # Create model
        custom_objects = {
            "learning_rate": lambda _: 0,
            "lr_schedule": lambda _: 0,
            "clip_range": lambda _: 0,
        }
        model = RecurrentPPO.load(
            PATH_TO_PRETRAINED_NET, env=envs, device="cpu", custom_objects=custom_objects
        )

        # EVALUATE
        eval_model = model
        eval_env = EnvironmentFactory.create(env_name, **config)

        # Enjoy trained agent
        perfs = []
        lens = []
        for i in range(num_episodes):
            lstm_states = None
            cum_rew = 0
            step = 0
            # eval_env.reset()
            # eval_env.step(np.zeros(39))
            obs = eval_env.reset()
            episode_starts = np.ones((1,), dtype=bool)
            done = False
            while not done:
                if render:
                    eval_env.sim.render(mode="window")
                action, lstm_states = eval_model.predict(
                    envs.normalize_obs(obs),
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, rewards, done, info = eval_env.step(action)
                episode_starts = done
                cum_rew += rewards
                step += 1
            lens.append(step)
            perfs.append(cum_rew)
            print("Episode", i, ", len:", step, ", cum rew: ", cum_rew)

            '''if (i + 1) % 10 == 0:
                len_error = np.std(lens) / np.sqrt(i + 1)
                perf_error = np.std(perfs) / np.sqrt(i + 1)

                print(f"\nEpisode {i+1}/{num_episodes}")
                print(f"Average len: {np.mean(lens):.2f} +/- {len_error:.2f}")
                print(f"Average rew: {np.mean(perfs):.2f} +/- {perf_error:.2f}\n")'''

        print(f"\nFinished evaluating {PATH_TO_PRETRAINED_NET}!")
        rwd_mean.append(np.mean(perfs))

    plt.plot(np.linspace(1,len(l_trained_models),len(l_trained_models)), rwd_mean)
    plt.xlabel('Curriculum step')
    plt.ylabel('Mean reward across episodes')
    plt.title('Rotation task, goal time period = [4.75, 5.25]')
    plt.savefig(os.path.join(ROOT_DIR,"SIL-Results/Rotation_perf.png"))
    plt.show()