import numpy as np
import os
import pandas as pd
import json
import torch
import subprocess
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.environment_factory import EnvironmentFactory
from definitions import ROOT_DIR
from myosuite.envs.myo.myochallenge.baoding_v1 import Task


# evaluation parameters:
render = True
save_df = False
# out_dir = "final_model_500_episodes_activations_info_cw"
out_dir = "step_12_500_episodes_activations_info_cw"
HOST = "alberto@amg5"
HOST_PROJECT_ROOT = "/media/disk2/alberto/MyoChallengeAnalysis"  # "/home/alberto/Dev/rl/MyoChallengeAnalysis"

eval_config = {
    "env_config": {
        "env_name": "CleanBaodingBalls",
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "alive": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0
        },
        "initial_phase": 1.5707963267948966,
        "limit_sds_angle": 0,
        "limit_init_angle": 0,
        "task_choice": "cw",
        "goal_time_period": [
            5,
            5
        ],
        "obs_keys": [
            "muscle_len",
            "muscle_vel",
            "muscle_force",
            "object1_pos",
            "object1_velp",
            "object2_pos",
            "object2_velp",
            "target1_pos",
            "target2_pos",
            "target1_err",
            "target2_err"
        ]
    },
    "experiment_path": os.path.join("output", "training", "2024-06-03", "10-09-05_5"),
    "checkpoint_num": 19_600_000,
    "task": None,  # Task.HOLD, Task.BAODING_CCW, Task.BAODING_CW, None,
    "num_episodes": 500,
    "seed": 42,
    # "random_phase": 0
}

def load_vecnormalize(env_path, base_env):
    venv = DummyVecEnv([lambda: base_env])
    print("env path", env_path)
    vecnormalize = VecNormalize.load(env_path, venv)
    return vecnormalize


def load_model(model_path):
    custom_objects = {
        "learning_rate": lambda _: 0,
        "lr_schedule": lambda _: 0,
        "clip_range": lambda _: 0,
    }
    model = RecurrentPPO.load(model_path, custom_objects=custom_objects, device="cpu")
    return model

def get_remote_checkpoint(experiment_path, checkpoint_num):
    if checkpoint_num is None:
        raise NotImplementedError("Selection of best checkpoint from the remote not implemented")
    file_names = [
        "args.json",
        "env_config.json",
        "*_config.json",
        f"rl_model_{checkpoint_num}_steps.zip",
        f"rl_model_vecnormalize_{checkpoint_num}_steps.pkl"
    ]
    file_paths = [os.path.join(f"{HOST}:{HOST_PROJECT_ROOT}", experiment_path, f) for f in file_names]
    os.makedirs(os.path.join(ROOT_DIR, experiment_path), exist_ok=True)
    subprocess.run(["rsync",  *file_paths, os.path.join(ROOT_DIR, experiment_path)])
    
if __name__ == "__main__":
    # Create test env and vecnormalize
    env = EnvironmentFactory.create(**eval_config["env_config"])

    if eval_config["experiment_path"] is None:
        model = PPO(policy="MultiInputPolicy", env=env)
        venv = DummyVecEnv([lambda: env])
        vecnormalize = VecNormalize(venv)
    else:
        env_path = os.path.join(ROOT_DIR, eval_config["experiment_path"], "rl_model_vecnormalize_{}_steps.pkl".format(eval_config["checkpoint_num"]))
        net_path = os.path.join(ROOT_DIR, eval_config["experiment_path"], "rl_model_{}_steps.zip".format(eval_config["checkpoint_num"]))
        if not os.path.exists(net_path):
            get_remote_checkpoint(eval_config["experiment_path"], eval_config["checkpoint_num"])
        vecnormalize = load_vecnormalize(env_path, env)
        model = load_model(net_path)
        
    vecnormalize.training = False
    vecnormalize.norm_reward = False

    # Enjoy trained agent
    perfs = []
    lens = []
    episode_data = []
    for i in range(eval_config["num_episodes"]):
        lstm_states = (np.zeros((1, 1, 256)), np.zeros((1, 1, 256)))
        cum_rew = 0
        step = 0
        obs = env.reset()
        episode_starts = torch.ones((1,))
        done = False
        if eval_config["task"] is not None:
            env.env.which_task = eval_config["task"]
        # obs = env.reset(random_phase=eval_config["random_phase"])
        obs = env.reset()
        while not done:
            if render:
                env.sim.render(mode="window")
            lstm_states_tensor = (torch.tensor(lstm_states[0], dtype=torch.float32).reshape(1, -1), torch.tensor(lstm_states[1], dtype=torch.float32).reshape(1, -1))     
            action, lstm_states = model.predict(
                vecnormalize.normalize_obs(obs),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            with torch.no_grad():
                features = model.policy.extract_features(torch.tensor(vecnormalize.normalize_obs(obs)).reshape(1, -1))
                lstm_out, _ = model.policy.lstm_actor(features, (lstm_states_tensor[0] * (1 - episode_starts), lstm_states_tensor[1] * (1 - episode_starts)))
                layer_1_out = model.policy.mlp_extractor.policy_net[1](model.policy.mlp_extractor.policy_net[0](lstm_out))
                layer_2_out = model.policy.mlp_extractor.policy_net[3](model.policy.mlp_extractor.policy_net[2](layer_1_out))
                action_pred = model.policy._get_action_dist_from_latent(layer_2_out).mode().clip(-1, 1)

            assert np.allclose(action_pred, action), print(action_pred, action)
            next_obs, rewards, done, _ = env.step(action)
            episode_data.append(
                [
                    i,
                    step,
                    obs,
                    action,
                    rewards,
                    next_obs,
                    env.last_ctrl,
                    env.rwd_dict,
                    np.squeeze(lstm_states[0]),
                    np.squeeze(lstm_states[1]),
                    np.squeeze(lstm_out.numpy()),
                    np.squeeze(layer_1_out.numpy()),
                    np.squeeze(layer_2_out.numpy()),
                    "ccw" if eval_config["task"] == Task.BAODING_CCW else "cw",
                    env.sim.model.body_mass[env.object1_bid],
                    env.sim.model.body_mass[env.object2_bid],
                    env.sim.model.geom_size[env.object1_gid][0],
                    env.sim.model.geom_size[env.object2_gid][0],
                    env.sim.model.geom_friction[env.object1_gid][0],
                    env.sim.model.geom_friction[env.object1_gid][1],
                    env.sim.model.geom_friction[env.object1_gid][2],
                    env.x_radius,
                    env.y_radius,
                    obs[0:23],
                    (next_obs[0:23] - obs[0:23]) / 0.0025
                ]
            )
            obs = next_obs
            episode_starts = done
            cum_rew += rewards
            step += 1

        lens.append(step)
        perfs.append(cum_rew)
        print("Episode", i, ", len:", step, ", cum rew: ", cum_rew)

        if (i + 1) % 10 == 0:
            len_error = np.std(lens) / np.sqrt(i + 1)
            perf_error = np.std(perfs) / np.sqrt(i + 1)

            print(f"\nEpisode {i+1}/{eval_config['num_episodes']}")
            print(f"Average len: {np.mean(lens):.2f} +/- {len_error:.2f}")
            print(f"Average rew: {np.mean(perfs):.2f} +/- {perf_error:.2f}\n")

    print(f"\nFinished evaluating {eval_config['net_path']}!")
    if save_df:
        df = pd.DataFrame(
            episode_data,
            columns=[
                "episode",
                "step",
                "observation",
                "action",
                "reward",
                "next_observation",
                "muscle_act",
                "rew_dict",
                "lstm_state_0",
                "lstm_state_1",
                "lstm_out",
                "layer_1_out",
                "layer_2_out",
                "task",
                "mass_1",
                "mass_2",
                "size_1",
                "size_2",
                "friction_0",
                "friction_1",
                "friction_2",
                "x_radius",
                "y_radius",
                "hand_pos",
                "hand_vel"
            ],
        )

        out_path = os.path.join(ROOT_DIR, "data", "rollouts", out_dir)
        os.makedirs(out_path, exist_ok=True)
        df.to_hdf(os.path.join(out_path, "data.hdf"), key="data")
        with open(os.path.join(out_path, "eval_config.json"), "w", encoding="utf8") as f:
            json.dump(eval_config, f, indent=4, default=lambda _: "<not serializable>")
        print("Saved to ", out_path)
