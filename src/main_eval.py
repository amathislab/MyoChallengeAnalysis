import numpy as np
import os
import pandas as pd
import json
import torch
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


eval_config = {
    "env_config": {
        "env_name": "CustomMyoBaodingBallsP1",
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0
        },
        "task_choice": "fixed",
        "goal_time_period": (5, 5)
    },
    "env_path": os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/12_period_5/env.pkl",
    ),
    "net_path": os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/12_period_5/model.zip",
    ),
    "task": Task.BAODING_CCW,  # Task.HOLD, Task.BAODING_CCW, Task.BAODING_CW, None,
    "num_episodes": 500,
    "seed": 42,
    "random_phase": 0
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
    model = RecurrentPPO.load(model_path, custom_objects=custom_objects)
    return model


if __name__ == "__main__":
    # Create test env and vecnormalize
    env = EnvironmentFactory.create(**eval_config["env_config"])

    vecnormalize = load_vecnormalize(eval_config["env_path"], env)
    vecnormalize.training = False
    vecnormalize.norm_reward = False

    # Load model
    model = load_model(eval_config["net_path"])

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
        obs = env.reset(random_phase=eval_config["random_phase"])
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
