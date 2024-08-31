import os
import shutil
import torch
from torch import nn
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from metrics.custom_callbacks import EnvDumpCallback
from train.trainer import BaodingTrainer


# define constants
ENV_NAME = "CustomMyoBaodingBallsP1"

now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", now)

# load_folder = "trained_models/baoding_phase2/alberto_518/"
PATH_TO_NORMALIZED_ENV = None  # load_folder + "training_env.pkl"
PATH_TO_PRETRAINED_NET = None  # load_folder + "best_model.zip"

# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "pos_dist_1": 1,
        "pos_dist_2": 1,
        "act_reg": 0,
        "alive": 1,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    },
    # # custom params for curriculum learning
    # "enable_rsi": False,
    # "rsi_probability": 0,
    # "balls_overlap": False,
    # "overlap_probability": 0,
    # "noise_fingers": 0,
    # "limit_init_angle": False,
    # "beta_init_angle": [0.9,0.9], # caution: doesn't work if limit_init_angle = False
    "goal_time_period": [5, 5],  # phase 2: (4, 6)
    # "goal_xrange": (0.020, 0.030),  # phase 2: (0.020, 0.030)
    # "goal_yrange": (0.022, 0.032),  # phase 2: (0.022, 0.032)
    # Randomization in physical properties of the baoding balls
    # "obj_size_range": (0.018, 0.022),
    # "obj_mass_range": (0.030, 0.300),
    # "obj_friction_change": (0.2, 0.001, 0.00002),
    "obs_keys": [
        'muscle_len', 'muscle_vel', 'muscle_force',
        'object1_pos', 'object1_velp',
        'object2_pos', 'object2_velp',
        'target1_pos', 'target2_pos',
        'target1_err', 'target2_err',
    ]
}

model_config = dict(
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=32,
    n_steps=128,
    learning_rate=2e-05,
    ent_coef=3e-06,
    clip_range=0.2,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    vf_coef=0.5,
    n_epochs=10,
    policy_kwargs=dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    ),
)

# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_config, num_env, start_index=0):
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(ENV_NAME, **env_config)
            env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    # Create and wrap the training and evaluations environments
    envs = make_parallel_envs(config, 16)
    if PATH_TO_NORMALIZED_ENV is not None:
        envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)
    else:
        envs = VecNormalize(envs, training=True, norm_obs=True, norm_reward=True)

    # eval_env = make_parallel_envs(config, num_env=1)
    # eval_env = VecNormalize.load(PATH_TO_NORMALIZED_ENV, eval_env)

    # Define callbacks for evaluation and saving the agent
    # eval_callback = EvalCallback(
    #     eval_env=eval_env,
    #     callback_on_new_best=EnvDumpCallback(TENSORBOARD_LOG, verbose=0),
    #     n_eval_episodes=10,
    #     best_model_save_path=TENSORBOARD_LOG,
    #     log_path=TENSORBOARD_LOG,
    #     eval_freq=10_000,
    #     deterministic=True,
    #     render=False,
    #     verbose=1,
    # )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=TENSORBOARD_LOG,
        save_vecnormalize=True,
        verbose=1,
    )

    # Define trainer
    trainer = BaodingTrainer(
        envs=envs,
        env_config=config,
        load_model_path=PATH_TO_PRETRAINED_NET,
        log_dir=TENSORBOARD_LOG,
        model_config=model_config,
        # callbacks=[eval_callback, checkpoint_callback],
        callbacks=[checkpoint_callback],
        timesteps=20_000_000,
    )

    # Train agent
    trainer.train(total_timesteps=trainer.timesteps)
    trainer.save()
