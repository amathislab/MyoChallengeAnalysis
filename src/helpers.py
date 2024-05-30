import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Iterable
from scipy.signal import butter, lfilter
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs.environment_factory import EnvironmentFactory
from definitions import TASK_TO_PRINT


def make_parallel_envs(env_name, env_config, num_env, start_index=0):
    # pylint: disable=redefined-outer-name
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(env_name, **env_config)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def set_config(period, rot_dir):
    return {
        "weighted_reward_keys": {
            "pos_dist_1": 0,
            "pos_dist_2": 0,
            "act_reg": 0,
            "alive": 0,
            "solved": 5,
            "done": 0,
            "sparse": 0,
        },
        "enable_rsi": False,
        "rsi_probability": 0,
        "balls_overlap": False,
        "overlap_probability": 0,
        "noise_fingers": 0,
        "limit_init_angle": 3.141592653589793,
        "goal_time_period": [period, period],
        "goal_xrange": [0.02, 0.03],
        "goal_yrange": [0.022, 0.032],
        "obj_size_range": [0.018, 0.024],
        "obj_mass_range": [0.03, 0.3],
        "obj_friction_change": [0.2, 0.001, 2e-05],
        "task_choice": "fixed",
        "rotation_direction": rot_dir,
    }


def get_episode_vel(episode_pos):
    episode_vel = np.zeros_like(episode_pos)
    episode_vel[1:, :] = episode_pos[1:] - episode_pos[:-1]  # 40 sim steps per second
    return episode_vel


def get_pos_vel_act(df):
    if "task" in df.keys():
        pos_list = (
            df.groupby(["episode", "task"])["observation"]
            .agg(lambda x: np.vstack(x)[:, :23])
            .tolist()
        )
    else:
        pos_list = (
            df.groupby(["episode"])["observation"]
            .agg(lambda x: np.vstack(x)[:, :23])
            .tolist()
        )
    vel_list = [get_episode_vel(episode_pos) for episode_pos in pos_list]
    pos = np.vstack(pos_list)
    vel = np.vstack(vel_list)
    muscle_act = np.vstack(df.muscle_act)
    return pos, vel, muscle_act


def get_exp_var_ratio(data, n_comp):
    pca = PCA(n_components=n_comp)
    pca.fit(data)
    return pca.explained_variance_ratio_


def get_dof_count(exp_var, threshold=0.85):
    cum_exp_var = np.cumsum(exp_var)
    for idx, val in enumerate(cum_exp_var):
        if val > threshold:
            return idx + 1


def ev(X, X_approx, model_mean):
    return 1 - np.sum((X - X_approx) ** 2) / np.sum((X - model_mean) ** 2)


def plot_explained_variance_ratio(
    exp_var,
    task_name,
    color,
    ax=None,
    fig=None,
    xtext_pos=(0, 0.96),
    ytext_pos=(0, 0.86),
    label=None,
):
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    if label is None:
        label = TASK_TO_PRINT[task_name]
    ax.step(
        range(1, len(exp_var) + 1),
        exp_var,
        where="mid",
        linewidth=3,
        color=color,
        label=label,
    )
    ax.set_xlabel("Number of PCs", fontsize=21)
    ax.set_ylabel("Cum. explained variance", fontsize=21)
    ax.tick_params(axis="both", labelsize=20)
    ax.axhline(y=0.95, color="black", linestyle="--", alpha=0.5)
    ax.axhline(y=0.85, color="black", linestyle="--", alpha=0.5)
    ax.text(*xtext_pos, "95%", color="black", fontsize=18)
    ax.text(*ytext_pos, "85%", color="black", fontsize=18)
    return fig, ax


def average_by_timestep(vec, timesteps):
    out_vec = []
    for ts in sorted(np.unique(timesteps)):
        out_vec.append(np.mean(vec[timesteps == ts], axis=0))
    return np.vstack(out_vec)


def measure_tangling(data):
    derivative = np.gradient(data, axis=0) * 40  # sample frequency
    epsilon = 1e-10
    Q_all = []
    for t in range(derivative.shape[0]):
        Q = (np.linalg.norm(derivative[t] - derivative, axis=1) ** 2) / (
            epsilon + np.linalg.norm(data[t] - data, axis=1) ** 2
        )
        Q = np.max(Q)
        Q_all.append(Q)

    return np.mean(Q_all)


def get_data_from_tb_log(path, y, x="step", tb_config=None):
    if tb_config is None:
        tb_config = {}

    event_acc = EventAccumulator(path, tb_config)
    event_acc.Reload()
    # print(event_acc.Tags())
    if not isinstance(y, Iterable):
        y = [y]

    out_dict = {}
    for attr_name in y:
        if attr_name in event_acc.Tags()["scalars"]:
            x_vals, y_vals = np.array(
                [(getattr(el, x), el.value) for el in event_acc.Scalars(attr_name)]
            ).T
            out_dict[attr_name] = (x_vals, y_vals)
        else:
            out_dict[attr_name] = None
    return out_dict


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
