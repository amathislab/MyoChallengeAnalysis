from definitions import ROOT_DIR
import joblib
import os
import numpy as np
from envs.environment_factory import EnvironmentFactory
from legacy.functions_notebook import set_config

num_steps_per_pose = 1000
pc_idx = 1
render = True

# Load the PCA of the pose
pca_path = os.path.join(ROOT_DIR, "data", "pca", "pca_pose_ccw.joblib")
pca = joblib.load(pca_path)

env_name = "CustomMyoBaodingBallsP2"

config = set_config(period=5, rot_dir="ccw")
rollouts = []

eval_env = EnvironmentFactory.create(env_name, **config)
frames = []

for pc_sign in [-1, 1]:
    eval_env.reset()
    for n in range(num_steps_per_pose):
        obs, rewards, done, info = eval_env.step(-np.ones(39))
        hand_pose = pca.mean_ + pc_sign * pca.components_[pc_idx]
        qpos = np.concatenate((hand_pose, np.zeros(14)))
        qpos[25] = 20
        qpos[32] = 20
        qvel = np.zeros(35)
        eval_env.sim.model.site_pos[eval_env.target1_sid, 2] = 20
        eval_env.sim.model.site_pos[eval_env.target2_sid, 2] = 20
        eval_env.set_state(qpos, qvel)
        if render:
            eval_env.sim.render(mode="window")