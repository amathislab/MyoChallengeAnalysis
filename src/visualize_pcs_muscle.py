from definitions import ROOT_DIR
import pickle
import os
from envs.environment_factory import EnvironmentFactory
from legacy.functions_notebook import set_config

num_episodes = 10_000
action_idx = 0
render = True

# Load the file from Basecamp : 'performance_actions_components_t'
performance_components = pickle.load(
    open(
        os.path.join(
            ROOT_DIR, "data", "basecamp", "performance_action_components_t.pkl"
        ),
        "rb",
    )
)
principal_actions = [d["components"] for d in performance_components][0]

env_name = "CustomMyoBaodingBallsP2"

config = set_config(period=5, rot_dir="cw")
rollouts = []

eval_env = EnvironmentFactory.create(env_name, **config)
frames = []

for n in range(num_episodes):
    eval_env.reset()

    qpos = eval_env.init_qpos.copy()
    qvel = eval_env.init_qvel.copy()
    qpos[25] = 10
    qpos[32] = 10
    eval_env.sim.model.site_pos[eval_env.target1_sid, 2] = 10
    eval_env.sim.model.site_pos[eval_env.target2_sid, 2] = 10
    eval_env.set_state(qpos, qvel)

    timestep = 0
    while timestep < 16:
        if render:
            eval_env.sim.render(mode="window")
        obs, rewards, done, info = eval_env.step(principal_actions[action_idx])
        timestep += 1
