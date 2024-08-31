from envs.environment_factory import EnvironmentFactory
from legacy.functions_notebook import set_config
import matplotlib.pyplot as plt
import pickle
import os
from definitions import ROOT_DIR
import time


# performance_components = pickle.load(open(os.path.join(ROOT_DIR, "data", "basecamp", "performance_action_components_t.pkl"),'rb'))
# principal_actions = [d['components'] for d in performance_components][0]
# plot_every = 30
# for action_idx in range(5):
#     env_name = "CustomMyoBaodingBallsP2"

#     config = set_config(period=5,rot_dir="cw")
#     rollouts = []

#     eval_env = EnvironmentFactory.create(env_name, **config)
#     frames = []

#     for n in range(10): # Just one episode
#         eval_env.reset()
#         qpos = eval_env.init_qpos.copy()
#         qvel = eval_env.init_qvel.copy()
#         qpos[25] = 10
#         qpos[32] = 10
#         eval_env.sim.model.site_pos[eval_env.target1_sid, 2] = 10
#         eval_env.sim.model.site_pos[eval_env.target2_sid, 2] = 10
#         eval_env.set_state(qpos, qvel)
#         timestep = 0
#         while timestep < 50 : 
#             # curr_frame = eval_env.render_camera_offscreen(['hand_top', 'hand_bottom', 'hand_side_inter', 'hand_side_exter', 'plam_lookat'])
#             # frames.append(curr_frame)
#             timestep += 1
#             obs, rewards, done, info = eval_env.step(principal_actions[action_idx])
#             eval_env.mj_render()

    # cam_frames = [l[1] for l in frames[::plot_every]]
    # num_frames = len(cam_frames)

    # print("Plotting component ", action_idx)
    # # Create a figure with a single row and the number of columns equal to the number of frames
    # fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 4, 4))

    # # Remove axes for all subplots
    # for ax in axes:
    #     ax.axis('off')

    # # Display each frame in its respective subplot
    # for i, frame in enumerate(cam_frames):
    #     axes[i].imshow(frame)

    # plt.tight_layout()  # Adjust spacing between subplots
    # # plt.savefig(os.path.join(ROOT_DIR, "data", "figures", "panel_3", f"principal_component_{action_idx}_frames.png"), format="png", dpi=600, bbox_inches="tight")
    # plt.show()
    
if __name__ == "__main__":

    env_config = {
        "weighted_reward_keys": {
            "pos_dist_1": 2,
            "pos_dist_2": 2,
            "act_reg": 0,
            # "alive": 1,
            "solved": 5,
            "done": 0,
            "sparse": 0,
        },
        "task_choice": "fixed",
        "obs_keys": [
            'muscle_len', 'muscle_vel', 'muscle_force',
            'object1_pos', 'object1_velp',
            'object2_pos', 'object2_velp',
            'target1_pos', 'target2_pos',
            'target1_err', 'target2_err',
        ]
    }

    env = EnvironmentFactory.create("CustomMyoBaodingBallsP1", **env_config)
    breakpoint()


