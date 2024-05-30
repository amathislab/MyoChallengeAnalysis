import os

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
NUM_MUSCLES = 39
NUM_JOINTS = 23

TASK_TO_PRINT = {
    "baoding": "Baoding",
    "early_baoding": "Baoding step 12",
    "hand_pose": "Hand Pose",
    "hand_reach": "Hand Reach",
    "pen": "Pen",
    "reorient": "Reorient"
}

ARM_MUSCLE_NAMES = [
    "ECRL",  # Hand muscles
    "ECRB",
    "ECU",
    "FCR",
    "FCU",
    "PL",
    "PT",
    "PQ",
]

FINGER_MUSCLE_NAMES = [
    "FDS5",
    "FDS4",
    "FDS3",
    "FDS2",
    "FDP5",
    "FDP4",
    "FDP3",
    "FDP2",
    "EDC5",
    "EDC4",
    "EDC3",
    "EDC2",
    "EDM",
    "EIP",
    "EPL",
    "EPB",
    "FPL",
    "APL",
    "OP",
    "RI2",
    "LU_RB2",
    "UI_UB2",
    "RI3",
    "LU_RB3",
    "UI_UB3",
    "RI4",
    "LU_RB4",
    "UI_UB4",
    "RI5",
    "LU_RB5",
    "UI_UB5",
]

HAND_MUSCLE_NAMES = ARM_MUSCLE_NAMES + FINGER_MUSCLE_NAMES

HAND_JOINT_NAMES = [
    "pro_sup",  # Hand joints
    "deviation",
    "flexion",
    "cmc_abduction",
    "cmc_flexion",
    "mp_flexion",
    "ip_flexion",
    "mcp2_flexion",
    "mcp2_abduction",
    "pm2_flexion",
    "md2_flexion",
    "mcp3_flexion",
    "mcp3_abduction",
    "pm3_flexion",
    "md3_flexion",
    "mcp4_flexion",
    "mcp4_abduction",
    "pm4_flexion",
    "md4_flexion",
    "mcp5_flexion",
    "mcp5_abduction",
    "pm5_flexion",
    "md5_flexion",
]

MAIN_DF_COLS = [
    "episode",
    "step",
    "observation",
    "action",
    "reward",
    "next_observation",
    "muscle_act",
    "rew_dict",
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
    "hand_vel",
]
