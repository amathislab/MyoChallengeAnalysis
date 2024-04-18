import os
from definitions import ROOT_DIR
import torch
import glob
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from functions import make_parallel_envs
import matplotlib.pyplot as plt

config ={
    "weighted_reward_keys": {
        "pos_dist_1": 0,
        "pos_dist_2": 0,
        "act_reg": 0,
        "alive": 0,
        "solved": 5,
        "done": 0,
        "sparse": 0
    },
    "task": "random",
    "enable_rsi": False,
    "rsi_probability": 0,
    "noise_palm": 0,
    "noise_fingers": 0,
    "noise_balls": 0,
    "goal_time_period": [
        10,
        10
    ],
    "goal_xrange": [
        0.025,
        0.025
    ],
    "goal_yrange": [
        0.028,
        0.028
    ],
    "drop_th": 1.3
}

steps = sorted(filter(os.path.isdir,glob.glob('/home/ingster/Bureau/SIL/Policy_analysis_learning_process/trained_models/curriculum_steps_complete_baoding_winner/'+'*')))

if __name__ == '__main__':

    env_name = 'CustomMyoBaodingBallsP1'
    envs = make_parallel_envs(env_name, config, num_env=1)

    # Create model
    custom_objects = {
        "learning_rate": lambda _: 0,
        "lr_schedule": lambda _: 0,
        "clip_range": lambda _: 0,
    }

    models = [None]*(len(steps))
    Wih = [None]*(len(steps)); Whh = [None]*(len(steps)); bih = [None]*(len(steps)); bhh = [None]*(len(steps))

    for i in range(len(steps)):
        PATH_TO_NORMALIZED_ENV = os.path.join(
            ROOT_DIR,
            steps[i]+'/env.pkl',
        )
        PATH_TO_PRETRAINED_NET = os.path.join(
            ROOT_DIR,
            steps[i]+'/model.zip',
        )

        envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)
        envs.training = False
        envs.norm_reward = False
        
        models[i] = RecurrentPPO.load(
                    PATH_TO_PRETRAINED_NET, env=envs, device="cpu", custom_objects=custom_objects
                )
        Wih[i], Whh[i], bih[i], bhh[i] = models[i].policy.lstm_actor.all_weights[0]

    
    dWih = [None]*(len(steps)-1); dWhh = [None]*(len(steps)-1); dbih = [None]*(len(steps)-1); dbhh = [None]*(len(steps)-1)
    for i in range(len(steps)-1) :
        dWih[i] = torch.norm(torch.subtract(Wih[i],Wih[i+1]),p='fro').detach().numpy()
        dWhh[i] = torch.norm(torch.subtract(Whh[i],Whh[i+1]),p='fro').detach().numpy()
        dbih[i] = torch.norm(torch.subtract(bih[i],bih[i+1]),p='fro').detach().numpy()
        dbhh[i] = torch.norm(torch.subtract(bhh[i],bhh[i+1]),p='fro').detach().numpy()


    step = [i for i in range(len(steps)-1)]
    figure2, axis = plt.subplots(2,2)
    axis[0,0].scatter(step,dWih,s=2); axis[0,0].set_title('Difference in Wih',fontsize=10)
    axis[0,1].scatter(step,dWhh,s=2); axis[0,1].set_title('Difference in Whh',fontsize=10)
    axis[1,0].scatter(step,dbih,s=2); axis[1,0].set_title('Difference in Bih',fontsize=10)
    axis[1,1].scatter(step,dbhh,s=2); axis[1,1].set_title('Difference in Bih',fontsize=10)
    plt.show()


'''
model.policy.lstm_actor.all_weights : [[weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0]]
model.policy.lstm_actor._all_weights : [['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']]
weight_ih_l0 : torch tensor of size [1024,86] 
weight_hh_l0 : torch tensor of size [1024,256]
bias_ih_l0 : torch tensor of size [1024]
bias_hh_l0 : torch tensor of size [1024]
'''
