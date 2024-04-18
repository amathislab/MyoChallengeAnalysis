from functions import measure_tangling
import numpy as np
from definitions import ROOT_DIR
import os
import pickle
import matplotlib.pyplot as plt

n_mass = 3
layers = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/layers_mass','rb'))

obs_ep1 = [d['observation'] for d in layers[0]]
obs_ep2 = [d['observation'] for d in layers[1]]
obs_ep3 = [d['observation'] for d in layers[2]]

q_obs_ep1 = measure_tangling(obs_ep1)
q_obs_ep2 = measure_tangling(obs_ep2)
q_obs_ep3 = measure_tangling(obs_ep3)
Q_obs = np.mean(np.mean(q_obs_ep1)+np.mean(q_obs_ep2)+np.mean(q_obs_ep3))

lstm_ep1 = np.array([d['LSTM hidden state'] for d in layers[0]])
lstm_ep2 = np.array([d['LSTM hidden state'] for d in layers[1]])
lstm_ep3 = np.array([d['LSTM hidden state'] for d in layers[2]])

q_lstm_ep1 = measure_tangling(lstm_ep1)
q_lstm_ep2 = measure_tangling(lstm_ep2)
q_lstm_ep3 = measure_tangling(lstm_ep3)
Q_lstm = np.mean(np.mean(q_lstm_ep1)+np.mean(q_lstm_ep2)+np.mean(q_lstm_ep3))

l1_ep1 = np.array([d['Linear layer 1'] for d in layers[0]])
l1_ep2 = np.array([d['Linear layer 1'] for d in layers[1]])
l1_ep3 = np.array([d['Linear layer 1'] for d in layers[2]])

q_l1_ep1 = measure_tangling(l1_ep1)
q_l1_ep2 = measure_tangling(l1_ep2)
q_l1_ep3 = measure_tangling(l1_ep3)
Q_l1 = np.mean(np.mean(q_l1_ep1)+np.mean(q_l1_ep2)+np.mean(q_l1_ep3))

l2_ep1 = np.array([d['Linear layer 2'] for d in layers[0]])
l2_ep2 = np.array([d['Linear layer 2'] for d in layers[1]])
l2_ep3 = np.array([d['Linear layer 2'] for d in layers[2]])

q_l2_ep1 = measure_tangling(l2_ep1)
q_l2_ep2 = measure_tangling(l2_ep2)
q_l2_ep3 = measure_tangling(l2_ep3)
Q_l2 = np.mean(np.mean(q_l2_ep1)+np.mean(q_l2_ep2)+np.mean(q_l2_ep3))

acts_ep1 = np.array([d['actions'] for d in layers[0]])
acts_ep2 = np.array([d['actions'] for d in layers[1]])
acts_ep3 = np.array([d['actions'] for d in layers[2]])

q_acts_ep1 = measure_tangling(acts_ep1)
q_acts_ep2 = measure_tangling(acts_ep2)
q_acts_ep3 = measure_tangling(acts_ep3)
Q_acts = np.mean(np.mean(q_acts_ep1)+np.mean(q_acts_ep2)+np.mean(q_acts_ep3))

for Qs, title, filename in zip([[q_obs_ep1,q_obs_ep2,q_obs_ep3],[q_lstm_ep1,q_lstm_ep2,q_lstm_ep3],[q_acts_ep1,q_acts_ep2,q_acts_ep3]],['Q Observations','Q LSTM','Q Actions'],['qobs-t','qlstm-t','qacts-t']):
    figure = plt.figure()
    for i in range(len(Qs)) :
        plt.scatter(np.arange(0,len(Qs[i]),1),Qs[i],s=7,label='Mass %s'%i,alpha=0.8)
    plt.ylabel(title,fontsize=21)
    plt.xlabel('Time step',fontsize=21)
    plt.xticks(fontsize=19,rotation=45)
    plt.yticks(fontsize=19)
    plt.subplots_adjust(left=0.17,bottom=0.2)
    plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Tangling/Full-space/%s.png'%filename))
    plt.show()

obs_per_mass = [obs_ep1,obs_ep2,obs_ep3]
lstm_per_mass = [lstm_ep1,lstm_ep2,lstm_ep3]
acts_per_mass = [acts_ep1,acts_ep2,acts_ep3]

for trans, title,filename in zip([obs_per_mass,lstm_per_mass],['Q observations','Q LSTM'],['obs-acts','lstm-acts']):
    figure = plt.figure()
    for i in range(n_mass):
        plt.scatter(measure_tangling(acts_per_mass[i]),measure_tangling(trans[i]),s=7,label='Mass %s'%i,alpha=0.8)
    plt.ylabel(title,fontsize=21)
    plt.xlabel('Q actions',fontsize=21)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    lgnd = plt.legend(fontsize=21,loc='center left', bbox_to_anchor=(0.7, 0.7))
    lgnd.legendHandles[0]._sizes = [40]
    lgnd.legendHandles[1]._sizes = [40]
    lgnd.legendHandles[2]._sizes = [40]
    plt.subplots_adjust(left=0.15,bottom=0.2)
    pt = (0, 0)
    plt.axline(pt, slope=1, color='black')
    plt.axis('square')
    plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Tangling/Full-space/%s.png'%filename))
    plt.show()