from functions import measure_tangling
import numpy as np
from definitions import ROOT_DIR
import os
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

n_mass = 3
layers = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/layers_mass','rb'))
len_ep1 = len(layers[0])
len_ep2 = len(layers[1])
len_ep3 = len(layers[2])

layers = [item for sublist in layers for item in sublist] # concatenation
n_comp = 25
pca = PCA(n_components=n_comp)

obs = np.array([d['observation'] for d in layers])
obs_trans = pca.fit_transform(obs)
q_obs_ep1 = measure_tangling(obs_trans[0:len_ep1])
q_obs_ep2 = measure_tangling(obs_trans[len_ep1:len_ep1+len_ep2])
q_obs_ep3 = measure_tangling(obs_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3])
Q_obs = np.mean(np.mean(q_obs_ep1)+np.mean(q_obs_ep2)+np.mean(q_obs_ep3))

lstm = np.array([d['LSTM hidden state'] for d in layers])
lstm_trans = pca.fit_transform(lstm)
q_lstm_ep1 = measure_tangling(lstm_trans[0:len_ep1])
q_lstm_ep2 = measure_tangling(lstm_trans[len_ep1:len_ep1+len_ep2])
q_lstm_ep3 = measure_tangling(lstm_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3])
Q_lstm = np.mean(np.mean(q_lstm_ep1)+np.mean(q_lstm_ep2)+np.mean(q_lstm_ep3))

l1 = np.array([d['Linear layer 1'] for d in layers])
l1_trans = pca.fit_transform(l1)
q_l1_ep1 = measure_tangling(l1_trans[0:len_ep1])
q_l1_ep2 = measure_tangling(l1_trans[len_ep1:len_ep1+len_ep2])
q_l1_ep3 = measure_tangling(l1_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3])
Q_l1 = np.mean(np.mean(q_l1_ep1)+np.mean(q_l1_ep2)+np.mean(q_l1_ep3))

l2 = np.array([d['Linear layer 2'] for d in layers])
l2_trans = pca.fit_transform(l2)
q_l2_ep1 = measure_tangling(l2_trans[0:len_ep1])
q_l2_ep2 = measure_tangling(l2_trans[len_ep1:len_ep1+len_ep2])
q_l2_ep3 = measure_tangling(l2_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3])
Q_l2 = np.mean(np.mean(q_l2_ep1)+np.mean(q_l2_ep2)+np.mean(q_l2_ep3))

acts = np.array([d['actions'] for d in layers])
acts_trans = pca.fit_transform(acts)
q_acts_ep1 = measure_tangling(acts_trans[0:len_ep1])
q_acts_ep2 = measure_tangling(acts_trans[len_ep1:len_ep1+len_ep2])
q_acts_ep3 = measure_tangling(acts_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3])
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
    plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Tangling/PCA/%s.png'%filename))
    plt.show()

obs_trans_per_mass = [obs_trans[0:len_ep1],obs_trans[len_ep1:len_ep1+len_ep2],obs_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3]]
lstm_trans_per_mass = [lstm_trans[0:len_ep1],lstm_trans[len_ep1:len_ep1+len_ep2],lstm_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3]]
acts_trans_per_mass = [acts_trans[0:len_ep1],acts_trans[len_ep1:len_ep1+len_ep2],acts_trans[len_ep1+len_ep2:len_ep1+len_ep2+len_ep3]]

for trans, title,filename in zip([obs_trans_per_mass,lstm_trans_per_mass],['Q observations','Q LSTM'],['obs-acts','lstm-acts']):
    figure = plt.figure()
    for i in range(n_mass):
        plt.scatter(measure_tangling(acts_trans_per_mass[i]),measure_tangling(trans[i]),s=7,label='Mass %s'%i,alpha=0.8)
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
    plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Tangling/PCA/%s.png'%filename))
    plt.show()
