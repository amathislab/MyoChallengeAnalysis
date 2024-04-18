import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from definitions import ROOT_DIR
import numpy as np

performance_components = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/performance_actions_components','rb'))
principal_actions = [d['components'] for d in performance_components][0]

fig = sns.heatmap(pd.DataFrame(principal_actions[:13]),cmap="coolwarm").get_figure()
plt.xlabel('Action dimensions',fontsize=21)
plt.ylabel('Principal actions',fontsize=21)
plt.yticks(rotation=0,fontsize=17)
plt.xticks(ticks=np.arange(1,40,3),labels=np.arange(1,40,3),rotation=45,fontsize=17)
plt.subplots_adjust(left=0.15,bottom=0.2)
plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Motor-synergies/Muscle-activations/principal_actions_heatmap.png'))