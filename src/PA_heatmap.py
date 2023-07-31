import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from definitions import ROOT_DIR

performance_components = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/performance_actions_components','rb'))
principal_actions = [d['components'] for d in performance_components][0]

fig = sns.heatmap(pd.DataFrame(principal_actions[:13]),cmap="coolwarm").get_figure()
plt.xlabel('Action dimensions',fontsize=21)
plt.ylabel('Principal actions',fontsize=21)
plt.yticks(fontsize=21)
plt.xticks(fontsize=21)
plt.subplots_adjust(left=0.15,bottom=0.15)
plt.savefig(os.path.join(ROOT_DIR,'SIL-Results/Motor-synergies/Muscle-activations/principal_actions_heatmap.png'))