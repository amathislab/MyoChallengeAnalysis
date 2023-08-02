import pickle
from functions import UMAP_physics
import os
from definitions import ROOT_DIR

n_config = 2
curriculum_step = '32_phase_2_smaller_rate_resume'
data_mass = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/Mass_'+curriculum_step+'_%s-configs' %n_config,'rb'))[0]
data_size = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/Size_'+curriculum_step+'_%s-configs' %n_config,'rb'))[0]
data_xradius = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/Xradius_'+curriculum_step+'_%s-configs' %n_config,'rb'))[0]
data_yradius = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/Yradius_'+curriculum_step+'_%s-configs' %n_config,'rb'))[0]

fp_mass = os.path.join(ROOT_DIR,'SIL-Results/UMAP/Mass/')
fp_size = os.path.join(ROOT_DIR,'SIL-Results/UMAP/Size/')
fp_xradius = os.path.join(ROOT_DIR,'SIL-Results/UMAP/Xradius/')
fp_yradius = os.path.join(ROOT_DIR,'SIL-Results/UMAP/Yradius/')

UMAP_physics(ball_parameter='mass',df=data_mass,fp=fp_mass,n_config=n_config,curriculum_step=curriculum_step)
UMAP_physics(ball_parameter='size',df=data_size,fp=fp_size,n_config=n_config,curriculum_step=curriculum_step)
UMAP_physics(ball_parameter='xradius',df=data_xradius,fp=fp_xradius,n_config=n_config,curriculum_step=curriculum_step)
UMAP_physics(ball_parameter='yradius',df=data_yradius,fp=fp_yradius,n_config=n_config,curriculum_step=curriculum_step)

def takeXradius(dict):
    return dict['xradius']

def takeYradius(dict):
    return dict['yradius']

n_config = 100
curriculum_step = '12_period_5'
data_xradius = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/Xradius_'+curriculum_step+'_%s-configs' %n_config,'rb'))[0]
data_xradius.sort(key=takeXradius)

data_yradius = pickle.load(open('/home/ingster/Bureau/SIL-BigResults/Yradius_'+curriculum_step+'_%s-configs' %n_config,'rb'))[0]
data_yradius.sort(key=takeYradius)

fp_xradius = os.path.join(ROOT_DIR,'SIL-Results/UMAP/Xradius/')
fp_yradius = os.path.join(ROOT_DIR,'SIL-Results/UMAP/Yradius/')
UMAP_physics(ball_parameter='xradius',df=data_xradius,fp=fp_xradius,n_config=n_config,curriculum_step=curriculum_step)
UMAP_physics(ball_parameter='yradius',df=data_yradius,fp=fp_yradius,n_config=n_config,curriculum_step=curriculum_step)