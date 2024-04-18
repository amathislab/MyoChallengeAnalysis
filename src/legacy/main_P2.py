import os
from definitions import ROOT_DIR
from functions import *
import pickle

# Generate data for P2 model on env 2 (saved outside the pushed file, as very heavy files)

if __name__ == '__main__':
    curriculum_step = '32_phase_2_smaller_rate_resume'
    n_config = 2
    num_ep = 100
    env_name = 'CustomMyoBaodingBallsP2'
    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/env.pkl",
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/32_phase_2_smaller_rate_resume/model.zip",
    )

    data_mass = get_layers(ball_parameter='mass',envir=PATH_TO_NORMALIZED_ENV,net=PATH_TO_PRETRAINED_NET,env_name=env_name,num_ep=num_ep,n_config=n_config)
    fp_mass = open('/home/ingster/Bureau/SIL-BigResults/Mass_'+curriculum_step+'_%s-configs' %n_config, 'wb')
    pickle.dump(data_mass,fp_mass)
    fp_mass.close()

    data_size = get_layers(ball_parameter='size',envir=PATH_TO_NORMALIZED_ENV,net=PATH_TO_PRETRAINED_NET,env_name=env_name,num_ep=num_ep,n_config=n_config)
    fp_size = open('/home/ingster/Bureau/SIL-BigResults/Size_'+curriculum_step+'_%s-configs' %n_config, 'wb')
    pickle.dump(data_size,fp_size)
    fp_mass.close()
  
    data_xradius = get_layers(ball_parameter='xradius',envir=PATH_TO_NORMALIZED_ENV,net=PATH_TO_PRETRAINED_NET,env_name=env_name,num_ep=num_ep,n_config=n_config)
    fp_xradius = open('/home/ingster/Bureau/SIL-BigResults/Xradius_'+curriculum_step+'_%s-configs' %n_config, 'wb')
    pickle.dump(data_xradius,fp_xradius)
    fp_xradius.close()

    data_yradius = get_layers(ball_parameter='yradius',envir=PATH_TO_NORMALIZED_ENV,net=PATH_TO_PRETRAINED_NET,env_name=env_name,num_ep=num_ep,n_config=n_config)
    fp_yradius = open('/home/ingster/Bureau/SIL-BigResults/Yradius_'+curriculum_step+'_%s-configs' %n_config, 'wb')
    pickle.dump(data_yradius,fp_yradius)
    fp_yradius.close()