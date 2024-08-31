import os
from definitions import ROOT_DIR
from legacy.functions import *
import pickle

# Generate data for P1 model on env 1 (saved outside the pushed file, as very heavy files)

if __name__ == '__main__':
    curriculum_step = '12_period_5'
    n_config = 100
    env_name = 'CustomMyoBaodingBallsP2'
    PATH_TO_NORMALIZED_ENV = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/12_period_5/env.pkl",
    )
    PATH_TO_PRETRAINED_NET = os.path.join(
        ROOT_DIR,
        "trained_models/curriculum_steps_complete_baoding_winner/12_period_5/model.zip",
    )

    data_xradius = get_layers_P1(ball_parameter='xradius',envir=PATH_TO_NORMALIZED_ENV,net=PATH_TO_PRETRAINED_NET,env_name=env_name,n_config=n_config)
    fp_xradius = open('/home/ingster/Bureau/SIL-BigResults/Xradius_'+curriculum_step+'_%s-configs' %n_config, 'wb')
    pickle.dump(data_xradius,fp_xradius)
    fp_xradius.close()

    data_yradius = get_layers_P1(ball_parameter='yradius',envir=PATH_TO_NORMALIZED_ENV,net=PATH_TO_PRETRAINED_NET,env_name=env_name,n_config=n_config)
    fp_yradius = open('/home/ingster/Bureau/SIL-BigResults/Yradius_'+curriculum_step+'_%s-configs' %n_config, 'wb')
    pickle.dump(data_yradius,fp_yradius)
    fp_yradius.close()