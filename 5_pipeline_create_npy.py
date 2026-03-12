from functions_get_data import *
import numpy as np
from utils_extraction import get_session_type_final
from utils_tt import *

# ARGUMENTS
sr = 30e3
t_pre = 0.5#0.2
t_post = 0.50#0.300
bin_width = 0.02#0.005 ou 0.00625 c'est la fréquence d'échantillonnage des positions
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)  


#session = 'MMELOIK_20241029_SESSION_00'
#path = '/Volumes/data6/eTheremin/MMELOIK/'+ session + '/'
path = '/auto/data6/eTheremin/SKIEUR/SKIEUR_20260310_SESSION_00/'
mock=True
#session_type = get_session_type_final(path)
#print(session_type)
#session_type = 'Playback' #TrackingOnly ou PbOnly



# vérifier qu'il n existe pas de tt.pkl, s'il n''existe pas alors on le créée, sinon c'est pas la peine.
# get_session_type pour le session_type

 
#2. Créer le data.npy et features.npy
#create_data_features_mock(path+'headstage_0', bin_width, sr, mock=mock)
create_data_features_new_version(path+'headstage_1/', bin_width, sr, mock=mock)
# version test de spike_sorting

#create_data_features_ss(path+'headstage_0/', bin_width, fs, mock=False)




