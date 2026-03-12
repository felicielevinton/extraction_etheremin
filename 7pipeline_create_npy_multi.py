from functions_get_data import *
import numpy as np
from utils_extraction import get_session_type_final
from utils_tt import *
from spike_sorting import *
from utils import *
import pandas as pd

# ARGUMENTS
sr = 30e3
t_pre = 0.5#0.2
t_post = 0.50#0.300
bin_width = 0.005#0.005 ou 0.00625 c'est la fréquence d'échantillonnage des positions
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)  

mock=True

root_directory_base = '/auto/data6/eTheremin/'

sheet_ids = {
    #"SKIEUR": "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU",
    #"HERCULE": "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU", 
    #"MMELOIK": "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU", 
   # "ALTAI" : "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU", 
    "SKIEUR" : "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU" 
}

sessions_pmc, sessions_a1 = [], []

# On filtre à la fois 'playback' et 'playback_block'
for sheet_name, sheet_id in sheet_ids.items():
    sessions_pmc.extend(get_sessions_pmc(
        sheet_name,
        sheet_id,
        session_filter=['playback'],
        timeline='all', 
        mounted = 'auto'
    ))
    sessions_a1.extend(get_sessions_aone(
        sheet_name,
        sheet_id,
        session_filter=['playback'],
        timeline='all', 
        mounted = 'auto'
    ))



# vérifier qu'il n existe pas de tt.pkl, s'il n''existe pas alors on le créée, sinon c'est pas la peine.
# get_session_type pour le session_type

 
#2. Créer le data.npy et features.npy
#create_data_features_mock(path+'headstage_0', bin_width, sr, mock=mock)
for path in sessions_a1 : 
    try : 
        create_data_features_new_version(path, bin_width, sr, mock=mock)
    except:
        pass
# version test de spike_sorting

#create_data_features_ss(path+'headstage_0/', bin_width, fs, mock=False)




