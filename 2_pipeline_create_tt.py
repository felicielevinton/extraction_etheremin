from functions_get_data import *
import numpy as np
from utils_extraction import get_session_type_final
from utils_tt import *

# ARGUMENTS
sr = 30e3
t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.005
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)

#path = '/auto/data2/eTheremin/ALTAI/ALTAI_20240726_SESSION_01/'
path = '/auto/data6/eTheremin/SKIEUR/SKIEUR_20260313_SESSION_00/'
#path = '/auto/data2/eTheremin/ALTAI/ALTAI_20240822_SESSION_00/'
#session = 'ALTAI_20240809_SESSION_00'
#path = '/Volumes/data2/eTheremin/ALTAI/'+ session + '/'
#session_type = get_session_type_final(path)
session_type = 'Playback' #TrackingOnly ou PbOnly ou Playback MappingChange
#session_type = 'Playback' #TrackingOnly ou PbOnly ou Playback MappingChange ou Playback



#create_tones_triggers_and_condition_V3(path, session_type)
if session_type == 'TrackingOnly' or session_type == 'PbOnly':
    create_tt_no_mock(path, mock=False)
    #create_tt(path) # peut etre ici que ca va beuguer
elif session_type == 'Playback':
    create_tt(path)
elif session_type =='MappingChange':
    create_tt_mc(path)
elif session_type =='Tonotopy':
    create_tt_tono(path, 0)

print(f'tt.pkl created,  for {path}')