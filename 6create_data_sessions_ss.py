# main_pipeline.py

import pandas as pd
from utils_tt import *




sheet_ids = {
    "SKIEUR": "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU"
    #"FETA": "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU", 
    #"MMELOIK": "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU", 
    #"NAPOLEON": "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU", 
    #"HERCULE": "1sFatSTXO0j3OONKstz7YN-mM04kNMjk_r7zo951yicU" 
}

# -------- usage ----------

base_data_path = '/auto/data6/eTheremin/SKIEUR/'
save_base_path = base_data_path

# Paramètres pour la création des spikes
fs = 30e3
t_pre = 0.2
t_post = 0.5
bin_width = 0.02
freq_min = 3
mock = True



def get_sessions(sheet_name, sheet_id, session_filter=None):
    """Retourne les chemins de toutes les sessions valides d'une feuille Google Sheet."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)

    # filtrage basique : uniquement celles marquées "yes"
    filtered = df[df['use'] == 'yes']

    # appliquer un filtre supplémentaire si demandé
    if session_filter is not None:
        filtered = filtered[filtered['type'].isin(session_filter)]

    sessions = filtered['session'].tolist()

    # règles pour headstages
    if sheet_name == "HERCULE":
        root_directory = f'/auto/data6/eTheremin/{sheet_name}/'
        headstages = [0]
    elif sheet_name == "ALTAI":
        root_directory = f'/auto/data2/eTheremin/{sheet_name}/'
        headstages = [0, 1]
    else:
        root_directory = f'/auto/data6/eTheremin/{sheet_name}/'
        headstages = [0,1]

    # construire les chemins
    paths = [f"{root_directory}{s}/headstage_{hs}/" for s in sessions for hs in headstages]
    return paths

sessions = []
for sheet_name, sheet_id in sheet_ids.items():
    sessions.extend(get_sessions(sheet_name, sheet_id, session_filter=['playback']))


for session in sessions:
    try:
        create_data_features_new_version_spike_sorted(session, bin_width, fs, mock=mock)

    except Exception as e:
        print(f"Session {session} already spike sorted: {e}")
        pass