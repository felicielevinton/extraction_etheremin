
from zetapy.dependencies import *
from kneed import DataGenerator, KneeLocator
from quick_extract import *
from get_data import *
from load_rhd import *
import matplotlib.pyplot as plt
from ExtractRecordings.manual.simple_sort import*
import pandas as pd
from PostProcessing.tools.utils import *
from multiprocessing import Pool
import json
from tqdm import tqdm
import pickle
fs = 30e3
n_blocs = 1

#### Attention, fichier à reprendre une fois qu'on aura bien lu le json automatiquement

#### et surtout séparer les fonctions et les utilitaires pour byobu


## fonctions utiles 
def extract_times_triggers(file_path):
    """"
    on veut récupérer les temps des triggers
    input : tt --> np.load(tt.npz)
    output : un tableau contenant les temps de tous les triggers
    c: condition =0 si tracking, 1 si playback
    """
    with open(file_path, 'rb') as file:
        tt = pickle.load(file)
    t_triggers = np.array(tt['triggers'])
    t_condition = np.array(tt['condition'])

    min_len = min(len(t_triggers), len(t_condition))

    t_triggers = t_triggers[:min_len]
    t_condition = t_condition[:min_len]

    print(len(t_triggers), len(t_condition))

    t_triggers_playback = t_triggers[t_condition == 1] # if i want only playback triggers
    
    return t_triggers_playback#t_triggers


def do_zetapy(x, cluster, triggers):
    
    
    triggers = np.hstack(triggers)
        
    a, b = getZeta(x * (1 / 30000), triggers * (1 / 30000))
    if a < 0.001:
        return cluster
        # good_clusters.append(cluster)
    else:
        return -1


def do_zetapy_wrapper(args):
    return do_zetapy(*args)


def check_responsiveness(triggers, spikes, path, clusters=None, tag=None):
    """
    Vérifie qu'une unité répond aux stimuli.
    """
    good_clusters = list()
    if clusters is not None:
        iterator = clusters
    else:
        #iterator = list(range(spikes.get_n_clusters()))
        iterator = list(range(len(spikes)))
        
    
    t = np.hstack(triggers)
    #args = [(spikes.get_spike_times(cluster=cluster), cluster, t) for cluster in iterator]
    args = [(np.array(spk[cluster]), cluster, t) for cluster in iterator]
    with Pool(processes=32) as pool:
        good_clusters = list(tqdm(pool.imap_unordered(do_zetapy_wrapper, args), total=len(iterator)))
    
    # for cluster in tqdm(iterator):
    #     x = spikes.get_spike_times(cluster=cluster)
    #     triggers = np.hstack(triggers)
    #     
    #     a, b = getZeta(x * (1 / 30000), triggers * (1 / 30000))
    #     if a < 0.001:
    #         good_clusters.append(cluster)
    good_clusters = np.array(good_clusters)
    good_clusters = good_clusters[~np.equal(good_clusters, -1)]
    
    if tag is not None:
        filename = f"good_clusters_{tag}.npy"
    else:
        filename = "good_clusters.npy"
    np.save(path+filename, good_clusters)
    return good_clusters





#### ici ca commence 
path = '/auto/data6/eTheremin/SKIEUR/SKIEUR_20260312_SESSION_01/headstage_0/' #  relance



# charger la recording length (nécessaire pour charger les spikes)    
#file = path+'recording_length.bin'
#with open(file, 'rb') as file:
    #recording_length = file.read()
#recording_length = recording_length.decode('utf-8')

    # Extract only the numbers using a simple filter
#recording_length = int(''.join(filter(str.isdigit, recording_length))) 

# Extraire les spikes
#spk = Spikes(path, recording_length=int(recording_length))  
spike_times = np.load(path+'/spike_times.npy', allow_pickle=True)
spike_clusters = np.load(path+'/spike_clusters.npy', allow_pickle=True)

spk = {}
for value, cluster in zip(spike_times, spike_clusters):
    if cluster not in spk:
        spk[cluster] = []
    spk[cluster].append(value)
#Extraire les temps des triggers
trig_times = extract_times_triggers(path+'tt.pkl')


#Zeta Test : 
check_responsiveness(trig_times, spk, path, clusters=None, tag=None)

print('all izz well')



