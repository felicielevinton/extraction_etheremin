from kneed import DataGenerator, KneeLocator
from quick_extract import *
from get_data import *
from load_rhd import *
import matplotlib.pyplot as plt
from ExtractRecordings.manual.simple_sort import*
import pandas as pd
from PostProcessing.tools.utils import *
from matplotlib.colors import ListedColormap, Normalize
from format_data import *
from skimage import measure
import matplotlib.colors as colors
from scipy.signal import find_peaks
from extract_data_total import *
import PostProcessing.tools.utils as ut
from PostProcessing.tools.extraction import *
from get_data import *
import re
import numpy as np
import os
import glob
import warnings
from copy import deepcopy
import json
import pickle
sr = 30e3
t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.02
#bin_width = 0.02
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
max_freq = 2
min_freq=2 #3 for A1
threshold = 2 #threshold for contour detection 3.2 is good


def get_triggers(path, analog_line):
    """"
    Récupérer les triggers en tracking
    
     - analog_line : numero de la ligne de triggers analogique. 
      (tracking0, playback1 et mock3 pour les xp de types Playback)
    """
    an_triggers = np.load(os.path.join(path, "analog_in.npy"))
    an_times = ut.extract_analog_triggers_compat(an_triggers[analog_line])
    frequencies, tones_total, triggers_spe, tag = get_data(path, trigs=an_times)
    return an_times, tones_total


def get_triggers_tracking(path):
    """"
    Récupérer les triggers en tracking
    """
    an_triggers = np.load(os.path.join(path, "analog_in.npy"))
    an_times = ut.extract_analog_triggers_compat(an_triggers[1])
    frequencies, tones_total, triggers_spe, tag = get_data(path, trigs=an_times)
    return an_times, tones_total

def get_triggers_playback(path):
    """
    Récupérer les triggers en playback

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    an_triggers = np.load(os.path.join(path, "analog_in.npy"))
    an_times = ut.extract_analog_triggers_compat(an_triggers[0])
    frequencies, tones_total, triggers_spe, tag = get_data(path, trigs=an_times)
    return an_times, tones_total

def get_triggers_tonotopy(path):
    an_triggers = np.load(os.path.join(path, "analog_in.npy"))
    an_times = ut.extract_analog_triggers_compat(an_triggers[0])
    frequencies, tones_total, triggers_spe, tag = get_data(path, trigs=an_times, tonotopy_only = True)
    return an_times, tones_total
    
    get_data(folder, trigs, tonotopy_only=True, tracking_only=False)

def create_tones_triggers_and_condition(path):
    """
    créer un fichier .pkl dans lequel j'ai : 
     - fréquences jouées (tones)
     - temps du triggers (triggers)
     - condition (condition) 0 pour tracking, 1 pour playback

    Args:
        path (_type_): _description_
    """
    triggers_tr, tones_total_tr = get_triggers_tracking(path)
    triggers_pb, tones_total_pb = get_triggers_playback(path)
    
    condition_tr = np.zeros(len(triggers_tr))
    condition_pb = np.ones(len(triggers_pb))
    
    trig_times = np.concatenate((triggers_tr, triggers_pb))
    tones = np.concatenate((tones_total_tr, tones_total_pb))
    condition = np.concatenate((condition_tr, condition_pb))
    
    sorted_indices = np.argsort(trig_times)
    sorted_triggers = trig_times[sorted_indices]
    sorted_tones = tones[sorted_indices]
    sorted_condition = condition[sorted_indices]
    
    tt = {
    'tones': sorted_tones,
    'triggers': sorted_triggers,
    'condition': sorted_condition
    }
    file_path = path+'/tt.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(tt, file)
    print('tt.pkl created')
    
    return None


def create_tones_triggers_and_condition_V2(path, session_type):
    """
    BEST VERSION
    créer un fichier .pkl dans lequel j'ai : 
     - fréquences jouées (tones)
     - temps du triggers (triggers)
     - condition (condition) 0 pour tracking, 1 pour playback

    Args:
        path (_type_): _description_
    """
    
    if session_type=='Playback':
        triggers_tr, tones_total_tr = get_triggers_tracking(path, )
        triggers_pb, tones_total_pb = get_triggers_playback(path)
        
        condition_tr = np.zeros(len(triggers_tr))
        condition_pb = np.ones(len(triggers_pb))
        
        trig_times = np.concatenate((triggers_tr, triggers_pb))
        tones = np.concatenate((tones_total_tr, tones_total_pb))
        condition = np.concatenate((condition_tr, condition_pb))
        
    elif session_type=='Tonotopy' or session_type=='PbOnly' or  session_type=='TrackingOnly' : 
        triggers_pb, tones_total_pb = get_triggers_tonotopy(path)
        condition_pb = np.ones(len(triggers_pb))
        
        trig_times = triggers_pb
        tones =  tones_total_pb
        condition = condition_pb
    
    sorted_indices = np.argsort(trig_times[:len(tones)])
    sorted_indices = sorted_indices[:-1]
    sorted_triggers = trig_times[sorted_indices]
    sorted_tones = tones[sorted_indices]
    sorted_condition = condition[sorted_indices]
    
    tt = {
    'tones': sorted_tones,
    'triggers': sorted_triggers,
    'condition': sorted_condition
    }
    file_path = path+'/tt.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(tt, file)
    print('tt.pkl created')
    
    return None



def create_tones_triggers_and_condition_V3(path, session_type):
    """
    BEST VERSION
    La c'est la version pour les xp tracking/playback avec les an_trig qui sont avec ce mapping : 
     - an0 = tracking
     - an1 = playback
     - an3 = mock
    créer un fichier .pkl dans lequel j'ai : 
     - fréquences jouées (tones)
     - temps du triggers (triggers)
     - condition (condition) 0 pour tracking, 1 pour playback

    Args:
        path (_type_): _description_
    """
    
    if session_type=='Playback':
        triggers_tr, tones_total_tr = get_triggers(path, analog_line=0)
        triggers_pb, tones_total_pb = get_triggers(path, analog_line=1)
        
        condition_tr = np.zeros(len(triggers_tr))
        condition_pb = np.ones(len(triggers_pb))
        
        trig_times = np.concatenate((triggers_tr, triggers_pb))
        tones = np.concatenate((tones_total_tr, tones_total_pb))
        condition = np.concatenate((condition_tr, condition_pb))
        
    elif session_type=='Tonotopy' or session_type=='PbOnly' or  session_type=='TrackingOnly' : 
        triggers_pb, tones_total_pb = get_triggers_tonotopy(path)
        condition_pb = np.ones(len(triggers_pb))
        
        trig_times = triggers_pb
        tones =  tones_total_pb
        condition = condition_pb
    
    sorted_indices = np.argsort(trig_times[:len(tones)])
    sorted_indices = sorted_indices[:-1]
    sorted_triggers = trig_times[sorted_indices]
    sorted_tones = tones[sorted_indices]
    sorted_condition = condition[sorted_indices]
    
    tt = {
    'tones': sorted_tones,
    'triggers': sorted_triggers,
    'condition': sorted_condition
    }
    file_path = path+'/tt.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(tt, file)
    print('tt.pkl created')
    
    return None


def create_tones_triggers_and_condition_V4(path, session_type):
    """
    BEST VERSION
    La c'est la version pour les xp tracking/playback avec les an_trig qui sont avec ce mapping : 
     - an0 = tracking
     - an1 = playback
     - an3 = mock
    créer un fichier .pkl dans lequel j'ai : 
     - fréquences jouées (tones)
     - temps du triggers (triggers)
     - condition (condition) 0 pour tracking, 1 pour playback
     - block (le block)

    Args:
        path (_type_): _description_
    """
    
    if session_type=='Playback':
        triggers_tr, tones_total_tr = get_triggers(path, analog_line=0)
        triggers_pb, tones_total_pb = get_triggers(path, analog_line=1)
        triggers_mck, tones_total_mck = get_triggers(path, analog_line=3)
        
        condition_tr = np.zeros(len(triggers_tr))
        condition_pb = np.ones(len(triggers_pb))

        
        trig_times = np.concatenate((triggers_tr, triggers_pb))
        trig_times_mck = triggers_mck
        tones = np.concatenate((tones_total_tr, tones_total_pb))
        tones_mck = tones_total_mck
        condition = np.concatenate((condition_tr, condition_pb))
        
    elif session_type=='Tonotopy' or session_type=='PbOnly' or  session_type=='TrackingOnly' : 
        triggers_pb, tones_total_pb = get_triggers_tonotopy(path)
        condition_pb = np.ones(len(triggers_pb))
        
        trig_times = triggers_pb
        tones =  tones_total_pb
        condition = condition_pb
    
    sorted_indices = np.argsort(trig_times[:len(tones)])
    sorted_indices = sorted_indices[:-1]
    sorted_triggers = trig_times[sorted_indices]
    sorted_tones = tones[sorted_indices]
    sorted_condition = condition[sorted_indices]
    
    tt = {
    'tones': sorted_tones,
    'triggers': sorted_triggers,
    'condition': sorted_condition, 
    'mock_triggers' : trig_times_mck,
    'tones_triggers' : tones_mck
    }
    file_path = path+'/tt.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(tt, file)
    print('tt.pkl created')
    
    return None






def create_data_features(path, bin_width, fs):
    
    #data = pd.read_hdf(path+'/data.h5')
    
    #file = path+'/recording_length.bin'
    #with open(file, 'rb') as file:
       # recording_length = file.read()
    #recording_length = recording_length.decode('utf-8')

    # Extract only the numbers using a simple filter
   # recording_length = int(''.join(filter(str.isdigit, recording_length)))

    #print(recording_length)
    #extraire recording_length OK ca marche

 
    spk_clusters = np.load(path+'/spike_clusters.npy', allow_pickle=True)
    spk_times = np.load(path+'/spike_times.npy', allow_pickle=True)

    clusters = {}
    for value, cluster in zip(spk_times, spk_clusters):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(value)

    ##NEURO
    t_spk, c_spk = [], [] #spike times, cluster
    #for cluster in range(spike.get_n_clusters()):
    for cluster in range(32):
        t_spk.append(clusters[cluster]) #spikes times
        c_spk.append(np.full_like(t_spk[cluster], cluster))
    t_spk = np.hstack(t_spk)
    c_spk = np.hstack(c_spk)

    # mettre en secondes 
    t_spk = t_spk/fs
    c_spk = c_spk


    ## faire les bins : 
    min_value = t_spk.min()  # Get the minimum value of 'spike_time'
    max_value = t_spk.max()  # Get the maximum value of 'spike_time'

    bins = np.arange(min_value, max_value + bin_width, bin_width)  # Define custom bin edges

    ## histogramme par cluster
    unique_clusters = np.unique(c_spk)

    histograms_per_cluster = {}

    for cluster in unique_clusters:
        spike_times_cluster = [time for time, clus in zip(t_spk, c_spk) if clus == cluster]
        # Now spike_times_cluster contains spike times for the current cluster
        
        # Perform histogram for the current cluster
        hist, bin_edges = np.histogram(spike_times_cluster, bins=bins)
        histograms_per_cluster[cluster] = (hist, bin_edges)

    print(histograms_per_cluster)
    data = [histograms_per_cluster[key][0] for key in histograms_per_cluster]
    np.save(path+f'/data_{bin_width}.npy', data)


    #### TRIGGERS
    tt_path = path+'/tt.pkl'
    with open(tt_path, 'rb') as file:
        tt = pickle.load(file)
        
    #sorted_indices = np.argsort(tt['triggers'])
    #sorted_triggers = tt['triggers'][sorted_indices]
    #sorted_tones = tt['tones'][sorted_indices]
    #sorted_condition = tt['condition'][sorted_indices]
    
    try :
        t_stim = tt['triggers']/fs
    except:
        t_stim = [x / fs for x in tt['triggers']]
    f_stim = tt['tones']
    type_stim = tt['condition']
    
    #attention
    t_stim = np.array(t_stim, dtype=float)
    type_stim = np.array(type_stim, dtype=float)
    
    unique_tones = sorted(np.unique(f_stim))
    
    print(f"Shape of t_stim: {t_stim.shape}")
    print(f"Shape of f_stim: {f_stim.shape}")
    print(f"Shape of bins: {bins.shape}")

    #need to interpolate between two stims to get the frequency in between
    # 1. True stims
    stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
    interpolated_freq = np.zeros(len(bins) - 1)

    previous_frequency = None
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]

        # Check if any stimuli fall within the current bin
        stimuli_in_bin = (t_stim >= bin_start) & (t_stim < bin_end)
        
        print(f"stimuli_in_bin indices: {np.where(stimuli_in_bin)}")
        print(f"f_stim values in bin {i}: {f_stim[stimuli_in_bin]}")
        if np.any(stimuli_in_bin):
            # If stimuli are present, set stimulus_presence to True for this bin
            stimulus_presence[i] = True

            # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
            # You can simply take the frequency of the first stimulus within the bin
            interpolated_freq[i] = f_stim[stimuli_in_bin][0]
            previous_frequency = interpolated_freq[i]  # Update previous frequency
        else:
            # If no stimulus in the bin, set bin_frequencies to the previous frequency
            if previous_frequency is not None:
                interpolated_freq[i] = previous_frequency
                
    interpolated_type_stim = np.interp(bins, t_stim, type_stim)



    # Create a dictionary to store information for each time bin
    features = {}
    for i, bin in enumerate(bins[:-1]):
        features[bin] = {
            'Played_frequency': interpolated_freq[i],
            'Condition': interpolated_type_stim[i],
            'Frequency_changes': stimulus_presence[i]
        }
        
        
        
    features = list(features.values())
    np.save(path+f'/features_{bin_width}.npy', features)
    
    np.save(path+'/unique_tones.npy', unique_tones)

    #with open(path+'/features.json', 'w') as json_file:
        #json.dump(features, json_file)   
        
        
    print('all izz well')
        

