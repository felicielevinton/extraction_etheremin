import spikeinterface.full as si
#import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spiketoolkit
import spikeinterface.widgets as sw
import numpy as np
from utils_extraction import *
import numpy as np
import spikeinterface
import zarr as zr
import os
from  pathlib import Path
import tqdm
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from probeinterface import Probe, ProbeGroup
import matplotlib
import pickle
matplotlib.use('Agg')
sr=30e3
fs = sr


root =  '/auto/data6/eTheremin/SKIEUR/SKIEUR_20260312_SESSION_01/'
path = root+'headstage_1/' 

neural_data = np.load(path +'/neural_data.npy')
sig = neural_data


n_cpus = os.cpu_count()

#s'il y a deja un gc on va le lire pour retirer les canaux morts et parfaire la CMR
#if os.path.isfile(path + 'good_clusters.npy'):
    #gc = np.load(path + 'good_clusters.npy')
    # ici je regroupe tous les numéros de canaux qui ne sont pas dans good clusters.
    #channels_to_remove = [num for num in range(32) if num not in gc]
    #print(channels_to_remove)
#else : 
gc = np.arange(32)
#channels_to_remove = None

full_raw_rec = se.NumpyRecording(traces_list=np.transpose(sig), sampling_frequency=sr)
# Convertir le type de données avant d'appliquer le filtre
full_raw_rec = full_raw_rec.astype('float32') 

print("Canaux avant suppression:", full_raw_rec.get_channel_ids())
raw_rec = full_raw_rec
#raw_rec = full_raw_rec.remove_channels(channels_to_remove) #,"CH12", "CH13","CH14", "CH15", "CH16", "CH17", "CH18", "CH19", "CH21", "CH22", "CH23","CH31"  ])
print("Canaux après suppression:", raw_rec.get_channel_ids())
recording_cmr = si.common_reference(raw_rec, reference='global', operator='median')
recording_f = si.bandpass_filter(recording_cmr, freq_min=300, freq_max=3000)
#np.save(path+'/recording_f.npy', recording_f.get_traces())
n_cpus = os.cpu_count()
n_jobs = n_cpus - 4
job_kwargs = dict(chunk_duration='5s', n_jobs=n_jobs, progress_bar=True)

peaks = detect_peaks(
        recording_f,
        method='by_channel',
        gather_mode="memory",
        peak_sign='neg',#neg
        detect_threshold= 3,#3,#3 ,#3.5,  # thresh = 3.32 for burrata # 3 pour ALTAI 4 pour oscypeck, 3 pour Napoleon
        exclude_sweep_ms=1, #avant c'etait 0.1 je teste à 1
        noise_levels=None,
        random_chunk_kwargs={},

        **job_kwargs,
)

peaks_array = np.array(peaks)
spk_times = peaks_array['sample_index'].tolist()
spk_clusters = peaks_array['channel_index'].tolist()
np.save(path+'/spike_times.npy',spk_times )
np.save(path+'/spike_clusters.npy',spk_clusters)

clusters = {}
for value, cluster in zip(spk_times, spk_clusters):
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append(value)
for cluster, values in clusters.items():
    print(f"Cluster {cluster}: {len(values)}")
    
#triggers
pkl_path = path+'tt.pkl'

with open(pkl_path, 'rb') as file:
    triggers_data = pickle.load(file)
an_times =  triggers_data['triggers']


#PLOT
t_pre = 0.2#
t_post = 0.50#0.300
bin_width = 0.02
# Créer les bins de temps"
psth_bins = np.arange(-t_pre, t_post, bin_width)
window = [-t_pre, t_post]

import numpy as np
import matplotlib.pyplot as plt

def compute_psth(spike_times, stimulus_times, bin_size, window):
    # Combine all spikes relative to stimulus times
    try:
        stimulus_times = stimulus_times/sr
    except:
        stimulus_times = [x / fs for x in stimulus_times]
    spike_times = spike_times/sr
    all_spikes = []
    for stim_time in stimulus_times:
        relative_spikes = spike_times - stim_time
        all_spikes.extend(relative_spikes[(relative_spikes >= window[0]) & (relative_spikes <= window[1])])
    
    # Create histogram
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    hist, bin_edges = np.histogram(all_spikes, bins=bins)
    
    # Normalize to get the firing rate
    psth = hist / (len(stimulus_times) * bin_size)
    #psth=hist
    return psth, bin_edges

num_plots, num_rows, num_columns = get_better_plot_geometry(gc)


fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Heatmaps clusters', y=1.02)
plt.subplots_adjust() 


stim_times = an_times
    #
for cluster in range(num_plots):
    if cluster < num_plots:
        row, col = get_plot_coords(cluster)
        spike_times = np.array(clusters[cluster])
        psth,edges = compute_psth(spike_times, stim_times, bin_width, window)
        axes[row, col].plot(psth_bins, psth)
        axes[row, col].axvline(0, c = 'black', linestyle='--')
        axes[row, col].set_title(f'Cluster {cluster}')
fig.tight_layout()
fig.savefig(path+'psth_figure_spikeinterface.png') 






# ca sert à rien mais au cas où

def get_fma_probe():
    """ Implements the FMA probe using Probeinterface.
    """
    ### The following distances are in mm:
    inter_hole_spacing = 0.4  # along one row
    inter_row_spacing = np.sqrt(0.4**2-0.2**2)  # between row/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240419_SESSION_01/headstage_1/psth_figure_spikeinterface.png

    # We need to remove the ground from these positions:
    mask = np.zeros((16, 2), dtype=bool)
    mask[[0, 15], [0, 1]] = True
    positions = positions[np.logical_not(mask.reshape(-1))]

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions,
                       shapes='circle', shape_params={'radius': 100})
    polygon = [(0, 0), (0, 16000), (800, 16000), (800, 0)]

    probe.set_device_channel_indices(mapping)

    return probe
