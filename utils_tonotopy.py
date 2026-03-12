from load_rhd import *
from quick_extract import *
from get_data import *
import PostProcessing.tools.heatmap as hm
from get_data import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *

import spikeinterface.full as si
#import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spiketoolkit

def load_rhd(path, save_path, digital=True, analog=True, accelerometer=True, filtered=True, export_to_dat=False):
    # Load data
    a = load.read_data(path, data_in_float=False)

    # Construct file paths based on options
    if filtered:
        neural_data_fn = "filtered_neural_data"
    else:
        neural_data_fn = "neural_data"

    if not export_to_dat:
        neural_data_fn += ".npy"
        np.save(os.path.join(save_path, neural_data_fn), a["amplifier_data"])
    else:
        neural_data_fn += ".dat"
        a["amplifier_data"].tofile(os.path.join(save_path, neural_data_fn), sep="", format="%U")

    # Save digital data if specified
    if digital:
        np.save(os.path.join(save_path, "dig_in.npy"), a["board_dig_in_data"])

    # Save analog data if specified
    if analog:
        np.save(os.path.join(save_path, "analog_in.npy"), a["board_adc_data"])

    # Save accelerometer data if specified
    if accelerometer:
        np.save(os.path.join(save_path, "accelerometer.npy"), a["aux_input_data"])

    return neural_data_fn  # Return the filename of the saved neural data



def filter_and_cmr(neural_data, sampling_rate, save_path):
    recording = se.NumpyRecording(traces_list=neural_data, sampling_frequency=sampling_rate)
    # Apply high-pass filter
    recording_highpass = spiketoolkit.highpass_filter(recording=recording, freq_min=300.)

    # Apply common median reference (CMR)
    recording_cmr = spiketoolkit.common_reference(recording=recording_highpass, reference='global', operator='median')
    traces = recording_cmr.get_traces()
    filtered_neural_data = [traces[channel_index] for channel_index in range(32)]
    np.save(save_path+'refiltered_neural_data.npy',filtered_neural_data )


def filter_and_cmr_chunked(neural_data, sampling_rate, save_path, chunk_size):
    """
    
    Fonction pour filtrer et appliquer le CMR sur le signal neural non filtré mais attention
    ici on découpe le signal en sous signaux pour aider la mémoire de l'ordi à pas mourir.
    Args:
        neural_data (_type_): _description_
        sampling_rate (_type_): _description_
        save_path (_type_): _description_
        chunk_size (_type_): taille des sous_tableaux
    """
    num_channels = neural_data.shape[0]
    num_samples = neural_data.shape[1]
    
    # Initialize an array to store the processed data
    filtered_neural_data = np.zeros((num_channels, num_samples))
    
    # Process data in chunks
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        chunk = neural_data[:, start:end]
        
        recording = se.NumpyRecording(traces_list=chunk, sampling_frequency=sampling_rate)
        
        # Apply high-pass filter
        recording_highpass = spiketoolkit.highpass_filter(recording=recording, freq_min=300.)
        
        # Apply common median reference (CMR)
        recording_cmr = spiketoolkit.common_reference(recording=recording_highpass, reference='global', operator='median')
        
        # Get the processed traces
        traces = recording_cmr.get_traces()
        
        # Store the processed chunk
        filtered_neural_data[:, start:end] = traces
    
    # Save the processed data
    np.save(save_path + 'refiltered_neural_data.npy', filtered_neural_data)


def quick_extract(path,  mode="relative", threshold=-3.7):
    root_dir = os.path.split(path)[0]
    data = ss.load_spike_data(path)
    print(data)
    channels = np.arange(data.shape[0])
    spike_times = np.empty(0, dtype=np.uint64)
    spike_clusters = np.empty(0, dtype=np.int32)
    assert mode in ("relative", "absolute"), "Mode is relative (from RMS calculation) or absolute (threshold in µV)."
    to_float = False
    if data.dtype == np.uint16:
        to_float = True
    if mode == "absolute":
        threshold = -60
    else:
        threshold = threshold
    for i, channel in tqdm(enumerate(channels)):
        print(i, channel)
        if to_float:
            chan = np.multiply(0.195, (data[channel].astype(np.int32) - 32768))
            spk, _ = ss.thresholder(chan, mode, threshold=threshold)
        else:
            spk, _ = ss.thresholder(data[channel], mode, threshold=threshold)
        cluster = np.full_like(spk, i)
        print(i)
        spike_times = np.hstack((spike_times, spk))
        spike_clusters = np.hstack((spike_clusters, cluster))
    idx = np.argsort(spike_times)
    spike_times = spike_times[idx]
    spike_clusters = spike_clusters[idx]
    print(spike_clusters)
    np.save(os.path.join(root_dir, "spike_times.npy"), spike_times)
    np.save(os.path.join(root_dir, "spike_clusters.npy"), spike_clusters)
    


def process_data(path):
    spk = ut.Spikes(path)
    an_triggers = np.load(os.path.join(path, "analog_in.npy"))
    an_times = ut.extract_analog_triggers_compat(an_triggers[0])
    frequencies, tones_total, triggers_spe, tag = get_data(path, trigs=an_times)
    l_spikes = list()
    hm_tonotopy = hm.Heatmap()
    hm_tonotopy.compute_heatmap(trigs=an_times, tone_sequence=tones_total, spikes=spk, t_pre=0.100, t_post=0.450,
                                 bin_size=0.002)
    # hm_tonotopy.plot("Tonotopy", folder=opt.folder, ext="png")
    #hm_tonotopy.plot_smooth_2d("Tonotopy", folder=folder, ext="png")
    #hm_tonotopy.save(folder=folder, typeof="tonotopy")
    np.save(path+'hm_tonotopy.npy',hm_tonotopy )
    return hm_tonotopy

def fast_tonotopy(path, n_clus):
    spk_times = np.load(path+'spike_times.npy')
    spk_clus = np.load(path+'spike_clusters.npy') 
    hm = process_data(path, path)
    tones = hm.get_tones()
    clusters = hm.get_clusters()
    heatmaps = hm.plot("tono", n_clus, folder=None, cmap="bwr", l_ex=None, r_ex=None, ext="png")
    gc = np.arange(start=0, stop=n_clus, step=1)
    
    plot_heatmap_bandwidth_tonotopy(heatmaps,3.7, gc,tones)
    
def plot_heatmap_bandwidth_tonotopy(heatmaps,threshold, gc,unique_tones, min_freq, max_freq, t_pre=0.1, t_post=0.5, bin_width=0.01):
    """""
    Best function pour déterminer la bandwidth et plotter la heatmap et les contours de la bandwidth
    input : heatmaps(contenant plusieurs clusters), le threshold pour la detection du pic, good_clusters
        unique_tones (les fréquences jouées), min_freq, max_freq : les indices des fréquences qu'on exclut (pas assez de présentations)
        condition : 'tracking' ou 'playback'
    output : save plot des heatmap avec la bandwidth entourée .png
            save tableau des heatmaps telles que plottée (avec les psth) .npy
            save tableau contenant les bandwidth de chaque cluster .npy
            
    """
    
    # pour les plots:

    #num_rows, num_columns = get_plot_geometry(gc)
    
    num_plots, num_rows, num_columns = get_better_plot_geometry(gc)


    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(24, 18))
    fig.suptitle('Heatmaps clusters', y=1.02)
    plt.subplots_adjust() 
    
     # Flatten the axis array if it's more than 1D
    if num_rows > 1 and num_columns > 1:
        axes = axes.flatten()
    #
    bandwidth = []
    plotted_heatmap = []
    peaks = []
    for cluster, ax in enumerate(axes):
        if cluster < num_plots:
            #print(cluster)
            heatmap_cluster = np.array(heatmaps[cluster])
            hm, peak = detect_peak(heatmaps, cluster)
            #heatmap_min = np.min(heatmap_cluster[min_freq:max_freq])
            #heatmap_max = np.max(abs(heatmap_cluster[min_freq:max_freq]))
            heatmap_min = np.min(heatmap_cluster)
            heatmap_max = np.max(abs(heatmap_cluster))
            #abs_max = max(abs(heatmap_min), abs(heatmap_max))
            abs_max = np.max(abs(heatmap_cluster))
            #abs_max = np.max(abs(heatmap_cluster[5:-5]))
            #contours = get_contour(hm, threshold)
            
        # Je retire la moyenne pre-stim ligne par ligne (fréquence par fréquence)
            t_0 = int(t_pre/bin_width)
            prestim_hm = heatmap_cluster[:, :t_0]
            mean_freq = np.mean(prestim_hm, axis=1)

            for i in range(heatmap_cluster.shape[0]):  # Parcours des lignes de A
                heatmap_cluster[i] -= mean_freq[i]
            
            
            smoothed = smooth_2d(heatmap_cluster, 3)
            
            #je mets des zeros aux frequences trop hautes et trop basses où je n'ai pas
            #assez de présentations
            lowf = np.zeros((min_freq+1, len(heatmap_cluster[0])))
            highf = np.zeros((max_freq+1, len(heatmap_cluster[0])))
            
            milieu = np.concatenate((lowf, smoothed))

            # Concaténation à l'arrière
            milieu = np.concatenate((milieu, highf))
            
            
            img = ax.pcolormesh(smoothed, cmap=create_centered_colormap(), vmin=-abs_max, vmax=abs_max)
            ax.set_yticks(np.arange(len(unique_tones)), unique_tones)
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(f'Cluster {gc[cluster]}')
            #ax.axvline(x=t_0, color='black', linestyle='--') # to print a vertical line at the stim onset time
        
           
            #cbar_ax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
            #fig.colorbar(img, cax=cbar_ax)
        # Hide any unused subplots
    for ax in axes[num_plots:]:
        ax.axis('off')
    plt.tight_layout()  