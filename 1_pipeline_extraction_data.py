from kneed import DataGenerator, KneeLocator
from quick_extract import *
from get_data import *
from load_rhd import *
import matplotlib.pyplot as plt
from ExtractRecordings.manual.simple_sort import*
import pandas as pd
from PostProcessing.tools.utils import *
from matplotlib.colors import ListedColormap, Normalize
#from format_data import *
from skimage import measure
import matplotlib.colors as colors
from scipy.signal import find_peaks
from extract_data_total import *
from utils_extraction import *
sr = 30e3
#t_pre = 0.2#0.2
#t_post = 0.30#0.300
#bin_width = 0.005
#bin_width = 0.02
#psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
n_headstages = 2 # nombre de headstages utilisés pendant la manip
#Fichier python pour pouvoir utiliser byobu pour faire tourner l'extraction en arrière plan 

#pour oscypek : /mnt/working2/felicie/data2/eTheremin/OSCYPEK/OSCYPEK
path = '/auto/data6/eTheremin/SKIEUR/SKIEUR_20260313_SESSION_00/'

extract_from_rhd(path, sampling_rate=sr, n_headstages=n_headstages, channels_to_remove=None)

copy_files(n_headstages, path)
session_type = get_session_type_final(path)