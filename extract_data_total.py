from load_rhd import *
from quick_extract import *
from get_data import *
import PostProcessing.tools.heatmap as hm
from get_data import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from load_rhd import *
from quick_extract import *
from get_data import *
from get_data import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import shutil

##### Fonctions pour extraire les données neuronales ######

def create_folder(path):
    """
    Checks if a folder exists and if not, creates it
    """
    # Specify the folder path
    folder_path = path

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If the folder doesn't exist, create it
        os.makedirs(folder_path)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)
        
        
def copy_files(n_headstages, path):
    """
    il faut copier les fichiers analog_in, dig_in et acc et tones dans les 
    folders headstage_0 et headstage_1
    n_headstages : nombre de headstages
    """
    
    path_0 = path+'headstage_0/'
    path_1 = path+'headstage_1/'
    
    tones_dir_0 = os.path.join(path_0, 'tones')
    os.makedirs(tones_dir_0, exist_ok=True)
    
      #copier le fichier de tones dans les dossiers headstage_0 et headstage_1
    for file_name in os.listdir(path+'tones/'):
        # Chemin complet du fichier source
        source_file = os.path.join(path+'tones/', file_name)
        # Copier le fichier dans le dossier de destination
        shutil.copy(source_file, tones_dir_0)
        if n_headstages>1:
            shutil.copy(source_file, path_1)
        
    shutil.copy(path+'analog_in.npy', path_0)
    shutil.copy(path+'dig_in.npy', path_0)    
    if n_headstages>1:
        shutil.copy(path+'analog_in.npy', path_1)
        shutil.copy(path+'dig_in.npy', path_1)
    print("All izz well")


def extract_from_rhd(path, sampling_rate, n_headstages, channels_to_remove=None):
    """
    Une seule fonction pour extraire depuis le fichier ephys.rhd jusqu'à ?
    input : path du folder où se trouve le fichier rhd
            channels_to_remove : list contenant les indices des channels à ne pas prendre en compte
            samplinge_rate : le sampling rate 
            n_headstages : nombre de headstages
            
    1 ere etape à appliquer sur le fichier ephys
    """
    load_rhd(path+'ephys.rhd', path, digital=True, analog=True, accelerometer=True, filtered=False, export_to_dat=False)
    neural_data = np.load(path + 'neural_data.npy')
    print('rhd loaded')
    
    # Divide into two folders for headstage 0 and headstage 1
    #créer les dossiers "headstage_0" et "headstage_1"
    path_0 = path + '/headstage_0'
    path_1 = path + '/headstage_1'
    create_folder(path_0)
    
    print('folder is created')
    #headstage 1
    neural_data_0 = neural_data[0:32]
    # Si jamais il faut retirer certains canaux 
    if channels_to_remove==None:
        neural_data_0 = neural_data[0:32]
        print('no channel to remove')
    else : 
        neural_data_0 = [neural_data_0[i] for i in range(len(neural_data_0)) if i not in channels_to_remove]
    np.save(path_0+'/neural_data.npy',neural_data_0 )
    
    #filter_and_cmr_chunked(neural_data_0, sampling_rate, path+'headstage_0/', int(len(neural_data_0)/10))
    ### je teste sans refiltrer ni le common CMR qui prend trop de CPU :(
    #extract spikes etc
    quick_extract(path_0+'/neural_data.npy')
    
    if n_headstages>1:
        create_folder(path_1)
    
        neural_data_1 = neural_data[32:64]
        # headstage 2
        if channels_to_remove==None:
            neural_data_1 = neural_data[32:64]
        else :
            neural_data_1 = [neural_data_1[i] for i in range(len(neural_data_1)) if i not in channels_to_remove]
        np.save(path_1+'/neural_data.npy',neural_data_1 )

        print('len(neural_data_1)= ',len(neural_data_1))
        #filter_and_cmr(neural_data_1, sampling_rate, path+'headstage_1/')
        quick_extract(path_1+'/neural_data.npy')
    
    print("All izz well")