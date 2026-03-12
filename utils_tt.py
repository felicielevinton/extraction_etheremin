from kneed import DataGenerator, KneeLocator
from quick_extract import *
from get_data import *
from load_rhd import *
import matplotlib.pyplot as plt
from ExtractRecordings.manual.simple_sort import*
import pandas as pd
from PostProcessing.tools.utils import *
import csv
from format_data import *
import pandas as pd
#from create_data import *
import os
import glob
import scipy.io
import math
from utils import *
from format_data import get_sem
import re
import json
import pickle


# fichier pour créer le tt.pkl de facon propre

def get_triggers(path, analog_line):
    """"
    Récupérer les triggers en tracking
    
     - analog_line : numero de la ligne de triggers analogique. 
      (tracking0, playback1 et mock3 pour les xp de types Playback)
    """
    an_triggers = np.load(os.path.join(path, "analog_in.npy"))
    an_times = ut.extract_analog_triggers_compat(an_triggers[analog_line])
    frequencies, tones_total, triggers_spe, tag = get_data(path, trigs=an_times, tracking_only=True)
    return an_times, tones_total



def get_triggers_tono(path, analog_line, tonotopy_only=True):
    """"
    Récupérer les triggers en tracking
    
     - analog_line : numero de la ligne de triggers analogique. 
      (tracking0, playback1 et mock3 pour les xp de types Playback)
    """
    an_triggers = np.load(os.path.join(path, "analog_in.npy"))
    an_times = ut.extract_analog_triggers_compat(an_triggers[analog_line])
    frequencies, tones_total, triggers_spe, tag = get_data(path, trigs=an_times, tonotopy_only=True)
    return an_times, tones_total
tonotopy_only=False,

def extract_number_from_filename(filename, type):
    """
    Extrait le numéro qui apparaît après le préfixe 'tracking_' dans une chaîne de caractères.
    
    Args:
    - filename (str): Le nom du fichier ou chaîne contenant le préfixe 'tracking_'.
    
    Returns:
    - int: Le numéro extrait, ou None si aucun numéro n'est trouvé.
    """
    # Utilisation d'une expression régulière pour rechercher un numéro après 'type'
    pattern = rf'{type}(\d+)'
    match = re.search(pattern, filename)
    
    if match:
        # Retourne le numéro extrait en tant qu'entier
        return int(match.group(1))
    else:
        # Aucun numéro trouvé
        return None
    


def get_tracking_tones(folder):
    all_files = glob.glob(os.path.join(folder+'/tones/', "*tracking_*.bin"))
    all_files.sort(key=lambda x: extract_number_from_filename(x, 'tracking_'))
    # Print all matching files
    print("Files matching the pattern:")
    for file in all_files:
        print(file)
    all_tones, all_blocs = [], []
    for file in all_files:
        
        # Load the binary file into a NumPy array
        tones = np.fromfile(file, dtype=np.double)
        
        # Append the tones data to the list
        all_tones.append(tones)
        blocs = np.full(len(tones),extract_number_from_filename(file, 'tracking_'))
        all_blocs.append(blocs)
    return all_tones, all_blocs

def get_playback_tones(folder):
    # Find all files matching the pattern
    all_files = glob.glob(os.path.join(folder, 'tones', "*playback_*.bin"))

    # Print all matching files
    print("Files matching the pattern:")
    for file in all_files:
        print(file)

    # Sort files by the number extracted from their filename
    all_files.sort(key=lambda x: extract_number_from_filename(x, 'playback_'))

    all_tones, all_blocs = [], []
    for file in all_files:
        # Load the binary file into a NumPy array
        tones = np.fromfile(file, dtype=np.double)
        
        # Append the tones data to the list
        all_tones.append(tones)
        
        # Create an array filled with the file number
        blocs = np.full(len(tones), extract_number_from_filename(file, 'playback_'))
        all_blocs.append(blocs)

    return all_tones, all_blocs

def get_tail_tones(folder):
    all_files = glob.glob(os.path.join(folder+'/tones/', "*tail*.bin"))
    all_files.sort(key=lambda x: extract_number_from_filename(x, 'tail_'))
    # Print all matching files
    print("Files matching the pattern:")
    for file in all_files:
        print(file)
    all_tones, all_blocs = [], []
    for file in all_files:
        
        # Load the binary file into a NumPy array
        tones = np.fromfile(file, dtype=np.double)
        
        # Append the tones data to the list
        all_tones.append(tones)
        blocs = np.full(len(tones),extract_number_from_filename(file, 'tail_'))
        all_blocs.append(blocs)
    return all_tones, all_blocs

def get_mock_tones(folder):
    all_files = glob.glob(os.path.join(folder+'/tones/', "*mock*.bin"))
    all_files.sort(key=lambda x: extract_number_from_filename(x, 'mock_'))
    # Print all matching files
    print("Files matching the pattern:")
    for file in all_files:
        print(file)
    all_tones, all_blocs = [], []
    for file in all_files:
        
        # Load the binary file into a NumPy array
        tones = np.fromfile(file, dtype=np.double)
        
        # Append the tones data to the list
        all_tones.append(tones)
        blocs = np.full(len(tones),extract_number_from_filename(file, 'mock_'))
        all_blocs.append(blocs)
    return all_tones, all_blocs

def get_mc_tones(folder):
    #tones from mapping change condition
    all_files = glob.glob(os.path.join(folder+'/tones/', "*mc*.bin"))
    all_files.sort(key=lambda x: extract_number_from_filename(x, 'mc'))
    # Print all matching files
    print("Files matching the pattern:")
    for file in all_files:
        print(file)
    all_tones, all_blocs = [], []
    for file in all_files:
        
        # Load the binary file into a NumPy array
        tones = np.fromfile(file, dtype=np.double)
        
        # Append the tones data to the list
        all_tones.append(tones)
        blocs = np.full(len(tones),extract_number_from_filename(file, 'mc_'))
        all_blocs.append(blocs)
    return all_tones, all_blocs


def read_json_file(json_file):
    """
    Reads a JSON file and extracts information from specific sections.
    
    Args:
    json_file (str): The path to the JSON file.
    
    Returns:
    list: A list of dictionaries containing the extracted information.
    """
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Liste pour stocker les informations
    extracted_data = []

    # Parcourir les blocs pour extraire les informations
    for block_name, block_data in json_data.items():
        if block_name.startswith("Block_"):
            # Initialiser les variables
            mock_fn = None
            playback_tones_fn = None
            tracking_tones_fn = None
            mapping_change_fn = None

            # Vérifier s'il y a une section 'playback'
            if 'playback'in block_data:
                playback_data = block_data['playback']
                mock_fn = playback_data.get('Mock_fn')
                playback_tones_fn = playback_data.get('Tones_fn')
            
            # Vérifier s'il y a une section 'tracking'
            if 'tracking' in block_data:
                try : 
                    tracking_data = block_data['Tracking']
                except : 
                    tracking_data = block_data['tracking']
                tracking_tones_fn = tracking_data.get('Tones_fn')
        
                
            # Vérifier s'il y a une section 'mapping change'
            if 'MappingChange' in block_data:
                mc_data = block_data['MappingChange']
                mapping_change_fn = mc_data.get('Tones_fn')


            # Si ni 'playback' ni 'tracking' ne sont présents, prendre 'Type' et 'Tones_fn'
            if not playback_tones_fn and not tracking_tones_fn and not mapping_change_fn:
                block_type = block_data.get('Type')
                tones_fn = block_data.get('Tones_fn')
                print(block_type)
                
                if block_type =='TrackingOnly':
                    tones_fn = block_data.get('Tones_fn')
                    extracted_data.append({
                        "Block": block_name,
                        "Type": block_type,
                        "Tracking Tones_fn": tones_fn
                })
                    
                elif block_type =='PlaybackOnly':
                    extracted_data.append({
                        "Block": block_name,
                        "Type": block_type,
                        "Playback Tones_fn": tones_fn
                })
                    
                else : 
                    extracted_data.append({
                    "Block": block_name,
                    "Type": block_type,
                    "Tail Tones_fn": tones_fn
                })
                
                #extracted_data.append({
                   # "Block": block_name,
                   # "Type": block_type,
                   # "Tail Tones_fn": tones_fn
                #})
            else:
                extracted_data.append({
                    "Block": block_name,
                    "Playback Mock_fn": mock_fn,
                    "Playback Tones_fn": playback_tones_fn,
                    "Tracking Tones_fn": tracking_tones_fn,
                    "Mapping Change Tones_fn": mapping_change_fn
                })
    
    return extracted_data



   
def read_json_file_old(json_file):
    """
    Reads a JSON file and extracts information from specific sections.
    
    Args:
    json_file (str): The path to the JSON file.
    
    Returns:
    list: A list of dictionaries containing the extracted information.
    """
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Liste pour stocker les informations
    extracted_data = []

    # Parcourir les blocs pour extraire les informations
    for block_name, block_data in json_data.items():
        if block_name.startswith("Block_"):
            # Initialiser les variables
            mock_fn = None
            playback_tones_fn = None
            tracking_tones_fn = None
            mapping_change_fn = None

            # Vérifier s'il y a une section 'playback'
            if 'playback'in block_data:
                playback_data = block_data['playback']
                mock_fn = playback_data.get('Mock_fn')
                playback_tones_fn = playback_data.get('Tones_fn')
            
            # Vérifier s'il y a une section 'tracking'
            if 'tracking' in block_data:
                tracking_data = block_data['Tracking']
                tracking_tones_fn = tracking_data.get('Tones_fn')
        
                
            # Vérifier s'il y a une section 'mapping change'
            if 'MappingChange' in block_data:
                mc_data = block_data['MappingChange']
                mapping_change_fn = mc_data.get('Tones_fn')

            # Si ni 'playback' ni 'tracking' ne sont présents, prendre 'Type' et 'Tones_fn'
            if not playback_tones_fn and not tracking_tones_fn and not mapping_change_fn:
                block_type = block_data.get('Type')
                tones_fn = block_data.get('Tones_fn')
                
                if block_type =='Tracking':
                    tones_fn = block_data.get('Tones_fn')
                    extracted_data.append({
                        "Block": block_name,
                        "Type": block_type,
                        "Tracking Tones_fn": tones_fn
                })
                    
                elif block_type =='Playback':
                    extracted_data.append({
                        "Block": block_name,
                        "Type": block_type,
                        "Playback Tones_fn": tones_fn
                })
                    
                else : 
                    extracted_data.append({
                    "Block": block_name,
                    "Type": block_type,
                    "Tail Tones_fn": tones_fn
                })
                
                #extracted_data.append({
                   # "Block": block_name,
                   # "Type": block_type,
                   # "Tail Tones_fn": tones_fn
                #})
            else:
                extracted_data.append({
                    "Block": block_name,
                    "Playback Mock_fn": mock_fn,
                    "Playback Tones_fn": playback_tones_fn,
                    "Tracking Tones_fn": tracking_tones_fn,
                    "Mapping Change Tones_fn": mapping_change_fn
                })
    
    return extracted_data


def comparer(valeur1, valeur2):
    print(valeur1, valeur2)
    if valeur1 != valeur2:
        raise ValueError(f"Erreur : {valeur1} et {valeur2} ne sont pas égales.")
    
    


def find_json(path):
    # Import glob within the function to avoid conflicts with any other 'glob' object
    import glob
    
    # Use glob to find all JSON files in the specified directory
    json_files = glob.glob(os.path.join(path, "*.json"))
    
    # Check if any JSON files were found
    if not json_files:
        print("Aucun fichier JSON trouvé dans le dossier spécifié.")
        return None
    
    # Return the absolute path of the first JSON file found
    return os.path.abspath(json_files[0])
    



# Fonction pour charger les fichiers binaires et concaténer les données
def concatenate_tones_and_labels(extracted_data, folder, mock=True):
    all_tones = []
    all_labels = []
    all_mock = []

    # Parcourir les blocs dans l'ordre
    for block in extracted_data:
        block_tones = []
        block_labels = []
        block_mock = []

        block_name = block.get('Block')
        #block_type = block.get('Type', '')
        
        tail_tones_fn = block.get('Tail Tones_fn')
        if tail_tones_fn:
            tail_tones_path = folder + '/' + tail_tones_fn
            tail_tones = np.fromfile(tail_tones_path, dtype=np.double)
            block_tones.append(tail_tones)
            
            # Créer les labels correspondants
            block_type = 'Tail'
            tail_labels = [(block_name, block_type)] * len(tail_tones)
            block_labels.append(tail_labels)
            print('tail length = ', len(tail_tones))
            
        # Charger les tracking tones si disponibles
        tracking_tones_fn = block.get('Tracking Tones_fn')
        if tracking_tones_fn:
            tracking_tones_path = folder + '/' + tracking_tones_fn
            tracking_tones = np.fromfile(tracking_tones_path, dtype=np.double)
            block_tones.append(tracking_tones)
            
            # Créer les labels correspondants
            block_type = 'Tracking'
            tracking_labels = [(block_name, block_type)] * len(tracking_tones)
            block_labels.append(tracking_labels)

        # Charger les playback tones si disponibles
        playback_tones_fn = block.get('Playback Tones_fn')
        mock_tones_fn = block.get('Playback Mock_fn')
        if playback_tones_fn:
            playback_tones_path = folder + '/' + playback_tones_fn
            playback_tones = np.fromfile(playback_tones_path, dtype=np.double)
            block_tones.append(playback_tones)
            if mock:  
                mock_tones_path = folder + '/' + mock_tones_fn
                mock_tones = np.fromfile(mock_tones_path, dtype=np.double)
                block_mock.append(mock_tones)
                
            block_type = 'Playback'
                # Créer les labels correspondants
            playback_labels = [(block_name, block_type)] * len(playback_tones)
            block_labels.append(playback_labels)
            
        # Mapping change
        mc_tones_fn = block.get('Mapping Change Tones_fn')
        if mc_tones_fn:
            mc_tones_path = folder + '/' + mc_tones_fn
            mc_tones = np.fromfile(mc_tones_path, dtype=np.double)
            block_tones.append(mc_tones)
            
            # Créer les labels correspondants
            block_type = 'Mapping Change'
            mc_labels = [(block_name, block_type)] * len(mc_tones)
            block_labels.append(mc_labels)
            

        # Concaténer les tons et labels de ce bloc
        if block_tones:
            concatenated_block_tones = np.concatenate(block_tones)
            concatenated_block_labels = np.concatenate(block_labels)
            all_tones.append(concatenated_block_tones)
            all_labels.append(concatenated_block_labels)
            all_mock.append(block_mock)
        try:
            comparer(len(playback_tones), len(tracking_tones))
        except:
            pass

    # Concaténer tous les blocs ensemble
    if all_tones:
        final_tones = np.concatenate(all_tones)
        final_labels = np.concatenate(all_labels)
        
        return final_tones, final_labels, all_mock
    else:
        return None, None, None
    
def convert_condition_block(final_tones, final_labels): 
    block = [row[0] for row in final_labels]
    condition = [row[1] for row in final_labels]


    conversion = {
        'Tail': -1,
        'Tracking': 0,
        'Playback': 1,
        'Mapping Change':2
    }

    # Transformation en valeurs numériques
    condition_= [conversion[val] for val in condition]

    return(condition_, block)

def save_tt(tones, triggers,block, condition, mock_triggers, mock_tones, path):

    #concatenate mock_tones:
    #try:
        #mock_tones = np.concatenate([element for sous_tableau in mock_tones for element in sous_tableau])
    #except:
       # mock_tones = np.array([])
    
    tt = {
        'tones': tones,
        'triggers': triggers,
        'block': block, 
        'condition' : condition,
        'mock_triggers' : mock_triggers,
        'mock_tones' : mock_tones
    }

    
    #filtrer le tt
    filtered_triggers = [
    trigger for trigger, block, condition in zip(tt['triggers'], tt['block'], tt['condition'])
    if block is not None and condition is not None
    ]

    # Mise à jour du dictionnaire avec la liste filtrée
    tt['triggers'] = filtered_triggers
     
    # save tt.pkl
    with open(path+'/tt.pkl', 'wb') as file:
        pickle.dump(tt, file)


def creer_tableau_blocs(A):
    # Initialiser un tableau pour stocker les numéros de blocs
    blocs = np.zeros_like(A, dtype=object)  # Un tableau de la même taille que A, rempli de zéros
    bloc_courant = 0  # Compteur du bloc
    blocs[0] = 'block_0'
    # Parcourir le tableau A
    for i in range(1, len(A)):
        # Si on passe de 0 à une valeur non-nulle, on démarre un nouveau bloc
        if A[i-1] != 0 and A[i] == 0 and A[i+1] == 0:
            print(A[i-1], A[i], A[i+1])
            bloc_courant += 1  # Nouveau bloc trouvé

        # Affecter le numéro du bloc courant aux éléments non-nuls
        if A[i] != 0:
            blocs[i] = f'block_{bloc_courant}'
        else:
            blocs[i] = f'block_{bloc_courant}'  # Les zéros appartiennent à aucun bloc (ou bloc zéro)
    
    return blocs


def create_tt(path) : 
    # Fonction pour créer le tt.pkl dans le cas d'une session playback classique (avec les mock )
    
    # get triggers
    triggers_tr, tones_total_tr = get_triggers(path+'headstage_0/', analog_line=0)
    triggers_pb, tones_total_pb = get_triggers(path+'headstage_0/', analog_line=2)   # c'est ligne 1 normalement mais 2 pour Hercule
    triggers_mck, tones_total_mck = get_triggers(path+'headstage_0/', analog_line=3)
        
    condition_tr = np.zeros(len(triggers_tr))
    condition_pb = np.ones(len(triggers_pb)) 
            
    trig_times = np.concatenate((triggers_tr, triggers_pb)) 
    tones = np.concatenate((tones_total_tr, tones_total_pb))
    condition = np.concatenate((condition_tr, condition_pb))
            

    print(len(tones))
    print(len(trig_times))
    
    sorted_indices = np.argsort(trig_times[:len(tones)])
    sorted_indices = sorted_indices[:-1]
    sorted_triggers = trig_times[sorted_indices]
    sorted_tones = tones[sorted_indices]
    sorted_condition = condition[sorted_indices]
    
    # we have triggers now let's get the tones
    json_path = find_json(path)
    extracted_data = read_json_file(json_path)
    tones, labels, mock_tones = concatenate_tones_and_labels(extracted_data, path+'headstage_0/tones')
    condition, block = convert_condition_block(tones, labels)
    save_tt(tones, sorted_triggers, block, condition, triggers_mck, tones_total_mck, path+'headstage_0')
    



def create_tt_playback_only(path, n_headstage) : 
    # Fonction pour créer le tt.pkl dans le cas d'une session playback classique (avec les mock )
    
    # get triggers
    triggers_pb, tones_total_pb = get_triggers(path+f'headstage_{n_headstage}/', analog_line=0)
        
    condition_pb = np.ones(len(triggers_pb)) 
            
    trig_times =  triggers_pb
    tones = tones_total_pb
    condition = condition_pb
    block = np.full(len(condition), 'Block_000')

    print(len(tones))
    print(len(trig_times))
    
    
    triggers_mck, mock_tones = [], []
    # we have triggers now let's get the tones

    save_tt(tones, triggers_pb, block, condition, triggers_mck, mock_tones, path+f'headstage_{n_headstage}/')
    
     
    
def create_tt_no_mock(path, mock=False): 
    # Fonction pour créer le tt.pkl dans le cas d'une session playback classique (avec les mock )
    #ne marche que pour Altaï --> pas les memes lignes de triggers avec Burrata notamment. 
    # get triggers
    triggers_tr, tones_total_tr = get_triggers(path+'headstage_0/', analog_line=0)
    triggers_pb, tones_total_pb = get_triggers(path+'headstage_0/', analog_line=1)
        
    condition_tr = np.zeros(len(triggers_tr))
    condition_pb = np.ones(len(triggers_pb)) 
            
    trig_times = np.concatenate((triggers_tr, triggers_pb)) 
    tones = np.concatenate((tones_total_tr, tones_total_pb))
    condition = np.concatenate((condition_tr, condition_pb))
            

    print(len(tones))
    print(len(trig_times))
    
    sorted_indices = np.argsort(trig_times[:len(tones)])
    #sorted_indices = np.argsort(trig_times)
    sorted_indices = sorted_indices[:-1]
    sorted_triggers = trig_times[sorted_indices]
    sorted_tones = tones[sorted_indices]
    sorted_condition = condition[sorted_indices]
    
    
    triggers_mck, mock_tones = [], []
    # we have triggers now let's get the tones
    json_path = find_json(path)
    extracted_data = read_json_file(json_path)
    tones, labels, mock_tones = concatenate_tones_and_labels(extracted_data, path+'headstage_0/tones', mock)
    condition, block = convert_condition_block(tones, labels)
    save_tt(tones, sorted_triggers, block, condition, triggers_mck, mock_tones, path+'headstage_0')



def create_tt_tono(path, n_headstage, mock=False): 
    # pour les sessions de tonotopies
    # Fonction pour créer le tt.pkl dans le cas d'une session playback classique (avec les mock )
    #ne marche que pour Altaï --> pas les memes lignes de triggers avec Burrata notamment. 
    # get triggers
    triggers_tr, tones_total_tr = get_triggers(path+f'headstage_{n_headstage}/', analog_line=0)

        
    condition_tr = np.zeros(len(triggers_tr))
            
    trig_times = triggers_tr
    tones = tones_total_tr
    condition = condition_tr
            

    print(len(tones))
    print(len(trig_times))
    
    sorted_indices = np.argsort(trig_times[:len(tones)])
    #sorted_indices = np.argsort(trig_times)
    sorted_indices = sorted_indices[:-1]
    sorted_triggers = trig_times[sorted_indices]
    sorted_tones = tones[sorted_indices]
    sorted_condition = condition[sorted_indices]
    
    
    triggers_mck, mock_tones = [], []
    # we have triggers now let's get the tones
    json_path = find_json(path)
    extracted_data = read_json_file(json_path)
    #tones, labels, mock_tones = concatenate_tones_and_labels(extracted_data, path+'headstage_0/tones', mock)
    #condition, block = convert_condition_block(tones, labels)
    block = np.full(len(condition), 'Block_000')
    save_tt(tones, sorted_triggers, block, condition, triggers_mck, mock_tones, path+f'/headstage_{n_headstage}')






def create_tt_no_burrata(path, mock=False): 
    # Fonction pour créer le tt.pkl dans le cas d'une session playback classique (avec les mock )
    # pour Burrata
    # get triggers
    triggers_tr, tones_total_tr = get_triggers(path+'headstage_0/', analog_line=1)
    triggers_pb, tones_total_pb = get_triggers(path+'headstage_0/', analog_line=0)
    triggers_mock, tones_total_mock = get_triggers(path+'headstage_0/', analog_line=0)
        
    condition_tr = np.zeros(len(triggers_tr))
    condition_pb = np.ones(len(triggers_pb)) 
            
    trig_times = np.concatenate((triggers_tr, triggers_pb)) 
    tones = np.concatenate((tones_total_tr, tones_total_pb))
    condition = np.concatenate((condition_tr, condition_pb))
            

    print(len(tones))
    print(len(trig_times))
    
    sorted_indices = np.argsort(trig_times[:len(tones)])
    sorted_indices = sorted_indices[:-1]
    sorted_triggers = trig_times[sorted_indices]
    sorted_tones = tones[sorted_indices]
    sorted_condition = condition[sorted_indices]
    
    
    triggers_mck, mock_tones = [], []
    # we have triggers now let's get the tones
    json_path = find_json(path)
    extracted_data = read_json_file(json_path)
    tones, labels, mock_tones = concatenate_tones_and_labels(extracted_data, path+'headstage_0/tones', mock)
    condition, block = convert_condition_block(tones, labels)
    save_tt(tones, sorted_triggers, block, condition, triggers_mck, mock_tones, path+'headstage_0')
    
    
def create_tt_mc(path) : 
    # Fonction pour créer le tt.pkl dans le cas d'une session mapping change 
    
    # get triggers
    triggers_tr, tones_total_tr = get_triggers(path+'headstage_0/', analog_line=0)
    triggers_mc, tones_total_mc = get_triggers(path+'headstage_0/', analog_line=1)
        
    condition_tr = np.zeros(len(triggers_tr))
    condition_mc = np.full_like(triggers_mc, 2)
            
    trig_times = np.concatenate((triggers_tr, triggers_mc))
    tones = np.concatenate((tones_total_tr, tones_total_mc))
    condition = np.concatenate((condition_tr, condition_mc)) 
        
    sorted_indices = np.argsort(trig_times[:len(tones)])
    sorted_indices = sorted_indices[:-1]
    sorted_triggers = trig_times[sorted_indices]
    sorted_tones = tones[sorted_indices]
    sorted_condition = condition[sorted_indices]
    block = creer_tableau_blocs(sorted_condition)
    # we have triggers now let's get the tones
    #json_path = find_json(path)
    #extracted_data = read_json_file(json_path)
    #tones, labels, mock_tones = concatenate_tones_and_labels(extracted_data, path+'headstage_0/tones')
    #condition, block = convert_condition_block(tones, labels)
    save_tt(sorted_tones, sorted_triggers,block, sorted_condition, None, None, path+'headstage_0')
    
def create_data_features_new_version(path, bin_width, fs, mock=True):
 
    spk_clusters = np.load(path+'/spike_clusters.npy', allow_pickle=True)
    spk_times = np.load(path+'/spike_times.npy', allow_pickle=True)

    clusters = {}
    for value, cluster in zip(spk_times, spk_clusters):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(value)

    ##NEURO
    n_clus = np.max(list(spk_clusters))
    t_spk, c_spk = [], [] #spike times, cluster
    #for cluster in range(spike.get_n_clusters()):
    for cluster in range(n_clus+1):
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
        
    
    t_stim = np.array(tt['triggers'])/fs
    f_stim = np.array(tt['tones'])
    type_stim = tt['condition']
    #block = [int(block[-1]) for block in tt['block']]
    #block = [int(block.split('_0')[1]) for block in tt['block']]
    block = [int(block.split('_')[-1]) for block in tt['block']]
    if mock:
        t_mock = np.array(tt['mock_triggers'])/fs
        f_mock = tt['mock_tones'] 
    
    #attention
    t_stim = np.array(t_stim, dtype=float)
    type_stim = np.array(type_stim, dtype=float)
    block = np.array(block, dtype=float)
    if mock : 
        t_mock = np.array(t_mock, dtype=float)
    
    
    
    unique_tones = sorted(np.unique(f_stim))
    
    
    # Attention on a envie que t_stim et f_stim soient de la même longueur
    if len(t_stim)>len(f_stim):
        t_stim = t_stim[:len(f_stim)]
        type_stim = type_stim[:len(f_stim)]
        block = block[:len(f_stim)]
        type_stim = type_stim[:len(f_stim)]
        print(f" ATTENTION Shape of t_stim: {t_stim.shape}")
        print(f"ATTENTION Shape of f_stim: {f_stim.shape}")
    elif len(f_stim)>len(t_stim):
        f_stim = f_stim[:len(t_stim)]
        type_stim = type_stim[:len(t_stim)]
        block = block[:len(t_stim)]
        type_stim = type_stim[:len(t_stim)]
        
        print(f" ATTENTION Shape of t_stim: {t_stim.shape}")
        print(f"ATTENTION Shape of f_stim: {f_stim.shape}")


    #need to interpolate between two stims to get the frequency in between
    # 1. True stims
    stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
    interpolated_freq = np.zeros(len(bins) - 1)
    interpolated_type_stim = np.zeros(len(bins) - 1)
    interpolated_block_stim = np.zeros(len(bins) - 1)
    previous_frequency = None
    previous_condition = None
    previous_block = None

    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]

        # Check if any stimuli fall within the current bin
        stimuli_in_bin = (t_stim >= bin_start) & (t_stim < bin_end)
        
        if np.any(stimuli_in_bin):
            print(stimuli_in_bin)
            # If stimuli are present, set stimulus_presence to True for this bin
            stimulus_presence[i] = True

            # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
            # You can simply take the frequency of the first stimulus within the bin
            interpolated_freq[i] = f_stim[stimuli_in_bin][0]
            interpolated_type_stim[i] = type_stim[stimuli_in_bin][0]
            interpolated_block_stim[i] = block[stimuli_in_bin][0]


            previous_frequency = interpolated_freq[i]  # Update previous frequency
            previous_condition = interpolated_type_stim[i]
            previous_block = interpolated_block_stim[i]

            
        else:
            # If no stimulus in the bin, set bin_frequencies to the previous frequency
            if previous_frequency is not None:
                interpolated_freq[i] = previous_frequency
                interpolated_type_stim[i] = previous_condition
                interpolated_block_stim[i] = previous_block
                
    #interpolated_type_stim = np.interp(bins, t_stim, type_stim)
    #interpolated_block_stim = np.interp(bins, t_stim, block)
    
    
      # 2. Mock stims
    if mock:
        mock_stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
        interpolated_mock_freq = np.zeros(len(bins) - 1)

        previous_frequency = None
        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]

            # Check if any stimuli fall within the current bin
            mock_in_bin = (t_mock >= bin_start) & (t_mock < bin_end)
            
            print(f"stimuli_in_bin indices: {np.where(mock_in_bin)}")
            print(f"f_stim values in bin {i}: {f_mock[mock_in_bin]}")
            if np.any(mock_in_bin):
                # If stimuli are present, set stimulus_presence to True for this bin
                mock_stimulus_presence[i] = True

                # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
                # You can simply take the frequency of the first stimulus within the bin
                interpolated_mock_freq[i] = f_mock[mock_in_bin][0]
                previous_frequency = interpolated_mock_freq[i]  # Update previous frequency
            else:
                # If no stimulus in the bin, set bin_frequencies to the previous frequency
                if previous_frequency is not None:
                    interpolated_mock_freq[i] = previous_frequency



        # Create a dictionary to store information for each time bin
        features = {}
        for i, bin in enumerate(bins[:-1]):
            features[bin] = {
                'Played_frequency': interpolated_freq[i],
                'Condition': interpolated_type_stim[i],
                'Block' : interpolated_block_stim[i],
                'Frequency_changes': stimulus_presence[i],
                'Mock_frequency': interpolated_mock_freq[i],
                'Mock_change' : mock_stimulus_presence[i]
                
            }
    else:
        features = {}
        for i, bin in enumerate(bins[:-1]):
            features[bin] = {
                'Played_frequency': interpolated_freq[i],
                'Condition': interpolated_type_stim[i],
                'Block' : interpolated_block_stim[i],
                'Frequency_changes': stimulus_presence[i]
            }      
        
        
        
    features = list(features.values())
    np.save(path+f'/features_{bin_width}.npy', features)
    
    np.save(path+'/unique_tones.npy', unique_tones)

    #with open(path+'/features.json', 'w') as json_file:
        #json.dump(features, json_file)   
        
    print(f"Shape of t_stim: {t_stim.shape}")
    print(f"Shape of f_stim: {f_stim.shape}")
    print(f"Shape of bins: {bins.shape}")
    print('all izz well')
        
    
    
def create_data_features_new_version_spike_sorted(path, bin_width, fs, mock=True):
 
    spk_clusters = np.load(path+'/spike_sorting/ss_spike_clusters.npy', allow_pickle=True)
    spk_times = np.load(path+'/spike_sorting/ss_spike_times.npy', allow_pickle=True)

    # je ne garde que les channels qui ont un cluster 0 (il faut bien du bruit ???)
    channels_with_cluster0 = np.unique(
        spk_clusters[spk_clusters[:, 1] == 0, 0]
    )

        # -----------------------------------
    mask_channels = np.isin(spk_clusters[:, 0], channels_with_cluster0)
    spk_clusters = spk_clusters[mask_channels]
    spk_times = spk_times[mask_channels]

    # -----------------------------------
    # 3️⃣ Retirer le cluster 0 (bruit)
    # -----------------------------------
    mask_not_noise = spk_clusters[:, 1] != 0
    spk_clusters = spk_clusters[mask_not_noise]
    spk_times = spk_times[mask_not_noise]


    # Retirer les clusters 0 de chaque channel -- cluster 0 c'est du bruit sur wave_clus --
    #mask = spk_clusters[:, 1] != 0
    #spk_clusters = spk_clusters[mask]
    #spk_times = spk_times[mask]
    ##



    
    spk_clusters = np.array([int(f"{int(a)}{int(b)}") for a, b in spk_clusters])
    clusters = {}
    for value, cluster in zip(spk_times, spk_clusters):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(value)

    ##NEURO
    n_clus = np.max(list(spk_clusters))
    t_spk, c_spk = [], []
    for cluster in clusters.keys():  # seulement les clusters existants
        t_spk.append(np.array(clusters[cluster]))
        c_spk.append(np.full_like(clusters[cluster], cluster))
    t_spk = np.hstack(t_spk)
    c_spk = np.hstack(c_spk)

    # mettre en secondes
    t_spk = t_spk /1000 # ce qui sort de waveclus est en ms


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

    data = [histograms_per_cluster[key][0] for key in histograms_per_cluster]
    np.save(path+f'/spike_sorting/data_{bin_width}.npy', data)
    np.save(path+f'/spike_sorting/unique_clusters_{bin_width}.npy', unique_clusters)

    #### TRIGGERS
    tt_path = path+'/tt.pkl'
    with open(tt_path, 'rb') as file:
        tt = pickle.load(file)
        
    
    t_stim = np.array(tt['triggers'])/fs
    f_stim = np.array(tt['tones'])
    type_stim = tt['condition']
    #block = [int(block[-1]) for block in tt['block']]
    #block = [int(block.split('_0')[1]) for block in tt['block']]
    block = [int(block.split('_')[-1]) for block in tt['block']]
    if mock:
        t_mock = np.array(tt['mock_triggers'])/fs
        f_mock = tt['mock_tones'] 
    
    #attention
    t_stim = np.array(t_stim, dtype=float)
    type_stim = np.array(type_stim, dtype=float)
    block = np.array(block, dtype=float)
    if mock : 
        t_mock = np.array(t_mock, dtype=float)
    
    
    
    unique_tones = sorted(np.unique(f_stim))
    
    
    # Attention on a envie que t_stim et f_stim soient de la même longueur
    if len(t_stim)>len(f_stim):
        t_stim = t_stim[:len(f_stim)]
        type_stim = type_stim[:len(f_stim)]
        block = block[:len(f_stim)]
        type_stim = type_stim[:len(f_stim)]
        #print(f" ATTENTION Shape of t_stim: {t_stim.shape}")
        #print(f"ATTENTION Shape of f_stim: {f_stim.shape}")
    elif len(f_stim)>len(t_stim):
        f_stim = f_stim[:len(t_stim)]
        type_stim = type_stim[:len(t_stim)]
        block = block[:len(t_stim)]
        type_stim = type_stim[:len(t_stim)]
        
        #print(f" ATTENTION Shape of t_stim: {t_stim.shape}")
        #print(f"ATTENTION Shape of f_stim: {f_stim.shape}")


    #need to interpolate between two stims to get the frequency in between
    # 1. True stims
    stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
    interpolated_freq = np.zeros(len(bins) - 1)
    interpolated_type_stim = np.zeros(len(bins) - 1)
    interpolated_block_stim = np.zeros(len(bins) - 1)
    previous_frequency = None
    previous_condition = None
    previous_block = None

    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]

        # Check if any stimuli fall within the current bin
        stimuli_in_bin = (t_stim >= bin_start) & (t_stim < bin_end)
        
        if np.any(stimuli_in_bin):
            # If stimuli are present, set stimulus_presence to True for this bin
            stimulus_presence[i] = True

            # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
            # You can simply take the frequency of the first stimulus within the bin
            interpolated_freq[i] = f_stim[stimuli_in_bin][0]
            interpolated_type_stim[i] = type_stim[stimuli_in_bin][0]
            interpolated_block_stim[i] = block[stimuli_in_bin][0]


            previous_frequency = interpolated_freq[i]  # Update previous frequency
            previous_condition = interpolated_type_stim[i]
            previous_block = interpolated_block_stim[i]

            
        else:
            # If no stimulus in the bin, set bin_frequencies to the previous frequency
            if previous_frequency is not None:
                interpolated_freq[i] = previous_frequency
                interpolated_type_stim[i] = previous_condition
                interpolated_block_stim[i] = previous_block
                
    #interpolated_type_stim = np.interp(bins, t_stim, type_stim)
    #interpolated_block_stim = np.interp(bins, t_stim, block)
    
    
      # 2. Mock stims
    if mock:
        mock_stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
        interpolated_mock_freq = np.zeros(len(bins) - 1)

        previous_frequency = None
        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]

            # Check if any stimuli fall within the current bin
            mock_in_bin = (t_mock >= bin_start) & (t_mock < bin_end)
            
            if np.any(mock_in_bin):
                # If stimuli are present, set stimulus_presence to True for this bin
                mock_stimulus_presence[i] = True

                # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
                # You can simply take the frequency of the first stimulus within the bin
                interpolated_mock_freq[i] = f_mock[mock_in_bin][0]
                previous_frequency = interpolated_mock_freq[i]  # Update previous frequency
            else:
                # If no stimulus in the bin, set bin_frequencies to the previous frequency
                if previous_frequency is not None:
                    interpolated_mock_freq[i] = previous_frequency



        # Create a dictionary to store information for each time bin
        features = {}
        for i, bin in enumerate(bins[:-1]):
            features[bin] = {
                'Played_frequency': interpolated_freq[i],
                'Condition': interpolated_type_stim[i],
                'Block' : interpolated_block_stim[i],
                'Frequency_changes': stimulus_presence[i],
                'Mock_frequency': interpolated_mock_freq[i],
                'Mock_change' : mock_stimulus_presence[i]
                
            }
    else:
        features = {}
        for i, bin in enumerate(bins[:-1]):
            features[bin] = {
                'Played_frequency': interpolated_freq[i],
                'Condition': interpolated_type_stim[i],
                'Block' : interpolated_block_stim[i],
                'Frequency_changes': stimulus_presence[i]
            }      
        
        
        
    features = list(features.values())
    np.save(path+f'/spike_sorting/features_{bin_width}.npy', features)
    
    np.save(path+'/unique_tones.npy', unique_tones)

    #with open(path+'/features.json', 'w') as json_file:
        #json.dump(features, json_file)   
        
    print(f"Shape of t_stim: {t_stim.shape}")
    print(f"Shape of f_stim: {f_stim.shape}")
    print(f"Shape of bins: {bins.shape}")
    print('all izz well')
        
    

def create_data_features_mock(path, bin_width, fs, mock=True):
    
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
    
    t_stim = np.array(tt['triggers'])/fs
    f_stim = tt['tones']
    type_stim = tt['condition']
    #block = [int(block[-1]) for block in tt['block']]
    block = [int(block.split('_0')[1]) for block in tt['block']]
    if mock:
        t_mock = np.array(tt['mock_triggers'])/fs
        f_mock = tt['mock_tones'] 
    
    #attention
    t_stim = np.array(t_stim, dtype=float)
    type_stim = np.array(type_stim, dtype=float)
    block = np.array(block, dtype=float)
    if mock : 
        t_mock = np.array(t_mock, dtype=float)
    
    
    
    unique_tones = sorted(np.unique(f_stim))
    
    print(f"Shape of t_stim: {t_stim.shape}")
    print(f"Shape of f_stim: {f_stim.shape}")
    print(f"Shape of bins: {bins.shape}")
    
    # Attention on a envie que t_stim et f_stim soient de la même longueur
    if len(t_stim)>len(f_stim):
        t_stim = t_stim[:len(f_stim)]
        type_stim = type_stim[:len(f_stim)]
        block = block[:len(f_stim)]
        type_stim = type_stim[:len(f_stim)]
        print(f" ATTENTION Shape of t_stim: {t_stim.shape}")
        print(f"ATTENTION Shape of f_stim: {f_stim.shape}")
    elif len(f_stim)>len(t_stim):
        f_stim = f_stim[:len(t_stim)]
        type_stim = type_stim[:len(t_stim)]
        block = block[:len(t_stim)]
        type_stim = type_stim[:len(t_stim)]
        
        print(f" ATTENTION Shape of t_stim: {t_stim.shape}")
        print(f"ATTENTION Shape of f_stim: {f_stim.shape}")


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
    interpolated_block_stim = np.interp(bins, t_stim, block)
    
    
      # 2. Mock stims
    if mock:
        mock_stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
        interpolated_mock_freq = np.zeros(len(bins) - 1)

        previous_frequency = None
        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]

            # Check if any stimuli fall within the current bin
            mock_in_bin = (t_mock >= bin_start) & (t_mock < bin_end)
            
            print(f"stimuli_in_bin indices: {np.where(mock_in_bin)}")
            print(f"f_stim values in bin {i}: {f_mock[mock_in_bin]}")
            if np.any(mock_in_bin):
                # If stimuli are present, set stimulus_presence to True for this bin
                mock_stimulus_presence[i] = True

                # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
                # You can simply take the frequency of the first stimulus within the bin
                interpolated_mock_freq[i] = f_mock[mock_in_bin][0]
                previous_frequency = interpolated_mock_freq[i]  # Update previous frequency
            else:
                # If no stimulus in the bin, set bin_frequencies to the previous frequency
                if previous_frequency is not None:
                    interpolated_mock_freq[i] = previous_frequency



        # Create a dictionary to store information for each time bin
        features = {}
        for i, bin in enumerate(bins[:-1]):
            features[bin] = {
                'Played_frequency': interpolated_freq[i],
                'Condition': interpolated_type_stim[i],
                'Block' : interpolated_block_stim[i],
                'Frequency_changes': stimulus_presence[i],
                'Mock_frequency': interpolated_mock_freq[i],
                'Mock_change' : mock_stimulus_presence[i]
                
            }
    else:
        features = {}
        for i, bin in enumerate(bins[:-1]):
            features[bin] = {
                'Played_frequency': interpolated_freq[i],
                'Condition': interpolated_type_stim[i],
                'Block' : interpolated_block_stim[i],
                'Frequency_changes': stimulus_presence[i]
            }      
        
        
        
    features = list(features.values())
    np.save(path+f'/features_{bin_width}.npy', features)
    
    np.save(path+'/unique_tones.npy', unique_tones)

    #with open(path+'/features.json', 'w') as json_file:
        #json.dump(features, json_file)   
        
    print(f"Shape of t_stim: {t_stim.shape}")
    print(f"Shape of f_stim: {f_stim.shape}")
    print(f"Shape of bins: {bins.shape}")
    print('all izz well')
        


def create_data_features_ss(path, clus, bin_width, fs, mock=False):
    # clus : numero du cluster spike sorté

    # version si spike_sorting c'est une version test
    
    #data = pd.read_hdf(path+'/data.h5')
    
    #file = path+'/recording_length.bin'
    #with open(file, 'rb') as file:
       # recording_length = file.read()
    #recording_length = recording_length.decode('utf-8')

    # Extract only the numbers using a simple filter
   # recording_length = int(''.join(filter(str.isdigit, recording_length)))

    #print(recording_length)
    #extraire recording_length OK ca marche

 
    spk_clusters = np.load(path+'ss_C' + str(clus) + '_spike_clusters.npy', allow_pickle=True)
    spk_times = np.load(path+'ss_C' + str(clus) + '_spike_times.npy', allow_pickle=True)

    clusters = {}
    for value, cluster in zip(spk_times, spk_clusters):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(value)

    ##NEURO
    t_spk, c_spk = [], [] #spike times, cluster
    #for cluster in range(spike.get_n_clusters()):
    #n_clus = len(np.unique(spk_clusters))
    n_0 = 0
    for cluster in np.unique(spk_clusters):
        t_spk.append(clusters[cluster]) #spikes times
        c_spk.append(np.full_like(t_spk[n_0], cluster))
        n_0 = n_0 +1
    t_spk = np.hstack(t_spk)
    c_spk = np.hstack(c_spk)

    # mettre en secondes 
    t_spk = t_spk/1000 # attention ce qui sort de wave_clus c'est en ms
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
    np.save(path+f'data_ss_channel_{clus}_{bin_width}.npy', data)


    #### TRIGGERS
    tt_path = path+'tt.pkl'
    with open(tt_path, 'rb') as file:
        tt = pickle.load(file)
        
    #sorted_indices = np.argsort(tt['triggers'])
    #sorted_triggers = tt['triggers'][sorted_indices]
    #sorted_tones = tt['tones'][sorted_indices]
    #sorted_condition = tt['condition'][sorted_indices]
    
    t_stim = np.array(tt['triggers'])/fs
    print(tt.keys)
    f_stim = tt['tones']
    type_stim = tt['condition']
    block = [int(block[-1]) for block in tt['block']]
    if mock:
        t_mock = np.array(tt['mock_triggers'])/fs
        f_mock = tt['mock_tones'] 
    
    #attention
    t_stim = np.array(t_stim, dtype=float)
    type_stim = np.array(type_stim, dtype=float)
    block = np.array(block, dtype=float)
    if mock : 
        t_mock = np.array(t_mock, dtype=float)
    
    
    
    unique_tones = sorted(np.unique(f_stim))
    
    print(f"Shape of t_stim: {t_stim.shape}")
    print(f"Shape of f_stim: {f_stim.shape}")
    print(f"Shape of bins: {bins.shape}")
    
    # Attention on a envie que t_stim et f_stim soient de la même longueur
    if len(t_stim)>len(f_stim):
        t_stim = t_stim[:len(f_stim)]
        type_stim = type_stim[:len(f_stim)]
        block = block[:len(f_stim)]
        type_stim = type_stim[:len(f_stim)]
        print(f" ATTENTION Shape of t_stim: {t_stim.shape}")
        print(f"ATTENTION Shape of f_stim: {f_stim.shape}")
    elif len(f_stim)>len(t_stim):
        f_stim = f_stim[:len(t_stim)]
        type_stim = type_stim[:len(t_stim)]
        block = block[:len(t_stim)]
        type_stim = type_stim[:len(t_stim)]
        
        print(f" ATTENTION Shape of t_stim: {t_stim.shape}")
        print(f"ATTENTION Shape of f_stim: {f_stim.shape}")


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
    interpolated_block_stim = np.interp(bins, t_stim, block)
    
    
      # 2. Mock stims
    if mock:
        mock_stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
        interpolated_mock_freq = np.zeros(len(bins) - 1)

        previous_frequency = None
        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]

            # Check if any stimuli fall within the current bin
            mock_in_bin = (t_mock >= bin_start) & (t_mock < bin_end)
            
            print(f"stimuli_in_bin indices: {np.where(mock_in_bin)}")
            print(f"f_stim values in bin {i}: {f_mock[mock_in_bin]}")
            if np.any(mock_in_bin):
                # If stimuli are present, set stimulus_presence to True for this bin
                mock_stimulus_presence[i] = True

                # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
                # You can simply take the frequency of the first stimulus within the bin
                interpolated_mock_freq[i] = f_mock[mock_in_bin][0]
                previous_frequency = interpolated_mock_freq[i]  # Update previous frequency
            else:
                # If no stimulus in the bin, set bin_frequencies to the previous frequency
                if previous_frequency is not None:
                    interpolated_mock_freq[i] = previous_frequency



        # Create a dictionary to store information for each time bin
        features = {}
        for i, bin in enumerate(bins[:-1]):
            features[bin] = {
                'Played_frequency': interpolated_freq[i],
                'Condition': interpolated_type_stim[i],
                'Block' : interpolated_block_stim[i],
                'Frequency_changes': stimulus_presence[i],
                'Mock_frequency': interpolated_mock_freq[i],
                'Mock_change' : mock_stimulus_presence[i]
                
            }
    else:
        features = {}
        for i, bin in enumerate(bins[:-1]):
            features[bin] = {
                'Played_frequency': interpolated_freq[i],
                'Condition': interpolated_type_stim[i],
                'Block' : interpolated_block_stim[i],
                'Frequency_changes': stimulus_presence[i]
            }      
        
        
        
    features = list(features.values())
    np.save(path+f'/features_{bin_width}.npy', features)
    
    np.save(path+'/unique_tones.npy', unique_tones)

    #with open(path+'/features.json', 'w') as json_file:
        #json.dump(features, json_file)   
        
    print(f"Shape of t_stim: {t_stim.shape}")
    print(f"Shape of f_stim: {f_stim.shape}")
    print(f"Shape of bins: {bins.shape}")
    print('all izz well')
        



# OLD VERSIONS

def read_json_file_OLD(json_file):
    """"
    arg : l'adresse du json file
    """
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Liste pour stocker les informations
    extracted_data = []

    # Parcourir les blocs pour extraire les informations
    for block_name, block_data in json_data.items():
        if block_name.startswith("Block_"):
            # Initialiser les variables
            mock_fn = None
            playback_tones_fn = None
            tracking_tones_fn = None
            mapping_change_fn = None

            # Vérifier s'il y a une section 'playback'
            if 'playback' in block_data:
                playback_data = block_data['playback']
                mock_tones_fn = playback_data.get('Mock_fn')
                playback_tones_fn = playback_data.get('Tones_fn')
            
            # Vérifier s'il y a une section 'tracking'
            if 'tracking' in block_data:
                tracking_data = block_data['tracking']
                tracking_tones_fn = tracking_data.get('Tones_fn')
                
            # Vérifier s'il y a une section 'mapping change'
            if 'mc' in block_data:
                tracking_data = block_data['mc']
                tracking_tones_fn = tracking_data.get('Tones_fn')

            # Si ni 'playback' ni 'tracking' ne sont présents, prendre 'Type' et 'Positions_fn'
            if not playback_tones_fn and not tracking_tones_fn:
                block_type = block_data.get('Type')
                tracking_tail_fn = block_data.get('Tones_fn')
                extracted_data.append({
                    "Block": block_name,
                    "Type": block_type,
                    "Tail Tones_fn": tracking_tail_fn
                })
            else:
                extracted_data.append({
                    "Block": block_name,
                    "Playback Mock_fn": mock_tones_fn,
                    "Playback Tones_fn": playback_tones_fn,
                    "Tracking Tones_fn": tracking_tones_fn
                })
    return extracted_data