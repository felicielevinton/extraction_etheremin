
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from load_rhd import *
from quick_extract import *
from get_data import *
import numpy as np
import matplotlib.pyplot as plt
import shutil


def get_session_type_final(path):
    """
    Fonction qui renvoie le type de la session parmi TrackingOnly, PlaybackOnly etc
    elle va chercher dans le fichier json le type de session
    """
    # List all files in the folder
    files = os.listdir(path)

    # Filter JSON files
    json_files = [file for file in files if file.endswith('.json')]
    # Check if only one JSON file is found
    if len(json_files) == 1:
        json_file_path = os.path.join(path, json_files[0])
        print("Found JSON file:", json_file_path)
        # Load the JSON data from file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        # Extract the "Type" field
        try : 
            type_value = data['Block_000']['Type']

            if type_value=="Pause":
                type_value = data['Block_001']['Type']
                
            print("Type:", type_value)
        except :
            type_value = [data[key]["Type"] for key in data if key.startswith("Experiment_")][1]
            print("Type:", type_value)       
            
            
    else:
        print("Error: No JSON files found.")
    return type_value

def get_triggers_mapping(path):
    """
    Fonction qui renvoie le mapping des triggers dans le fichier json le type de session
    si c'est v2 alors le mapping c'est : #ANALOG_TRIGGERS_MAPPING = {"MAIN": 1, "PLAYBACK": 0, "MOCK": 3, "TARGET": 2}
    sinon c'est ANALOG_TRIGGERS_MAPPING = {"MAIN": 1, "PLAYBACK": 0}
    """
    analog_triggers_mapping = {"MAIN": 1, "PLAYBACK": 0}
   # List all files in the folder
    files = os.listdir(path)

    # Filter JSON files
    json_files = [file for file in files if file.endswith('.json')]

    # Check if only one JSON file is found
    if len(json_files) == 1:
        json_file_path = os.path.join(path, json_files[0])
        print("Found JSON file:", json_file_path)
        # Load the JSON data from file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
# Extraire la valeur de la clé "Version"
        version = data.get("Version", "La clé 'Version' n'existe pas.")
        print('version : ', version)
        if version=='v2':
            analog_triggers_mapping = {"MAIN": 1, "PLAYBACK": 0, "MOCK": 3, "TARGET": 2}     
    return analog_triggers_mapping