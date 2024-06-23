import os
import h5py
import numpy as np
import csv
from collections import defaultdict
import json
file_args = "args_colab.json" if 'COLAB_GPU' in os.environ else 'args.json'

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd
# load the arguments from the json file
with open(f'./args/{file_args}', 'r') as f:
    args = json.load(f)
args = defaultdict_from_json(args)


def read_landmarks_from_csv(csv_path):
    landmarks = []
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            landmarks.append([float(val) for val in row])
    return np.array(landmarks)


root_dir = args["gazeMpiimage_dir"]
hdf5_path = args["gazeMpiimage_dir"] + "/face_lmk.hdf5"
def preprocess_landmarks_to_hdf5(root_dir, hdf5_path):
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    landmarks = read_landmarks_from_csv(csv_path)
                    # Use a unique key for each landmark data
                    key = os.path.relpath(csv_path, root_dir).replace(os.sep, '_')
                    hdf5_file.create_dataset(key, data=landmarks)
                    
preprocess_landmarks_to_hdf5(root_dir, hdf5_path)