import os
import numpy as np
import cv2
import csv
import h5py
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter

class Mpiigaze(Dataset): 
    def __init__(self, pathorg, root, transform, train, angle, fold=0):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.lines = []
        self.angle = angle
        self.landmarks_dict = {}  # Dictionary to store landmarks
        path = pathorg.copy()
        self.hdf5_path = os.path.join(root, "face_lmk.hdf5")
        if not train:
            self.angle = 90
        if train:
            path.pop(fold)
        else:
            path = path[fold]
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    lines = f.readlines()
                    lines.pop(0)
                    self.orig_list_len += len(lines)
                    for line in lines:
                        gaze2d = line.strip().split(" ")[7]
                        label = np.array(gaze2d.split(",")).astype("float")
                        if abs((label[0] * 180 / np.pi)) <= angle and abs((label[1] * 180 / np.pi)) <= angle:
                            self.lines.append(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0] * 180 / np.pi)) <= 42 and abs((label[1] * 180 / np.pi)) <= 42:
                        self.lines.append(line)
        
   
        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len - len(self.lines), angle))
        
    def __len__(self):
        return len(self.lines)
  
    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        gaze2d = line[7]
        face = line[0]

        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0] * 180 / np.pi
        yaw = label[1] * 180 / np.pi
        key = face.replace('.jpg', '.csv').replace('/', '_').replace('\\', '_')
        key = '_lmk_'.join(key.rsplit('_', 1))  # Adjust the key format
        # Retrieve preprocessed landmarks
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            landmark = torch.from_numpy(hdf5_file[key][:])
        cont_labels = torch.FloatTensor([pitch, yaw])

        return landmark.float(), cont_labels.float()
    
class Mpiigaze_nofold(Dataset): 
    def __init__(self, pathorg, root, transform, angle, train=True):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.lines = []
        self.angle = angle
        self.landmarks_dict = {}  # Dictionary to store landmarks
        self.hdf5_path = os.path.join(root, "face_lmk.hdf5")
        paths = pathorg.copy()
        if not train:
            self.angle = 90
        if isinstance(paths, list):
            for path in paths:
                self._load_data_from_file(path)
        else:
            self._load_data_from_file(paths)

        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len - len(self.lines), angle))
        
    def _load_data_from_file(self, path):
        with open(path) as f:
            lines = f.readlines()
            lines.pop(0)
            self.orig_list_len += len(lines)
            for line in lines:
                gaze2d = line.strip().split(" ")[7]
                label = np.array(gaze2d.split(",")).astype("float")
                if abs((label[0] * 180 / np.pi)) <= self.angle and abs((label[1] * 180 / np.pi)) <= self.angle:
                    self.lines.append(line)
        
    def __len__(self):
        return len(self.lines)
  
    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        gaze2d = line[7]
        face = line[0]

        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0] * 180 / np.pi
        yaw = label[1] * 180 / np.pi
        key = face.replace('.jpg', '.csv').replace('/', '_').replace('\\', '_')
        key = '_lmk_'.join(key.rsplit('_', 1))  # Adjust the key format
        # Retrieve preprocessed landmarks
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            landmark = torch.from_numpy(hdf5_file[key][:])
        cont_labels = torch.FloatTensor([pitch, yaw])

        return landmark.float(), cont_labels.float()   
# Example usage
# Assuming pathorg and root are defined
# dataset = Mpiigaze(pathorg, root, transform=None, train=True, angle=30, fold=0)
