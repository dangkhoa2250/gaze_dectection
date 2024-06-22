import cv2
import numpy as np
import mediapipe as mp
import os
import shutil
import csv
import torch
from collections import defaultdict
import json
import logging
import warnings

# Tắt các cảnh báo của PyTorch
logging.getLogger('torch').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')


def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

mp_face_mesh = mp.solutions.face_mesh

def add_padding(image, desired_size=300):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

def get_landmark_vector(image):
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        img_h, img_w = image.shape[:2]
        
        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )
            # Find the maximum x and y values
            max_x = np.max(mesh_points[:, 0])
            max_y = np.max(mesh_points[:, 1])
            
            # Normalize the coordinates based on the maximum x and y values
            normalized_points = mesh_points / [max_x, max_y]
            
            # Flatten the array to create a single vector of x and y coordinates
            landmark_vector = normalized_points.flatten()
            return landmark_vector * 10
        else:
            return np.array([])  # Return an empty array if no landmarks are detected

def save_landmarks_to_csv(landmarks, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(landmarks)

def create_or_reset_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def create_face_lmk_data():
    file_args = "../args/args.json"
    
    # load the arguments from the json file
    with open(f'./args/{file_args}', 'r') as f:
        args = json.load(f)
    args = defaultdict_from_json(args)
    dataset = args['dataset']
    
    for fold in os.listdir(args["gazeMpiimage_dir"]):
        per_image_folder = os.path.join(args["gazeMpiimage_dir"], fold, "face")
        csv_foler = os.path.join(args["gazeMpiimage_dir"], fold, "face_lmk")
        result_file = args["data_preprocess_path"] + "/error_file.txt"
        create_or_reset_folder(csv_foler)
        for image in os.listdir(per_image_folder):
            image_path = os.path.join(per_image_folder, image)
            csv_path = os.path.join(csv_foler, image.replace(".jpg", ".csv"))
            image = cv2.imread(image_path)
            padded_image = add_padding(image)
            landmark_vector = get_landmark_vector(padded_image)
            save_landmarks_to_csv(landmark_vector, csv_path)
            if len(landmark_vector) != 956:
                # write to text file error file in args["data_preprocess_path"]
                with open(result_file, "a") as f:
                    f.write(f"Error in file {csv_path}\n")
        # write to error_file.txt when finish a fold
        with open(result_file, "a") as f:
            f.write(f"Finish fold {fold}\n")
                    
# Example usage:
# image_path = r'C:\Users\dangk\B4\shisen\data\MPIIFaceGaze\preprocessed\Image\p00\face\1.jpg'
# folder_path = r'C:\Users\dangk\B4\shisen\data\MPIIFaceGaze\preprocessed\Image\p00\face_lmk'
# csv_path = os.path.join(folder_path, '1.csv')

# # Create or reset folder
# create_or_reset_folder(folder_path)

# # Read image and process
# image = cv2.imread(image_path)
# padded_image = add_padding(image)

# # Get normalized landmark vector
# landmark_vector = get_landmark_vector(padded_image)

# Save landmarks to CSV
# save_landmarks_to_csv(landmark_vector, csv_path)

# print(f"Normalized Landmark Vector saved to {csv_path}")

if __name__ == '__main__':
    create_face_lmk_data()