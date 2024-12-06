"""
Script to train model for tubule NORI segmentation
"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import tifffile
from tifffile import TiffFile
import cv2
import os
from imantics import Polygons, Mask
import json

import zipfile
from ultralytics import YOLO
import shutil
from matplotlib.patches import Polygon
from utilities import (make_dataset_directory,
                       load_images)
import time

# Load config

with open(os.path.join(os.path.join('files', 'user_config.json')), 'r', encoding='utf-8') as f:
    config = json.load(f)

# Sample's info
nori_images = config['data_information']['nori_images']
protein_layer = config['data_information']['protein_layer']
lipid_layer = config['data_information']['lipid_layer']
tubule_masks_layer = config['data_information']['tubule_masks_layer']

# Model's info
model_name = config['model_information']['model_name']
modifications = config['model_information']['modifications']
crop_size = config['model_information']['crop_size']

# Create train directory
make_dataset_directory(os.path.join('files', 'train'), model_name)

# Split to train and validation subsets
all_files = os.listdir(nori_images)
test_groups = all_files[int(len(all_files)*0.8):]

train_images = list(filter(lambda p: (p not in test_groups), all_files))
val_images = list(filter(lambda p: (p in test_groups), all_files))

# save train subset
for file_name in train_images[0:]:
    file_name_save = file_name.split('.')[0]
    directory = 'train'

    (all_images,
     all_masks,
     all_images_names) = load_images(nori_images,
                                    file_name,
                                    protein_layer,
                                    lipid_layer,
                                    tubule_masks_layer,
                                    crop_size=crop_size)

time.sleep(1000)
