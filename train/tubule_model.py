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
from utilities import (get_polygons,
                       make_dataset_directory,
                       load_images,
                       save_polygons,
                       save_subset)

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
dataset_folder = os.path.join('files', 'train', model_name)

# Split to train and validation subsets
all_files = os.listdir(nori_images)
test_groups = all_files[int(len(all_files)*0.8):]

train_images = list(filter(lambda p: (p not in test_groups), all_files))
val_images = list(filter(lambda p: (p in test_groups), all_files))

# save train subset
save_subset(train_images,
            'train',
            modifications=modifications)
# save validation subset
save_subset(val_images,
            'val',
            modifications=False)
# for file_name in train_images[0:]:
#     file_name_save = file_name.split('.')[0]
#     directory = 'train'

#     (all_images,
#      all_masks,
#      all_images_names) = load_images(nori_images,
#                                     file_name,
#                                     protein_layer,
#                                     lipid_layer,
#                                     tubule_masks_layer,
#                                     crop_size=crop_size)

#     for image, mask, image_name in zip(all_images, all_masks, all_images_names):
#         # Extract all pokygons from mask
#         all_polygons = get_polygons(mask)
#         # Convert array to RGB format
#         im = (image*255).transpose((1, 2, 0)).astype(np.uint8)
#         width = im.shape[0]
#         height = im.shape[1]
#         # Two modes for train dataset
#         if modifications:
#             # Apply specific augmentation
#             for k in np.arange(1, 2.5, 1):
#                 for prob in np.arange(0, 0.02, 0.01):
#                     if (k>=1) | (prob>=0):
#                         resized_image = change_resolution_pil(Image.fromarray(im),
#                                                               int(width/k),
#                                                               int(height/k))
#                         restore_image = change_resolution_pil(resized_image,
#                                                               int(width),
#                                                               int(height))
#                         noise_im = add_salt_and_pepper_noise(np.array(restore_image),
#                                                              salt_prob=prob,
#                                                              pepper_prob=prob)
#                         # Keep water layer 0
#                         noise_im[:, :, 2] = 0
#                         # Convert to image
#                         noise_im = Image.fromarray(noise_im)
#                         file_name_save_new = '_'.join([image_name, str(k), str(prob)]) + '.jpg'
#                         file_path_save = os.path.join(dataset_folder,
#                                                       'images',
#                                                       directory,
#                                                       file_name_save_new)
#                         noise_im.save(file_path_save,
#                                       format="JPEG",
#                                       quality=100,
#                                       optimize=False)
#                         save_polygons(all_polygons,
#                                       file_name_save_new,
#                                       directory,
#                                       dataset_folder)
#         else:
#             file_name_save_new = image_name + '.jpg'
#             file_path_save = os.path.join(dataset_folder,
#                                           'images',
#                                            directory,
#                                            file_name_save_new)
#             im = Image.fromarray(im)
#             im.save(file_path_save,
#                     format="JPEG",
#                     quality=100,
#                     optimize=False)
#             save_polygons(all_polygons,
#                           file_name_save,
#                           directory,
#                           dataset_folder)

# # save validation subset
# for file_name in val_images[0:]:
#     file_name_save = file_name.split('.')[0]
#     directory = 'val'

#     (all_images,
#      all_masks,
#      all_images_names) = load_images(nori_images,
#                                     file_name,
#                                     protein_layer,
#                                     lipid_layer,
#                                     tubule_masks_layer,
#                                     crop_size=crop_size)

#     for image, mask, image_name in zip(all_images, all_masks, all_images_names):
#         # Extract all pokygons from mask
#         all_polygons = get_polygons(mask)
#         # Convert array to RGB format
#         im = (image*255).transpose((1, 2, 0)).astype(np.uint8)
#         width = im.shape[0]
#         height = im.shape[1]

#         file_name_save_new = image_name + '.jpg'
#         file_path_save = os.path.join(dataset_folder,
#                                         'images',
#                                         directory,
#                                         file_name_save_new)
#         im = Image.fromarray(im)
#         im.save(file_path_save,
#                 format="JPEG",
#                 quality=100,
#                 optimize=False)
#         save_polygons(all_polygons,
#                         file_name_save,
#                         directory,
#                         dataset_folder)

time.sleep(1000)
