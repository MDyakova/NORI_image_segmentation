"""
Functions for train scripts
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
from tqdm import tqdm_notebook

import zipfile
from ultralytics import YOLO
import shutil
from matplotlib.patches import Polygon
import json

from utilities_augmentation import (add_salt_and_pepper_noise,
                                    change_resolution_pil)

# Load config

with open(os.path.join(os.path.join('train_directory', 'user_config.json')), 'r', encoding='utf-8') as f:
    config = json.load(f)

# Sample's info
nori_images = config['data_information']['nori_images']
protein_layer = config['data_information']['protein_layer']
lipid_layer = config['data_information']['lipid_layer']
tubule_masks_layer = config['data_information']['tubule_masks_layer']

# Model's info
model_name = config['tubule_yolo_model']['model_information']['model_name']
modifications = config['tubule_yolo_model']['model_information']['modifications']
crop_size = config['tubule_yolo_model']['model_information']['crop_size']

# Path to save dataset
dataset_folder = os.path.join('datasets', model_name)

def make_dataset_directory(dataset_folder, dataset_name):
    """
    Make directory for dataset
    """
    shutil.rmtree(os.path.join(dataset_folder, dataset_name),
                  ignore_errors=True)
    os.makedirs(os.path.join(dataset_folder, dataset_name),
                exist_ok=True)
    for folder in ['images', 'labels']:
        for set_folder in ['train', 'val']:
            os.makedirs(os.path.join(os.path.join(dataset_folder, dataset_name),
                                     folder, set_folder), exist_ok=True)

# polygons for test results
def get_polygons_predict(mask_label):
    """
    Convert masks to polygons
    """
    polygons = Mask((mask_label>0)).polygons()
    polygons = polygons.points
    return polygons

def get_polygons(mask):
    """
    Convert masks to polygons and fix sizes
    """
    height, width = mask.shape
    all_polygons = []

    # Get coordinates
    polygons = get_polygons_predict(mask)

    for polygon in polygons:
        x, y = polygon[:, 0], polygon[:, 1]
        contour = np.array([[x[i], y[i]] for i in range(len(x))],
                           dtype=np.int32).reshape((-1, 1, 2))
        area = cv2.contourArea(contour)
        # Filter extrimal values
        if (area>100) & (len(x)>10):
            mask_i = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask_i,
                             [contour],
                             contourIdx=-1,
                             color=255,
                             thickness=cv2.FILLED)
            # Increase polygon size
            kernel = np.ones((3, 3), np.uint8)
            mask_new = cv2.dilate(mask_i,
                                  kernel,
                                  iterations=4)
            # Get new coordinates
            polygons_new = get_polygons_predict(mask_new)
            polygon_nornalized = []
            for (x,y) in polygons_new[0]:
                polygon_nornalized.append(x/width)
                polygon_nornalized.append(y/height)
            all_polygons.append(polygon_nornalized)
    return all_polygons

def load_images(file_name,
                crop_size=640):
    """
    Load and preprocessing big tiff images and masks to tiles
    """
    with TiffFile(os.path.join(nori_images, file_name)) as tif:
      image = tif.asarray()

      # Extract masks
      mask = image[tubule_masks_layer]

      # Extract nori_data
      protein = image[protein_layer, :, :]
      lipid = image[lipid_layer, :, :]
      # While is empty
      water = image[lipid_layer, :, :]*0
      image = np.stack([protein, lipid, water], axis=0)

    height, width = image[0].shape
    all_images = []
    all_masks = []
    all_images_names = []

    # Make crops with 50% overlap
    for i in range(0, height - crop_size, crop_size//2):
        for j in range(0, width - crop_size, crop_size//2):
            image_crop = image[:, i:i+crop_size, j:j+crop_size]
            mask_crop = mask[i:i+crop_size, j:j+crop_size]
            all_layers = []
            # Filter extremal higher values
            for layer in range(0, 3):
                image_layer = image_crop[layer]
                percentile_99 = np.percentile(image_layer, 99)
                image_layer = np.where((image_layer>percentile_99), percentile_99, image_layer)
                if layer==2:
                    image_layer = image_layer*0
                else:
                    image_layer = image_layer/image_layer.max()

                all_layers.append(image_layer)
            image_crop = np.array(all_layers)
            all_images.append(image_crop)
            crop_name = '_'.join([file_name.split('.')[0], str(i), str(j)])
            all_images_names.append(crop_name)
            all_masks.append(mask_crop)

    return all_images, all_masks, all_images_names

# save polygons to txt
def save_polygons(all_polygons, file_name_save, directory):
    """
    Save polygon coordinates to txt
    """
    with open(os.path.join(dataset_folder,
                           'labels',
                           directory,
                           file_name_save.replace('.jpg', '.txt')),
                           'w', encoding='utf-8') as file:
      for polygon in all_polygons:
          for p_, p in enumerate(polygon):
              if p_ == len(polygon) - 1:
                  file.write('{}\n'.format(p))
              elif p_ == 0:
                  file.write('0 {} '.format(p))
              else:
                  file.write('{} '.format(p))

def save_subset(images, directory, modifications=False):
    """
    Save train and validation datasets for YOLO segmentation format
    """
    for file_name in images[0:]:
        file_name_save = file_name.split('.')[0]

        (all_images,
        all_masks,
        all_images_names) = load_images(file_name,
                                        crop_size=crop_size)

        for image, mask, image_name in zip(all_images, all_masks, all_images_names):
            # Extract all pokygons from mask
            all_polygons = get_polygons(mask)
            # Convert array to RGB format
            im = (image*255).transpose((1, 2, 0)).astype(np.uint8)
            width = im.shape[0]
            height = im.shape[1]
            # Two modes for train dataset
            if modifications:
                # Apply specific augmentation
                for k in np.arange(1, 2.5, 1):
                    for prob in np.arange(0, 0.02, 0.01):
                        if (k>=1) | (prob>=0):
                            resized_image = change_resolution_pil(Image.fromarray(im),
                                                                int(width/k),
                                                                int(height/k))
                            restore_image = change_resolution_pil(resized_image,
                                                                int(width),
                                                                int(height))
                            noise_im = add_salt_and_pepper_noise(np.array(restore_image),
                                                                salt_prob=prob,
                                                                pepper_prob=prob)
                            # Keep water layer 0
                            noise_im[:, :, 2] = 0
                            # Convert to image
                            noise_im = Image.fromarray(noise_im)
                            file_name_save_new = '_'.join([image_name, str(k), str(prob)]) + '.jpg'
                            file_path_save = os.path.join(dataset_folder,
                                                        'images',
                                                        directory,
                                                        file_name_save_new)
                            noise_im.save(file_path_save,
                                        format="JPEG",
                                        quality=100,
                                        optimize=False)
                            save_polygons(all_polygons,
                                        file_name_save_new,
                                        directory)
            else:
                file_name_save_new = image_name + '.jpg'
                file_path_save = os.path.join(dataset_folder,
                                            'images',
                                            directory,
                                            file_name_save_new)
                im = Image.fromarray(im)
                im.save(file_path_save,
                        format="JPEG",
                        quality=100,
                        optimize=False)
                save_polygons(all_polygons,
                            file_name_save_new,
                            directory)

def make_model_config(config_path, object_type):
    """
    Create config for ultralitic library
    """
    with open(config_path,
            'w',
            encoding='utf-8') as file:
        file.write(f'path: {model_name}\n')
        file.write('train: images/train\n')
        file.write('val: images/val\n')
        file.write('nc: 1\n')
        file.write(f'names: ["{object_type}"]\n')
