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

def load_images(nori_images,
                file_name,
                protein_layer,
                lipid_layer,
                tubule_masks_layer,
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
                image_layer = image_layer/image_layer.max()
                if layer==2:
                    image_layer = image_layer*0

                all_layers.append(image_layer)
            image_crop = np.array(all_layers)
            all_images.append(image_crop)
            crop_name = '_'.join([file_name.split('.')[0], str(i), str(j)])
            all_images_names.append(crop_name)
            all_masks.append(mask_crop)

    return all_images, all_masks, all_images_names
