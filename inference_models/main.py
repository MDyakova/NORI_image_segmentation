"""
Script to launch all model segmentation for new samples
"""

import sys
import os
# Append the parent directory of `train_models` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import tifffile
from tifffile import TiffFile
import cv2
import json
import os
from imantics import Polygons, Mask
from tqdm import tqdm_notebook

import zipfile
import shutil
from matplotlib.patches import Polygon

from ultralytics import YOLO
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models

from utilities import (make_output_directory,
                       image_filter,
                       image_to_unet)
from train_models.model_utilities import UNet

import time

# time.sleep(10000)

if __name__ == "__main__":

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(os.path.join(os.path.join('inference_directory', 'inference_config.json')),
            'r',
            encoding='utf-8') as f:
        config = json.load(f)

    # Sample's info
    nori_images = config['data_information']['nori_images']
    protein_layer = config['data_information']['protein_layer']
    lipid_layer = config['data_information']['lipid_layer']

    # Output's info
    output_folder = config['output_information']['output_folder']

    # Model's info
    tubule_model_path = config['models']['tubule_model']
    nuclei_model_path = config['models']['nuclei_model']
    crop_size = config['models']['crop_size']

    # Create output directory
    make_output_directory(output_folder)
    images_save = os.path.join(output_folder, 'images')
    labels_save = os.path.join(output_folder, 'labels')
    tiff_save = os.path.join(output_folder, 'tiff_files')

    # Load YOLO model
    tubule_model = YOLO(tubule_model_path)

    # Load Unet model
    nuclei_model = UNet().to(device)
    nuclei_model.load_state_dict(torch.load(nuclei_model_path,
                                            map_location=device))
    nuclei_model.eval()

    # All test samples
    image_names = os.listdir(nori_images)
    # Existing results
    saved_labels = os.listdir(labels_save)

    for file_name in image_names:
        file_name_save = file_name.split('.')[0]
        labels_name = file_name_save + '.csv'
        if ('.tif' in file_name) & (labels_name not in saved_labels):
            # Load image
            image_path = os.path.join(nori_images, file_name)
            with TiffFile(image_path) as tif:
                image = tif.asarray()
            # Filter outliers
            image_full = image_filter(image)
            image_full = Image.fromarray((image_full*255).transpose((1, 2, 0)).astype(np.uint8))

            height, width = image[0].shape
            # Make empty masks for all segmented objects
            mask_for_tubules = np.zeros((height, width, 4), dtype=int)
            mask__for_nuclei = np.zeros((height, width), dtype=int)
            mask__for_lumen = np.zeros((height, width), dtype=int)
            mask__for_prob = np.zeros((height, width), dtype=int)
            step = 0
            mask_n = 1
            mask_nucl_id = 1
            mask_lumen_id = 1
            nucleus_polygons = []
            lumen_polygons = []

            # Make tiles with 50% interchange
            for i in range(0, height - crop_size//2, crop_size//2):
                for j in range(0, width - crop_size//2, crop_size//2):
                    # Determine mask layer to save polygons for crop
                    if (i%crop_size==0) & (j%crop_size==0):
                        layer = 0
                    elif (i%crop_size==0) & (j%crop_size!=0):
                        layer = 1
                    elif (i%crop_size!=0) & (j%crop_size==0):
                        layer = 2
                    else:
                        layer = 3

                    # Make crop and filter them
                    image_crop = image[:, i:i+crop_size, j:j+crop_size]
                    image_crop = image_filter(image, image_crop, is_crop=True)
                    image_crop = (image_crop*255).transpose((1, 2, 0)).astype(np.uint8)
                    # image_crop = (image_crop * 255).astype(np.uint8)
                    height_crop, width_crop, _ = image_crop.shape

                    # Get tubule predictions
                    tubule_results = tubule_model(image_crop)

                    # Get nuclei predictions
                    image_for_unet = image_to_unet(image_crop, crop_size).to(device)
                    nuclei_results = nuclei_model(image_for_unet)
                    nuclei_results = (nuclei_results > 0.9).float()  # Binarize prediction
                    nuclei_results = nuclei_results.squeeze().cpu().numpy()
                    nuclei_results = np.array(Image.fromarray(nuclei_results.astype(np.uint8)*255)
                                              .resize((height_crop, width_crop)))
                    break
                break
            break

time.sleep(1000)

