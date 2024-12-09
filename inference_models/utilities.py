"""
Function to launch models
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
import shutil
from matplotlib.patches import Polygon


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

def make_output_directory(output_folder):
    """
    Make directory for outputs
    """
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)

    folders = ['images', 'labels', 'tiff_files']
    for folder in folders:
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)

def image_filter(image, image_crop=None, is_crop=False):
    """
    Function remove outliers
    """
    all_layers = []
    if is_crop:
        for layer in range(0, 3):
            image_layer = image_crop[layer]
            image_layer_all = image[layer]
            all_percentile = []
            percentile_99 = np.percentile(image_layer, 99)
            image_layer = np.where((image_layer>percentile_99), percentile_99, image_layer)
            # Normalize each layer
            image_layer = image_layer/image_layer.max()
            # Keep water layer is 0
            if layer==2:
                image_layer = image_layer*0
            all_layers.append(image_layer)
    else:
        for layer in range(0, 3):
            image_layer = image[layer]
            all_percentile = []
            # Compute percentile 99 for tiles to remove big extrime areas
            for step_i in range(image_layer.shape[0]//256):
                for step_j in range(image_layer.shape[1]//256):
                    image_layer_crop = image_layer[step_i*256:(step_i+1)*256, step_j*256:(step_j+1)*256]
                    percentile_99 = np.percentile(image_layer_crop, 99)
                    all_percentile.append(percentile_99)
            percentile_99 = np.median(all_percentile)
            image_layer = np.where((image_layer>percentile_99), percentile_99, image_layer)
            # Normalize each layer
            image_layer = image_layer/image_layer.max()
            # Keep water layer is 0
            if layer==2:
                image_layer = image_layer*0
            all_layers.append(image_layer)
    filtered_image = np.array(all_layers)
    return filtered_image

#
def image_to_unet(image_crop, crop_size):
    """
    Function to load a single image for unet model
    """
    # Define the transformation to match the model input
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),  # Resize to match your input size
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # Convert image to unet format
    img = Image.fromarray(image_crop.astype('uint8')).convert("RGB")
    # img = Image.fromarray(image).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension
