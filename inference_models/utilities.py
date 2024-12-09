"""
Function to launch models
"""

# import libraries
import sys
import os
# Append the parent directory of `train_models` to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

# polygons for test results
def get_polygons_predict(mask_label):
    """
    Convert masks to polygons
    """
    polygons = Mask((mask_label>0)).polygons()
    polygons = polygons.points
    return polygons

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

def tubule_contours(tubule_results,
                    mask_for_tubules,
                    mask_for_prob,
                    tubule_prob,
                    width_crop,
                    height_crop,
                    crop_size,
                    step_i,
                    step_j,
                    layer,
                    mask_n
                    ):
    """
    Function convert yolo predictions to masks
    """
    for result in tubule_results:
        try:
            for mask, prob in zip(result.masks.data, result.boxes.conf.cpu().numpy()):
                if (np.array(mask.cpu()).sum()>2000):
                    if prob>tubule_prob:
                        # Convert mask to polygons
                        # Model can return complex masks.
                        # In this case function finds few polygons.
                        mask = np.array(mask.cpu()) * 255
                        mask = cv2.resize(mask, (width_crop, height_crop))
                        polygons = get_polygons_predict(mask)

                        # For each polygon set keep only with high area
                        all_poligons_i = []
                        for polygon in polygons:
                            x, y = polygon[:, 0], polygon[:, 1]
                            contour_i = np.array([[x[i], y[i]] for i in range(len(x))], dtype=np.int32).reshape((-1, 1, 2))
                            area_i = cv2.contourArea(contour_i)
                            if area_i>2000:
                                all_poligons_i.append([polygon, area_i])

                        # For each good polygon make mask and add unique id and probability to whole masks
                        if len(all_poligons_i)>0:
                            polygon = all_poligons_i[np.argmax([i[1] for i in all_poligons_i])][0]
                            x, y = polygon[:, 0], polygon[:, 1]
                            contour_i = np.array([[x[i], y[i]] for i in range(len(x))], dtype=np.int32).reshape((-1, 1, 2))
                            mask = np.zeros((width_crop, height_crop), dtype=np.uint8)
                            cv2.drawContours(mask, [contour_i], contourIdx=-1, color=255, thickness=cv2.FILLED)

                            mask_for_tubules[step_i:step_i+crop_size, step_j:step_j+crop_size, layer] = np.where(mask>0, int(mask_n),
                                                                                mask_for_tubules[step_i:step_i+crop_size, step_j:step_j+crop_size, layer])
                            mask_for_prob[step_i:step_i+crop_size, step_j:step_j+crop_size] = np.where((mask>0), prob*100,
                                                                            mask_for_prob[step_i:step_i+crop_size, step_j:step_j+crop_size])
                        mask_n+=1
        except Exception as e:
            print(e)
    return mask_for_tubules, mask_for_prob, mask_n