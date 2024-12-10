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
from matplotlib.colors import ListedColormap

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
                       image_to_unet,
                       tubule_contours,
                       find_similar_contours_fast,
                       get_polygons_predict,
                       lumen_predict,
                       save_tiff)

from train_models.model_utilities import UNet
# from train_models.dataset_utilities import get_polygons_predict


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
    tubule_prob = config['models']['tubule_prob']
    nuclei_prob = config['models']['nuclei_prob']
    lumen_coef = config['models']['lumen_coef']
    lumen_cluster_size = config['models']['lumen_cluster_size']
    distance_threshold = config['models']['distance_threshold']

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
            mask_for_nuclei = np.zeros((height, width), dtype=int)
            mask_for_prob = np.zeros((height, width), dtype=int)
            mask_n = 1

            # Make tiles with 50% interchange
            for step_i in range(0, height - crop_size//2, crop_size//2):
                for step_j in range(0, width - crop_size//2, crop_size//2):
                    # Determine mask layer to save polygons for crop
                    if (step_i%crop_size==0) & (step_j%crop_size==0):
                        layer = 0
                    elif (step_i%crop_size==0) & (step_j%crop_size!=0):
                        layer = 1
                    elif (step_i%crop_size!=0) & (step_j%crop_size==0):
                        layer = 2
                    else:
                        layer = 3

                    # Make crop and filter them
                    image_crop = image[:, step_i:step_i+crop_size, step_j:step_j+crop_size]
                    image_crop = image_filter(image, image_crop, is_crop=True)
                    image_crop = (image_crop*255).transpose((1, 2, 0)).astype(np.uint8)
                    image_crop_for_yolo = cv2.cvtColor(image_crop, cv2.COLOR_RGB2BGR)
                    height_crop, width_crop, _ = image_crop.shape

                    # Get tubule predictions
                    tubule_results = tubule_model(image_crop_for_yolo)

                    # Get nuclei predictions
                    image_for_unet = image_to_unet(image_crop, crop_size).to(device)

                    with torch.no_grad():
                        nuclei_results = nuclei_model(image_for_unet)
                        nuclei_results = (nuclei_results > nuclei_prob).float()  # Binarize prediction
                        nuclei_results = nuclei_results.squeeze().cpu().numpy()
                    nuclei_results = np.array(Image.fromarray(nuclei_results.astype(np.uint8)*255)
                                              .resize((width_crop, height_crop)))

                    # Add tubule polygons to whole masks
                    (mask_for_tubules,
                     mask_for_prob,
                     mask_n) = tubule_contours(tubule_results,
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
                                                )

                    # Add nuclei polygons to whole masks
                    mask_for_nuclei[step_i:step_i+crop_size,
                                    step_j:step_j+crop_size] = np.where(nuclei_results>0, 1,
                                                mask_for_nuclei[step_i:step_i+crop_size,
                                                               step_j:step_j+crop_size])

            # Join all crop masks to real contours
            all_masks = find_similar_contours_fast(mask_for_tubules)
            all_contours = list(np.sort(pd.unique(all_masks.reshape(-1)))[::-1])

            # Make picture for control
            fig, ax = plt.subplots(1, 2, figsize=(36, 18))
            ax[0].imshow(image_full)
            ax[0].set_title(file_name_save)
            ax[1].imshow(image_full)

            # Add tubule contours to image and save polygon coordinates
            polygons_save = []
            while len(all_contours)>0:
                contour_id = all_contours.pop()
                if contour_id>0:
                    mask = np.where(all_masks == contour_id, 1, 0)
                    polygons = get_polygons_predict(mask)
                    for polygon in polygons:
                        x, y = polygon[:, 0], polygon[:, 1]
                        x = np.append(x, x[0])
                        y = np.append(y, y[0])
                        ax[1].add_patch(Polygon(np.stack([x, y]).T,
                                                fill=False,
                                                edgecolor='white',
                                                  linewidth=2))
                        polygons_save.append([file_name_save, 'tubule', x, y])

            polygons_save = pd.DataFrame(polygons_save,
                                         columns=['file_name',
                                                    'type', 'x', 'y'])
            polygons_save.to_csv(os.path.join(labels_save,
                                              file_name_save + '.csv'),
                                              sep=';', index=False)

            # Found lumen for each tubule contour
            mask_for_lumen = lumen_predict(image_full,
                                            all_masks,
                                            lumen_coef,
                                            distance_threshold,
                                            lumen_cluster_size)

            # Save results to tiff file
            save_tiff(image,
                      tiff_save,
                      file_name_save,
                      all_masks,
                      mask_for_nuclei,
                      mask_for_lumen,
                      mask_for_prob)

            # Add nuclei and lumen masks to control image
            cmap = ListedColormap(['none', 'blue'])
            ax[1].imshow(mask_for_lumen, cmap=cmap, alpha=0.8)

            cmap = ListedColormap(['none', 'white'])
            ax[1].imshow(mask_for_nuclei, cmap=cmap, alpha=0.8)

            # Save control image
            file_path_save = os.path.join(images_save,
                                          file_name_save + '.png')
            plt.savefig(file_path_save, dpi=600, format='png', bbox_inches='tight')
            plt.close()

# time.sleep(1000)

