"""
Script to launch all model segmentation for new samples
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

from utilities import make_output_directory
from train_models.model_utilities import UNet

import time

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

    # Create output directory
    make_output_directory(output_folder)

    # Load YOLO model
    tubule_model = YOLO(tubule_model_path)

    # Load Unet model
    model_nuclei = UNet().to(device)
    model_nuclei.load_state_dict(torch.load(nuclei_model_path,
                                            map_location=device))
    model_nuclei.eval()


