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
from tqdm import tqdm_notebook
import json

import zipfile
from ultralytics import YOLO
import shutil
from matplotlib.patches import Polygon
from utilities import make_dataset_directory
import time

# Load config

with open(os.path.join(os.path.join('files', 'user_config.json')), 'r', encoding='utf-8') as f:
    config = json.load(f)

# Sample's info
nori_images = config['data_information']['nori_images']
protein_layer = config['data_information']['protein_layer']
lipid_layer = config['data_information']['lipid_layer']

# Model's info
model_name = config['model_information']['model_name']

# Create train directory
make_dataset_directory(os.path.join('files', 'train'), model_name)

time.sleep(1000)
