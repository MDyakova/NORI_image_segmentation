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
