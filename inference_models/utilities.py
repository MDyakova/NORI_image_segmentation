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