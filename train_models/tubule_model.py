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
import json

import zipfile
from ultralytics import YOLO, settings
import shutil
from matplotlib.patches import Polygon
from yolo_model_utilities import (make_dataset_directory,
                                  save_subset,
                                   make_model_config)
import time

if __name__ == "__main__":

    # Load config
    with open(os.path.join(os.path.join('train_directory','user_config.json')), 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Sample's info
    nori_images = config['data_information']['nori_images']
    protein_layer = config['data_information']['protein_layer']
    lipid_layer = config['data_information']['lipid_layer']
    tubule_masks_layer = config['data_information']['tubule_masks_layer']

    # Model's info
    model_name = config['model_information']['model_name']
    modifications = config['model_information']['modifications']
    crop_size = config['model_information']['crop_size']

    # Train config
    model_config = config['train_config']['data']
    epochs = config['train_config']['epochs']
    imgsz = config['train_config']['imgsz']
    batch = config['train_config']['batch']
    patience = config['train_config']['patience']
    overlap_mask = config['train_config']['overlap_mask']
    augment = config['train_config']['augment']
    object_type = config['train_config']['object_type']

    # # Create train directory
    make_dataset_directory(os.path.join('datasets'), model_name)
    dataset_folder = os.path.join('datasets', model_name)

    # Update a setting
    # settings.update({"datasets_dir": dataset_folder})

    # Split to train and validation subsets
    all_files = os.listdir(nori_images)
    test_groups = all_files[int(len(all_files)*0.8):]

    train_images = list(filter(lambda p: (p not in test_groups), all_files))
    val_images = list(filter(lambda p: (p in test_groups), all_files))

    # save train subset
    save_subset(train_images,
                'train',
                modifications=modifications)
    # save validation subset
    save_subset(val_images,
                'val',
                modifications=False)

    # # Create config for yolo training
    config_path = os.path.join('datasets',
                            model_name + '.yaml')
    make_model_config(config_path, object_type)

    # load a pretrained model (recommended for training)
    model = YOLO('yolov8n-seg.pt')

    # train the model
    model.train(data=config_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                overlap_mask=overlap_mask,
                augment=augment,
                name=model_name,
                exist_ok=True,
                project=os.path.join('train_directory',
                                    'models')
                                    )
    time.sleep(1000)
