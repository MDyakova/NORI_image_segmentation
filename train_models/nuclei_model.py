"""
Script to train model for nuclei NORI segmentation
"""

# import libraries
import os
import json

from dataset_utilities import (make_dataset_directory,
                       train_val_split)

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
    model_name = config['nuclei_unet_model']['model_information']['model_name']
    modifications = config['nuclei_unet_model']['model_information']['modifications']
    crop_size = config['nuclei_unet_model']['model_information']['crop_size']

    # Create train directory
    make_dataset_directory(os.path.join('datasets'), model_name)
    dataset_folder = os.path.join('datasets', model_name)

    # Split to train and validation subsets
    train_images, val_images = train_val_split(nori_images)

    time.sleep(10000)