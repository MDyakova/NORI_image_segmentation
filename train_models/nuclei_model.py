"""
Script to train model for nuclei NORI segmentation
"""

# import libraries
import os
import json
from torch.utils.data import Dataset, DataLoader

from dataset_utilities import (make_dataset_directory,
                                train_val_split,
                                save_subset)
from model_utilities import (UnetDataset,
                             get_train_augmentations,
                             get_val_augmentations,
                             unet_train)

import time

if __name__ == "__main__":

    # Load config
    with open(os.path.join(os.path.join('train_directory','train_config.json')), 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Sample's info
    nori_images = config['data_information']['nori_images']
    protein_layer = config['data_information']['protein_layer']
    lipid_layer = config['data_information']['lipid_layer']
    tubule_masks_layer = config['data_information']['tubule_masks_layer']
    nuclei_masks_layer = config['data_information']['tubule_masks_layer']

    # Model's info
    model_name = config['nuclei_unet_model']['model_information']['model_name']
    modifications = config['nuclei_unet_model']['model_information']['modifications']
    crop_size = config['nuclei_unet_model']['model_information']['crop_size']

    # Train config
    epochs = config['nuclei_unet_model']['model_config']['epochs']
    batch = config['nuclei_unet_model']['model_config']['batch']
    patience = config['nuclei_unet_model']['model_config']['patience']
    learning_rate = config['nuclei_unet_model']['model_config']['learning_rate']

    # Create train directory
    make_dataset_directory(os.path.join('datasets'),
                           model_name,
                           target_name='masks')
    dataset_folder = os.path.join('datasets', model_name)

    # Split to train and validation subsets
    train_images, val_images = train_val_split(nori_images)

    # save train subset
    save_subset(train_images,
                'train',
                dataset_folder,
                crop_size,
                object='nuclei',
                modifications=modifications,
                model_type='unet')
    # save validation subset
    save_subset(val_images,
                'val',
                dataset_folder,
                crop_size,
                object='nuclei',
                modifications=False,
                model_type='unet')

    # Dataset directories
    train_image_dir = os.path.join(dataset_folder, 'images', 'train')
    train_mask_dir = os.path.join(dataset_folder, 'masks', 'train')

    val_image_dir = os.path.join(dataset_folder, 'images', 'val')
    val_mask_dir = os.path.join(dataset_folder, 'masks', 'val')


    # Define training and validation datasets
    train_dataset = UnetDataset(image_dir=train_image_dir,
                                mask_dir=train_mask_dir,
                                augmentations=get_train_augmentations())
    val_dataset = UnetDataset(image_dir=val_image_dir,
                                mask_dir=val_mask_dir,
                                augmentations=get_val_augmentations())

    # Define DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # Train model
    model_dir = os.path.join('train_directory',
                                    'models')
    unet_train(train_loader,
               val_loader,
               model_dir,
               model_name,
               num_epochs=epochs,
               lr=learning_rate,
               patience=patience)


    time.sleep(10000)