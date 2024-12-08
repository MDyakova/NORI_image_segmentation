"""
Script to train model for tubule NORI segmentation
"""

# import libraries
import os
import json
from ultralytics import YOLO
from dataset_utilities import ( save_subset,
                                   make_yolo_config,
                                   make_dataset_directory,
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
    model_name = config['tubule_yolo_model']['model_information']['model_name']
    modifications = config['tubule_yolo_model']['model_information']['modifications']
    crop_size = config['tubule_yolo_model']['model_information']['crop_size']

    # Train config
    epochs = config['tubule_yolo_model']['train_config']['epochs']
    imgsz = config['tubule_yolo_model']['train_config']['imgsz']
    batch = config['tubule_yolo_model']['train_config']['batch']
    patience = config['tubule_yolo_model']['train_config']['patience']
    overlap_mask = config['tubule_yolo_model']['train_config']['overlap_mask']
    object_type = config['tubule_yolo_model']['train_config']['object_type']

    # Create train directory
    make_dataset_directory(os.path.join('datasets'), model_name)
    dataset_folder = os.path.join('datasets', model_name)

    # Update a setting
    # settings.update({"datasets_dir": dataset_folder})

    # Split to train and validation subsets
    train_images, val_images = train_val_split(nori_images)

    # save train subset
    save_subset(train_images,
                'train',
                dataset_folder,
                crop_size,
                target_type='labels',
                modifications=modifications)
    # save validation subset
    save_subset(val_images,
                'val',
                dataset_folder,
                crop_size,
                target_type='labels',
                modifications=False)

    # # Create config for yolo training
    config_path = os.path.join('datasets',
                            model_name + '.yaml')
    make_yolo_config(config_path, object_type, model_name)

    # load a pretrained model (recommended for training)
    model = YOLO('yolov8n-seg.pt')
    model_dir = os.path.join('train_directory',
                                    'models')

    # train the model
    model.train(data=config_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                overlap_mask=overlap_mask,
                name=model_name,
                exist_ok=True,
                project=model_dir)

    # Rename model
    model_path = os.path.join(model_dir, model_name, 'weights', 'best.pt')
    model_custom_path = os.path.join(model_dir, model_name, 'weights', model_name + '.pt')
    os.rename(model_path, model_custom_path)
