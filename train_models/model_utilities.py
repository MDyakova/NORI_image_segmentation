"""
Function to train models
"""

# import libraries
import numpy as np


def make_yolo_config(config_path, object_type, model_name):
    """
    Create config for ultralitic library
    """
    with open(config_path,
            'w',
            encoding='utf-8') as file:
        file.write(f'path: {model_name}\n')
        file.write('train: images/train\n')
        file.write('val: images/val\n')
        file.write('nc: 1\n')
        file.write(f'names: ["{object_type}"]\n')