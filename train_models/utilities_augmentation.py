"""
Function for data augmentation
"""

import numpy as np
from PIL import Image

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Function add noise like small white and black dots.
    """
    noisy_image = np.copy(image)
    salt_mask = np.random.choice([0, 1], size=image.shape, p=[1 - salt_prob, salt_prob])
    pepper_mask = np.random.choice([0, 1], size=image.shape, p=[1 - pepper_prob, pepper_prob])
    noisy_image[salt_mask == 1] = 255  # Salt
    noisy_image[pepper_mask == 1] = 0  # Pepper
    return noisy_image

def change_resolution_pil(image, width, height):
    """
    Function dicrease image resolution
    for bad quality of data
    """
    resized_image = image.resize((width, height),
                                 Image.Resampling.LANCZOS)
    return resized_image

