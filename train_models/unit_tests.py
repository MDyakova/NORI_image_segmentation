"""
Unit tests for train functions
"""

# import libraries
import numpy as np
from PIL import Image
import torch
from dataset_utilities import (
    get_polygons_predict,
    change_resolution_pil,
    add_salt_and_pepper_noise,
)
from model_utilities import calculate_dice

def test_get_polygons_predict():
    """
    Test converting masks to polygons
    """

    test_mask = np.zeros((100, 100))
    test_mask[30:50, 40:70] = 1

    polygons = get_polygons_predict(test_mask)

    assert (polygons[0][0][0] == 40) & (
        polygons[0][0][1] == 30
    ), "get_polygons_predict function incorrect"


def test_change_resolution_pil():
    """
    Test resize of image
    """
    test_array = np.random.rand(100, 100)
    test_image = Image.fromarray(test_array)

    resized_image = change_resolution_pil(test_image, 20, 20)
    assert resized_image.size == (20, 20), "change_resolution_pil function incorrect"


def test_add_salt_and_pepper_noise():
    """
    Test function to add salt and papper noize
    """

    # Test salt modification
    test_array = np.zeros((100, 100))
    noisy_image_salt = add_salt_and_pepper_noise(
        test_array, salt_prob=0.1, pepper_prob=0
    )
    salt_pixels = np.mean(noisy_image_salt == 255)
    assert np.allclose(
        salt_pixels, 0.1, atol=1e-2
    ), "add_salt_and_pepper_noise function incorrect"

    # Test papper modification
    test_array = np.ones((100, 100))
    noisy_image_papper = add_salt_and_pepper_noise(
        test_array, salt_prob=0, pepper_prob=0.1
    )
    paper_pixels = np.mean(noisy_image_papper == 0)
    assert np.allclose(
        paper_pixels, 0.1, atol=1e-2
    ), "add_salt_and_pepper_noise function incorrect"


def test_calculate_dice():
    """
    Test function for dice coefficient
    """
    # Make test target array
    target_array = np.zeros((100, 100))
    target_array[0:10, 0:10] = 1
    target_array[20:30, 20:40] = 1
    target_array[50:80, 50:80] = 1
    # Make test prediction array
    preds_array = np.zeros((100, 100))
    preds_array[0:10, 0:10] = 1
    preds_array[20:30, 20:35] = 1
    preds_array[50:70, 50:70] = 1

    # Calculate dice
    dice_coeff = calculate_dice(
        torch.from_numpy(target_array), torch.from_numpy(preds_array), threshold=0.5
    )
    assert np.allclose(dice_coeff, 0.7, atol=1e-2), "calculate_dice function incorrect"
