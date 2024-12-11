"""
Tests for inference functions
"""

# import libraries
import numpy as np
from utilities import (
    image_filter,
    get_polygons_predict,
    image_to_unet,
    find_similar_contours_fast,
    hierarchical_clustering,
)

def test_image_filter():
    """
    Test filter of extremal values for each nori layer
    """
    test_array = np.random.randint(low=0, high=1001, size=(3, 1000, 1000))
    changed_array = test_array.copy()
    test_array[0, 40:41, 40:41] = 1000
    changed_array[0, 40:41, 40:41] = 5000
    filtered_image = image_filter(changed_array)
    assert np.allclose(
        np.mean(test_array[0]) / 1000, np.mean(filtered_image[0]), atol=1e-2
    ), "image_filter function incorrect"


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


def test_image_to_unet():
    """
    Test function to transform array to unet format
    """
    test_array = np.random.randint(low=0, high=255, size=(1000, 1000, 3))
    transformed_image = image_to_unet(test_array, 640)

    assert list(transformed_image.shape) == [
        1,
        3,
        640,
        640,
    ], "image_to_unet function incorrect"

    assert transformed_image[0][0].mean() < 1, "image_to_unet function incorrect"


def test_find_similar_contours_fast():
    """
    Test function to join small contours to one
    """

    image_for_masks = np.zeros((1000, 1000, 4))
    image_for_masks[100:200, 100:200, 0] = 1
    image_for_masks[130:210, 130:210, 1] = 2
    image_for_masks[500:600, 500:600, 2] = 3
    all_contours = find_similar_contours_fast(image_for_masks)

    unique_id = np.unique(all_contours.reshape(-1), return_counts=True)
    assert (len(unique_id[0]) == 3) & (
        unique_id[1][1] == 11500
    ), "find_similar_contours_fast function incorrect"


def test_hierarchical_clustering():
    """
    Test hierarchical clustering for lumen
    """
    test_array = np.random.randint(low=120, high=255, size=(100, 100, 3))
    test_array[30:70, 50:70, :] = np.random.randint(low=0, high=90, size=(40, 20, 3))

    lumen_coords = np.where(test_array < 100)
    filtered_coords = hierarchical_clustering(
        lumen_coords, distance_threshold=20, min_cluster_size=30
    )
    assert (np.allclose(filtered_coords.max(), 70, atol=5)) & (
        np.allclose(filtered_coords.min(), 30, atol=5)
    ), "hierarchical_clustering function incorrect"
