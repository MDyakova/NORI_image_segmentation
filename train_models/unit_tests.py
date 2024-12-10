"""
Unit tests for main functions
"""

# import libraries
import os
import json
import numpy as np
from ultralytics import YOLO
from dataset_utilities import (get_polygons_predict)
from model_utilities import make_yolo_config

def test_get_polygons_predict():
    """
    Test converting masks to polygons
    """

    test_mask = np.zeros((100, 100))
    test_mask[30:50, 40:70] = 1

    polygons = get_polygons_predict(test_mask)

    assert ((polygons[0][0][0] == 40)
            & (polygons[0][0][1] == 30)
            ), "get_polygons_predict function incorrect"
