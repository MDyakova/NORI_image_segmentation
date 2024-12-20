"""
Function to launch models
"""

# import libraries
import os
import shutil
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from imantics import Mask
import torch
from torch import nn
from torchvision import transforms
from sklearn.cluster import AgglomerativeClustering


def make_output_directory(output_folder):
    """
    Make directory for outputs
    """
    os.makedirs(output_folder, exist_ok=True)

    folders = ["images", "labels", "tiff_files"]
    for folder in folders:
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)


def image_filter(image, image_crop=None, is_crop=False):
    """
    Function remove outliers
    """
    all_layers = []
    if is_crop:
        for layer in range(0, 3):
            image_layer = image_crop[layer]
            all_percentile = []
            percentile_99 = np.percentile(image_layer, 99)
            image_layer = np.where(
                (image_layer > percentile_99), percentile_99, image_layer
            )
            # Normalize each layer
            image_layer = image_layer / image_layer.max()
            # Keep water layer is 0
            if layer == 2:
                image_layer = image_layer * 0
            all_layers.append(image_layer)
    else:
        for layer in range(0, 3):
            image_layer = image[layer]
            all_percentile = []
            # Compute percentile 99 for tiles to remove big extrime areas
            for step_i in range(image_layer.shape[0] // 256):
                for step_j in range(image_layer.shape[1] // 256):
                    image_layer_crop = image_layer[
                        step_i * 256 : (step_i + 1) * 256,
                        step_j * 256 : (step_j + 1) * 256,
                    ]
                    percentile_99 = np.percentile(image_layer_crop, 99)
                    all_percentile.append(percentile_99)
            percentile_99 = np.median(all_percentile)
            image_layer = np.where(
                (image_layer > percentile_99), percentile_99, image_layer
            )
            # Normalize each layer
            image_layer = image_layer / image_layer.max()
            # Keep water layer is 0
            if layer == 2:
                image_layer = image_layer * 0
            all_layers.append(image_layer)
    filtered_image = np.array(all_layers)
    return filtered_image


# polygons for test results
def get_polygons_predict(mask_label):
    """
    Convert masks to polygons
    """
    polygons = Mask((mask_label > 0)).polygons()
    polygons = polygons.points
    return polygons


def image_to_unet(image_crop, crop_size):
    """
    Function to load a single image for unet model
    """
    # Define the transformation to match the model input
    transform = transforms.Compose(
        [
            transforms.Resize(
                (crop_size, crop_size)
            ),  # Resize to match your input size
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    # Convert image to unet format
    img = Image.fromarray(image_crop.astype("uint8")).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension


def tubule_contours(
    tubule_results,
    mask_for_tubules,
    mask_for_prob,
    tubule_prob,
    width_crop,
    height_crop,
    crop_size,
    step_i,
    step_j,
    layer,
    mask_n,
):
    """
    Function convert yolo predictions to masks
    """
    for result in tubule_results:
        try:
            for mask, prob in zip(result.masks.data, result.boxes.conf.cpu().numpy()):
                if np.array(mask.cpu()).sum() > 2000:
                    if prob > tubule_prob:
                        # Convert mask to polygons
                        # Model can return complex masks.
                        # In this case function finds few polygons.
                        mask = np.array(mask.cpu()) * 255
                        mask = cv2.resize(mask, (width_crop, height_crop))
                        polygons = get_polygons_predict(mask)

                        # For each polygon set keep only with high area
                        all_poligons_i = []
                        for polygon in polygons:
                            x_coords, y_coords = polygon[:, 0], polygon[:, 1]
                            contour_i = np.array(
                                [[x_coords[step_i], y_coords[step_i]]
                                    for step_i in range(len(x_coords))],
                                dtype=np.int32
                            ).reshape((-1, 1, 2))
                            area_i = cv2.contourArea(contour_i)
                            if area_i > 2000:
                                all_poligons_i.append([polygon, area_i])

                        # For each good polygon make mask
                        # and add unique id and probability to whole masks
                        if len(all_poligons_i) > 0:
                            polygon = all_poligons_i[
                                np.argmax([i[1] for i in all_poligons_i])
                            ][0]
                            x_coords, y_coords = polygon[:, 0], polygon[:, 1]
                            contour_i = np.array(
                                [[x_coords[step_i], y_coords[step_i]]
                                    for step_i in range(len(x_coords))],
                                dtype=np.int32
                            ).reshape((-1, 1, 2))
                            mask = np.zeros((height_crop, width_crop), dtype=np.uint8)
                            cv2.drawContours(
                                mask,
                                [contour_i],
                                contourIdx=-1,
                                color=255,
                                thickness=cv2.FILLED,
                            )

                            mask_for_tubules[
                                step_i : step_i + crop_size,
                                step_j : step_j + crop_size,
                                layer,
                            ] = np.where(
                                mask > 0,
                                int(mask_n),
                                mask_for_tubules[
                                    step_i : step_i + crop_size,
                                    step_j : step_j + crop_size,
                                    layer,
                                ],
                            )
                            mask_for_prob[
                                step_i : step_i + crop_size, step_j : step_j + crop_size
                            ] = np.where(
                                (mask > 0),
                                prob * 100,
                                mask_for_prob[
                                    step_i : step_i + crop_size, step_j : step_j + crop_size
                                ],
                            )
                        mask_n += 1
        except Exception as e:
            print(e)
    return mask_for_tubules, mask_for_prob, mask_n


def join_pairs(pairs):
    """
    Function to join mask pairs
    """
    # Create a dictionary to map each element to its group
    groups = defaultdict(set)

    # Iterate over pairs and add them to the corresponding group
    for pair in pairs:
        member_a, member_b = pair
        # Merge the groups of a and b
        group_a = groups[member_a]
        group_b = groups[member_b]
        combined_group = group_a | group_b | {member_a, member_b}
        # Update both a and b's groups with the combined group
        for elem in combined_group:
            groups[elem] = combined_group

    # Remove duplicates and return the result
    unique_groups = set(frozenset(group) for group in groups.values())
    return [list(group) for group in unique_groups]


def find_similar_contours_fast(image_for_masks):
    """
    Function join small contours on different crops to big real size contours
    """
    mask_id, numbers = np.unique(image_for_masks.reshape(-1), return_counts=True)
    mask_numbers = dict(zip(mask_id, numbers))
    groups_short = [
        image_for_masks[i, j, :]
        for i in range(image_for_masks.shape[0])
        for j in range(image_for_masks.shape[1])
    ]
    groups_short = [sorted(list(set(group)))[1:] for group in groups_short]
    groups_short = [group for group in groups_short if len(group) > 1]

    # Dictionary to store pair counts
    pair_counts = defaultdict(int)
    # Loop through each sublist and generate pairs
    for sublist in groups_short:
        for pair in combinations(
            sorted(sublist), 2
        ):  # Generate sorted pairs to avoid (1, 2) and (2, 1) being counted separately
            pair_counts[pair] += 1
    # Convert the defaultdict to a regular dictionary for better readability (optional)
    pair_counts = dict(pair_counts)

    pairs = []
    for key, value in pair_counts.items():
        value_0 = mask_numbers[key[0]]
        value_1 = mask_numbers[key[1]]
        k = value / min(value_0, value_1)
        if k > 0.2:
            pairs.append(key)

    merged_pairs = join_pairs(pairs)

    for group in merged_pairs:
        main_id = group[0]
        for similar_id in group[1:]:
            image_for_masks[image_for_masks == similar_id] = main_id
    image_for_masks_max = image_for_masks.max(axis=2)
    return image_for_masks_max


# prompt: ierarchical clustering for lumen coords


def hierarchical_clustering(lumen_coords, distance_threshold=5, min_cluster_size=75):
    """
    Function join dark dots to clusters
    """

    # Reshape the coordinates for clustering
    coords = np.column_stack((lumen_coords[0], lumen_coords[1]))

    # Apply hierarchical clustering
    agg_clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold
    )
    agg_clustering.fit(coords)

    # Get the cluster labels
    cluster_labels = agg_clustering.labels_

    # # Count the number of points in each cluster
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    # Filter out clusters that are smaller than the minimum size
    valid_clusters = unique_labels[counts >= min_cluster_size]

    # # Create a mask for coordinates belonging to valid clusters
    valid_coords_mask = np.isin(cluster_labels, valid_clusters)

    # Filter the coordinates and labels based on the valid clusters
    filtered_coords = coords[valid_coords_mask]

    return filtered_coords


def lumen_predict(
    full_image, all_masks, lumen_coeff, distance_threshold, lumen_cluster_size
):
    """
    Function recognize lumen clusters on whole nori image
    """
    # Make layer for protein, lipid and water (still is 0)

    im_mean = np.array(full_image).max(axis=2)
    mask_for_lumen = np.zeros(im_mean.shape, dtype=np.uint8)

    # Find clusters inside each tubule contour
    all_tubules_id = pd.unique(all_masks.reshape(-1))
    for tubule_id in all_tubules_id:
        if tubule_id != 0:
            mask = np.zeros(all_masks.shape, dtype=np.uint8)
            mask[np.where(all_masks == tubule_id)] = 1
            mask_label = np.where(mask > 0, im_mean, 255)
            lumen_coords = np.where(mask_label < lumen_coeff)
            # Filter very extrime results for bit black areas
            if (len(lumen_coords[0]) > 30) & (len(lumen_coords[0]) < 100000):
                # Get cluster coordinates
                filtered_coords = hierarchical_clustering(
                    lumen_coords,
                    distance_threshold=distance_threshold,
                    min_cluster_size=lumen_cluster_size,
                )
                # Save lumen to mask
                for coord in filtered_coords:
                    mask_for_lumen[int(coord[0]), int(coord[1])] = 1
    return mask_for_lumen


def save_tiff(
    image,
    directory_tiff_save,
    file_name_save,
    all_masks,
    image_for_nucl,
    image_for_lumen,
    image_for_prob,
):
    """
    Save nori layers and all masks to tiff file
    """
    save_layers = [0, 1, 2]
    file_name_save = os.path.join(directory_tiff_save, file_name_save + ".tiff")
    tiff_layers = []
    for layer_id, layer in enumerate(image):
        if layer_id in save_layers:
            tiff_layers.append(Image.fromarray(layer))
    tiff_layers.append(Image.fromarray(all_masks.astype(np.uint8)))
    tiff_layers.append(Image.fromarray(image_for_nucl.astype(np.uint8)))
    tiff_layers.append(Image.fromarray(image_for_lumen.astype(np.uint8)))
    tiff_layers.append(Image.fromarray(image_for_prob.astype(np.uint8)))
    layers_converted = [layer.convert("F") for layer in tiff_layers]
    layers_converted[0].save(
        file_name_save, save_all=True, append_images=layers_converted[1:]
    )


class UNet(nn.Module):
    """
    Define the U-Net Model
    """

    def __init__(self):
        super(UNet, self).__init__()

        # Downsampling path
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Upsampling path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Add convolutional block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_x):
        """
        Main network architecture
        """
        enc1 = self.enc1(input_x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final(dec1))
