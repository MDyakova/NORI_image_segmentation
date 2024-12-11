"""
Functions to prepare datasets for models
"""

# import libraries
import os
import shutil
import numpy as np
from PIL import Image
import cv2
from tifffile import TiffFile
from imantics import Mask

def make_dataset_directory(dataset_folder, dataset_name, target_name="labels"):
    """
    Make directory for dataset
    """
    shutil.rmtree(os.path.join(dataset_folder, dataset_name), ignore_errors=True)
    os.makedirs(os.path.join(dataset_folder, dataset_name), exist_ok=True)
    if target_name == "labels":
        folders = ["images", "labels"]
    elif target_name == "masks":
        folders = ["images", "masks"]
    for folder in folders:
        for set_folder in ["train", "val"]:
            os.makedirs(
                os.path.join(dataset_folder, dataset_name, folder, set_folder),
                exist_ok=True,
            )


def train_val_split(nori_images):
    """
    Split dataset to train and validation subsets
    """
    all_files = os.listdir(nori_images)
    test_groups = all_files[int(len(all_files) * 0.8) :]

    train_images = list(filter(lambda p: (p not in test_groups), all_files))
    val_images = list(filter(lambda p: (p in test_groups), all_files))
    return train_images, val_images


# polygons for test results
def get_polygons_predict(mask_label):
    """
    Convert masks to polygons
    """
    polygons = Mask((mask_label > 0)).polygons()
    polygons = polygons.points
    return polygons


def get_polygons(mask, fix_contour=True):
    """
    Convert masks to polygons and fix sizes
    """
    height, width = mask.shape
    all_polygons = []

    # Get coordinates
    polygons = get_polygons_predict(mask)

    for polygon in polygons:
        coord_x, coord_y = polygon[:, 0], polygon[:, 1]
        contour = np.array(
            [[coord_x[step_i], coord_y[step_i]] for step_i in range(len(coord_x))], dtype=np.int32
        ).reshape((-1, 1, 2))
        area = cv2.contourArea(contour)
        # Filter extrimal values
        if (area > 100) & (len(coord_x) > 10):
            mask_i = np.zeros(mask.shape, dtype=np.uint8)
            if fix_contour:
                cv2.drawContours(
                    mask_i, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED
                )
                # Increase polygon size
                kernel = np.ones((3, 3), np.uint8)
                mask_new = cv2.dilate(mask_i, kernel, iterations=4)
            else:
                mask_new = mask_i
            # Get new coordinates
            polygons_new = get_polygons_predict(mask_new)
            polygon_nornalized = []
            for coord_x, coord_y in polygons_new[0]:
                polygon_nornalized.append(coord_x / width)
                polygon_nornalized.append(coord_y / height)
            all_polygons.append(polygon_nornalized)
    return all_polygons


def load_images(
    nori_images, file_name, mask_layer, protein_layer, lipid_layer, crop_size=640
):
    """
    Load and preprocessing big tiff images and masks to tiles
    """
    with TiffFile(os.path.join(nori_images, file_name)) as tif:
        image = tif.asarray()

        # Extract masks
        mask = image[mask_layer]

        # Extract nori_data
        protein = image[protein_layer, :, :]
        lipid = image[lipid_layer, :, :]
        # While is empty
        water = image[lipid_layer, :, :] * 0
        image = np.stack([protein, lipid, water], axis=0)

    height, width = image[0].shape
    all_images = []
    all_masks = []
    all_images_names = []

    # Make crops with 50% overlap
    for i in range(0, height - crop_size, crop_size // 2):
        for j in range(0, width - crop_size, crop_size // 2):
            image_crop = image[:, i : i + crop_size, j : j + crop_size]
            mask_crop = mask[i : i + crop_size, j : j + crop_size]
            all_layers = []
            # Filter extremal higher values
            for layer in range(0, 3):
                image_layer = image_crop[layer]
                percentile_99 = np.percentile(image_layer, 99)
                image_layer = np.where(
                    (image_layer > percentile_99), percentile_99, image_layer
                )
                if layer == 2:
                    image_layer = image_layer * 0
                else:
                    image_layer = image_layer / image_layer.max()

                all_layers.append(image_layer)
            image_crop = np.array(all_layers)
            all_images.append(image_crop)
            crop_name = "_".join([file_name.split(".")[0], str(i), str(j)])
            all_images_names.append(crop_name)
            all_masks.append(mask_crop)

    return all_images, all_masks, all_images_names


# save polygons to txt
def save_polygons(all_polygons, file_name_save, directory, dataset_folder):
    """
    Save polygon coordinates to txt
    """
    with open(
        os.path.join(
            dataset_folder, "labels", directory, file_name_save.replace(".jpg", ".txt")
        ),
        "w",
        encoding="utf-8",
    ) as file:
        for polygon in all_polygons:
            for poly_, poly in enumerate(polygon):
                if poly_ == len(polygon) - 1:
                    file.write(f"{poly}\n")
                elif poly_ == 0:
                    file.write(f"0 {poly} ")
                else:
                    file.write(f"{poly} ")


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Function add noise like small white and black dots.
    """
    noisy_image = np.copy(image)
    salt_mask = np.random.choice([0, 1], size=image.shape,
                                 p=[1 - salt_prob, salt_prob])
    pepper_mask = np.random.choice(
        [0, 1], size=image.shape, p=[1 - pepper_prob, pepper_prob]
    )
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


def save_train_data(
    image, dataset_folder, directory, file_name_save_new, target, model_type
):
    """
    Function save all neccesary data to train model
    """
    file_path_save = os.path.join(
        dataset_folder, "images", directory, file_name_save_new
    )
    image = Image.fromarray(image)
    image.save(file_path_save, format="JPEG", quality=100, optimize=False)
    if model_type == "yolo":
        save_polygons(target, file_name_save_new, directory, dataset_folder)
    elif model_type == "unet":
        target = Image.fromarray((target).astype(np.uint8))
        mask_path_save = os.path.join(
            dataset_folder, "masks", directory, file_name_save_new
        )
        target.save(mask_path_save, format="JPEG", quality=100, optimize=False)


def save_subset(
    images,
    nori_images,
    directory,
    dataset_folder,
    crop_size,
    tubule_masks_layer,
    nuclei_masks_layer,
    protein_layer,
    lipid_layer,
    object="tubule",
    modifications=False,
    model_type="yolo",
):
    """
    Save train and validation datasets for YOLO segmentation format
    """
    if object == "tubule":
        mask_layer = tubule_masks_layer
    elif object == "nuclei":
        mask_layer = nuclei_masks_layer

    for file_name in images[0:]:
        (all_images, all_masks, all_images_names) = load_images(
            nori_images,
            file_name,
            mask_layer,
            protein_layer,
            lipid_layer,
            crop_size=crop_size,
        )

        for image, mask, image_name in zip(all_images, all_masks, all_images_names):
            # Extract all pokygons from mask
            all_polygons = get_polygons(mask)

            # Select target
            if model_type == "yolo":
                target = all_polygons
            elif model_type == "unet":
                target = mask

            # Convert array to RGB format
            im_transp = (image * 255).transpose((1, 2, 0)).astype(np.uint8)
            width = im_transp.shape[0]
            height = im_transp.shape[1]

            # Two modes for train dataset
            if modifications:
                # Apply specific augmentation
                for k in np.arange(1, 2.5, 1):
                    for prob in np.arange(0, 0.02, 0.01):
                        if (k >= 1) | (prob >= 0):
                            resized_image = change_resolution_pil(
                                Image.fromarray(im_transp), int(width / k), int(height / k)
                            )
                            restore_image = change_resolution_pil(
                                resized_image, int(width), int(height)
                            )
                            noise_im = add_salt_and_pepper_noise(
                                np.array(restore_image),
                                salt_prob=prob,
                                pepper_prob=prob,
                            )
                            # Keep water layer 0
                            noise_im[:, :, 2] = 0
                            # Convert to image
                            # noise_im = Image.fromarray(noise_im)
                            file_name_save_new = (
                                "_".join([image_name, str(k), str(prob)]) + ".jpg"
                            )
                            save_train_data(
                                noise_im,
                                dataset_folder,
                                directory,
                                file_name_save_new,
                                target,
                                model_type,
                            )
            else:
                file_name_save_new = image_name + ".jpg"
                save_train_data(
                    im_transp,
                    dataset_folder,
                    directory,
                    file_name_save_new,
                    target,
                    model_type,
                )
