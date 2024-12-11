"""
Function to train models
"""

# import libraries
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def make_yolo_config(config_path, object_type, model_name):
    """
    Create config for ultralitic library
    """
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(f"path: {model_name}\n")
        file.write("train: images/train\n")
        file.write("val: images/val\n")
        file.write("nc: 1\n")
        file.write(f'names: ["{object_type}"]\n')


def get_train_augmentations():
    """
    Define training augmentations including resizing
    """
    return A.Compose(
        [
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_augmentations():
    """
    Define validation augmentations (only resizing and normalization)
    """
    return A.Compose(
        [
            A.Resize(640, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class UnetDataset(Dataset):
    """
    Create dataset for Unet model
    """

    def __init__(self, image_dir, mask_dir, augmentations=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = os.listdir(image_dir)
        self.augmentations = augmentations

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Apply augmentations if available
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Normalize mask to [0, 1] if needed
        mask = (mask > 0).float()  # Convert to binary mask with values 0 or 1
        mask = mask.unsqueeze(0)  # Add channel dimension

        return image, mask


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


def calculate_dice(preds, targets, threshold=0.5):
    """
    Function to compute Dice coefficient
    """
    preds = (preds > threshold).float()  # Binarize predictions
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection) / (
        preds.sum() + targets.sum() + 1e-8
    )  # Add small epsilon for stability
    return dice.item()


def unet_train(
    train_loader,
    val_loader,
    model_dir,
    model_name,
    num_epochs=50,
    learning_rate=1e-4,
    patience=20
):
    """
    Train and save unet model
    """
    shutil.rmtree(os.path.join(model_dir, model_name), ignore_errors=True)
    os.makedirs(os.path.join(model_dir, model_name), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training with Dice metric and best model saving
    best_dice = 0.0  # Initialize the best Dice coefficient
    epoch_no_saved = 0  # Epochs number without model saving
    best_model_path = os.path.join(model_dir, model_name, model_name + ".pth")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate Dice coefficient for the batch
                batch_dice = calculate_dice(outputs, masks)
                val_dice += batch_dice

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)  # Average Dice over all batches

        # Save the model if it has the best Dice coefficient so far
        is_saved = 0
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            is_saved = 1
            epoch_no_saved = 0
        else:
            epoch_no_saved += 1

        results_table = [[f"{epoch+1}", train_loss, val_loss, val_dice, is_saved]]
        columns = (
            "Epoch",
            "Train Loss",
            "Validation Loss",
            "Validation Dice",
            "Model saved",
        )
        results_table = pd.DataFrame(results_table, columns=columns)
        if epoch == 0:
            results_table.to_csv(
                os.path.join(model_dir, model_name, "results.csv"), index=None
            )
        else:
            results_table.to_csv(
                os.path.join(model_dir, model_name, "results.csv"),
                index=None,
                mode="a",
                header=None,
            )

        if epoch_no_saved > patience:
            break
