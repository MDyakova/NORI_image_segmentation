# NoRI Image Segmentation

This repository contains tools for segmenting tubules and associated features from whole kidneys imaged using **normalised stimulated Raman spectroscopy (NoRI) microscopy**, a cutting-edge approach developed at Marc Kirschner's lab at Harvard.

## 🚀 **Project Description**
Traditional histopathology relies on qualitative and subjective assessments. This project leverages NoRI microscopy to deliver **quantitative insights** into protein and lipid distributions in biological samples at sub-cellular resolution. The approach is being applied to study kidney tissue biology in health and disease.

### 🧬 **Unique Technology**
The NoRI microscopy platform measures the distribution of proteins and lipids at the cellular level. Machine-learning models provide comparisons between:
- Young vs. old cells.
- Healthy vs. diseased tissues.
- Male vs. Female samples and so on.

These insights support decision-making to **treat, protect, and extend the functional lifespan** of cells.

### 🌟 **Goals and Impact**
The project aims to:
- Extend the **physiological health span** of tissues.
- Advance understanding of tissue aging and disease progression.
- Develop new diagnostic tools and treatments for age-related conditions.

---

## 📁 **Repository Structure**

The repository includes a `work_directory` with the following structure:

```
work_directory/
├── datasets/              # Input datasets for training and inference
├── train_directory/       # Directory for training-related data
│   ├── data/              # Training input data
│   ├── models/            # Trained models storage
│   └── train_config.json  # Configuration file for training
└── inference_directory/   # Directory for inference-related data
    ├── data/              # Inference input data
    └── inference_config.json # Configuration file for inference
```

---

## 🛠 **Getting Started**

### 1. **Model Inference**

Run pre-trained models on new NoRI datasets.

1. Clone the repository:
   ```bash
   git clone https://github.com/MDyakova/NORI_image_segmentation.git
   cd NORI_image_segmentation
   ```

2. Pull the Docker image:
   ```bash
   docker pull mdyakova/nori_segmentation:v1.0
   ```

3. Launch the inference container:
   ```bash
   docker run -v ${PWD}/work_directory/inference_directory:/inference_models/inference_directory -d mdyakova/nori_segmentation:v1.0 main.py
   ```

Ensure your input data is placed in `work_directory/inference_directory/data`, and the configuration file `inference_config.json` is updated with necessary details.

---

### 2. **Training New Models**

Train neural network models for segmenting objects in NoRI microscopy images.

1. Clone the repository:
   ```bash
   git clone https://github.com/MDyakova/NORI_image_segmentation.git
   cd NORI_image_segmentation
   ```

2. Pull the Docker image:
   ```bash
   docker pull mdyakova/nori_segmentation_train:v1.0
   ```

3. Launch the training container for the desired model:
   - **YOLO Model**:
     ```bash
     docker run -v ${PWD}/work_directory/train_directory:/train_models/train_directory -v ${PWD}/work_directory/datasets:/train_models/datasets -d mdyakova/nori_segmentation_train:v1.0 tubule_model.py
     ```
   - **U-Net Model**:
     ```bash
     docker run -v ${PWD}/work_directory/train_directory:/train_models/train_directory -v ${PWD}/work_directory/datasets:/train_models/datasets -d mdyakova/nori_segmentation_train:v1.0 nuclei_model.py
     ```

Ensure your training input data is placed in `work_directory/train_directory/data`, and the configuration file `train_config.json` is updated with necessary details.

---

## 🛠 **Configuration Details**

### **Inference Configuration**
The `inference_config.json` file provides necessary information for running the inference process.

#### Example: `inference_config.json`
```json
{
    "data_information": {
        "nori_images": "inference_directory/data/{your_nori_samples}",
        "protein_layer": 0,
        "lipid_layer": 1
    },
    "output_information": {
        "output_folder": "inference_directory/{folder_for_segmentation_results}"
    },
    "models": {
        "tubule_model": "default_models/best_tubule_model.pt",
        "nuclei_model": "default_models/best_nuclei_model.pth",
        "crop_size": 640,
        "tubule_prob": 0.5,
        "nuclei_prob": 0.5,
        "lumen_coef": 100,
        "lumen_cluster_size": 100,
        "distance_threshold": 200
    }
}
```

#### Parameter Details:
- **`data_information`**
  - `nori_images`: Path to your NoRI sample files for inference. Replace `{your_nori_samples}` with the location of your input data.
  - `protein_layer`: The layer in the NoRI TIFF file corresponding to protein data.
  - `lipid_layer`: The layer in the NoRI TIFF file corresponding to lipid data.

- **`output_information`**
  - `output_folder`: Path where segmentation results will be saved. Replace `{folder_for_segmentation_results}` with the desired output directory.

- **`models`**
  - `tubule_model`: Path to the model for tubule segmentation. Use `default_models/best_tubule_model.pt` for default settings, or specify your custom model.
  - `nuclei_model`: Path to the model for nuclei segmentation. Use `default_models/best_nuclei_model.pth` for default settings, or specify your custom model.
  - `crop_size`: Size of image crops used during inference (default: `640`).
  - `tubule_prob`: Probability threshold for tubule segmentation (default: `0.5`).
  - `nuclei_prob`: Probability threshold for nuclei segmentation (default: `0.5`).
  - `lumen_coef`: Coefficient for lumen intensity adjustment (default: `100`).
  - `lumen_cluster_size`: Minimum cluster size for lumen detection (default: `100`).
  - `distance_threshold`: Distance threshold for post-processing (default: `200`).

---

### **Training Configuration**
The `train_config.json` file provides necessary information for training the models.

#### Example: `train_config.json`
```json
{
    "data_information": {
        "nori_images": "train_directory/data/mouse_kidney",
        "protein_layer": 0,
        "lipid_layer": 1,
        "tubule_masks_layer": 10,
        "nuclei_masks_layer": 9
    },
    "tubule_yolo_model": {
        "model_information": {
            "model_name": "kidney_tubule_model_12-06-2024",
            "modifications": true,
            "crop_size": 640
        },
        "model_config": {
            "epochs": 200,
            "imgsz": 640,
            "batch": 16,
            "patience": 30,
            "overlap_mask": false,
            "object_type": "tubule"
        }
    },
    "nuclei_unet_model": {
        "model_information": {
            "model_name": "kidney_nuclei_model_12-06-2024",
            "modifications": false,
            "crop_size": 640
        },
        "model_config": {
            "epochs": 200,
            "batch": 8,
            "patience": 30,
            "learning_rate": 0.0001
        }
    }
}
```

#### Parameter Details:
- **`data_information`**
  - `nori_images`: Path to your NoRI sample files for training.
  - `protein_layer`: The layer in the NoRI TIFF file corresponding to protein data.
  - `lipid_layer`: The layer in the NoRI TIFF file corresponding to lipid data.
  - `tubule_masks_layer`: The layer in the NoRI TIFF file containing tubule segmentation masks.
  - `nuclei_masks_layer`: The layer in the NoRI TIFF file containing nuclei segmentation masks.

- **`tubule_yolo_model`**
  - `model_information`: Details about the YOLO model for tubule segmentation.
    - `model_name`: Name for the model being trained.
    - `modifications`: Whether custom modifications are applied to the model (true/false).
    - `crop_size`: Size of image crops used for training.
  - `model_config`: YOLO-specific training parameters.
    - `epochs`: Number of training epochs.
    - `imgsz`: Image size for YOLO training.
    - `batch`: Batch size for training.
    - `patience`: Early stopping patience parameter.
    - `overlap_mask`: Whether to allow overlapping masks (true/false).
    - `object_type`: Object type being segmented (e.g., "tubule").

- **`nuclei_unet_model`**
  - `model_information`: Details about the U-Net model for nuclei segmentation.
    - `model_name`: Name for the model being trained.
    - `modifications`: Whether custom modifications are applied to the model (true/false).
    - `crop_size`: Size of image crops used for training.
  - `model_config`: U-Net-specific training parameters.
    - `epochs`: Number of training epochs.
    - `batch`: Batch size for training.
    - `patience`: Early stopping patience parameter.
    - `learning_rate`: Learning rate for training.

---

## 📊 **Segmentation Results**

The segmentation process generates the following outputs:

1. **Images**
   - Annotated NoRI images (protein and lipid layers) with tubule contours, nuclei masks, and lumen masks.

2. **Labels**
   - Object coordinates and attributes for tubules, nuclei, and lumen.

3. **TIFF Files**
   - Multi-layer TIFF files with:
     1. NoRI Protein Level
     2. NoRI Lipid Level
     3. Tubule Masks
     4. Nuclei Masks
     5. Lumen Masks
     6. Tubule Masks Probabilities

---

### 📫 **Contact**
For questions or contributions, please contact:
**Mariia Diakova**
- GitHub: [MDyakova](https://github.com/MDyakova)
- email: m.dyakova.ml@gmail.com

---

Let me know if further edits are required!