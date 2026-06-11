# Liver Tumor Segmentation using 3D Attention U-Net

Deep learning–based automated liver and tumor segmentation from abdominal CT scans using **3D U-Net** and **3D Attention U-Net** architectures on the **LiTS (Liver Tumor Segmentation)** dataset.

This project focuses on improving medical image segmentation by handling challenges such as **class imbalance, low contrast boundaries, anatomical variability, and small tumor detection** in volumetric CT scans.

---

## Project Overview

Liver tumor segmentation is a critical task in medical imaging for:

* Tumor detection and diagnosis
* Treatment planning and radiotherapy
* Disease progression monitoring
* Organ boundary delineation

Traditional segmentation methods struggle with irregular tumor boundaries and noisy CT scans. This project explores **3D deep learning architectures** capable of learning volumetric features directly from CT volumes for improved segmentation performance.

---

## Problem Statement

Liver tumor segmentation from CT scans is challenging due to:

* Low contrast between liver and surrounding tissues
* High anatomical variability among patients
* Small tumor regions causing severe class imbalance
* Complex 3D volumetric structure of CT scans

The objective of this project is to build an automated framework capable of accurately segmenting **liver and tumor regions** from CT images while maintaining computational efficiency.

---

## Dataset

### LiTS (Liver Tumor Segmentation Challenge)

The project uses the **LiTS dataset**, a publicly available benchmark dataset of 3D CT scans for liver tumor segmentation.

### Dataset Details

* **Total CT Volumes:** 201
* **Training Volumes:** 131
* **Test Volumes:** 70
* **Image Resolution:** ~512 × 512 per slice
* **Total Slices:** 58,000+

### Label Encoding

| Label | Class      |
| ----- | ---------- |
| 0     | Background |
| 1     | Liver      |
| 2     | Tumor      |


### File Format

* **Format:** NIfTI (`.nii`, `.nii.gz`)
* **Data Type:** 3D Volumetric CT Scans
* **Volume Shape:** `(Height × Width × Slices)`
* **Metadata Included:**

  * Spatial resolution (voxel spacing)
  * Image orientation
  * Anatomical positioning information

Each CT scan is paired with a corresponding **ground-truth segmentation mask**, ensuring a one-to-one relationship between medical images and annotations.

### Directory Structure

```text
dataset/
├── images/
│   ├── volume-0.nii
│   ├── volume-1.nii
│   └── ...
│
├── labels/
│   ├── segmentation-0.nii
│   ├── segmentation-1.nii
│   └── ...
```

### Data Organization

* `images/` → Raw abdominal CT scan volumes
* `labels/` → Expert-annotated segmentation masks
* **One-to-one mapping** between each CT volume and its corresponding mask


## Tech Stack

**Languages & Libraries**

* Python
* PyTorch
* NumPy
* OpenCV
* Scikit-learn
* Matplotlib
* Nibabel

**Deep Learning Concepts**

* 3D U-Net
* 3D Attention U-Net
* Dice Loss
* Weighted Cross Entropy Loss
* Medical Image Segmentation
* Data Augmentation

---

## Project Workflow

```text
Raw CT Scans (.nii)
        ↓
Data Cleaning & Preprocessing
        ↓
3D Patch Extraction
        ↓
3D U-Net / Attention U-Net
        ↓
Training & Validation
        ↓
Evaluation (Dice Score, IoU)
        ↓
Tumor Segmentation Output
```

---

## Data Preprocessing

The following preprocessing techniques were implemented:

* Volume–mask verification
* CT scan normalization
* Hounsfield Unit (HU) clipping `[-200, 250]`
* Z-score normalization
* Removal of non-informative slices
* 3D patch extraction `(16 × 128 × 128)`
* Tumor-focused patch selection
* Data augmentation

### Augmentations Used

* Horizontal/Vertical flips
* Rotation
* Intensity scaling
* Gaussian noise
* Gamma correction

These preprocessing steps help improve robustness and reduce overfitting.

---

## Model Architecture

### 1. 3D U-Net

A baseline **3D U-Net** model was implemented for volumetric medical image segmentation.

Key Features:

* Encoder–decoder architecture
* Skip connections for feature preservation
* Volumetric feature extraction
* Pixel-level segmentation

---

### 2. 3D Attention U-Net

An enhanced **3D Attention U-Net** was implemented to improve segmentation precision.

Additional Features:

* Attention Gates (AGs)
* Better focus on tumor regions
* Reduced background interference
* Improved boundary localization

The Attention U-Net architecture performed better for difficult tumor regions due to selective feature learning.

---

## Loss Function

A **Combined Loss Function** was used:

### Dice Loss

Used for improving overlap between predicted and ground truth segmentation masks.

### Weighted Cross Entropy Loss

Used to address severe class imbalance.

**Class Weights:**

* Background → 0.2
* Liver → 0.3
* Tumor → 0.5

---

## Evaluation Metrics

Model performance was evaluated using:

* **Dice Similarity Coefficient (DSC)**
* **Intersection over Union (IoU)**
* **Validation Accuracy**

These metrics provide both overlap-based and region-based performance evaluation.

---

## Training Configuration

| Parameter              | Value             |
| ---------------------- | ----------------- |
| Optimizer              | Adam              |
| Learning Rate          | 1e-4              |
| Weight Decay           | 1e-5              |
| LR Scheduler           | ReduceLROnPlateau |
| Train–Validation Split | 80:20             |
| Random Seed            | 42                |

---

## Repository Structure

```text
tumor-segmentation/
│── attention_unet1.ipynb
│── data_cleaning1.ipynb
│── README.md
│── requirements.txt
│── sample_outputs/
```

---

## Results

The project demonstrates successful segmentation of:

* Liver regions
* Tumor regions
* Volumetric CT structures

Performance evaluation was conducted using **Dice Score** and **IoU metrics**.

> Add prediction screenshots in the `sample_outputs/` folder for stronger GitHub presentation.

---

## Future Improvements

* Transformer-based segmentation models
* Better small tumor detection
* GAN-based synthetic augmentation
* Multi-organ segmentation
* Real-time clinical deployment optimization

---


