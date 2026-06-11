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

**Notebook:** `Datacleaning(1).ipynb`

To ensure consistent training and improve segmentation quality, multiple preprocessing and data preparation steps were applied to the **LiTS CT dataset** before model training.

### Steps

#### 1. Load NIfTI Volumes

Medical CT scans and segmentation masks are loaded using the `nibabel` library.

```python
load_nii(path)
```

Each `.nii` / `.nii.gz` file is converted into a NumPy array (`float32`) for efficient volumetric processing.

---

#### 2. Volume–Mask Verification

A one-to-one mapping is maintained between CT volumes and segmentation masks.

```text
volume-X.nii → segmentation-X.nii
```

Files with missing masks are automatically skipped to prevent training inconsistencies.

---

#### 3. Orientation Correction

LiTS volumes are originally stored in:

```text
(H, W, D)
```

All CT volumes and masks are transposed into a depth-first representation:

```text
(D, H, W)
```

This ensures compatibility with **3D convolutional neural networks** and volumetric patch extraction.

---

#### 4. Hounsfield Unit (HU) Clipping

CT intensity values are clipped to retain only medically relevant soft-tissue information.

```python
clip_hu(volume, min_hu=-200, max_hu=250)
```

HU values outside the range `[-200, 250]` are suppressed to reduce noise from irrelevant anatomical structures such as air and bones.

---

#### 5. Intensity Normalization

Z-score normalization is applied independently to each volume for stable training.

```python
normalize(volume)
```

Normalization formula:

```text
(volume - mean) / (std + 1e-8)
```

This reduces inter-scan intensity variation and improves model convergence.

---

#### 6. Mask Cleaning

Segmentation masks are converted into unsigned integer format for multi-class segmentation.

```python
clean_mask(mask)
```

Classes used:

| Label | Class      |
| ----- | ---------- |
| 0     | Background |
| 1     | Liver      |
| 2     | Tumor      |

---

#### 7. Removal of Empty Slices

Non-informative slices containing only background pixels are removed.

```python
remove_empty_slices(volume, mask)
```

This step:

* Reduces unnecessary computation
* Mitigates class imbalance
* Improves training efficiency

Only slices containing liver or tumor annotations are retained.

---

#### 8. Train–Validation Split

A **volume-wise split** is performed using:

```python
train_test_split()
```

Configuration:

```text
Training Set   → 80%
Validation Set → 20%
Random Seed    → 42
```

The split is performed at the **patient volume level** to avoid data leakage between training and validation datasets.

---

#### 9. 3D Patch Extraction

To enable efficient volumetric learning, overlapping **3D patches** are extracted from each CT volume.

```python
patch_size = (16, 128, 128)
stride     = (8, 64, 64)
```

A sliding window extracts patches in three spatial dimensions (`Depth × Height × Width`).

---

#### 10. Tumor-Aware Patch Sampling

To address severe class imbalance, patches are selected using class-aware sampling:

| Patch Content   | Action                       |
| --------------- | ---------------------------- |
| Tumor present   | Always included              |
| Liver present   | Always included              |
| Background only | Included with 5% probability |

This strategy increases exposure to clinically important tumor regions while reducing redundant background samples.

---

#### 11. Dataset Generation

Extracted patches are stored as NumPy arrays for faster model training.

```python
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)

np.save("X_val.npy", X_val)
np.save("Y_val.npy", Y_val)
```

Generated outputs:

```text
X_train.npy → Training image patches
Y_train.npy → Training segmentation masks
X_val.npy   → Validation image patches
Y_val.npy   → Validation segmentation masks
```

These processed datasets are later used as inputs for training the **3D U-Net** and **3D Attention U-Net** models.


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



The baseline model is a **3D U-Net** with 2 encoder levels and 2 decoder levels.

```
Input (1 × 16 × 128 × 128)
        │
   DoubleConv → 32 ch                 ← skip x1
        │
   MaxPool3d + DoubleConv → 64 ch     ← skip x2
        │
   MaxPool3d + DoubleConv → 128 ch    (bottleneck)
        │
   ConvTranspose3d + concat(x2) + DoubleConv → 64 ch
        │
   ConvTranspose3d + concat(x1) + DoubleConv → 32 ch
        │
   Conv3d(1×1×1) → 3 classes

```
**DoubleConv block:**
```
Conv3d → BatchNorm3d → ReLU → Conv3d → BatchNorm3d → ReLU
```

**Weight Initialization:** Kaiming Normal for all `Conv3d` layers; biases set to zero.

### Key Components

- `DoubleConv`: two consecutive 3×3×3 convolutions with BN + ReLU
- `Down`: `MaxPool3d(2)` followed by `DoubleConv`
- `Up
---

### 2. 3D Attention U-Net

An enhanced **3D Attention U-Net** was implemented to improve segmentation precision.


 The Attention U-Net extends the standard U-Net with **Attention Gates** on every skip connection. These gates learn to suppress irrelevant feature activations before they are concatenated in the decoder.

```
Input (1 × 16 × 128 × 128)
        │
   DoubleConv → 32 ch                     ← skip x1
        │
   MaxPool3d + DoubleConv → 64 ch         ← skip x2
        │
   MaxPool3d + DoubleConv → 128 ch        ← skip x3
        │
   MaxPool3d + DoubleConv → 256 ch        (bottleneck)
        │
   ConvTranspose3d + AttGate(x3) + DoubleConv → 128 ch
        │
   ConvTranspose3d + AttGate(x2) + DoubleConv → 64 ch
        │
   ConvTranspose3d + AttGate(x1) + DoubleConv → 32 ch
        │
   Conv3d(1×1×1) → 3 classes
```

Additional Features:

* Attention Gates (AGs)
* Better focus on tumor regions
* Reduced background interference
* Improved boundary localization
The Attention U-Net is **one level deeper** than the standard U-Net (256-channel bottleneck vs 128), giving it greater representational capacity.


---

## Loss Function

To handle **class imbalance** in liver tumor segmentation, a **Combined Loss Function** was used by integrating **Weighted Cross-Entropy Loss** with **Soft Dice Loss**.

### Combined Loss

```text
L_total = L_CE + L_Dice
```

### Weighted Cross-Entropy Loss

Class weights were assigned to improve learning for underrepresented tumor regions:

```python
weight = [0.2, 0.3, 0.5]
# Background, Liver, Tumor
```

A higher weight was given to the **tumor class** to reduce imbalance between background, liver, and tumor voxels.

### Soft Dice Loss

Dice Loss was computed on **foreground classes (Liver + Tumor)** to improve overlap between predicted and ground-truth segmentation masks.

$$
L_{\text{Dice}} =
\frac{1}{C}
\sum_{c=1}^{C}
\left(
1 -
\frac{
2 \sum p_c \cdot t_c + \varepsilon
}{
\sum p_c + \sum t_c + \varepsilon
}
\right)
$$

where:

* **p₍c₎** → predicted softmax probability
* **t₍c₎** → ground truth target
* **ε** → small constant to avoid division by zero

### Why Combined Loss?

* **Cross-Entropy Loss** → improves voxel-wise classification
* **Dice Loss** → improves segmentation overlap

Together, they improve **tumor boundary detection** and overall segmentation performance.


### Evaluation Metrics

**Dice Score (per class):**
$$\text{Dice}_c = \frac{2 \cdot |P_c \cap T_c| + \varepsilon}{|P_c| + |T_c| + \varepsilon}$$

**IoU / Jaccard Score (per class):**
$$\text{IoU}_c = \frac{|P_c \cap T_c|}{|P_c \cup T_c|}$$

Reported per class (Background, Liver, Tumor) as well as **Mean Dice (Liver + Tumor)** as the primary model selection criterion.

---

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
## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/liver-tumor-segmentation.git
cd liver-tumor-segmentation
```

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn nibabel opencv-python
pip install jupyter notebook
```

---

### 4. Download the Dataset

This project uses the **LiTS (Liver Tumor Segmentation)** dataset.

Download both parts from Kaggle:

* **Part 1:** https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation
* **Part 2:** https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation-part-2

**Both parts together form the complete dataset.**

After downloading, organize the dataset as:

```text
dataset/
├── images/
│   ├── volume-0.nii
│   ├── volume-1.nii
│   └── ...
│
├── masks/
│   ├── segmentation-0.nii
│   ├── segmentation-1.nii
│   └── ...
```

---

### 5. GPU Setup (Recommended)

Training 3D segmentation models is computationally intensive, so **GPU (CUDA)** is recommended.

Verify GPU availability:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected output:

```text
True
NVIDIA GPU Name
```

If no GPU is detected, training will automatically run on CPU (slower).

---

## How to Run

### Step 1 — Launch Jupyter Notebook

```bash
jupyter notebook
```

---

### Step 2 — Data Preprocessing

Open and run all cells in **`data_cleaning1.ipynb`**.

Update dataset paths:

```python
image_dir = "path/to/images"
mask_dir = "path/to/masks"
```

This notebook will:

* Load CT volumes and masks
* Apply preprocessing (HU clipping, normalization, empty slice removal)
* Generate 3D patches
* Create train–validation split
* Save:

```text
X_train.npy
Y_train.npy
X_val.npy
Y_val.npy
```

---

### Step 3 — Train the 3D Attention U-Net

Open and run all cells in **`attention_unet1.ipynb`**.

The notebook will:

* Load preprocessed datasets
* Train the **3D Attention U-Net**
* Compute **Dice Score** and **IoU** metrics
* Save the trained model checkpoint

---

### Step 4 — Run Inference

Update the paths:

```python
img_path = "path/to/volume.nii"
mask_path = "path/to/segmentation.nii"
```

Run the inference and visualization cells to generate predicted segmentation masks.

---

## Results

The project demonstrates successful segmentation of:

* Liver regions
* Tumor regions
* Volumetric CT structures



<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/2273a933-347e-4dea-9b65-713d29a76290" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/ab2e5129-7d0d-4ff0-abe2-94a8741d1f53" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/24a8abdf-4f60-4733-8e2e-0c2c81a95613" />
<img width="584" height="455" alt="image" src="https://github.com/user-attachments/assets/2f413679-8815-4181-9033-07612be1ebb5" />

<img width="1013" height="316" alt="image" src="https://github.com/user-attachments/assets/5ef44847-7445-44e3-9a89-ffec44ce5ce8" />
<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/57dcd10d-d9a9-43df-9d25-c63dbf738103" />
<img width="567" height="435" alt="image" src="https://github.com/user-attachments/assets/8b15449e-17da-4efa-8dc3-251847d2f605" />

---

## Future Improvements

* Transformer-based segmentation models
* Better small tumor detection
* GAN-based synthetic augmentation
* Multi-organ segmentation
* Real-time clinical deployment optimization

---
> **Computational Note:**
> Due to the high computational cost of **3D medical image segmentation**, experiments in this project were conducted on a **subset of 10 LiTS volumes**, which could be trained efficiently on a laptop GPU (**NVIDIA RTX 4060 Laptop GPU, CUDA 12.1**).
> Training on the complete **LiTS dataset (131 training volumes)** would require significantly higher memory and compute resources, typically a **high-memory workstation or cloud GPU environment** for efficient training.


