# PhysioNet ECG Image Digitization - 3rd Place Solution

This repository contains the 3rd place solution for the [PhysioNet ECG Image Digitization Challenge 2025](https://physionet.org/content/ecg-image-digitization-challenge/1.0.0/).

## Project Overview

The goal of this challenge is to extract digital ECG signals from scanned ECG images (paper recordings). This solution uses a multi-stage deep learning pipeline:

- **Stage 0**: Keypoint detection and image normalization
- **Stage 1**: Grid detection and perspective rectification  
- **Stage 2**: Pixel-level waveform segmentation and signal extraction

### Solution Writeup

Detailed solution description: [3rd Place Solution Writeup](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/3rd-place-solution)

### Acknowledgments

The Stage 0 and Stage 1 implementations are adapted from:
- **hengck23's Kaggle notebook**: [ECG Image Digitization - Demo Submission](https://www.kaggle.com/code/hengck23/demo-submission) 

**Checkpoints**: Download `stage0-last.checkpoint.pth` and `stage1-last.checkpoint.pth` from [hengck23's Kaggle Dataset](https://www.kaggle.com/datasets/hengck23/hengck23-demo-submit-physionet) and place them in `checkpoints/`

## Repository Structure

```
.
├── checkpoints/                    # Pre-trained model checkpoints
│   ├── stage0-last.checkpoint.pth
│   └── stage1-last.checkpoint.pth
├── configs/                        # Configuration files
│   ├── base.py
│   └── stage2.py
├── data/                          # Data directory (not in repo)
│   ├── train/                     # Training images and CSVs
│   ├── train.csv                  # Metadata (fs, sig_len)
│   └── train_folds.csv           # Cross-validation folds
├── data_preparation/              # Data preparation scripts
│   ├── prepare_inputs.py          # Generate rectified images (Stage 0+1)
│   ├── generate_masks.py          # Generate training masks
│   ├── mask_generator_highres.py  # Alternative mask generator
│   ├── prepare_resized_rectified_sample.py
│   └── resize_rectified_images.py
├── datasets/                      # PyTorch dataset classes
│   └── stage2_dataset.py
├── models/                        # Model architectures
│   ├── backbones.py
│   ├── decoders.py
│   └── stage2_net.py
├── utils/                         # Utility functions
│   ├── augmentation.py
│   ├── common.py
│   ├── geometry.py
│   ├── grid_utils.py
│   ├── kaggle_metric.py
│   ├── metrics.py
│   ├── postprocess.py
│   └── visualization.py
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Requirements

- Python 3.10+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place.git
cd physionet-ecg-image-digitization-3rd-place
```

2. Create a new conda environment:
```bash
conda create -n physionet-ecg python=3.10 -y
conda activate physionet-ecg
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from [PhysioNet](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data) and place it in the `data/` directory:
```
data/
├── train/
│   ├── {sample_id}/
│   │   ├── {sample_id}.csv      # Ground truth signals
│   │   └── {sample_id}-{type}.png  # ECG images
│   └── ...
├── train.csv                     # Metadata
└── train_folds.csv              # Cross-validation splits
```

## Data Preparation

### Step 1: Generate Rectified Images (Stage 0 + Stage 1)

This script processes raw ECG images through Stage 0 and Stage 1 to generate rectified images (4400×1700):

```bash
# From project root
python data_preparation/prepare_inputs.py --workers 2

```

**What it does:**
- Stage 0: Detects 9 keypoints (lead labels) and normalizes image orientation
- Stage 1: Detects grid points and performs perspective rectification
- Output: Rectified images saved to `outputs/stage1/rectified_kaggle_data_4400x1700/`

**Output files per image:**
- `{image_id}.rect.png` - Rectified image (4400×1700)
- `{image_id}.homo.npy` - Homography matrix
- `{image_id}.rotation.npy` - Rotation info
- `{image_id}.gridpoint_xy.npy` - Grid point coordinates

**Options:**
- `--workers`: Number of parallel workers (default: 8)
- `--debug`: Process only first 10 samples for testing
- `--debug-samples N`: Process only N samples

### Step 2: Generate Training Masks

This script generates 4-channel pixel masks (4352×1696) from CSV signal data:

```bash
python data_preparation/generate_masks.py --workers 16
```

**What it does:**
- Reads ECG signals from CSV files
- Converts signals to pixel coordinates using:
  - Sampling rate (fs)
  - Zero mV positions
  - mV to pixel conversion (79 pixels/mV)
- Generates 4-channel masks representing 4 horizontal bands:
  - Channel 0: I, aVR, V1, V4
  - Channel 1: II, aVL, V2, V5
  - Channel 2: III, aVF, V3, V6
  - Channel 3: II-rhythm (full 10s)

**Output:**
- `{sample_id}.mask.npy` saved to `data/mask_4352x1696/`

**Options:**
- `--workers`: Number of parallel workers (default: 16)
- `--debug`: Process only first 10 samples for testing

## Stage 2 Training

Once data preparation is complete, you can train Stage 2 models:

```bash
python train.py
```

## Model Architectures

### Stage 2 model
Multiple architectures supported:
- **ResNet34**
- **HRNet**
- **ConvNeXt V2** (best performance)
- **EfficientNetV2**
