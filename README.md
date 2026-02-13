# Brain Tumor MRI Classification

A deep learning model for classifying brain tumors from MRI scans into four categories using **EfficientNetB0** with transfer learning.

| | |
|---|---|
| **Framework** | TensorFlow / Keras 3 |
| **Architecture** | EfficientNetB0 (ImageNet pre-trained) |
| **Input** | 224 × 224 RGB MRI images |
| **Classes** | Glioma, Meningioma, Pituitary, No Tumor |
| **Test Accuracy** | **93.75%** (1600 images) |

---

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Project Structure](#project-structure)
- [Inference — Custom Images](#inference--custom-images)

---

## Dataset

### Primary Dataset
**[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** by Masoud Nickparvar

| Split | Images | Classes |
|-------|--------|---------|
| Training | ~5,712 | 4 |
| Testing | 1,600 | 4 |

Classes: `glioma` · `meningioma` · `notumor` · `pituitary`

### Cross-Dataset Validation
**[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)** by Navoneel — a binary (tumor / no tumor) dataset used to test generalization on unseen data distributions.

---

## Model Architecture

```
EfficientNetB0 (frozen/partially unfrozen)
    │
    ▼
GlobalAveragePooling2D
    │
BatchNormalization → Dropout(0.4)
    │
Dense(256, ReLU)
    │
BatchNormalization → Dropout(0.3)
    │
Dense(128, ReLU) → Dropout(0.2)
    │
Dense(4, Softmax)
```

- **Base model:** EfficientNetB0 pre-trained on ImageNet (expects `[0, 255]` input — no rescaling)
- **Custom head:** BatchNorm + Dropout cascade for regularization
- **Total params:** ~4.3M (only ~300K trainable in Phase 1)

---

## Training Strategy

### Phase 1 — Feature Extraction
- Base model **completely frozen**
- Learning rate: `1e-3`
- Epochs: up to 10 (with EarlyStopping)

### Phase 2 — Fine-Tuning
- Top **30%** of EfficientNet layers unfrozen
- Learning rate: `1e-4` (10× lower)
- Epochs: up to 20 (with EarlyStopping)

### Callbacks
| Callback | Purpose |
|----------|---------|
| `ModelCheckpoint` | Saves best model by `val_loss` |
| `EarlyStopping` | Stops when `val_loss` stalls |
| `ReduceLROnPlateau` | Halves LR after 3 epochs without improvement |

### Data Augmentation (Keras 3 compatible)
- Random rotation (±18°)
- Random translation (10%)
- Random zoom (10%)
- Horizontal flip
- Random contrast & brightness (10%)

---

## Results

### Test Set Performance (1,600 images)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 0.98 | 0.82 | 0.89 | 400 |
| Meningioma | 0.89 | 0.93 | 0.91 | 400 |
| No Tumor | 0.95 | 1.00 | 0.97 | 400 |
| Pituitary | 0.93 | 1.00 | 0.97 | 400 |
| **Overall** | **0.94** | **0.94** | **0.93** | **1600** |

### Cross-Dataset Validation (253 images)

| Metric | Value |
|--------|-------|
| Specificity (Healthy) | 100.00% |
| Sensitivity (Tumor) | 57.42% |
| Overall Accuracy | 73.91% |

> The lower cross-dataset sensitivity is expected — the external dataset contains different MRI acquisition protocols and labeling criteria.

---

## Setup & Usage

### Prerequisites
- Google Colab (recommended) or a local GPU environment
- Kaggle API key (`kaggle.json`) in Google Drive root
- Google Drive mounted

### Quick Start

1. **Open** `tumor.ipynb` in Google Colab
2. **Run cells 1–4** — downloads dataset, loads imports, preprocesses data, and loads the saved model from Drive
3. **Run cell 5** — automatically skipped if model is loaded (trains only if no saved model exists)
4. **Run cell 6** — evaluation with confusion matrix and classification report
5. **Run cell 8** — inference on your own MRI images (place them in `Drive/Test_Images/`)

### Using a Pre-Trained Model

Place `best_tumor_classifier_v1.keras` in the root of your Google Drive. The notebook will detect and load it automatically — no training required.

---

## Project Structure

```
Brain-Tumor-Classification/
├── tumor.ipynb          # Main notebook (training, evaluation, inference)
├── README.md            # This file
```

**Google Drive files** (created during execution):
```
MyDrive/
├── kaggle.json                        # Kaggle API credentials
├── best_tumor_classifier_v1.keras     # Final saved model (~49 MB)
├── tumor_classifier_model.keras       # Best checkpoint during training (~18 MB)
└── Test_Images/                       # Drop custom MRI images here for inference
```

---

## Inference — Custom Images

1. Create a folder `Test_Images/` in the root of your Google Drive
2. Upload MRI scan images (`.jpg`, `.png`, `.jpeg`)
3. Run cell 8 in the notebook

Each image will display with:
- **Predicted class** (Glioma / Meningioma / Pituitary / No Tumor)
- **Confidence score** (softmax probability)
- **Filename** for reference

---

## Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup & Download | Mount Drive, download Kaggle dataset |
| 2 | Imports & Config | Libraries, constants, model paths |
| 3 | Preprocessing | Train/val split, augmentation, prefetch |
| 4 | Model Architecture | Load saved model or build from scratch |
| 5 | Training | 2-phase training with callbacks |
| 6 | Evaluation | Confusion matrix, classification report |
| 7 | Training Curves | Loss & accuracy plots (if trained) |
| 8 | Custom Inference | Predict on your own MRI images |
| 9 | Cross-Dataset Test | Generalization test on external data |
| 10 | Save Model | Save/overwrite model to Drive |

---

## Tech Stack

- **Python** 3.12
- **TensorFlow** 2.x / **Keras** 3.x
- **EfficientNetB0** (ImageNet weights)
- **scikit-learn** (metrics)
- **Matplotlib** / **Seaborn** (visualization)
- **Google Colab** (runtime)

---

## License

This project is for educational and research purposes.

Dataset credits:
- [Masoud Nickparvar — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Navoneel — Brain MRI Images for Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
