# mnist-cnn-from-scratch

This project is modular CNN pipeline for handwritten digit classification on MNIST, built from scratch. It covers every step from data loading and model definition to training, evaluation, and model persistence.

---

<img width="701" height="372" alt="image" src="https://github.com/user-attachments/assets/1eb0762b-9e5d-415d-ae51-398eabfb9ef8" />


## Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | MNIST — 60 000 train / 10 000 test greyscale 28×28 images |
| **Task** | 10-class digit classification (0 – 9) |
| **Model** | Lightweight 2-block CNN (~93 k parameters) |
| **Framework** | TensorFlow / Keras |

---

## Repository Structure

```
.
├── main.py                 # End-to-end pipeline (train + evaluate + visualise)
├── requirements.txt        # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── data.py             # MNIST loading & preprocessing
│   ├── model.py            # CNN architecture definition
│   ├── train.py            # Compilation, training loop, evaluation, model saving
│   └── visualize.py        # All visualisation functions (plots, Grad-CAM, etc.)
└── outputs/                # Generated after running main.py
    ├── mnist_cnn.keras
    ├── training_history.png
    ├── sample_predictions.png
    ├── confusion_matrix.png
    ├── classification_report.png
    ├── conv_filters.png
    ├── feature_maps.png
    ├── gradcam.png
    └── tsne_embeddings.png
```

---

## Features

- **Modular source code** — each concern (data, model, training, visualisation)
  lives in its own module with full docstrings and type annotations.
- **CLI interface** — configure epochs, batch size, learning rate, and subset
  size from the command line.
- **Callbacks** — EarlyStopping and ReduceLROnPlateau are wired in by default.
- **Rich visualisations** — training curves, sample predictions, confusion
  matrix, per-class metrics, convolutional filters, feature maps, and Grad-CAM
  heatmaps are generated automatically and saved to `outputs/`.

---

## Getting Started

### Prerequisites

- **Python 3.12** (TensorFlow does not yet support 3.13+)
- pip

### Installation

```bash

# Create a virtual environment (recommended)
py -3.12 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Full pipeline

```bash
python main.py
```

The trained model and all visualisation plots are saved to `outputs/`.

### Quick experiment with a smaller dataset

```bash
python main.py --subset 5000 --epochs 10
```

---

## Generated Visualisations

| File | Description |
|------|-------------|
| `training_history.png` | Loss and accuracy curves (train vs. validation) |
| `sample_predictions.png` | Grid of test images with true/predicted labels |
| `confusion_matrix.png` | Normalised confusion matrix |
| `classification_report.png` | Per-class precision, recall, and F1-score bar chart |
| `conv_filters.png` | Learned kernels of the first Conv2D layer |
| `feature_maps.png` | Intermediate activations from conv1 and conv2 |
| `gradcam.png` | Grad-CAM heatmap overlays on sample images |
| `tsne_embeddings.png` | 2-D t-SNE projection of learned feature embeddings (dense1) |

---

## Model Architecture

```
Input  (28, 28, 1)
  │
  ├─ Conv2D  32 filters, 3×3, ReLU
  ├─ MaxPool 2×2
  │
  ├─ Conv2D  64 filters, 3×3, ReLU
  ├─ MaxPool 2×2
  │
  ├─ Flatten
  ├─ Dense   128, ReLU
  └─ Dense   10, Softmax   → class probabilities
```

**Loss:** Sparse Categorical Cross-Entropy  
**Optimiser:** Adam (default lr = 0.001)  
**Metrics:** Accuracy


