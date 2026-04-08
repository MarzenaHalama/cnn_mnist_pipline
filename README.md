# mnist-cnn-from-scratch

<img width="753" height="183" alt="image" src="https://github.com/user-attachments/assets/1a27cada-fb02-473c-8cb1-66c376ffc077" />


This project is modular CNN pipeline for handwritten digit classification on MNIST, built from scratch. It covers every step from data loading and model definition to training, evaluation, and model persistence.

---

<img width="701" height="372" alt="image" src="https://github.com/user-attachments/assets/1eb0762b-9e5d-415d-ae51-398eabfb9ef8" />
<img width="701" height="689" alt="image" src="https://github.com/user-attachments/assets/db328b6d-f152-43f7-829e-ff0c9254fa8f" />


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
│   ├── model.py            # CNN architecture definition (baseline)
│   ├── model2.py           # Deeper CNN with BatchNorm, Dropout & augmentation
│   ├── train.py            # Compilation, training loop, evaluation, model saving
│   └── visualize.py        # All visualisation functions (plots, Grad-CAM, etc.)
└── outputs_[i]        

```

---

## Features

- **Modular source code** — each concern (data, model, training, visualisation)
  lives in its own module with full docstrings and type annotations.
- **Two model variants** — a lightweight baseline CNN and a deeper CNN (v2) with
  BatchNorm, Dropout, and built-in data augmentation for small training sets.
- **CLI interface** — configure epochs, batch size, learning rate, model variant,
  and subset size from the command line.
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

### Model 1 — baseline CNN, 5 000 training samples

```bash
python main.py --model 1 --subset 5000 --epochs 20 --batch-size 64 --lr 0.001 --output-dir output_model_1
```

### Model 2 — deeper CNN, 500 training samples

```bash
python main.py --model 2 --subset 500 --epochs 20 --batch-size 64 --lr 0.001 --output-dir output_model_2
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

### Model 1 — Baseline CNN (~93 k params)

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

### Model 2 — Deeper CNN with regularisation (~400 k params)

```
Input  (28, 28, 1)
  │
  ├─ RandomRotation + RandomZoom  (augmentation, train only)
  │
  ├─ Conv2D 32, 3×3, same → BN → ReLU
  ├─ Conv2D 32, 3×3, same → BN → ReLU
  ├─ MaxPool 2×2 → Dropout 0.25
  │
  ├─ Conv2D 64, 3×3, same → BN → ReLU
  ├─ Conv2D 64, 3×3, same → BN → ReLU
  ├─ MaxPool 2×2 → Dropout 0.25
  │
  ├─ Conv2D 128, 3×3, same → BN → ReLU
  ├─ MaxPool 2×2 → Dropout 0.25
  │
  ├─ Flatten
  ├─ Dense 256, ReLU → BN → Dropout 0.4
  ├─ Dense 128, ReLU → Dropout 0.4
  └─ Dense 10, Softmax   → class probabilities
```

**Loss:** Sparse Categorical Cross-Entropy  
**Optimiser:** Adam (default lr = 0.001)  
**Metrics:** Accuracy


