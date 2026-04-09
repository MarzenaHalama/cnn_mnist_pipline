# mnist-cnn-from-scratch

<img width="753" height="183" alt="image" src="https://github.com/user-attachments/assets/1a27cada-fb02-473c-8cb1-66c376ffc077" />


This project is modular CNN pipeline for handwritten digit classification on MNIST, built from scratch. It covers every step from data loading and model definition to training, evaluation, and model persistence.

---
## Model Architecture

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
├── export_model.py         # Export a saved .keras model to .pkl format
├── comparison.ipynb        # Notebook comparing model variants
├── requirements.txt        # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── data.py             # MNIST loading & preprocessing
│   ├── model.py            # CNN architecture definition (baseline)
│   ├── model2.py           # Deeper CNN with BatchNorm, Dropout & augmentation
│   ├── train.py            # Compilation, training loop, evaluation, model saving (.keras & .pkl)
│   └── visualize.py        # All visualisation functions (plots, Grad-CAM, etc.)
├── output_model_1/         # Saved model & plots for baseline CNN
├── output_model_2/         # Saved model & plots for deeper CNN v2
└── outputs/                # Default output directory

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

### Export model to pickle

```bash
python export_model.py --input output_model_1/mnist_cnn.keras --output output_model_1/mnist_cnn.pkl
```

---

## Training Parameters

| Parameter | Value |
|-----------|-------|
| **Loss function** | Sparse Categorical Cross-Entropy |
| **Optimiser** | Adam (default lr = 0.001) |
| **Metrics** | Accuracy |
| **Callbacks** | EarlyStopping (patience 5/15), ReduceLROnPlateau (factor 0.5, patience 3) |
| **Validation split** | 10 % of training data |
| **Default epochs** | 20 |
| **Default batch size** | 64 |


