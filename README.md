# mnist-cnn-from-scratch

This project is a production-style, modular CNN pipeline for handwritten digit classification on MNIST — built from scratch. It covers every step from data loading and model definition to training, evaluation, and model persistence.

---

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
├── main.py                 # End-to-end pipeline (train + evaluate + save)
├── requirements.txt        # Python dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── data.py             # MNIST loading & preprocessing
│   ├── model.py            # CNN architecture definition
│   ├── train.py            # Compilation, training loop, evaluation, model saving
.

```

---

## Features

- **Modular source code** — each concern (data, model, training)
  lives in its own module with full docstrings and type annotations.
- **CLI interface** — configure epochs, batch size, learning rate, and subset
  size from the command line.
- **Callbacks** — EarlyStopping and ReduceLROnPlateau are wired in by default.

---

## Getting Started

### Prerequisites

- **Python 3.10 – 3.12** (TensorFlow does not yet support 3.13+)
- pip

### Installation

```bash

# Create a virtual environment (recommended)
python -m venv .venv
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

The trained model is saved to `outputs/`.

### Quick experiment with a smaller dataset

```bash
python main.py --subset 5000 --epochs 10
```

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


