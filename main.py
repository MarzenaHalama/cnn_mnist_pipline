"""MNIST CNN Classifier — end-to-end pipeline.

Run this script to train a CNN on MNIST, evaluate it, and save the model.

Usage
-----
    python main.py                      # full 60 k training set
    python main.py --subset 5000        # quick run with 5 000 samples
    python main.py --epochs 30 --batch-size 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data import load_mnist
from src.model import build_cnn
from src.train import compile_model, evaluate, save_model, train
from src.visualize import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_conv_filters,
    plot_feature_maps,
    plot_gradcam,
    plot_sample_predictions,
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a CNN on MNIST and generate visualisations.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use only the first N training samples (default: all 60 000).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for saved figures and model.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the full training and visualisation pipeline."""
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)


    print("\n=== Loading data ===")
    (x_train, y_train), (x_test, y_test) = load_mnist(subset_size=args.subset)
    print(f"Train: {x_train.shape}  Test: {x_test.shape}")

    print("\n=== Building model ===")
    model = build_cnn()
    model.summary()


    print("\n=== Training ===")
    compile_model(model, learning_rate=args.lr)
    history = train(
        model,
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\n=== Evaluation ===")
    metrics = evaluate(model, x_test, y_test)
    print(f"Test loss: {metrics['loss']:.4f}  Test accuracy: {metrics['accuracy']:.4f}")

  
    save_model(model, out / "mnist_cnn.keras")

    # --- Visualisations ---
    print("\n=== Generating visualisations ===")
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    plot_training_history(history.history, save_path=str(out / "training_history.png"))
    plot_sample_predictions(model, x_test, y_test, save_path=str(out / "sample_predictions.png"))
    plot_confusion_matrix(y_test, y_pred, save_path=str(out / "confusion_matrix.png"))
    plot_classification_report(y_test, y_pred, save_path=str(out / "classification_report.png"))
    plot_conv_filters(model, layer_name="conv1", save_path=str(out / "conv_filters.png"))
    plot_feature_maps(model, x_test[0], save_path=str(out / "feature_maps.png"))
    plot_gradcam(model, x_test, y_test, save_path=str(out / "gradcam.png"))

    print(f"\nAll outputs saved to {out.resolve()}")


if __name__ == "__main__":
    main()
