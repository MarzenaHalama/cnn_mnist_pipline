"""Export a saved Keras model to pickle format.

Usage
-----
    python export_model.py                                          # default: outputs/mnist_cnn.keras -> outputs/mnist_cnn.pkl
    python export_model.py --input output_model_1/mnist_cnn.keras   # custom input
    python export_model.py --input output_model_2/mnist_cnn_v2.keras --output model_v2.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import keras

from src.train import save_model_pkl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Keras model to pickle format.")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/mnist_cnn.keras",
        help="Path to the saved .keras model file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination .pkl file path (default: same name with .pkl extension).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")

    output_path = args.output or input_path.with_suffix(".pkl")

    print(f"Loading model from {input_path}")
    model = keras.models.load_model(input_path)

    save_model_pkl(model, output_path)


if __name__ == "__main__":
    main()
