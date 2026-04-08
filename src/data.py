"""Data loading and preprocessing utilities for the MNIST dataset."""

from __future__ import annotations

from typing import Tuple

import keras
import numpy as np
from numpy.typing import NDArray


def load_mnist(
    subset_size: int | None = None,
) -> Tuple[
    Tuple[NDArray[np.float32], NDArray[np.int64]],
    Tuple[NDArray[np.float32], NDArray[np.int64]],
]:
    """Load the MNIST dataset, normalise pixel values, and add a channel dim.

    Parameters
    ----------
    subset_size : int | None
        If provided, only the first *subset_size* training samples are kept.
        Useful for quick experiments on limited hardware.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        Training and test splits ready for a Conv2D model.
        Images are float32 in [0, 1] with shape ``(N, 28, 28, 1)``.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    if subset_size is not None:
        x_train = x_train[:subset_size]
        y_train = y_train[:subset_size]

    x_train = _preprocess(x_train)
    x_test = _preprocess(x_test)

    return (x_train, y_train), (x_test, y_test)


def _preprocess(images: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Scale pixel values to [0, 1] and add a trailing channel dimension.

    Parameters
    ----------
    images : NDArray[np.uint8]
        Raw images of shape ``(N, 28, 28)``.

    Returns
    -------
    NDArray[np.float32]
        Processed images of shape ``(N, 28, 28, 1)``.
    """
    images = images.astype(np.float32) / 255.0
    return np.expand_dims(images, axis=-1)
