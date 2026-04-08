from __future__ import annotations

from pathlib import Path
from typing import Any

import keras
import numpy as np
from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau
from keras.models import Model
from numpy.typing import NDArray


def compile_model(
    model: Model,
    learning_rate: float = 1e-3,
) -> None:
    """Compile model with Adam optimiser and sparse cross-entropy loss.

    Parameters:
    model,
    learning_rate : float.
    
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train(
    model: Model,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    epochs: int = 20,
    batch_size: int = 64,
    validation_split: float = 0.1,
    early_stopping_patience: int = 5,
) -> History:
    """Train the model and return the training history.

    Parameters:
    model,
    x_train - training images of shape (N, 28, 28, 1).
    y_train - integer class labels.
    epochs : int
    batch_size : int
    validation_split : float

    Returns:
    History - object containing per-epoch metrics.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks
    )
    return history


def evaluate(
    model: Model,
    x_test: NDArray[np.float32],
    y_test: NDArray[np.int64],
) -> dict[str, float]:
    """Evaluate the model on the test set.

    Parameters:
    model,
    x_test,
    y_test,

    Returns:
    dict[str, float]
    """
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return {"loss": loss, "accuracy": accuracy}


def save_model(model: Model, path: str | Path = "saved_model") -> None:
    """Persist the model to disk in the Keras native format.

    Parameters:
    model,
    path - destination directory or ``.keras`` file path.
    """
    model.save(path)
    print(f"Model saved to {path}")
