from __future__ import annotations

import keras
from keras import layers
from keras.models import Model


def build_cnn(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
) -> Model:
    """Build a simple convolutional neural network for image classification.

    Parameters
    input_shape - spatial dimensions plus channels, e.g. ``(28, 28, 1)``.
    num_classes - int.


    Returns:
    keras.Model
    
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", name="conv1"),
            layers.MaxPooling2D((2, 2), name="pool1"),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", name="conv2"),
            layers.MaxPooling2D((2, 2), name="pool2"),
            # Classifier head
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense1"),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="mnist_cnn",
    )
    return model
