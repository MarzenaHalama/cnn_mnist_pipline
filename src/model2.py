from __future__ import annotations

import keras
from keras import layers
from keras.models import Model


def build_cnn_v2(
    input_shape: tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    dropout_rate: float = 0.25,
) -> Model:
    """Build a deeper CNN with BatchNorm, Dropout and data augmentation.


    Parameters:
    input_shape,
    num_classes,
    dropout_rate,

    Returns:
    keras.Model
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Classifier head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="mnist_cnn_v2")
    return model
