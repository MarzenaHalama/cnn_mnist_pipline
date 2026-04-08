from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

CLASS_NAMES: list[str] = [str(i) for i in range(10)]



def plot_training_history(
    history_dict: dict[str, list[float]],
    save_path: str | None = None,
) -> Figure:
    """Plot loss and accuracy curves for training and validation.

    Parameters
    
    history_dict,
    save_path.

    Returns:
    Figure

    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(history_dict["loss"], label="Train loss", color="#5e17eb", linewidth=2.5)
    ax_loss.plot(history_dict["val_loss"], label="Val loss", color="#373737", linewidth=2.5)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss Curve")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(history_dict["accuracy"], label="Train accuracy", color="#cb6ce6", linewidth=2.5)
    ax_acc.plot(history_dict["val_accuracy"], label="Val accuracy", color="#373737", linewidth=2.5)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy Curve")
    ax_acc.legend()
    ax_acc.grid(True)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig



def plot_sample_predictions(
    model: Model,
    images: NDArray[np.float32],
    labels: NDArray[np.int64],
    num_samples: int = 16,
    save_path: str | None = None,
) -> Figure:
    """Display a grid of test images with true and predicted labels.

   

    Parameters:
    model, 
    images,
    labels - GT,
    num_samples : int,
    save_path,

    Returns:
    Figure
    """
    idx = np.random.choice(len(images), num_samples, replace=False)
    sample_imgs = images[idx]
    sample_labels = labels[idx]
    preds = np.argmax(model.predict(sample_imgs, verbose=0), axis=1)

    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.asarray(axes).flatten()

    for i in range(num_samples):
        axes[i].imshow(sample_imgs[i].squeeze(), cmap="gray")
        colour = "green" if preds[i] == sample_labels[i] else "red"
        axes[i].set_title(
            f"True: {sample_labels[i]}  Pred: {preds[i]}",
            fontsize=9,
            color=colour,
        )
        axes[i].axis("off")
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Sample Predictions", fontsize=14)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_confusion_matrix(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    normalize: bool = True,
    save_path: str | None = None,
) -> Figure:
    """Render a colour-coded confusion matrix.

    Parameters:
    y_true,
    y_pred,
    save_path.

    Returns:
    Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    fmt = ".2f" if normalize else "d"
    norm = "true" if normalize else None

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm if not normalize else confusion_matrix(y_true, y_pred, normalize="true"),
        display_labels=CLASS_NAMES,
    )
    disp.plot(cmap="Purples", ax=ax, values_format=fmt, text_kw={"fontsize": 18})
    ax.set_title("Confusion Matrix" + (" (Normalised)" if normalize else ""), fontsize=20)
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
    ax.tick_params(labelsize=15)
    # Align colorbar height with the matrix
    im = ax.images[0]
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Remove the default colorbar created by ConfusionMatrixDisplay
    if disp.im_.colorbar is not None:
        disp.im_.colorbar.remove()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_classification_report(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    save_path: str | None = None,
) -> Figure:
    """Bar chart of per-class precision, recall, and F1-score.

    Parameters
    y_true, 
    y_pred,
    save_path.

    Returns:
    Figure
    """
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True,
        zero_division=0,
    )

    classes = CLASS_NAMES
    precision = [report[c]["precision"] for c in classes]
    recall = [report[c]["recall"] for c in classes]
    f1 = [report[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#cb6ce6", alpha=0.6)
    ax.bar(x, recall, width, label="Recall", color="#373737", alpha=0.3)
    ax.bar(x + width, f1, width, label="F1-score", color="#5e17eb", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig



def plot_conv_filters(
    model: Model,
    layer_name: str = "conv1",
    save_path: str | None = None,
) -> Figure:
    """Visualise the learned kernels of a Conv2D layer.

    Parameters
    model, 
    layer_name : str - convolutional layer name to visualise, e.g. "conv1".
    save_path.

    Returns:
    Figure
    """
    filters, _biases = model.get_layer(layer_name).get_weights()
    n_filters = filters.shape[-1]
    cols = 8
    rows = int(np.ceil(n_filters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.asarray(axes).flatten()
    for i in range(n_filters):
        kernel = filters[:, :, 0, i]
        axes[i].imshow(kernel, cmap="viridis")
        axes[i].set_title(f"F{i}", fontsize=7)
        axes[i].axis("off")
    for i in range(n_filters, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Filters — layer '{layer_name}'", fontsize=13)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_feature_maps(
    model: Model,
    image: NDArray[np.float32],
    layer_names: Sequence[str] = ("conv1", "conv2"),
    max_maps: int = 16,
    save_path: str | None = None,
) -> Figure:
    """Show intermediate feature-map activations for a single image.

    Parameters:
    model,
    image,
    layer_names - names of layers whose outputs to visualise.
    max_maps : int - maximum number of feature maps to display per layer.
    save_path.

    Returns:
    Figure
    """
    outputs = [model.get_layer(n).output for n in layer_names]
    activation_model = Model(inputs=model.inputs, outputs=outputs)
    activations = activation_model.predict(image[np.newaxis], verbose=0)

    n_layers = len(layer_names)
    fig, axes = plt.subplots(
        n_layers, max_maps, figsize=(max_maps * 1.2, n_layers * 1.5),
    )
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    for row, (name, act) in enumerate(zip(layer_names, activations)):
        n_show = min(act.shape[-1], max_maps)
        for col in range(max_maps):
            ax = axes[row, col]
            if col < n_show:
                ax.imshow(act[0, :, :, col], cmap="viridis")
                if col == 0:
                    ax.set_ylabel(name, fontsize=9)
            ax.axis("off")

    fig.suptitle("Feature Map Activations", fontsize=13)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig



def compute_gradcam(
    model: Model,
    image: NDArray[np.float32],
    layer_name: str = "conv2",
    class_index: int | None = None,
) -> NDArray[np.float32]:
    """Compute a Grad-CAM heatmap for image.

    Parameters:
    model,
    image,
    layer_name : str - target convolutional layer name,
    class_index - int - target class index for which to compute Grad-CAM. If ``None``, the predicted class is used.

    Returns:
    Heatmap as 2-D array of shape (H, W) with values in [0, 1].
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.outputs[0]],
    )

    img_tensor = tf.cast(image[np.newaxis], tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1).numpy()
    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam /= cam.max()
    return cam


def plot_gradcam(
    model: Model,
    images: NDArray[np.float32],
    labels: NDArray[np.int64],
    layer_name: str = "conv2",
    num_samples: int = 8,
    save_path: str | None = None,
) -> Figure:
    """Overlay Grad-CAM heatmaps on a set of sample images.

    Parameters:
    model,
    images,
    labels,
    layer_name,
    num_samples : int -how many images to display.
    save_path.

    Returns:
    Figure
    """
    idx = np.random.choice(len(images), num_samples, replace=False)
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4.5))

    for i, ix in enumerate(idx):
        img = images[ix]
        pred = int(np.argmax(model.predict(img[np.newaxis], verbose=0)))
        cam = compute_gradcam(model, img, layer_name=layer_name)

        # Original image
        axes[0, i].imshow(img.squeeze(), cmap="gray")
        axes[0, i].set_title(f"True: {labels[ix]}", fontsize=8)
        axes[0, i].axis("off")

        # Grad-CAM overlay
        cam_resized = tf.image.resize(
            cam[..., np.newaxis], (28, 28),
        ).numpy().squeeze()
        axes[1, i].imshow(img.squeeze(), cmap="gray")
        axes[1, i].imshow(cam_resized, cmap="jet", alpha=0.5)
        axes[1, i].set_title(f"Pred: {pred}", fontsize=8)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
    fig.suptitle(f"Grad-CAM (layer: {layer_name})", fontsize=13)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_tsne_embeddings(
    model: Model,
    images: NDArray[np.float32],
    labels: NDArray[np.int64],
    layer_name: str = "dense1",
    num_samples: int = 3000,
    perplexity: float = 30.0,
    save_path: str | None = None,
) -> Figure:
    """2-D t-SNE projection of the learned feature embeddings.

    Parameters:
    model,
    images,
    labels,
    layer_name : str - layer whose output to use as embedding.
    num_samples : int - number of images to include (t-SNE is O(n^2)).
    perplexity : float - t-SNE perplexity.
    save_path.

    Returns:
    Figure
    """
    from sklearn.manifold import TSNE

    idx = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    sample_imgs = images[idx]
    sample_labels = labels[idx]

    embedding_model = Model(
        inputs=model.inputs,
        outputs=model.get_layer(layer_name).output,
    )
    embeddings = embedding_model.predict(sample_imgs, verbose=0)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=sample_labels,
        cmap="tab10",
        s=8,
        alpha=0.7,
    )
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit class", fontsize=12)
    ax.set_title(f"t-SNE of '{layer_name}' embeddings ({len(sample_labels)} samples)", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def _maybe_save(fig: Figure, path: str | None) -> None:
    """Save *fig* to *path* if it is not ``None``."""
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {path}")
