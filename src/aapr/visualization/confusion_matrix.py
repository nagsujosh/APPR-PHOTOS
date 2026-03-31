from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return a dense confusion matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    normalize: bool = True,
    title: str = "Utility Confusion Matrix",
):
    """Plot and save a confusion matrix heatmap."""
    save_path = Path(save_path)
    matrix = compute_confusion_matrix(y_true, y_pred, len(class_names))

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        display = np.divide(
            matrix,
            np.maximum(row_sums, 1),
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sums != 0,
        )
        fmt = ".2f"
    else:
        display = matrix
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()

