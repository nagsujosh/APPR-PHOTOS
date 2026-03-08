import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    log_file: str | Path = None,
    metrics_history: list[dict] = None,
    save_path: str | None = None,
):
    """Plot loss and metric curves from training history.

    Accepts either a JSON log file or a list of epoch metric dicts.
    """
    if log_file:
        with open(log_file) as f:
            metrics_history = json.load(f)

    if not metrics_history:
        return

    epochs = range(len(metrics_history))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    if "loss" in metrics_history[0]:
        train_loss = [m.get("train_loss", m.get("loss", 0)) for m in metrics_history]
        val_loss = [m.get("val_loss", 0) for m in metrics_history]
        axes[0, 0].plot(epochs, train_loss, label="Train")
        axes[0, 0].plot(epochs, val_loss, label="Val")
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()
        axes[0, 0].set_xlabel("Epoch")

    # Utility UAR
    if "utility_uar" in metrics_history[0] or "train_utility_uar" in metrics_history[0]:
        train_uar = [m.get("train_utility_uar", m.get("utility_uar", 0)) for m in metrics_history]
        val_uar = [m.get("val_utility_uar", 0) for m in metrics_history]
        axes[0, 1].plot(epochs, train_uar, label="Train")
        axes[0, 1].plot(epochs, val_uar, label="Val")
        axes[0, 1].set_title("Utility UAR")
        axes[0, 1].legend()
        axes[0, 1].set_xlabel("Epoch")

    # Speaker ID accuracy (privacy)
    key = "privacy_speaker_id_acc"
    train_key = f"train_{key}"
    val_key = f"val_{key}"
    if any(k in metrics_history[0] for k in [key, train_key]):
        train_spk = [m.get(train_key, m.get(key, 0)) for m in metrics_history]
        val_spk = [m.get(val_key, 0) for m in metrics_history]
        axes[1, 0].plot(epochs, train_spk, label="Train")
        axes[1, 0].plot(epochs, val_spk, label="Val")
        axes[1, 0].set_title("Speaker ID Accuracy (lower = more private)")
        axes[1, 0].legend()
        axes[1, 0].set_xlabel("Epoch")

    # Lambda schedule
    if "lambda" in metrics_history[0]:
        lambdas = [m["lambda"] for m in metrics_history]
        axes[1, 1].plot(epochs, lambdas)
        axes[1, 1].set_title("Privacy Lambda Schedule")
        axes[1, 1].set_xlabel("Epoch")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
