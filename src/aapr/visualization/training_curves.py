import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)(?P<retrain>\s+\[ADV RETRAIN\])?\s+\|\s+"
    r"Train UAR:\s+(?P<train_uar>[0-9.]+)\s+\|\s+"
    r"Val UAR:\s+(?P<val_uar>[0-9.]+)\s+\|\s+"
    r"Lambda:\s+(?P<lambda>[0-9.]+)"
)


def parse_epoch_metrics_from_log(log_file: str | Path) -> list[dict]:
    """Parse the latest training run from the text train.log file."""
    log_file = Path(log_file)
    if not log_file.exists():
        return []

    runs: list[list[dict]] = []
    current_run: list[dict] = []

    with open(log_file, encoding="utf-8") as handle:
        for line in handle:
            if "INFO - Device:" in line:
                if current_run:
                    runs.append(current_run)
                current_run = []

            match = EPOCH_LINE_RE.search(line)
            if not match:
                continue

            current_run.append(
                {
                    "epoch": int(match.group("epoch")),
                    "train_utility_uar": float(match.group("train_uar")),
                    "val_utility_uar": float(match.group("val_uar")),
                    "lambda": float(match.group("lambda")),
                    "is_retrain": bool(match.group("retrain")),
                }
            )

    if current_run:
        runs.append(current_run)

    return runs[-1] if runs else []


def plot_training_curves(
    log_file: str | Path = None,
    metrics_history: list[dict] = None,
    save_path: str | None = None,
):
    """Plot loss and metric curves from training history.

    Accepts either a JSON metrics file, a text train.log file, or a list of
    epoch metric dicts.
    """
    if log_file:
        with open(log_file, encoding="utf-8") as handle:
            first_char = handle.read(1)
        if first_char == "[":
            with open(log_file, encoding="utf-8") as handle:
                metrics_history = json.load(handle)
        else:
            metrics_history = parse_epoch_metrics_from_log(log_file)

    if not metrics_history:
        return

    epochs = [m.get("epoch", idx) for idx, m in enumerate(metrics_history)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    retrain_epochs = [m["epoch"] for m in metrics_history if m.get("is_retrain")]
    for ax in axes.ravel():
        for epoch in retrain_epochs:
            ax.axvspan(epoch - 0.5, epoch + 0.5, color="#f59e0b", alpha=0.08)

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

    if retrain_epochs:
        axes[1, 1].text(
            0.02,
            0.92,
            "Shaded bands = adversary retraining",
            transform=axes[1, 1].transAxes,
            fontsize=9,
            color="#92400e",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
