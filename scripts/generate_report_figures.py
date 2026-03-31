#!/usr/bin/env python
"""Generate report-ready figures from a trained checkpoint."""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from aapr.data.utils import create_dataloaders
from aapr.models.adversary import MultiHeadAdversary
from aapr.models.privacy_filter import PrivacyFilter
from aapr.models.task_model import TaskModel
from aapr.utils.device import get_device, load_checkpoint
from aapr.utils.logging import setup_logger
from aapr.utils.seed import set_seed
from aapr.visualization.confusion_matrix import plot_confusion_matrix
from aapr.visualization.embeddings import plot_embeddings
from aapr.visualization.saliency import (
    compute_input_saliency,
    overlay_saliency,
    plot_saliency_grid,
)
from aapr.visualization.training_curves import (
    parse_epoch_metrics_from_log,
    plot_training_curves,
)
from scripts.train import build_dataset, build_feature_extractor


def load_models(cfg: dict, checkpoint_path: Path, device: torch.device):
    dataset = build_dataset(cfg)
    feature_extractor = build_feature_extractor(cfg)

    filter_cfg = cfg["model"]["filter"]
    privacy_filter = PrivacyFilter(
        input_dim=feature_extractor.output_dim,
        hidden_dim=filter_cfg.get("hidden_dim", 256),
        output_dim=filter_cfg.get("output_dim", 128),
        num_layers=filter_cfg.get("num_layers", 3),
        use_vib=filter_cfg.get("use_vib", True),
        vib_beta=filter_cfg.get("vib_beta", 0.001),
    )
    task_model = TaskModel(
        input_dim=filter_cfg.get("output_dim", 128),
        hidden_dim=cfg["model"]["task"].get("hidden_dim", 128),
        num_classes=dataset.num_utility_classes,
        dropout=cfg["model"]["task"].get("dropout", 0.2),
    )

    adv_heads = dict(cfg["model"]["adversary"].get("heads", {"gender": 2, "speaker_id": 10}))
    adv_heads["speaker_id"] = dataset.num_speakers
    adversary = MultiHeadAdversary(
        input_dim=filter_cfg.get("output_dim", 128),
        trunk_dim=cfg["model"]["adversary"].get("trunk_dim", 128),
        heads=adv_heads,
        dropout=cfg["model"]["adversary"].get("dropout", 0.3),
    )

    state = load_checkpoint(checkpoint_path, device, weights_only=False)
    if "feature_extractor" in state:
        feature_extractor.load_state_dict(state["feature_extractor"])
    privacy_filter.load_state_dict(state["privacy_filter"])
    task_model.load_state_dict(state["task_model"])
    adversary.load_state_dict(state["adversary"])

    feature_extractor.to(device).eval()
    privacy_filter.to(device).eval()
    task_model.to(device).eval()
    adversary.to(device).eval()

    return dataset, feature_extractor, privacy_filter, task_model, adversary, state


@torch.no_grad()
def collect_embeddings(
    privacy_filter,
    feature_extractor,
    loader,
    device,
    max_batches=None,
    max_examples=None,
):
    """Collect before/after pooled embeddings from a dataloader."""
    all_before, all_after = [], []
    all_speakers, all_utility = [], []

    num_examples = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device)
        features = feature_extractor(images)
        filtered, _ = privacy_filter(features)

        all_before.append(features.mean(dim=2).cpu().numpy())
        all_after.append(filtered.mean(dim=2).cpu().numpy())
        all_speakers.append(batch["privacy_labels"]["speaker_id"].numpy())
        all_utility.append(batch["utility_label"].numpy())
        num_examples += len(batch["utility_label"])
        if max_examples is not None and num_examples >= max_examples:
            break

    return (
        np.concatenate(all_before),
        np.concatenate(all_after),
        np.concatenate(all_speakers),
        np.concatenate(all_utility),
    )


@torch.no_grad()
def collect_test_predictions(
    privacy_filter,
    task_model,
    adversary,
    feature_extractor,
    loader,
    device,
    max_batches=None,
):
    """Collect utility and privacy predictions for the test split."""
    utility_preds, utility_labels = [], []
    privacy_metrics = {}

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device)
        features = feature_extractor(images)
        filtered, _ = privacy_filter(features)
        utility_logits = task_model(filtered)
        privacy_logits = adversary(filtered)

        utility_preds.append(utility_logits.argmax(dim=1).cpu().numpy())
        utility_labels.append(batch["utility_label"].numpy())

        for key, logits in privacy_logits.items():
            labels = batch["privacy_labels"].get(key)
            if labels is None:
                continue
            valid = labels >= 0
            if valid.sum() == 0:
                continue
            metric = privacy_metrics.setdefault(key, {"pred": [], "true": []})
            metric["pred"].append(logits[valid].argmax(dim=1).cpu().numpy())
            metric["true"].append(labels[valid].numpy())

    results = {
        "utility_true": np.concatenate(utility_labels),
        "utility_pred": np.concatenate(utility_preds),
    }
    for key, arrays in privacy_metrics.items():
        results[f"privacy_{key}_true"] = np.concatenate(arrays["true"])
        results[f"privacy_{key}_pred"] = np.concatenate(arrays["pred"])
    return results


def select_saliency_samples(subset, max_samples: int) -> list[dict]:
    """Pick a small diverse set of qualitative examples from the test split."""
    selected_by_combo = {}

    for idx in range(len(subset)):
        sample = subset[idx]
        if sample["privacy_labels"]["gender"] < 0:
            continue

        combo = (
            int(sample["utility_label"]),
            int(sample["privacy_labels"]["gender"]),
        )
        selected_by_combo.setdefault(combo, sample)
        if len(selected_by_combo) >= max_samples:
            break

    return list(selected_by_combo.values())[:max_samples]


def generate_saliency_rows(
    samples: list[dict],
    feature_extractor,
    privacy_filter,
    task_model,
    adversary,
    utility_names: list[str],
    device: torch.device,
) -> list[dict]:
    """Build qualitative saliency rows for report-ready plotting."""
    rows = []
    gender_names = {0: "male", 1: "female"}
    modules = [feature_extractor, privacy_filter, task_model, adversary]

    def utility_forward(x):
        features = feature_extractor(x)
        filtered, _ = privacy_filter(features)
        return task_model(filtered)

    def gender_forward(x):
        features = feature_extractor(x)
        filtered, _ = privacy_filter(features)
        return adversary(filtered)["gender"]

    for sample in samples:
        image = sample["image"].to(device)
        utility_target = int(sample["utility_label"])
        gender_target = int(sample["privacy_labels"]["gender"])

        utility_saliency = compute_input_saliency(
            image, utility_forward, utility_target, modules
        )
        privacy_saliency = compute_input_saliency(
            image, gender_forward, gender_target, modules
        )

        rows.append(
            {
                "original": image.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0),
                "original_title": (
                    f"Image\n"
                    f"utility={utility_names[utility_target]}, "
                    f"gender={gender_names.get(gender_target, str(gender_target))}"
                ),
                "utility_overlay": overlay_saliency(image, utility_saliency),
                "utility_title": f"Utility saliency\n(target={utility_names[utility_target]})",
                "privacy_overlay": overlay_saliency(image, privacy_saliency, cmap_name="viridis"),
                "privacy_title": (
                    f"Privacy saliency\n(gender target="
                    f"{gender_names.get(gender_target, str(gender_target))})"
                ),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_log", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/report_figures")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--embedding_samples", type=int, default=2000)
    parser.add_argument("--saliency_samples", type=int, default=4)
    parser.add_argument("--skip_training_curves", action="store_true")
    parser.add_argument("--skip_confusion", action="store_true")
    parser.add_argument("--skip_embeddings", action="store_true")
    parser.add_argument("--skip_saliency", action="store_true")
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="Limit test batches for faster figure generation when running on CPU.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device(args.device)
    logger = setup_logger("aapr")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = checkpoint_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else checkpoint_dir.parent / "config.json"
    train_log = Path(args.train_log) if args.train_log else checkpoint_dir.parent / "train.log"

    with open(config_path, encoding="utf-8") as handle:
        cfg = json.load(handle)

    set_seed(cfg.get("seed", 42))

    dataset, feature_extractor, privacy_filter, task_model, adversary, state = load_models(
        cfg, checkpoint_path, device
    )
    loaders = create_dataloaders(
        dataset,
        batch_size=cfg["dataset"].get("batch_size", 32),
        train_ratio=cfg["dataset"].get("train_ratio", 0.7),
        val_ratio=cfg["dataset"].get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    history = parse_epoch_metrics_from_log(train_log)
    if history and not args.skip_training_curves:
        plot_training_curves(metrics_history=history, save_path=output_dir / "training_curves.png")
        logger.info(f"Saved training curves to {output_dir / 'training_curves.png'}")

    predictions = None
    if not args.skip_confusion:
        predictions = collect_test_predictions(
            privacy_filter,
            task_model,
            adversary,
            feature_extractor,
            loaders["test"],
            device,
            max_batches=args.max_eval_batches,
        )
        plot_confusion_matrix(
            predictions["utility_true"],
            predictions["utility_pred"],
            class_names=list(dataset.utility_label_names),
            save_path=output_dir / "utility_confusion_matrix.png",
        )
        logger.info(
            f"Saved confusion matrix to {output_dir / 'utility_confusion_matrix.png'}"
        )

    if not args.skip_embeddings:
        before, after, speakers, utility = collect_embeddings(
            privacy_filter,
            feature_extractor,
            loaders["test"],
            device,
            max_batches=args.max_eval_batches,
            max_examples=args.embedding_samples,
        )
        plot_embeddings(
            before,
            after,
            speakers,
            utility,
            method="tsne",
            save_path=str(output_dir / "embeddings_tsne.png"),
            max_samples=args.embedding_samples,
        )
        logger.info(f"Saved embedding plot to {output_dir / 'embeddings_tsne.png'}")

    if not args.skip_saliency:
        saliency_samples = select_saliency_samples(loaders["test"].dataset, args.saliency_samples)
        rows = generate_saliency_rows(
            saliency_samples,
            feature_extractor,
            privacy_filter,
            task_model,
            adversary,
            list(dataset.utility_label_names),
            device,
        )
        plot_saliency_grid(rows, output_dir / "qualitative_saliency.png")
        logger.info(f"Saved saliency plot to {output_dir / 'qualitative_saliency.png'}")

    summary = {
        "checkpoint": str(checkpoint_path),
        "best_epoch": state.get("epoch"),
        "best_val_metrics": state.get("metrics", {}),
        "num_test_examples": int(len(predictions["utility_true"])) if predictions else None,
        "max_eval_batches": args.max_eval_batches,
        "figure_dir": str(output_dir),
    }
    with open(output_dir / "figure_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info(f"Saved figure summary to {output_dir / 'figure_summary.json'}")


if __name__ == "__main__":
    main()
