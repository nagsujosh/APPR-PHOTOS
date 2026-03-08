#!/usr/bin/env python
"""Generate all visualization plots."""
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from aapr.utils.device import get_device
from aapr.utils.seed import set_seed
from aapr.utils.logging import setup_logger
from aapr.visualization.embeddings import plot_embeddings
from aapr.visualization.pareto_plot import plot_pareto_frontier


def collect_embeddings(privacy_filter, feature_extractor, loader, device):
    """Collect before/after embeddings from a dataloader."""
    privacy_filter.eval()
    if feature_extractor:
        feature_extractor.eval()

    all_before, all_after = [], []
    all_speakers, all_emotions = [], []

    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)
            features = feature_extractor(waveform) if feature_extractor else waveform
            filtered, _ = privacy_filter(features)

            all_before.append(features.mean(dim=2).cpu().numpy())
            all_after.append(filtered.mean(dim=2).cpu().numpy())
            all_speakers.append(batch["privacy_labels"]["speaker_id"].numpy())
            all_emotions.append(batch["utility_label"].numpy())

    return (
        np.concatenate(all_before),
        np.concatenate(all_after),
        np.concatenate(all_speakers),
        np.concatenate(all_emotions),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/plots")
    parser.add_argument("--pareto_results", type=str, default=None)
    args = parser.parse_args()

    device = get_device("auto")
    set_seed(42)
    logger = setup_logger("aapr")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = args.config or str(checkpoint_dir.parent / "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    # Build models and load checkpoint
    from scripts.train import build_dataset, build_feature_extractor
    from aapr.data.utils import create_dataloaders
    from aapr.models.privacy_filter import PrivacyFilter
    from aapr.models.task_model import TaskModel
    from aapr.models.adversary import MultiHeadAdversary

    dataset = build_dataset(cfg)
    loaders = create_dataloaders(dataset, batch_size=32, seed=42)

    feature_extractor = build_feature_extractor(cfg)
    filter_cfg = cfg["model"]["filter"]
    privacy_filter = PrivacyFilter(
        input_dim=feature_extractor.output_dim,
        hidden_dim=filter_cfg["hidden_dim"],
        output_dim=filter_cfg["output_dim"],
        use_vib=filter_cfg["use_vib"],
    )

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    privacy_filter.load_state_dict(state["privacy_filter"])
    privacy_filter.to(device)
    feature_extractor.to(device)

    # t-SNE embeddings
    logger.info("Generating embedding plots...")
    before, after, speakers, emotions = collect_embeddings(
        privacy_filter, feature_extractor, loaders["test"], device
    )
    plot_embeddings(
        before, after, speakers, emotions,
        method="tsne", save_path=str(output_dir / "embeddings_tsne.png"),
    )
    logger.info(f"Saved embeddings plot to {output_dir / 'embeddings_tsne.png'}")

    # Pareto frontier
    if args.pareto_results:
        logger.info("Generating Pareto frontier plot...")
        plot_pareto_frontier(
            args.pareto_results,
            save_path=str(output_dir / "pareto_frontier.png"),
        )
        logger.info(f"Saved Pareto plot to {output_dir / 'pareto_frontier.png'}")


if __name__ == "__main__":
    main()
