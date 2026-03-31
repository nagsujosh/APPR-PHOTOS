#!/usr/bin/env python
"""Standalone evaluation script."""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from aapr.data.utils import create_dataloaders
from aapr.evaluation.evaluator import Evaluator
from aapr.models.adversary import MultiHeadAdversary
from aapr.models.privacy_filter import PrivacyFilter
from aapr.models.task_model import TaskModel
from aapr.utils.device import get_device, load_checkpoint
from aapr.utils.logging import setup_logger
from aapr.utils.seed import set_seed
from scripts.train import build_dataset, build_feature_extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON generated during training",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    logger = setup_logger("aapr")

    checkpoint_dir = Path(args.checkpoint).parent
    config_path = Path(args.config) if args.config else checkpoint_dir.parent / "config.json"

    with open(config_path) as handle:
        cfg = json.load(handle)

    set_seed(cfg.get("seed", 42))

    dataset = build_dataset(cfg)
    loaders = create_dataloaders(
        dataset,
        batch_size=cfg["dataset"].get("batch_size", 32),
        seed=cfg.get("seed", 42),
        num_workers=cfg["dataset"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    feature_extractor = build_feature_extractor(cfg)
    filter_cfg = cfg["model"]["filter"]
    input_dim = feature_extractor.output_dim

    privacy_filter = PrivacyFilter(
        input_dim=input_dim,
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

    state = load_checkpoint(args.checkpoint, device, weights_only=False)
    if "feature_extractor" in state:
        feature_extractor.load_state_dict(state["feature_extractor"])
    privacy_filter.load_state_dict(state["privacy_filter"])
    task_model.load_state_dict(state["task_model"])
    adversary.load_state_dict(state["adversary"])

    privacy_filter.to(device)
    task_model.to(device)
    adversary.to(device)
    feature_extractor.to(device)

    evaluator = Evaluator(device)
    results = evaluator.evaluate(
        privacy_filter, task_model, adversary, loaders["test"], feature_extractor
    )

    logger.info("Test Results:")
    for key, value in sorted(results.items()):
        logger.info(f"  {key}: {value:.4f}")

    output_path = checkpoint_dir.parent / "test_results.json"
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
