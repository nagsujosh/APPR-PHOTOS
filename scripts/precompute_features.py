#!/usr/bin/env python
"""Precompute image features and cache to disk."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aapr.data.utils import speaker_stratified_split
from aapr.features.feature_cache import precompute_features
from aapr.utils.config import apply_overrides, load_config
from aapr.utils.device import get_device
from aapr.utils.logging import setup_logger
from aapr.utils.seed import set_seed
from scripts.train import build_dataset, build_feature_extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    device = get_device(args.device)
    set_seed(cfg.get("seed", 42))
    logger = setup_logger("aapr")

    dataset = build_dataset(cfg)
    train_set, val_set, test_set = speaker_stratified_split(
        dataset,
        train_ratio=cfg["dataset"].get("train_ratio", 0.7),
        val_ratio=cfg["dataset"].get("val_ratio", 0.15),
        seed=cfg.get("seed", 42),
    )

    extractor = build_feature_extractor(cfg)
    default_cache = cfg["feature"].get("cache_dir", "data/processed/features")
    cache_dir = Path(args.cache_dir if args.cache_dir else default_cache)

    logger.info(f"Precomputing features for {len(dataset)} images")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Device: {device}")

    for split_name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        logger.info(f"Processing {split_name} split ({len(split_data)} samples)")
        precompute_features(
            split_data,
            extractor,
            cache_dir=cache_dir,
            split=split_name,
            batch_size=args.batch_size,
            device=device,
        )

    logger.info(f"Features cached to {cache_dir}")


if __name__ == "__main__":
    main()
