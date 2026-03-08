#!/usr/bin/env python
"""Precompute SSL features and cache to disk."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aapr.utils.device import get_device
from aapr.utils.seed import set_seed
from aapr.utils.logging import setup_logger
from aapr.data.cremad import CremaDDataset
from aapr.data.mderma import MDERMADataset
from aapr.data.tame import TAMEDataset
from aapr.data.utils import speaker_stratified_split
from aapr.features.ssl_embeddings import SSLEmbeddingExtractor
from aapr.features.feature_cache import precompute_features


DATASET_MAP = {
    "cremad": CremaDDataset,
    "mderma": MDERMADataset,
    "tame": TAMEDataset,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_MAP.keys())
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="data/processed")
    parser.add_argument("--model", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = get_device(args.device)
    set_seed(42)
    logger = setup_logger("aapr")

    logger.info(f"Precomputing features for {args.dataset} using {args.model}")
    logger.info(f"Device: {device}")

    # Load dataset
    dataset = DATASET_MAP[args.dataset](root=args.data_root)
    train_set, val_set, test_set = speaker_stratified_split(dataset)

    # Feature extractor
    extractor = SSLEmbeddingExtractor(model_name=args.model, freeze_ssl=True)

    cache_dir = Path(args.cache_dir) / f"{args.dataset}_{args.model.split('/')[-1]}"

    for split_name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        logger.info(f"Processing {split_name} split ({len(split_data)} samples)")
        precompute_features(
            split_data, extractor, cache_dir,
            split=split_name, batch_size=args.batch_size, device=device,
        )

    logger.info(f"Features cached to {cache_dir}")


if __name__ == "__main__":
    main()
