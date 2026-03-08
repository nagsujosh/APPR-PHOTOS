#!/usr/bin/env python
"""Standalone evaluation script."""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from aapr.utils.device import get_device
from aapr.utils.seed import set_seed
from aapr.utils.logging import setup_logger
from aapr.data.cremad import CremaDDataset
from aapr.data.mderma import MDERMADataset
from aapr.data.tame import TAMEDataset
from aapr.data.utils import create_dataloaders
from aapr.features.mel_spectrogram import MelSpectrogramExtractor
from aapr.features.ssl_embeddings import SSLEmbeddingExtractor
from aapr.models.privacy_filter import PrivacyFilter
from aapr.models.task_model import TaskModel
from aapr.models.adversary import MultiHeadAdversary
from aapr.evaluation.evaluator import Evaluator


DATASET_MAP = {
    "cremad": CremaDDataset,
    "mderma": MDERMADataset,
    "tame": TAMEDataset,
}


def build_dataset(cfg):
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    cls = DATASET_MAP[name]
    kwargs = {"root": ds_cfg["root"], "sample_rate": ds_cfg.get("sample_rate", 16000)}
    if "max_length_sec" in ds_cfg:
        kwargs["max_length_sec"] = ds_cfg["max_length_sec"]
    if name == "tame" and "num_pain_bins" in ds_cfg:
        kwargs["num_pain_bins"] = ds_cfg["num_pain_bins"]
    return cls(**kwargs)


def build_feature_extractor(cfg):
    feat_cfg = cfg["feature"]
    if feat_cfg["type"] == "melspec":
        return MelSpectrogramExtractor(
            sample_rate=cfg["dataset"].get("sample_rate", 16000),
            n_fft=feat_cfg.get("n_fft", 2048),
            hop_length=feat_cfg.get("hop_length", 512),
            n_mels=feat_cfg.get("n_mels", 128),
        )
    elif feat_cfg["type"] == "hubert":
        return SSLEmbeddingExtractor(
            model_name=feat_cfg.get("hubert_model", "facebook/hubert-base-ls960"),
            freeze_ssl=feat_cfg.get("freeze_ssl", True),
        )
    else:
        raise ValueError(f"Unknown feature type: {feat_cfg['type']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config JSON (from training output)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    logger = setup_logger("aapr")

    checkpoint_dir = Path(args.checkpoint).parent
    config_path = args.config if args.config else checkpoint_dir.parent / "config.json"

    with open(config_path) as f:
        cfg = json.load(f)

    set_seed(cfg.get("seed", 42))

    dataset = build_dataset(cfg)
    loaders = create_dataloaders(
        dataset,
        batch_size=cfg["dataset"].get("batch_size", 32),
        seed=cfg.get("seed", 42),
    )

    filter_cfg = cfg["model"]["filter"]
    feature_extractor = build_feature_extractor(cfg)
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
        num_classes=cfg["dataset"].get("num_utility_classes", 6),
    )
    adversary = MultiHeadAdversary(
        input_dim=filter_cfg.get("output_dim", 128),
        trunk_dim=cfg["model"]["adversary"].get("trunk_dim", 128),
        heads=cfg["model"]["adversary"].get("heads", {"gender": 2, "speaker_id": 91}),
    )

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    privacy_filter.load_state_dict(state["privacy_filter"])
    task_model.load_state_dict(state["task_model"])
    adversary.load_state_dict(state["adversary"])

    privacy_filter.to(device)
    task_model.to(device)
    adversary.to(device)
    feature_extractor.to(device)

    evaluator = Evaluator(device)
    results = evaluator.evaluate(
        privacy_filter, task_model, adversary,
        loaders["test"], feature_extractor,
    )

    logger.info("Test Results:")
    for k, v in sorted(results.items()):
        logger.info(f"  {k}: {v:.4f}")

    output_path = checkpoint_dir.parent / "test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
