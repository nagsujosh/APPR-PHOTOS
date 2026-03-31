#!/usr/bin/env python
"""Lambda sweep for Pareto frontier analysis."""
import argparse
import copy
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
from aapr.evaluation.pareto import run_lambda_sweep
from aapr.models.adversary import MultiHeadAdversary
from aapr.models.privacy_filter import PrivacyFilter
from aapr.models.task_model import TaskModel
from aapr.training.trainer import Trainer
from aapr.utils.config import load_config
from aapr.utils.device import get_device
from aapr.utils.logging import setup_logger
from aapr.utils.seed import set_seed
from aapr.visualization.pareto_plot import plot_pareto_frontier
from scripts.train import build_dataset, build_feature_extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="outputs/pareto")
    args = parser.parse_args()

    logger = setup_logger("aapr")
    device = get_device("auto")
    base_cfg = load_config(args.config)
    base_cfg["training"]["num_epochs"] = args.epochs

    def train_with_lambda(lambda_val: float):
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["lambda_privacy"] = lambda_val
        cfg["output"]["dir"] = f"{args.output_dir}/lambda_{lambda_val}"
        cfg["output"]["checkpoint_dir"] = f"{args.output_dir}/lambda_{lambda_val}/checkpoints"

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
            dropout=filter_cfg.get("dropout", 0.1),
        )
        task_model = TaskModel(
            input_dim=filter_cfg.get("output_dim", 128),
            hidden_dim=cfg["model"]["task"].get("hidden_dim", 128),
            num_classes=dataset.num_utility_classes,
            dropout=cfg["model"]["task"].get("dropout", 0.2),
        )

        heads = dict(cfg["model"]["adversary"]["heads"])
        heads["speaker_id"] = dataset.num_speakers
        adversary = MultiHeadAdversary(
            input_dim=filter_cfg.get("output_dim", 128),
            trunk_dim=cfg["model"]["adversary"].get("trunk_dim", 128),
            heads=heads,
            dropout=cfg["model"]["adversary"].get("dropout", 0.3),
        )

        trainer = Trainer(
            privacy_filter=privacy_filter,
            task_model=task_model,
            adversary=adversary,
            feature_extractor=feature_extractor,
            device=device,
            lambda_privacy=lambda_val,
            lambda_warmup_epochs=cfg["training"].get("lambda_warmup_epochs", 10),
            adversary_refresh_interval=cfg["training"].get("adversary_refresh_interval", 20),
            adversary_retrain_epochs=cfg["training"].get("adversary_retrain_epochs", 5),
            lr_main=cfg["training"].get("lr_main", 1e-3),
            lr_adversary=cfg["training"].get("lr_adversary", 5e-4),
            grad_clip=cfg["training"].get("grad_clip", 1.0),
            num_epochs=cfg["training"]["num_epochs"],
            checkpoint_dir=cfg["output"]["checkpoint_dir"],
        )
        trainer.fit(loaders["train"], loaders["val"], num_epochs=cfg["training"]["num_epochs"])

        feature_extractor.to(device)
        evaluator = Evaluator(device)
        return evaluator.evaluate(
            privacy_filter, task_model, adversary, loaders["test"], feature_extractor
        )

    results = run_lambda_sweep(train_with_lambda, args.lambdas, args.output_dir)
    plot_pareto_frontier(results, save_path=f"{args.output_dir}/pareto_frontier.png")
    logger.info(f"Pareto frontier saved to {args.output_dir}/pareto_frontier.png")


if __name__ == "__main__":
    main()
