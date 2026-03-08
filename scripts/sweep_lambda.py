#!/usr/bin/env python
"""Lambda sweep for Pareto frontier analysis."""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aapr.utils.config import load_config, apply_overrides
from aapr.utils.seed import set_seed
from aapr.utils.device import get_device
from aapr.utils.logging import setup_logger
from aapr.evaluation.pareto import run_lambda_sweep
from aapr.visualization.pareto_plot import plot_pareto_frontier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--lambdas", type=float, nargs="+",
                       default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="outputs/pareto")
    args = parser.parse_args()

    logger = setup_logger("aapr")
    device = get_device("auto")

    base_cfg = load_config(args.config)
    base_cfg["training"]["num_epochs"] = args.epochs

    def train_with_lambda(lambda_val):
        """Train model with given lambda and return test metrics."""
        import copy
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["lambda_privacy"] = lambda_val
        cfg["output"]["dir"] = f"{args.output_dir}/lambda_{lambda_val}"
        cfg["output"]["checkpoint_dir"] = f"{args.output_dir}/lambda_{lambda_val}/checkpoints"

        set_seed(cfg.get("seed", 42))

        # Import and run training
        from scripts.train import build_dataset, build_feature_extractor
        from aapr.data.utils import create_dataloaders
        from aapr.models.privacy_filter import PrivacyFilter
        from aapr.models.task_model import TaskModel
        from aapr.models.adversary import MultiHeadAdversary
        from aapr.training.trainer import Trainer
        from aapr.evaluation.evaluator import Evaluator

        dataset = build_dataset(cfg)
        loaders = create_dataloaders(dataset, batch_size=cfg["dataset"]["batch_size"],
                                     seed=cfg.get("seed", 42))
        feature_extractor = build_feature_extractor(cfg)
        input_dim = feature_extractor.output_dim
        filter_cfg = cfg["model"]["filter"]

        pf = PrivacyFilter(input_dim=input_dim, hidden_dim=filter_cfg["hidden_dim"],
                          output_dim=filter_cfg["output_dim"], use_vib=filter_cfg["use_vib"])
        tm = TaskModel(input_dim=filter_cfg["output_dim"], num_classes=cfg["dataset"].get("num_utility_classes", 6))
        adv = MultiHeadAdversary(input_dim=filter_cfg["output_dim"],
                                 heads=cfg["model"]["adversary"]["heads"])

        trainer = Trainer(pf, tm, adv, feature_extractor, device,
                         lambda_privacy=lambda_val,
                         checkpoint_dir=cfg["output"]["checkpoint_dir"])
        trainer.fit(loaders["train"], loaders["val"], num_epochs=cfg["training"]["num_epochs"])

        evaluator = Evaluator(device)
        feature_extractor.to(device)
        return evaluator.evaluate(pf, tm, adv, loaders["test"], feature_extractor)

    results = run_lambda_sweep(train_with_lambda, args.lambdas, args.output_dir)

    # Plot
    plot_pareto_frontier(results, save_path=f"{args.output_dir}/pareto_frontier.png")
    logger.info(f"Pareto frontier saved to {args.output_dir}/pareto_frontier.png")


if __name__ == "__main__":
    main()
