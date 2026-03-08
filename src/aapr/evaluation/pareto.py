import json
import logging
from pathlib import Path

logger = logging.getLogger("aapr")


def run_lambda_sweep(
    build_and_train_fn,
    lambda_values: list[float] = None,
    output_dir: str = "outputs/pareto",
) -> list[dict]:
    """Sweep lambda values and collect utility-privacy tradeoff results.

    Args:
        build_and_train_fn: callable(lambda_val) -> dict of metrics
        lambda_values: list of lambda values to sweep
        output_dir: where to save results
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for lam in lambda_values:
        logger.info(f"Lambda sweep: lambda={lam}")
        metrics = build_and_train_fn(lam)
        metrics["lambda"] = lam
        results.append(metrics)

        # Save incrementally
        with open(output_path / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    return results
