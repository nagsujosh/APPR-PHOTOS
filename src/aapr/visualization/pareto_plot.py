import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_frontier(
    results: list[dict] | str,
    utility_key: str = "utility_uar",
    privacy_key: str = "privacy_speaker_id_acc",
    save_path: str | None = None,
):
    """Plot Pareto frontier of utility vs privacy tradeoff.

    Args:
        results: list of dicts with lambda, utility, and privacy metrics
                 or path to JSON file
        utility_key: metric key for utility (higher = better)
        privacy_key: metric key for privacy (lower = more private)
    """
    if isinstance(results, (str, Path)):
        with open(results) as f:
            results = json.load(f)

    lambdas = [r["lambda"] for r in results]
    utility = [r.get(utility_key, 0) for r in results]
    privacy = [r.get(privacy_key, 0) for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(privacy, utility, c=lambdas, cmap="viridis", s=100, zorder=3)
    plt.colorbar(scatter, label="Lambda")

    # Connect points
    sorted_idx = np.argsort(privacy)
    ax.plot(
        [privacy[i] for i in sorted_idx],
        [utility[i] for i in sorted_idx],
        "k--", alpha=0.3, zorder=2,
    )

    # Annotate
    for i, lam in enumerate(lambdas):
        ax.annotate(f"λ={lam}", (privacy[i], utility[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel(f"Privacy ({privacy_key}) ← more private")
    ax.set_ylabel(f"Utility ({utility_key}) ↑ better")
    ax.set_title("Privacy-Utility Pareto Frontier")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
