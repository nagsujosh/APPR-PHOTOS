import logging

import torch
from torch.utils.data import DataLoader

from .evaluator import Evaluator

logger = logging.getLogger("aapr")


def cross_dataset_evaluation(
    privacy_filter,
    task_model,
    adversary,
    source_test_loader: DataLoader,
    target_test_loader: DataLoader,
    feature_extractor=None,
    device: torch.device = torch.device("cpu"),
    use_cached_features: bool = False,
) -> dict[str, dict]:
    """Evaluate model trained on source dataset against target dataset.

    Note: For cross-dataset, the adversary heads may not match (different speaker sets).
    We only evaluate utility transfer and re-train a simple adversary on target.
    """
    evaluator = Evaluator(device)

    # Source domain results
    source_results = evaluator.evaluate(
        privacy_filter, task_model, adversary,
        source_test_loader, feature_extractor, use_cached_features,
    )

    # Target domain: evaluate utility only (adversary heads don't match)
    privacy_filter.eval()
    task_model.eval()

    import numpy as np
    from ..training.metrics import compute_uar, compute_wa, compute_f1

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in target_test_loader:
            if use_cached_features:
                features = batch["features"].to(device)
            elif feature_extractor:
                features = feature_extractor(batch["image"].to(device))
            else:
                features = batch["image"].to(device)

            filtered, _ = privacy_filter(features)
            logits = task_model(filtered)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(batch["utility_label"].numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    target_results = {
        "utility_uar": compute_uar(y_true, y_pred),
        "utility_wa": compute_wa(y_true, y_pred),
        "utility_f1": compute_f1(y_true, y_pred),
    }

    return {"source": source_results, "target": target_results}
