import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..training.metrics import (
    compute_uar, compute_wa, compute_f1, compute_mi_estimate, compute_deid_rate,
)

logger = logging.getLogger("aapr")


class Evaluator:
    """Full test pipeline with all utility and privacy metrics."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        privacy_filter,
        task_model,
        adversary,
        loader: DataLoader,
        feature_extractor=None,
        use_cached_features: bool = False,
        baseline_speaker_acc: float | None = None,
    ) -> dict[str, float]:
        privacy_filter.eval()
        task_model.eval()
        adversary.eval()

        all_utility_preds, all_utility_labels = [], []
        all_privacy_preds = {}
        all_privacy_labels = {}
        all_filtered_embeddings = []
        all_speaker_labels = []

        for batch in loader:
            if use_cached_features:
                features = batch["features"].to(self.device)
            elif feature_extractor:
                features = feature_extractor(batch["image"].to(self.device))
            else:
                features = batch["image"].to(self.device)

            utility_labels = batch["utility_label"]
            privacy_labels = batch["privacy_labels"]

            filtered, _ = privacy_filter(features)
            utility_logits = task_model(filtered)
            privacy_logits = adversary(filtered)

            all_utility_preds.append(utility_logits.argmax(1).cpu().numpy())
            all_utility_labels.append(utility_labels.numpy())

            # Collect filtered embeddings for MI estimation
            pooled = filtered.mean(dim=2).cpu().numpy()
            all_filtered_embeddings.append(pooled)

            for k, logits in privacy_logits.items():
                labels = privacy_labels.get(k)
                if labels is None:
                    continue
                valid = labels >= 0
                if valid.sum() == 0:
                    continue
                all_privacy_preds.setdefault(k, []).append(logits[valid].argmax(1).cpu().numpy())
                all_privacy_labels.setdefault(k, []).append(labels[valid].numpy())

            if "speaker_id" in privacy_labels:
                valid = privacy_labels["speaker_id"] >= 0
                all_speaker_labels.append(privacy_labels["speaker_id"][valid].numpy())

        # Compute utility metrics
        y_true = np.concatenate(all_utility_labels)
        y_pred = np.concatenate(all_utility_preds)
        results = {
            "utility_uar": compute_uar(y_true, y_pred),
            "utility_wa": compute_wa(y_true, y_pred),
            "utility_f1": compute_f1(y_true, y_pred),
        }

        # Compute privacy metrics
        for k in all_privacy_preds:
            p_true = np.concatenate(all_privacy_labels[k])
            p_pred = np.concatenate(all_privacy_preds[k])
            results[f"privacy_{k}_acc"] = compute_wa(p_true, p_pred)
            results[f"privacy_{k}_uar"] = compute_uar(p_true, p_pred)

        # MI estimation
        embeddings = np.concatenate(all_filtered_embeddings)
        if all_speaker_labels:
            speaker_labels = np.concatenate(all_speaker_labels)
            if len(speaker_labels) == len(embeddings):
                results["mi_speaker"] = compute_mi_estimate(embeddings, speaker_labels)

        # De-identification rate
        if baseline_speaker_acc is not None and "privacy_speaker_id_acc" in results:
            results["deid_rate"] = compute_deid_rate(
                baseline_speaker_acc, results["privacy_speaker_id_acc"]
            )

        return results
