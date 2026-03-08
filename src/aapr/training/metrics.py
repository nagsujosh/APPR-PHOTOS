import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    roc_curve,
)


def compute_uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted Average Recall (balanced accuracy)."""
    return balanced_accuracy_score(y_true, y_pred)


def compute_wa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 score."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Equal Error Rate for speaker verification.

    y_true: binary (same speaker = 1, different = 0)
    y_scores: similarity scores
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def compute_deid_rate(
    original_speaker_acc: float, filtered_speaker_acc: float
) -> float:
    """De-identification rate: relative drop in speaker ID accuracy."""
    if original_speaker_acc == 0:
        return 0.0
    return 1.0 - filtered_speaker_acc / original_speaker_acc


def compute_mi_estimate(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Simple MI estimation using kNN-based method.

    Estimates I(Z; S) where Z are embeddings and S are discrete labels.
    Uses classification accuracy as an upper bound proxy.
    """
    from sklearn.neighbors import KNeighborsClassifier

    n = len(labels)
    if n < 10:
        return 0.0

    k = min(5, n // 2)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, labels)
    acc = knn.score(embeddings, labels)

    num_classes = len(np.unique(labels))
    chance = 1.0 / num_classes
    if acc <= chance:
        return 0.0
    return np.log2(num_classes) * (acc - chance) / (1 - chance)


class MetricTracker:
    """Accumulates predictions over batches for epoch-level metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.utility_preds = []
        self.utility_labels = []
        self.privacy_preds = {}
        self.privacy_labels = {}
        self.losses = []

    def update(
        self,
        utility_logits,
        utility_labels,
        privacy_logits: dict = None,
        privacy_labels: dict = None,
        loss: float = 0.0,
    ):
        self.utility_preds.append(utility_logits.argmax(dim=1).cpu().numpy())
        self.utility_labels.append(utility_labels.cpu().numpy())
        self.losses.append(loss)

        if privacy_logits:
            for k, v in privacy_logits.items():
                labels = privacy_labels.get(k)
                if labels is None:
                    continue
                valid = labels >= 0
                if valid.sum() == 0:
                    continue
                self.privacy_preds.setdefault(k, []).append(
                    v[valid].argmax(dim=1).cpu().numpy()
                )
                self.privacy_labels.setdefault(k, []).append(
                    labels[valid].cpu().numpy()
                )

    def compute(self) -> dict[str, float]:
        y_true = np.concatenate(self.utility_labels)
        y_pred = np.concatenate(self.utility_preds)

        metrics = {
            "utility_uar": compute_uar(y_true, y_pred),
            "utility_wa": compute_wa(y_true, y_pred),
            "utility_f1": compute_f1(y_true, y_pred),
            "loss": np.mean(self.losses),
        }

        for k in self.privacy_preds:
            p_true = np.concatenate(self.privacy_labels[k])
            p_pred = np.concatenate(self.privacy_preds[k])
            metrics[f"privacy_{k}_acc"] = compute_wa(p_true, p_pred)
            metrics[f"privacy_{k}_uar"] = compute_uar(p_true, p_pred)

        return metrics
