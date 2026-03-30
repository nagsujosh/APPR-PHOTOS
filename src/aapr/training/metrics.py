import numpy as np


def _as_numpy(y) -> np.ndarray:
    return np.asarray(y)


def compute_uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted average recall over classes present in y_true."""
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    classes = np.unique(y_true)
    if classes.size == 0:
        return 0.0

    recalls = []
    for cls in classes:
        mask = y_true == cls
        denom = mask.sum()
        if denom == 0:
            recalls.append(0.0)
        else:
            recalls.append(float((y_pred[mask] == cls).sum() / denom))
    return float(np.mean(recalls))


def compute_wa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted accuracy (plain accuracy over all samples)."""
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over classes present in y_true."""
    y_true = _as_numpy(y_true)
    y_pred = _as_numpy(y_pred)
    classes = np.unique(y_true)
    if classes.size == 0:
        return 0.0

    f1_scores = []
    for cls in classes:
        tp = np.logical_and(y_pred == cls, y_true == cls).sum()
        fp = np.logical_and(y_pred == cls, y_true != cls).sum()
        fn = np.logical_and(y_pred != cls, y_true == cls).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Equal error rate for binary verification labels."""
    y_true = _as_numpy(y_true).astype(int)
    y_scores = _as_numpy(y_scores).astype(float)
    if y_true.size == 0:
        return 0.0

    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    if pos == 0 or neg == 0:
        return 0.0

    order = np.argsort(-y_scores)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / pos
    fpr = fp / neg
    fnr = 1.0 - tpr

    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_deid_rate(original_speaker_acc: float, filtered_speaker_acc: float) -> float:
    """Relative drop in speaker-ID accuracy."""
    if original_speaker_acc == 0:
        return 0.0
    return 1.0 - filtered_speaker_acc / original_speaker_acc


def compute_mi_estimate(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Lightweight MI proxy using nearest-centroid classification accuracy."""
    embeddings = _as_numpy(embeddings)
    labels = _as_numpy(labels)
    n = len(labels)
    if n < 10:
        return 0.0

    classes = np.unique(labels)
    if classes.size < 2:
        return 0.0

    centroids = []
    for cls in classes:
        centroids.append(embeddings[labels == cls].mean(axis=0))
    centroids = np.stack(centroids, axis=0)  # (C, D)

    diffs = embeddings[:, None, :] - centroids[None, :, :]
    dists = np.sum(diffs * diffs, axis=2)  # (N, C)
    preds = classes[np.argmin(dists, axis=1)]
    acc = float((preds == labels).mean())

    num_classes = len(classes)
    chance = 1.0 / num_classes
    if acc <= chance:
        return 0.0
    return float(np.log2(num_classes) * (acc - chance) / (1.0 - chance))


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
            for key, logits in privacy_logits.items():
                labels = privacy_labels.get(key)
                if labels is None:
                    continue
                valid = labels >= 0
                if valid.sum() == 0:
                    continue
                self.privacy_preds.setdefault(key, []).append(
                    logits[valid].argmax(dim=1).cpu().numpy()
                )
                self.privacy_labels.setdefault(key, []).append(labels[valid].cpu().numpy())

    def compute(self) -> dict[str, float]:
        y_true = np.concatenate(self.utility_labels)
        y_pred = np.concatenate(self.utility_preds)

        metrics = {
            "utility_uar": compute_uar(y_true, y_pred),
            "utility_wa": compute_wa(y_true, y_pred),
            "utility_f1": compute_f1(y_true, y_pred),
            "loss": float(np.mean(self.losses) if self.losses else 0.0),
        }

        for key in self.privacy_preds:
            p_true = np.concatenate(self.privacy_labels[key])
            p_pred = np.concatenate(self.privacy_preds[key])
            metrics[f"privacy_{key}_acc"] = compute_wa(p_true, p_pred)
            metrics[f"privacy_{key}_uar"] = compute_uar(p_true, p_pred)

        return metrics
