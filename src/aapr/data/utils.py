from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


def speaker_stratified_split(
    dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset by speakers ensuring no speaker leakage across splits."""
    rng = np.random.RandomState(seed)
    speaker_ids = dataset.get_speaker_ids()

    speaker_to_indices = defaultdict(list)
    for idx, spk in enumerate(speaker_ids):
        speaker_to_indices[spk].append(idx)

    speakers = sorted(speaker_to_indices.keys())
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train : n_train + n_val]
    test_speakers = speakers[n_train + n_val :]

    train_indices = [i for s in train_speakers for i in speaker_to_indices[s]]
    val_indices = [i for s in val_speakers for i in speaker_to_indices[s]]
    test_indices = [i for s in test_speakers for i in speaker_to_indices[s]]

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def make_class_balanced_sampler(subset: Subset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that up-samples minority emotion classes.

    Each sample gets weight = 1 / count(its_class), so every class contributes
    equally in expectation regardless of dataset-level imbalance.
    """
    labels = [subset.dataset[i]["utility_label"] for i in subset.indices]
    label_counts = defaultdict(int)
    for lbl in labels:
        label_counts[lbl] += 1

    weights = [1.0 / label_counts[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate batch of samples into tensors."""
    def _to_long_tensor(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.stack([v.long() for v in values])
        return torch.tensor(values, dtype=torch.long)

    payload = {}
    if "image" in batch[0]:
        payload["image"] = torch.stack([b["image"] for b in batch])
    if "features" in batch[0]:
        payload["features"] = torch.stack([b["features"] for b in batch])

    utility_labels = _to_long_tensor([b["utility_label"] for b in batch])

    privacy_keys = batch[0]["privacy_labels"].keys()
    privacy_labels = {
        k: _to_long_tensor([b["privacy_labels"][k] for b in batch])
        for k in privacy_keys
    }

    payload["utility_label"] = utility_labels
    payload["privacy_labels"] = privacy_labels
    return payload


def create_dataloaders(
    dataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
    use_weighted_sampler: bool = False,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Create train/val/test dataloaders with speaker-stratified splits.

    Args:
        use_weighted_sampler: if True, the training loader uses a class-balanced
            WeightedRandomSampler instead of uniform shuffle to mitigate
            class-imbalance in utility labels.
    """
    train_set, val_set, test_set = speaker_stratified_split(
        dataset, train_ratio, val_ratio, seed
    )

    if use_weighted_sampler:
        sampler = make_class_balanced_sampler(train_set)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, sampler=sampler,
            collate_fn=collate_fn, num_workers=num_workers, drop_last=True,
            pin_memory=pin_memory, persistent_workers=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, drop_last=True,
            pin_memory=pin_memory, persistent_workers=num_workers > 0,
        )

    return {
        "train": train_loader,
        "val": DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=num_workers > 0,
        ),
    }
