from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def speaker_stratified_split(
    dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset by speakers ensuring no speaker leakage across splits."""
    rng = np.random.RandomState(seed)
    speaker_ids = dataset.get_speaker_ids()

    # Group indices by speaker
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


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate batch of samples into tensors."""
    waveforms = torch.stack([b["waveform"] for b in batch])
    utility_labels = torch.tensor([b["utility_label"] for b in batch], dtype=torch.long)

    # Collect all privacy label keys
    privacy_keys = batch[0]["privacy_labels"].keys()
    privacy_labels = {
        k: torch.tensor([b["privacy_labels"][k] for b in batch], dtype=torch.long)
        for k in privacy_keys
    }

    return {
        "waveform": waveforms,
        "utility_label": utility_labels,
        "privacy_labels": privacy_labels,
    }


def create_dataloaders(
    dataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create train/val/test dataloaders with speaker-stratified splits."""
    train_set, val_set, test_set = speaker_stratified_split(
        dataset, train_ratio, val_ratio, seed
    )

    return {
        "train": DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, drop_last=True,
        ),
        "val": DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers,
        ),
        "test": DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers,
        ),
    }
