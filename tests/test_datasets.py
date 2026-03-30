"""Test dataset utilities (no real data required)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import Dataset

from aapr.data.utils import speaker_stratified_split, collate_fn


class MockDataset(Dataset):
    """Mock dataset for testing splits and collation."""

    def __init__(self, n=100, num_speakers=10):
        self.n = n
        self.speaker_ids = [i % num_speakers for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 224, 224),
            "utility_label": idx % 6,
            "privacy_labels": {
                "speaker_id": self.speaker_ids[idx],
                "gender": idx % 2,
            },
            "metadata": {"filename": f"test_{idx}.jpg"},
        }

    def get_speaker_ids(self):
        return self.speaker_ids


class TestSpeakerStratifiedSplit:
    def test_no_speaker_leakage(self):
        ds = MockDataset(n=200, num_speakers=20)
        train, val, test = speaker_stratified_split(ds)

        train_speakers = set(ds.speaker_ids[i] for i in train.indices)
        val_speakers = set(ds.speaker_ids[i] for i in val.indices)
        test_speakers = set(ds.speaker_ids[i] for i in test.indices)

        assert len(train_speakers & val_speakers) == 0
        assert len(train_speakers & test_speakers) == 0
        assert len(val_speakers & test_speakers) == 0

    def test_split_covers_all(self):
        ds = MockDataset(n=100, num_speakers=10)
        train, val, test = speaker_stratified_split(ds)
        total = len(train) + len(val) + len(test)
        assert total == len(ds)

    def test_approximate_ratios(self):
        ds = MockDataset(n=1000, num_speakers=100)
        train, val, test = speaker_stratified_split(ds, 0.7, 0.15)
        # Ratios won't be exact due to speaker grouping, but should be close
        assert len(train) > len(val)
        assert len(train) > len(test)


class TestCollate:
    def test_collate_batch(self):
        ds = MockDataset()
        batch = [ds[i] for i in range(4)]
        collated = collate_fn(batch)

        assert collated["image"].shape == (4, 3, 224, 224)
        assert collated["utility_label"].shape == (4,)
        assert collated["privacy_labels"]["speaker_id"].shape == (4,)
        assert collated["privacy_labels"]["gender"].shape == (4,)
