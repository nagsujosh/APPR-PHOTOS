from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CachedFeatureDataset(Dataset):
    """Wraps a dataset with precomputed features loaded from disk."""

    def __init__(self, cache_dir: str | Path, split: str = "train"):
        self.cache_dir = Path(cache_dir) / split
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No cached features found in {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx], weights_only=False)


def precompute_features(
    dataset,
    feature_extractor: torch.nn.Module,
    cache_dir: str | Path,
    split: str = "train",
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
):
    """Extract and cache features for an entire dataset split."""
    cache_path = Path(cache_dir) / split
    cache_path.mkdir(parents=True, exist_ok=True)

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc=f"Caching {split}")):
            waveform = batch["waveform"].to(device)
            features = feature_extractor(waveform).cpu()

            sample = {
                "features": features.squeeze(0),  # (D, T')
                "utility_label": batch["utility_label"].squeeze(0),
                "privacy_labels": {k: v.squeeze(0) for k, v in batch["privacy_labels"].items()},
            }
            torch.save(sample, cache_path / f"{idx:06d}.pt")
