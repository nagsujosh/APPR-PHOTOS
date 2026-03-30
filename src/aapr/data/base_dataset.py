from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class PrivacyDataset(Dataset, ABC):
    """Abstract base for privacy-preserving utility datasets.

    Each sample returns:
        image: (C, H, W) float tensor
        utility_label: int (emotion class or pain level)
        privacy_labels: dict with keys like 'speaker_id', 'gender', 'age'
        metadata: dict with any extra info (e.g. filename)
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def num_utility_classes(self) -> int: ...

    @property
    @abstractmethod
    def num_speakers(self) -> int: ...

    @property
    @abstractmethod
    def utility_label_names(self) -> list[str]: ...
