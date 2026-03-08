from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class SpeechPrivacyDataset(Dataset, ABC):
    """Abstract base for all speech privacy datasets.

    Each sample returns:
        waveform: (1, T) float tensor
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
