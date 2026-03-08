"""CombinedDataset: merges CREMA-D and MDER-MA into a shared 4-class emotion space.

Emotion mapping (4 common classes):
  0: anger   (CREMA-D: ANG, MDER-MA: angry)
  1: happy   (CREMA-D: HAP, MDER-MA: happy)
  2: neutral (CREMA-D: NEU, MDER-MA: neutral)
  3: sad     (CREMA-D: SAD, MDER-MA: sad)

CREMA-D DIS and FEA samples are excluded from combined training.
Speaker IDs are remapped to a global collision-free index across datasets.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from .cremad import CremaDDataset
from .mderma import MDERMADataset

# Maps each dataset's original label to the 4-class common space (-1 = excluded)
CREMAD_TO_COMMON = {0: 0, 1: -1, 2: -1, 3: 1, 4: 2, 5: 3}   # ANG,DIS,FEA,HAP,NEU,SAD
MDERMA_TO_COMMON = {0: 0, 1: 1, 2: 2, 3: 3}                   # already 0-3
COMMON_NAMES = ["anger", "happy", "neutral", "sad"]


class CombinedEmotionDataset(Dataset):
    """Joint CREMA-D + MDER-MA dataset with unified 4-class emotion labels.

    Handles:
    - Label remapping to common 4-class space (DIS/FEA excluded)
    - Speaker ID remapping: CREMA-D speakers 0..N-1, MDER-MA speakers N..N+M-1
    - Union privacy_labels: speaker_id, gender (age -1 for CREMA-D samples)
    - get_speaker_ids() for speaker-stratified splitting
    """

    def __init__(
        self,
        cremad_root: str | Path | None = None,
        mderma_root: str | Path | None = None,
        sample_rate: int = 16000,
        max_length_sec: float = 5.0,
    ):
        self.sample_rate = sample_rate
        self.max_length_sec = max_length_sec

        self.samples = []   # list of (dataset_name, local_idx, global_speaker_idx)
        self._cremad = None
        self._mderma = None

        speaker_offset = 0

        # Load CREMA-D
        if cremad_root is not None and Path(cremad_root).exists():
            self._cremad = CremaDDataset(cremad_root, sample_rate, max_length_sec)
            cremad_speaker_count = self._cremad.num_speakers
            for local_idx in range(len(self._cremad)):
                raw = self._cremad[local_idx]
                common_label = CREMAD_TO_COMMON.get(raw["utility_label"], -1)
                if common_label == -1:
                    continue  # skip DIS and FEA
                orig_spk = raw["privacy_labels"]["speaker_id"]  # already 0-based from CremaDDataset
                self.samples.append({
                    "dataset": "cremad",
                    "local_idx": local_idx,
                    "utility_label": common_label,
                    "global_speaker_idx": speaker_offset + orig_spk,
                })
            speaker_offset += cremad_speaker_count
        else:
            cremad_speaker_count = 0

        # Load MDER-MA
        if mderma_root is not None and Path(mderma_root).exists():
            self._mderma = MDERMADataset(mderma_root, sample_rate, max_length_sec)
            for local_idx in range(len(self._mderma)):
                raw = self._mderma[local_idx]
                common_label = MDERMA_TO_COMMON.get(raw["utility_label"], -1)
                if common_label == -1:
                    continue
                orig_spk = raw["privacy_labels"]["speaker_id"]
                self.samples.append({
                    "dataset": "mderma",
                    "local_idx": local_idx,
                    "utility_label": common_label,
                    "global_speaker_idx": speaker_offset + orig_spk,
                })
            speaker_offset += self._mderma.num_speakers

        self.total_speakers = speaker_offset

        if len(self.samples) == 0:
            raise RuntimeError(
                "CombinedEmotionDataset: no samples loaded. "
                "Provide at least one valid dataset root."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.samples[idx]
        ds_name = entry["dataset"]

        if ds_name == "cremad":
            raw = self._cremad[entry["local_idx"]]
        else:
            raw = self._mderma[entry["local_idx"]]

        result = dict(raw)
        result["utility_label"] = entry["utility_label"]
        result["privacy_labels"] = dict(raw["privacy_labels"])
        result["privacy_labels"]["speaker_id"] = entry["global_speaker_idx"]
        # Normalise: CREMA-D may not have age; MDER-MA may not have age
        result["privacy_labels"].setdefault("age", -1)
        result["metadata"]["dataset"] = ds_name
        return result

    def get_speaker_ids(self) -> list[int]:
        return [s["global_speaker_idx"] for s in self.samples]

    @property
    def num_utility_classes(self) -> int:
        return 4

    @property
    def num_speakers(self) -> int:
        return self.total_speakers

    @property
    def utility_label_names(self) -> list[str]:
        return COMMON_NAMES

    def stats(self) -> dict:
        from collections import Counter
        emotion_counts = Counter(s["utility_label"] for s in self.samples)
        dataset_counts = Counter(s["dataset"] for s in self.samples)
        return {
            "total_samples": len(self.samples),
            "total_speakers": self.total_speakers,
            "emotion_counts": {COMMON_NAMES[k]: v for k, v in sorted(emotion_counts.items())},
            "dataset_counts": dict(dataset_counts),
        }
