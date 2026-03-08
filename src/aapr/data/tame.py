import csv
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .base_dataset import SpeechPrivacyDataset


class TAMEDataset(SpeechPrivacyDataset):
    """TAME dataset: 7039 utterances, 51 speakers, pain levels 0-10.

    Expects a metadata CSV with columns: filename, pain_level, speaker_id, gender, age
    """

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 16000,
        max_length_sec: float = 5.0,
        num_pain_bins: int = 4,
        file_list: list[dict] | None = None,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_length = int(max_length_sec * sample_rate)
        self.num_pain_bins = num_pain_bins

        if file_list is not None:
            self.samples = file_list
        else:
            self.samples = self._scan_dataset()

        all_speakers = sorted(set(s["speaker_id"] for s in self.samples))
        self.speaker_to_idx = {s: i for i, s in enumerate(all_speakers)}

    def _pain_to_bin(self, pain_level: int) -> int:
        """Bin continuous pain levels into discrete classes."""
        if self.num_pain_bins == 4:
            if pain_level == 0:
                return 0  # no pain
            elif pain_level <= 3:
                return 1  # mild
            elif pain_level <= 6:
                return 2  # moderate
            else:
                return 3  # severe
        return min(pain_level, self.num_pain_bins - 1)

    def _scan_dataset(self) -> list[dict]:
        samples = []
        meta_file = self.root / "metadata.csv"

        if not meta_file.exists():
            return samples

        with open(meta_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                filepath = self.root / row["filename"]
                if filepath.exists():
                    samples.append({
                        "filepath": filepath,
                        "pain_level": int(row["pain_level"]),
                        "speaker_id": row["speaker_id"],
                        "gender": int(row.get("gender", -1)),
                        "age": int(row.get("age", -1)),
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        waveform, sr = torchaudio.load(sample["filepath"])

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, : self.max_length]
        elif waveform.shape[1] < self.max_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_length - waveform.shape[1]))

        return {
            "waveform": waveform,
            "utility_label": self._pain_to_bin(sample["pain_level"]),
            "privacy_labels": {
                "speaker_id": self.speaker_to_idx[sample["speaker_id"]],
                "gender": sample["gender"],
                "age": sample["age"],
            },
            "metadata": {
                "filename": sample["filepath"].name,
                "raw_pain_level": sample["pain_level"],
            },
        }

    @property
    def num_utility_classes(self) -> int:
        return self.num_pain_bins

    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_idx)

    @property
    def utility_label_names(self) -> list[str]:
        if self.num_pain_bins == 4:
            return ["no_pain", "mild", "moderate", "severe"]
        return [f"level_{i}" for i in range(self.num_pain_bins)]

    def get_speaker_ids(self) -> list:
        return [s["speaker_id"] for s in self.samples]
