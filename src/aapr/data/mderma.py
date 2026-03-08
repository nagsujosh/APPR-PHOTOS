import csv
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .base_dataset import SpeechPrivacyDataset

EMOTION_MAP = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
EMOTION_NAMES = ["angry", "happy", "neutral", "sad"]


class MDERMADataset(SpeechPrivacyDataset):
    """MDER-MA dataset: 5288 clips, 4 emotions.

    Expects a metadata CSV with columns: filename, emotion, speaker_id, gender
    """

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 16000,
        max_length_sec: float = 5.0,
        file_list: list[dict] | None = None,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_length = int(max_length_sec * sample_rate)

        if file_list is not None:
            self.samples = file_list
        else:
            self.samples = self._scan_dataset()

        all_speakers = sorted(set(s["speaker_id"] for s in self.samples))
        self.speaker_to_idx = {s: i for i, s in enumerate(all_speakers)}

    def _scan_dataset(self) -> list[dict]:
        samples = []
        meta_file = self.root / "metadata.csv"

        if meta_file.exists():
            with open(meta_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filepath = self.root / row["filename"]
                    if filepath.exists():
                        samples.append({
                            "filepath": filepath,
                            "emotion": EMOTION_MAP.get(row["emotion"].lower(), -1),
                            "speaker_id": row["speaker_id"],
                            "gender": int(row.get("gender", -1)),
                        })
        else:
            # Fallback: scan directories structured as emotion/speaker/file.wav
            for emotion_dir in sorted(self.root.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                emotion_name = emotion_dir.name.lower()
                if emotion_name not in EMOTION_MAP:
                    continue
                for wav_file in sorted(emotion_dir.rglob("*.wav")):
                    speaker_id = wav_file.parent.name if wav_file.parent != emotion_dir else "unknown"
                    samples.append({
                        "filepath": wav_file,
                        "emotion": EMOTION_MAP[emotion_name],
                        "speaker_id": speaker_id,
                        "gender": -1,
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
            "utility_label": sample["emotion"],
            "privacy_labels": {
                "speaker_id": self.speaker_to_idx[sample["speaker_id"]],
                "gender": sample["gender"],
            },
            "metadata": {"filename": sample["filepath"].name},
        }

    @property
    def num_utility_classes(self) -> int:
        return 4

    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_idx)

    @property
    def utility_label_names(self) -> list[str]:
        return EMOTION_NAMES

    def get_speaker_ids(self) -> list:
        return [s["speaker_id"] for s in self.samples]
