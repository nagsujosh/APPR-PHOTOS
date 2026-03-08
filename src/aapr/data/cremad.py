import csv
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .base_dataset import SpeechPrivacyDataset

EMOTION_MAP = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
EMOTION_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
GENDER_MAP = {"Male": 0, "Female": 1}


class CremaDDataset(SpeechPrivacyDataset):
    """Crema-D dataset: 7442 clips, 91 actors, 6 emotions.

    Filename format: {ActorID}_{Sentence}_{Emotion}_{Intensity}.wav
    Demographics from VideoDemographics.csv: ActorID, Age, Sex, Race, Ethnicity
    """

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 16000,
        max_length_sec: float = 5.0,
        file_list: list[Path] | None = None,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_length = int(max_length_sec * sample_rate)

        # Load demographics
        self.demographics = self._load_demographics()

        # Collect audio files
        if file_list is not None:
            self.files = file_list
        else:
            audio_dir = self.root / "AudioWAV"
            if not audio_dir.exists():
                # Try flat structure
                audio_dir = self.root
            self.files = sorted(audio_dir.glob("*.wav"))

        # Build speaker mapping
        all_speakers = sorted(set(self._parse_speaker_id(f) for f in self.files))
        self.speaker_to_idx = {s: i for i, s in enumerate(all_speakers)}

    def _load_demographics(self) -> dict:
        demo = {}
        demo_file = self.root / "VideoDemographics.csv"
        if not demo_file.exists():
            return demo
        with open(demo_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                actor_id = int(row["ActorID"])
                demo[actor_id] = {
                    "age": int(row["Age"]),
                    "gender": GENDER_MAP.get(row["Sex"], -1),
                    "race": row.get("Race", ""),
                }
        return demo

    @staticmethod
    def _parse_speaker_id(filepath: Path) -> int:
        return int(filepath.stem.split("_")[0])

    @staticmethod
    def _parse_emotion(filepath: Path) -> int:
        emotion_code = filepath.stem.split("_")[2]
        return EMOTION_MAP.get(emotion_code, -1)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        filepath = self.files[idx]
        waveform, sr = torchaudio.load(filepath)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, : self.max_length]
        elif waveform.shape[1] < self.max_length:
            pad = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        speaker_id = self._parse_speaker_id(filepath)
        emotion = self._parse_emotion(filepath)
        demo = self.demographics.get(speaker_id, {})

        return {
            "waveform": waveform,
            "utility_label": emotion,
            "privacy_labels": {
                "speaker_id": self.speaker_to_idx[speaker_id],
                "gender": demo.get("gender", -1),
                "age": demo.get("age", -1),
            },
            "metadata": {"filename": filepath.name, "raw_speaker_id": speaker_id},
        }

    @property
    def num_utility_classes(self) -> int:
        return 6

    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_idx)

    @property
    def utility_label_names(self) -> list[str]:
        return EMOTION_NAMES

    def get_speaker_ids(self) -> list[int]:
        """Return raw speaker IDs for stratified splitting."""
        return [self._parse_speaker_id(f) for f in self.files]
