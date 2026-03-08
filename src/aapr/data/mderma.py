"""MDER-MA dataset loader.

Supports multiple file-system layouts produced by the Mendeley Data release:

  Layout A — emotion folders (most common):
    data/raw/mderma/
      angry/    speaker_01_001.wav  ...
      happy/    ...
      neutral/  ...
      sad/      ...

  Layout B — speaker then emotion sub-folders:
    data/raw/mderma/
      speaker_01/
        angry/  001.wav ...

  Layout C — flat with metadata CSV:
    data/raw/mderma/
      metadata.csv   (columns: filename, emotion, speaker_id, gender)
      *.wav

If metadata.csv is present it takes priority over directory scanning.
The gender column can be 'M'/'F', 'male'/'female', '0'/'1', or an integer.
"""

import csv
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .base_dataset import SpeechPrivacyDataset

EMOTION_MAP = {"angry": 0, "anger": 0, "happy": 1, "happiness": 1,
               "neutral": 2, "sad": 3, "sadness": 3}
EMOTION_NAMES = ["angry", "happy", "neutral", "sad"]

GENDER_MAP = {"m": 0, "male": 0, "f": 1, "female": 1, "0": 0, "1": 1}


def _parse_gender(value: str) -> int:
    if value is None:
        return -1
    v = str(value).strip().lower()
    return GENDER_MAP.get(v, int(v) if v.isdigit() else -1)


class MDERMADataset(SpeechPrivacyDataset):
    """MDER-MA: Moroccan Arabic emotional speech, 5288 clips, 4 emotions.

    Reference: Ouali & El Garouani, Data in Brief 62:112005, 2025.
    Download: https://www.sciencedirect.com/science/article/pii/S2352340925007292
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

        if len(self.samples) == 0:
            raise RuntimeError(
                f"MDERMADataset found no samples in '{self.root}'. "
                "Check that the data is extracted correctly. "
                "See docs/procedures.md for the expected directory structure."
            )

        all_speakers = sorted(set(s["speaker_id"] for s in self.samples))
        self.speaker_to_idx = {s: i for i, s in enumerate(all_speakers)}

    def _scan_dataset(self) -> list[dict]:
        # Priority 1: metadata CSV
        meta_file = self.root / "metadata.csv"
        if meta_file.exists():
            return self._scan_from_csv(meta_file)

        # Priority 2: emotion-folder layout (Layout A)
        samples = self._scan_emotion_folders()
        if samples:
            return samples

        # Priority 3: speaker → emotion layout (Layout B)
        return self._scan_speaker_folders()

    def _scan_from_csv(self, meta_file: Path) -> list[dict]:
        samples = []
        with open(meta_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filepath = self.root / row["filename"]
                if not filepath.exists():
                    filepath = self._find_audio(row["filename"])
                if filepath is None or not filepath.exists():
                    continue
                emotion = EMOTION_MAP.get(row.get("emotion", "").strip().lower(), -1)
                samples.append({
                    "filepath": filepath,
                    "emotion": emotion,
                    "speaker_id": row.get("speaker_id", filepath.parent.name),
                    "gender": _parse_gender(row.get("gender", "")),
                })
        return samples

    def _scan_emotion_folders(self) -> list[dict]:
        """Layout A: root/emotion_name/speaker?_*.wav"""
        samples = []
        for emotion_dir in sorted(self.root.iterdir()):
            if not emotion_dir.is_dir():
                continue
            ename = emotion_dir.name.strip().lower()
            if ename not in EMOTION_MAP:
                continue
            eid = EMOTION_MAP[ename]
            for wav in sorted(emotion_dir.rglob("*.wav")):
                # Try to infer speaker from filename prefix or parent folder
                spk = self._infer_speaker(wav, emotion_dir)
                samples.append({
                    "filepath": wav,
                    "emotion": eid,
                    "speaker_id": spk,
                    "gender": -1,
                })
        return samples

    def _scan_speaker_folders(self) -> list[dict]:
        """Layout B: root/speaker_id/emotion_name/*.wav"""
        samples = []
        for spk_dir in sorted(self.root.iterdir()):
            if not spk_dir.is_dir():
                continue
            spk_id = spk_dir.name
            for emotion_dir in sorted(spk_dir.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                ename = emotion_dir.name.strip().lower()
                if ename not in EMOTION_MAP:
                    continue
                eid = EMOTION_MAP[ename]
                for wav in sorted(emotion_dir.glob("*.wav")):
                    samples.append({
                        "filepath": wav,
                        "emotion": eid,
                        "speaker_id": spk_id,
                        "gender": -1,
                    })
        return samples

    def _infer_speaker(self, wav: Path, emotion_dir: Path) -> str:
        """Best-effort speaker ID from filename or subfolder."""
        if wav.parent != emotion_dir:
            return wav.parent.name   # subfolder = speaker
        # Try common naming patterns: spk01_001.wav, S01_F_sad_01.wav
        stem = wav.stem
        parts = stem.replace("-", "_").split("_")
        if parts:
            return parts[0]
        return "unknown"

    def _find_audio(self, filename: str) -> Path | None:
        """Search recursively for a filename in root."""
        matches = list(self.root.rglob(filename))
        return matches[0] if matches else None

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
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_length - waveform.shape[1])
            )

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
