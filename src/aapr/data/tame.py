"""TAME Pain dataset loader.

Supports multiple PhysioNet-style layouts:

  Layout A — metadata CSV (preferred):
    data/raw/tame/
      metadata.csv   (columns: filename, pain_level, speaker_id, gender, age)
      audio/  *.wav

  Layout B — PhysioNet subject folders:
    data/raw/tame/
      RECORDS             (one relative path per line)
      ANNOTATORS          (optional)
      subject_001/
        session_01/  *.wav

  Layout C — subject folder, annotation sidecar:
    data/raw/tame/
      subject_001/
        recording_001.wav
        recording_001.txt  (single line: pain_level)

Pain levels 0-10 are binned into 4 discrete classes by default:
  0: no pain (0), 1: mild (1-3), 2: moderate (4-6), 3: severe (7-10)

Reference: Tu-Quyen Dao et al., PhysioNet, January 2025.
           doi: 10.13026/20e2-1g10
Download:  https://doi.org/10.13026/20e2-1g10  (PhysioNet credentialed access)
"""

import csv
import re
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .base_dataset import SpeechPrivacyDataset

GENDER_MAP = {"m": 0, "male": 0, "f": 1, "female": 1, "0": 0, "1": 1}


def _parse_gender(value: str) -> int:
    if value is None:
        return -1
    v = str(value).strip().lower()
    return GENDER_MAP.get(v, int(v) if v.isdigit() else -1)


class TAMEDataset(SpeechPrivacyDataset):
    """TAME Pain: speech-based pain assessment, 7039 utterances, 51 speakers.

    Reference: Dao et al., PhysioNet 2025. doi: 10.13026/20e2-1g10
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

        if len(self.samples) == 0:
            raise RuntimeError(
                f"TAMEDataset found no samples in '{self.root}'. "
                "Ensure PhysioNet download is complete and metadata.csv exists. "
                "See docs/procedures.md for preparation steps."
            )

        all_speakers = sorted(set(s["speaker_id"] for s in self.samples))
        self.speaker_to_idx = {s: i for i, s in enumerate(all_speakers)}

    # ------------------------------------------------------------------ #
    # Scanning                                                             #
    # ------------------------------------------------------------------ #
    def _scan_dataset(self) -> list[dict]:
        # Priority 1: metadata.csv
        meta_file = self.root / "metadata.csv"
        if meta_file.exists():
            return self._scan_from_csv(meta_file)

        # Priority 2: PhysioNet RECORDS file
        records_file = self.root / "RECORDS"
        if records_file.exists():
            return self._scan_from_records(records_file)

        # Priority 3: subject folder scan with sidecar annotations
        return self._scan_subject_folders()

    def _scan_from_csv(self, meta_file: Path) -> list[dict]:
        samples = []
        with open(meta_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support multiple filename column names
                fname = row.get("filename") or row.get("file") or row.get("path", "")
                filepath = self.root / fname
                if not filepath.exists():
                    filepath = self._find_audio(fname)
                if filepath is None or not filepath.exists():
                    continue
                try:
                    pain = int(row.get("pain_level", row.get("pain", 0)))
                except ValueError:
                    pain = 0
                samples.append({
                    "filepath": filepath,
                    "pain_level": pain,
                    "speaker_id": row.get("speaker_id", row.get("subject_id", "unknown")),
                    "gender": _parse_gender(row.get("gender", row.get("sex", ""))),
                    "age": int(row.get("age", -1)) if row.get("age", "").isdigit() else -1,
                })
        return samples

    def _scan_from_records(self, records_file: Path) -> list[dict]:
        """PhysioNet-style RECORDS file: one relative path per line."""
        samples = []
        lines = records_file.read_text().strip().splitlines()

        # Try to load a demographics file if present
        demo = self._load_physionet_demographics()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            wav = self.root / (line if line.endswith(".wav") else line + ".wav")
            if not wav.exists():
                continue
            spk = self._extract_subject_id(wav)
            pain = self._read_sidecar_pain(wav)
            d = demo.get(spk, {})
            samples.append({
                "filepath": wav,
                "pain_level": pain,
                "speaker_id": spk,
                "gender": d.get("gender", -1),
                "age": d.get("age", -1),
            })
        return samples

    def _scan_subject_folders(self) -> list[dict]:
        """Layout: root/subject_*/  with optional pain sidecar .txt files."""
        samples = []
        demo = self._load_physionet_demographics()

        subject_dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and re.match(r"(subject|sub|spk|s)[-_]?\d+", d.name, re.IGNORECASE)
        )
        if not subject_dirs:
            # Flat structure: just scan all wav files
            subject_dirs = [self.root]

        for subj_dir in subject_dirs:
            spk = subj_dir.name
            d = demo.get(spk, {})
            for wav in sorted(subj_dir.rglob("*.wav")):
                pain = self._read_sidecar_pain(wav)
                samples.append({
                    "filepath": wav,
                    "pain_level": pain,
                    "speaker_id": spk,
                    "gender": d.get("gender", -1),
                    "age": d.get("age", -1),
                })
        return samples

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _load_physionet_demographics(self) -> dict[str, dict]:
        """Try to load demographics from common PhysioNet filenames."""
        demo = {}
        for fname in ("PATIENTS", "participants.tsv", "demographics.csv",
                      "subjects.csv", "subject_info.csv"):
            p = self.root / fname
            if not p.exists():
                continue
            sep = "\t" if fname.endswith(".tsv") else ","
            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=sep)
                for row in reader:
                    sid = row.get("subject_id") or row.get("id") or row.get("SUBJECT_ID", "")
                    demo[sid] = {
                        "gender": _parse_gender(row.get("sex", row.get("gender", ""))),
                        "age": int(row.get("age", -1)) if str(row.get("age", "")).isdigit() else -1,
                    }
            break
        return demo

    def _extract_subject_id(self, wav: Path) -> str:
        """Infer subject ID from path: prefers parent folder, then filename prefix."""
        parent = wav.parent.name
        if re.match(r"(subject|sub|spk|s)[-_]?\d+", parent, re.IGNORECASE):
            return parent
        # Fall back to numeric prefix of filename
        m = re.match(r"([a-zA-Z]*\d+)", wav.stem)
        return m.group(1) if m else wav.parent.name

    def _read_sidecar_pain(self, wav: Path) -> int:
        """Read pain annotation from a sidecar .txt with the same stem."""
        txt = wav.with_suffix(".txt")
        if txt.exists():
            try:
                content = txt.read_text().strip().split()[0]
                return int(float(content))
            except (ValueError, IndexError):
                pass
        # Try _annotation.txt suffix
        ann = wav.parent / (wav.stem + "_annotation.txt")
        if ann.exists():
            try:
                return int(float(ann.read_text().strip().split()[0]))
            except (ValueError, IndexError):
                pass
        return 0   # default: no pain

    def _find_audio(self, filename: str) -> Path | None:
        matches = list(self.root.rglob(filename))
        return matches[0] if matches else None

    def _pain_to_bin(self, pain_level: int) -> int:
        if self.num_pain_bins == 4:
            if pain_level == 0:
                return 0
            elif pain_level <= 3:
                return 1
            elif pain_level <= 6:
                return 2
            else:
                return 3
        return min(pain_level, self.num_pain_bins - 1)

    # ------------------------------------------------------------------ #
    # Dataset interface                                                    #
    # ------------------------------------------------------------------ #
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
