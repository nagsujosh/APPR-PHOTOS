import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base_dataset import PrivacyDataset

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
GENDER_MAP = {"m": 0, "male": 0, "f": 1, "female": 1, "0": 0, "1": 1}


def _parse_gender(value: Any) -> int:
    if value is None:
        return -1
    text = str(value).strip().lower()
    if not text:
        return -1
    if text in GENDER_MAP:
        return GENDER_MAP[text]
    if text.isdigit():
        return int(text)
    return -1


def _parse_age(value: Any) -> int:
    if value is None:
        return -1
    text = str(value).strip()
    return int(text) if text.isdigit() else -1


class PhotoPrivacyDataset(PrivacyDataset):
    """Generic image dataset for privacy-preserving representation learning.

    Supported layouts:
    - Folder layout: root/<class_name>/<image>
    - Folder + speaker layout: root/<class_name>/<speaker_id>/<image>
    - Metadata layout: root/metadata.csv with filename, utility label, speaker_id...
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        metadata_filename: str = "metadata.csv",
        image_extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

        metadata_path = self.root / metadata_filename
        if metadata_path.exists():
            samples = self._scan_from_metadata(metadata_path)
        else:
            samples = self._scan_from_folders()

        if not samples:
            raise RuntimeError(
                f"PhotoPrivacyDataset found no images in '{self.root}'. "
                "Expected class folders or metadata.csv entries."
            )

        self._build_label_maps(samples)
        self.samples = [self._encode_sample(sample) for sample in samples]

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in self.image_extensions

    def _scan_from_metadata(self, metadata_path: Path) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        with open(metadata_path, encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rel = (
                    row.get("filename")
                    or row.get("filepath")
                    or row.get("path")
                    or row.get("image")
                    or ""
                ).strip()
                if not rel:
                    continue

                image_path = self.root / rel
                if not image_path.exists() or not self._is_image(image_path):
                    continue

                utility_value = (
                    row.get("utility_label")
                    or row.get("label")
                    or row.get("class")
                    or row.get("emotion")
                    or row.get("task")
                    or ""
                )
                if str(utility_value).strip() == "":
                    continue

                samples.append(
                    {
                        "path": image_path,
                        "utility_raw": utility_value,
                        "speaker_raw": (
                            row.get("speaker_id")
                            or row.get("subject_id")
                            or row.get("person_id")
                            or row.get("identity")
                            or image_path.parent.name
                        ),
                        "gender": _parse_gender(row.get("gender")),
                        "age": _parse_age(row.get("age")),
                    }
                )
        return samples

    def _scan_from_folders(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        class_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        for class_dir in class_dirs:
            class_name = class_dir.name
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or not self._is_image(path):
                    continue
                # speaker id is optional: class/<speaker>/<image>
                if path.parent == class_dir:
                    speaker = path.stem.split("_")[0]
                else:
                    speaker = path.parent.name
                samples.append(
                    {
                        "path": path,
                        "utility_raw": class_name,
                        "speaker_raw": speaker,
                        "gender": -1,
                        "age": -1,
                    }
                )
        return samples

    def _build_label_maps(self, samples: list[dict[str, Any]]) -> None:
        utility_values = [sample["utility_raw"] for sample in samples]
        if all(str(value).strip().isdigit() for value in utility_values):
            utility_ids = sorted({int(str(value).strip()) for value in utility_values})
            self.utility_to_idx = {value: idx for idx, value in enumerate(utility_ids)}
            self.utility_names = [str(value) for value in utility_ids]
        else:
            utility_ids = sorted({str(value).strip() for value in utility_values})
            self.utility_to_idx = {value: idx for idx, value in enumerate(utility_ids)}
            self.utility_names = utility_ids

        speaker_values = sorted({str(sample["speaker_raw"]).strip() for sample in samples})
        self.speaker_to_idx = {speaker: idx for idx, speaker in enumerate(speaker_values)}

    def _encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        utility_value = sample["utility_raw"]
        if isinstance(next(iter(self.utility_to_idx.keys())), int):
            utility_key = int(str(utility_value).strip())
        else:
            utility_key = str(utility_value).strip()
        speaker_key = str(sample["speaker_raw"]).strip()

        return {
            "path": sample["path"],
            "utility_label": self.utility_to_idx[utility_key],
            "speaker_id": self.speaker_to_idx[speaker_key],
            "gender": sample["gender"],
            "age": sample["age"],
            "raw_utility": utility_key,
            "raw_speaker": speaker_key,
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)  # (C, H, W)
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.squeeze(0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = self._load_image(sample["path"])

        return {
            "image": image,
            "utility_label": sample["utility_label"],
            "privacy_labels": {
                "speaker_id": sample["speaker_id"],
                "gender": sample["gender"],
                "age": sample["age"],
            },
            "metadata": {
                "filename": sample["path"].name,
                "path": str(sample["path"]),
                "raw_utility": sample["raw_utility"],
                "raw_speaker": sample["raw_speaker"],
            },
        }

    @property
    def num_utility_classes(self) -> int:
        return len(self.utility_names)

    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_idx)

    @property
    def utility_label_names(self) -> list[str]:
        return self.utility_names

    def get_speaker_ids(self) -> list[int]:
        return [sample["speaker_id"] for sample in self.samples]
