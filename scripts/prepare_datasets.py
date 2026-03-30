#!/usr/bin/env python
"""Photo dataset verification and metadata utilities."""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aapr.data.image_dataset import DEFAULT_EXTENSIONS, PhotoPrivacyDataset


def is_image(path: Path) -> bool:
    return path.suffix.lower() in DEFAULT_EXTENSIONS


def verify_dataset(root: Path) -> dict:
    if not root.exists():
        return {"status": "MISSING", "path": str(root)}

    image_files = [p for p in root.rglob("*") if p.is_file() and is_image(p)]
    metadata_path = root / "metadata.csv"
    return {
        "status": "OK" if image_files else "EMPTY",
        "path": str(root),
        "num_images": len(image_files),
        "has_metadata_csv": metadata_path.exists(),
    }


def build_metadata(root: Path) -> Path:
    rows = []
    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        class_name = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or not is_image(image_path):
                continue
            # Optional speaker hierarchy: class/speaker/image
            if image_path.parent == class_dir:
                speaker_id = image_path.stem.split("_")[0]
            else:
                speaker_id = image_path.parent.name
            rows.append(
                {
                    "filename": str(image_path.relative_to(root)),
                    "utility_label": class_name,
                    "speaker_id": speaker_id,
                    "gender": "",
                    "age": "",
                }
            )

    if not rows:
        raise RuntimeError("No images found while building metadata.csv")

    output = root / "metadata.csv"
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filename", "utility_label", "speaker_id", "gender", "age"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {output}")
    print(f"Classes: {dict(Counter(row['utility_label'] for row in rows))}")
    print(f"Speakers: {len(set(row['speaker_id'] for row in rows))}")
    return output


def print_stats(root: Path):
    dataset = PhotoPrivacyDataset(root=root)
    labels = Counter(dataset[i]["utility_label"] for i in range(len(dataset)))
    missing_gender = sum(
        1 for i in range(len(dataset)) if dataset[i]["privacy_labels"]["gender"] == -1
    )
    print(f"Photos: {len(dataset)} samples | {dataset.num_speakers} speakers")
    print(f"Label distribution: {dict(labels)}")
    print(f"Missing gender labels: {missing_gender}")
    print(f"Class names: {dataset.utility_label_names}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/raw/photos")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--build-metadata", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)

    if args.verify:
        result = verify_dataset(root)
        print("Dataset verification:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    if args.build_metadata:
        build_metadata(root)

    if args.stats:
        print_stats(root)

    if not (args.verify or args.build_metadata or args.stats):
        parser.print_help()


if __name__ == "__main__":
    main()
