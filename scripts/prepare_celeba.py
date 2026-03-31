#!/usr/bin/env python
"""Download CelebA and build repo-compatible metadata.

This script uses torchvision's official CelebA dataset integration, which
downloads the aligned image zip and annotation files via the upstream Google
Drive links. torchvision requires the `gdown` package for this download path.

The prepared dataset layout is:

    data/raw/celeba/
      metadata.csv
      celeba/img_align_celeba/000001.jpg
      ...

metadata.csv rows point at the downloaded aligned images and expose:
  - utility_label: binary face attribute (default: Smiling)
  - speaker_id: CelebA identity label
  - gender: derived from the Male attribute
  - age: left blank
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/celeba"))
    parser.add_argument(
        "--utility-attr",
        type=str,
        default="Smiling",
        help="CelebA binary attribute to use as the utility label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from torchvision.datasets import CelebA
    except Exception as exc:  # pragma: no cover - import guidance
        print(
            "ERROR: torchvision is required to prepare CelebA. "
            "Install the project environment first.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        import gdown  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guidance
        print(
            "ERROR: gdown is required because torchvision downloads CelebA "
            "from Google Drive. Install it with: pip install gdown",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = CelebA(
        root=str(output_root),
        split="all",
        target_type=["attr", "identity"],
        download=True,
    )

    attr_names = list(dataset.attr_names)
    if args.utility_attr not in attr_names:
        raise SystemExit(
            f"Unknown utility attribute '{args.utility_attr}'. "
            f"Expected one of: {', '.join(attr_names)}"
        )
    if "Male" not in attr_names:
        raise SystemExit("CelebA metadata missing the 'Male' attribute.")

    utility_idx = attr_names.index(args.utility_attr)
    gender_idx = attr_names.index("Male")

    metadata_path = output_root / "metadata.csv"
    rows_written = 0
    with open(metadata_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "utility_label", "speaker_id", "gender", "age"],
        )
        writer.writeheader()

        for filename, attr_row, identity in zip(dataset.filename, dataset.attr, dataset.identity):
            utility_value = int(attr_row[utility_idx].item())
            gender_value = int(attr_row[gender_idx].item())
            writer.writerow(
                {
                    "filename": str(Path("celeba") / "img_align_celeba" / filename),
                    "utility_label": args.utility_attr.lower() if utility_value == 1 else f"not_{args.utility_attr.lower()}",
                    "speaker_id": int(identity.item()),
                    "gender": gender_value,
                    "age": "",
                }
            )
            rows_written += 1

    print(f"Wrote {rows_written} rows to {metadata_path}")
    print(f"Utility attribute: {args.utility_attr}")
    print("Root:", output_root)
    print("Images:", output_root / "celeba" / "img_align_celeba")


if __name__ == "__main__":
    main()
