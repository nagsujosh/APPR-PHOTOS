#!/bin/bash
# Download and prepare a default photo dataset (CIFAR-10) for appr-photos.
# Usage:
#   bash scripts/download_data.sh
#   bash scripts/download_data.sh data/raw/photos

set -euo pipefail

OUTPUT_DIR="${1:-data/raw/photos}"
TMP_DIR="/tmp/appr_photos_cifar10"
ARCHIVE="${TMP_DIR}/cifar-10-python.tar.gz"
LEGACY_ARCHIVE="/tmp/cifar-10-python.tar.gz"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${TMP_DIR}" "${OUTPUT_DIR}"

echo "==> Downloading CIFAR-10 archive"
if [[ -f "${ARCHIVE}" ]]; then
  echo "    Using cached archive: ${ARCHIVE}"
elif [[ -f "${LEGACY_ARCHIVE}" ]]; then
  echo "    Using cached archive: ${LEGACY_ARCHIVE}"
  cp "${LEGACY_ARCHIVE}" "${ARCHIVE}"
else
  curl -L --fail "${URL}" -o "${ARCHIVE}"
fi

echo "==> Converting CIFAR-10 to image-folder format in ${OUTPUT_DIR}"
if ! "${PYTHON_BIN}" -c "import numpy, PIL" >/dev/null 2>&1; then
  echo "ERROR: numpy/Pillow not found in ${PYTHON_BIN}."
  echo "Activate your project env first (for NVIDIA: conda activate appr-photos-cuda)."
  exit 1
fi

"${PYTHON_BIN}" - "${ARCHIVE}" "${OUTPUT_DIR}" <<'PY'
import csv
import pickle
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

archive = Path(sys.argv[1])
output_root = Path(sys.argv[2]).resolve()
extract_root = Path("/tmp/cifar10_extracted")

extract_root.mkdir(parents=True, exist_ok=True)
if not (extract_root / "cifar-10-batches-py").exists():
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(extract_root)

base = extract_root / "cifar-10-batches-py"
if not base.exists():
    raise SystemExit("Extraction failed: missing cifar-10-batches-py")

output_root.mkdir(parents=True, exist_ok=True)
for child in list(output_root.iterdir()):
    if child.is_dir():
        shutil.rmtree(child)
    else:
        child.unlink()

meta = pickle.load(open(base / "batches.meta", "rb"), encoding="bytes")
labels = [x.decode("utf-8") for x in meta[b"label_names"]]
batch_files = [base / f"data_batch_{i}" for i in range(1, 6)] + [base / "test_batch"]

total = 0
for batch_path in batch_files:
    batch = pickle.load(open(batch_path, "rb"), encoding="bytes")
    data = batch[b"data"]
    y = batch[b"labels"]
    filenames = [f.decode("utf-8") for f in batch[b"filenames"]]

    for i in range(len(y)):
        class_name = labels[y[i]]
        # CIFAR has no real identity labels. We create stable pseudo-speaker buckets
        # so speaker-stratified splitting in this project still works.
        speaker_id = f"speaker_{(total + i) % 200:03d}"
        out_dir = output_root / class_name / speaker_id
        out_dir.mkdir(parents=True, exist_ok=True)

        arr = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        stem = Path(filenames[i]).stem
        img.save(out_dir / f"{batch_path.stem}_{stem}.png")

    total += len(y)

rows = []
for class_dir in sorted(output_root.iterdir()):
    if not class_dir.is_dir():
        continue
    for speaker_dir in sorted(class_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for img_file in sorted(speaker_dir.glob("*.png")):
            rows.append(
                {
                    "filename": str(img_file.relative_to(output_root)),
                    "utility_label": class_dir.name,
                    "speaker_id": speaker_dir.name,
                    "gender": "",
                    "age": "",
                }
            )

with open(output_root / "metadata.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=["filename", "utility_label", "speaker_id", "gender", "age"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} images to {output_root}")
print("Classes:", ", ".join(labels))
PY

echo "==> Verifying dataset"
"${PYTHON_BIN}" scripts/prepare_datasets.py --verify --stats --root "${OUTPUT_DIR}"
echo "==> Dataset ready"
