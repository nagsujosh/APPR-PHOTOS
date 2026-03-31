#!/bin/bash
# Dataset bootstrap helper for appr-photos.
#
# CelebA is the only supported dataset bootstrap path:
#   bash scripts/download_data.sh celeba data/raw/celeba

set -euo pipefail

DATASET="${1:-celeba}"
OUTPUT_DIR="${2:-data/raw/celeba}"
PYTHON_BIN="${PYTHON_BIN:-python}"

prepare_celeba() {
  local output_dir="${1}"

  cat <<EOF
==> Downloading and preparing CelebA

Dataset:
  - images: CelebA aligned face images
  - utility label: Smiling / not_smiling
  - privacy labels: identity (speaker_id), gender

Output dir: ${output_dir}
EOF

  "${PYTHON_BIN}" scripts/prepare_celeba.py --output-root "${output_dir}"

  echo "==> Verifying dataset"
  "${PYTHON_BIN}" scripts/prepare_datasets.py --verify --stats --root "${output_dir}"
  echo "==> CelebA dataset ready"
}

case "${DATASET}" in
  celeba)
    prepare_celeba "${OUTPUT_DIR}"
    ;;
  *)
    cat <<EOF
Unknown dataset '${DATASET}'.

Usage:
  bash scripts/download_data.sh celeba [OUTPUT_DIR]
EOF
    exit 1
    ;;
esac
