#!/bin/bash
# Create a CUDA-enabled conda environment for NVIDIA GPU training.
#
# Usage:
#   bash scripts/setup_nvidia_env.sh
#   bash scripts/setup_nvidia_env.sh appr-photos-cuda 3.10 cu124
#
# Args:
#   1: env name      (default: appr-photos-cuda)
#   2: python ver    (default: 3.10)
#   3: CUDA tag      (default: cu124)  # choose cu121/cu124 to match your driver

set -euo pipefail

ENV_NAME="${1:-appr-photos-cuda}"
PY_VER="${2:-3.10}"
CUDA_TAG="${3:-cu124}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Creating conda env: ${ENV_NAME} (python ${PY_VER})"
conda create -y -n "${ENV_NAME}" python="${PY_VER}" pip

echo "==> Installing CUDA-enabled PyTorch (${CUDA_TAG})"
conda run -n "${ENV_NAME}" pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install \
  torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> Installing project dependencies"
conda run -n "${ENV_NAME}" pip install -e "${PROJECT_ROOT}"

echo "==> GPU sanity check"
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    print("gpu_name:", torch.cuda.get_device_name(0))
PY

echo "==> Done. Activate with: conda activate ${ENV_NAME}"
