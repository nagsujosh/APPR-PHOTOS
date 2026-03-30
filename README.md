# appr-photos

**Adversary-Adaptive Representation Learning for Privacy-Preserving Image Tasks**

CS750 Course Project — University of New Hampshire

## Overview

This project learns image representations that preserve utility labels (for example, emotion or category) while suppressing sensitive biometric information (identity, gender, age).

### Architecture

```
Image → CNN Feature Extractor
      → Privacy Filter (Conv1D + VIB)
      → Task Model (Attention Pooling + Classifier)  [utility]
      → GRL → Multi-Head Adversary                   [privacy]
```

**Training objective:** `min_{θ,φ} max_{ψ} L_utility - λ · L_privacy`

## NVIDIA GPU Quick Start

```bash
bash scripts/setup_nvidia_env.sh appr-photos-cuda 3.10 cu124
conda activate appr-photos-cuda
nvidia-smi
```

If your driver requires a different wheel, use `cu121` or another CUDA tag:

```bash
bash scripts/setup_nvidia_env.sh appr-photos-cuda 3.10 cu121
```

## Dataset Download

One-command default dataset bootstrap (CIFAR-10 -> image folder + metadata):

```bash
bash scripts/download_data.sh
```

This writes to `data/raw/photos` and creates `metadata.csv`.

## Custom Dataset Preparation

Organize images under `data/raw/photos`:

```text
data/raw/photos/
  <class_name>/
    <speaker_id>/image_001.jpg
    <speaker_id>/image_002.jpg
```

Or place `metadata.csv` in `data/raw/photos/` with fields such as:
`filename,utility_label,speaker_id,gender,age`.

Verification and metadata generation:

```bash
python scripts/prepare_datasets.py --verify --root data/raw/photos
python scripts/prepare_datasets.py --build-metadata --root data/raw/photos
python scripts/prepare_datasets.py --stats --root data/raw/photos
```

## Training

```bash
# NVIDIA-optimized config
python scripts/train.py --config configs/experiment/photos_nvidia.yaml

# Baseline image training
python scripts/train.py --config configs/experiment/photos_baseline.yaml

# Precompute features for faster training loops (optional)
python scripts/precompute_features.py --config configs/experiment/photos_baseline.yaml
python scripts/train.py --config configs/experiment/photos_cached.yaml

# Override config values from CLI
python scripts/train.py --config configs/default.yaml training.num_epochs=50 training.lambda_privacy=0.5
```

## CPU / Non-NVIDIA Setup

```bash
conda create -n appr-photos-cpu -y python=3.10 pip
conda activate appr-photos-cpu
pip install torch torchvision
pip install -e .
```

## Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/photos_nvidia/checkpoints/best_model.pt
```

## Lambda Sweep (Pareto Frontier)

```bash
python scripts/sweep_lambda.py --config configs/experiment/photos_baseline.yaml --epochs 50
```

## Visualization

```bash
python scripts/visualize.py --checkpoint outputs/photos_nvidia/checkpoints/best_model.pt
```

## Tests

```bash
pytest tests/ -v
```

## Evaluation Metrics

**Utility** (higher is better): UAR, Weighted Accuracy, Macro F1  
**Privacy** (lower is more private): identity accuracy, gender accuracy, de-identification rate, MI(Z; S)

## Project Structure

```text
src/aapr/
├── data/          # Photo dataset loaders and split/collation utils
├── features/      # CNN image feature extractor + feature cache
├── models/        # Privacy filter, task model, adversary, GRL
├── training/      # Adversarial trainer, losses, schedulers, metrics
├── evaluation/    # Evaluator, cross-dataset, Pareto analysis
├── visualization/ # Embeddings, training curves, Pareto plots
└── utils/         # Config, logging, seed, device detection
```
# APPR-PHOTOS
