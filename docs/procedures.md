# appr-photos procedures

This document describes the end-to-end workflow for photo-based privacy-preserving training.

## 1. NVIDIA Environment (Recommended)

```bash
bash scripts/setup_nvidia_env.sh appr-photos-cuda 3.10 cu124
conda activate appr-photos-cuda
nvidia-smi
```

If needed, swap CUDA wheel tag:

```bash
bash scripts/setup_nvidia_env.sh appr-photos-cuda 3.10 cu121
```

## 2. Dataset Structure

Use one of the following:

### Folder layout

```text
data/raw/photos/
  class_a/
    subject_01/
      img_001.jpg
      img_002.jpg
  class_b/
    subject_02/
      img_003.jpg
```

### Metadata layout

```text
data/raw/photos/
  metadata.csv
  images/...
```

Supported metadata columns:
- `filename` or `path`
- `utility_label` or `label` or `class`
- `speaker_id` (optional)
- `gender` (optional)
- `age` (optional)

## 3. Auto Download (Default)

```bash
bash scripts/download_data.sh
```

## 4. Verify and Prepare

```bash
python scripts/prepare_datasets.py --verify --root data/raw/photos
python scripts/prepare_datasets.py --build-metadata --root data/raw/photos
python scripts/prepare_datasets.py --stats --root data/raw/photos
```

## 5. Train

```bash
python scripts/train.py --config configs/experiment/photos_nvidia.yaml
```

```bash
python scripts/train.py --config configs/experiment/photos_baseline.yaml
```

## 6. Optional: Precompute Features

```bash
python scripts/precompute_features.py --config configs/experiment/photos_baseline.yaml
python scripts/train.py --config configs/experiment/photos_cached.yaml
```

## 7. Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/photos_nvidia/checkpoints/best_model.pt
```

## 8. Privacy-Utility Tradeoff Sweep

```bash
python scripts/sweep_lambda.py \
  --config configs/experiment/photos_baseline.yaml \
  --epochs 50 \
  --output_dir outputs/pareto
```

## 9. Visualizations

```bash
python scripts/visualize.py \
  --checkpoint outputs/photos_baseline/checkpoints/best_model.pt \
  --pareto_results outputs/pareto/sweep_results.json \
  --output_dir outputs/plots
```
