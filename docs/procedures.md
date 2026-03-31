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

## 2. Recommended Dataset: CelebA

This repo's recommended dataset is CelebA, downloaded through torchvision.
The practical task setup is:
- utility: `Smiling` vs `not_smiling`
- privacy: `identity` and `gender`

Download and prepare it with:

```bash
bash scripts/download_data.sh celeba data/raw/celeba
```

This produces:

```text
data/raw/celeba/
  celeba/
    img_align_celeba/*.jpg
  metadata.csv
```

## 3. Generic Dataset Structure

If you are preparing your own image dataset instead, use one of the following:

### Folder layout

```text
data/raw/celeba/
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
data/raw/celeba/
  metadata.csv
  images/...
```

Supported metadata columns:
- `filename` or `path`
- `utility_label` or `label` or `class`
- `speaker_id` (optional)
- `gender` (optional)
- `age` (optional)

## 4. Verify and Prepare

```bash
python scripts/prepare_datasets.py --verify --root data/raw/celeba
python scripts/prepare_datasets.py --stats --root data/raw/celeba
```

## 5. Train

```bash
python scripts/train.py --config configs/experiment/celeba_nvidia.yaml
```

```bash
python scripts/train.py --config configs/experiment/celeba_baseline.yaml
```

## 6. Optional: Precompute Features

```bash
python scripts/precompute_features.py --config configs/experiment/celeba_baseline.yaml
python scripts/train.py --config configs/experiment/celeba_cached.yaml
```

## 7. Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/celeba_nvidia/checkpoints/best_model.pt
```

## 8. Privacy-Utility Tradeoff Sweep

```bash
python scripts/sweep_lambda.py \
  --config configs/experiment/celeba_baseline.yaml \
  --epochs 20 \
  --output_dir outputs/pareto
```

## 9. Visualizations

```bash
python scripts/visualize.py \
  --checkpoint outputs/celeba_baseline/checkpoints/best_model.pt \
  --pareto_results outputs/pareto/sweep_results.json \
  --output_dir outputs/plots
```

## 10. Report Figures

Generate a clean figure bundle for the report or slides:

```bash
python scripts/generate_report_figures.py \
  --checkpoint outputs/celeba_nvidia/checkpoints/best_model.pt \
  --output_dir outputs/report_figures
```

This saves:
- `training_curves.png`
- `utility_confusion_matrix.png`
- `embeddings_tsne.png`
- `qualitative_saliency.png`
- `figure_summary.json`
