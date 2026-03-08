#!/bin/bash
# Run all experiments for AAPR-Speech
set -e

echo "=== AAPR-Speech Full Experiment Suite ==="
echo "Using conda environment: cs750"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate cs750

# 1. Main experiments
echo ""
echo "=== Experiment 1: Crema-D + Mel-Spectrogram ==="
python scripts/train.py --config configs/experiment/cremad_melspec.yaml

echo ""
echo "=== Experiment 2: Crema-D + HuBERT ==="
python scripts/train.py --config configs/experiment/cremad_hubert.yaml

# 2. Ablation: No VIB
echo ""
echo "=== Ablation: No VIB ==="
python scripts/train.py --config configs/experiment/ablation_novib.yaml

# 3. Ablation: No adversary refresh
echo ""
echo "=== Ablation: No Adversary Refresh ==="
python scripts/train.py --config configs/experiment/cremad_melspec.yaml \
    training.adversary_refresh_interval=0

# 4. Ablation: Single attribute adversary (gender only)
echo ""
echo "=== Ablation: Gender-Only Adversary ==="
python scripts/train.py --config configs/experiment/cremad_melspec.yaml \
    model.adversary.heads.speaker_id=0

# 5. Lambda sweep
echo ""
echo "=== Lambda Sweep for Pareto Frontier ==="
python scripts/sweep_lambda.py --config configs/experiment/cremad_melspec.yaml \
    --epochs 50 --output_dir outputs/pareto

# 6. Evaluation
echo ""
echo "=== Evaluation ==="
python scripts/evaluate.py --checkpoint outputs/cremad_melspec/checkpoints/best_model.pt
python scripts/evaluate.py --checkpoint outputs/cremad_hubert/checkpoints/best_model.pt

# 7. Visualization
echo ""
echo "=== Generating Plots ==="
python scripts/visualize.py \
    --checkpoint outputs/cremad_melspec/checkpoints/best_model.pt \
    --pareto_results outputs/pareto/sweep_results.json \
    --output_dir outputs/plots

echo ""
echo "=== All experiments complete ==="
echo "Results in outputs/"
