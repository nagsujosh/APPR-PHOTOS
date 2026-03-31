#!/bin/bash
# Run all experiments for appr-photos
set -e

echo "=== appr-photos Full Experiment Suite ==="
echo "Using conda environment: cs750"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate cs750

# 1. Main experiment
echo ""
echo "=== Experiment 1: CelebA Baseline ==="
python scripts/train.py --config configs/experiment/celeba_baseline.yaml

# 2. Ablation: No VIB
echo ""
echo "=== Ablation: No VIB ==="
python scripts/train.py --config configs/experiment/celeba_baseline.yaml \
    model.filter.use_vib=false \
    output.dir=outputs/celeba_ablation_novib \
    output.checkpoint_dir=outputs/celeba_ablation_novib/checkpoints \
    output.log_dir=outputs/celeba_ablation_novib/logs \
    output.tensorboard_dir=outputs/celeba_ablation_novib/tensorboard

# 3. Ablation: No adversary refresh
echo ""
echo "=== Ablation: No Adversary Refresh ==="
python scripts/train.py --config configs/experiment/celeba_baseline.yaml \
    training.adversary_refresh_interval=0

# 4. Ablation: Utility-only sanity run
echo ""
echo "=== Ablation: Utility-Only ==="
python scripts/train.py --config configs/experiment/celeba_baseline.yaml \
    training.lambda_privacy=0

# 5. Lambda sweep
echo ""
echo "=== Lambda Sweep for Pareto Frontier ==="
python scripts/sweep_lambda.py --config configs/experiment/celeba_baseline.yaml \
    --epochs 20 --output_dir outputs/pareto

# 6. Evaluation
echo ""
echo "=== Evaluation ==="
python scripts/evaluate.py --checkpoint outputs/celeba_baseline/checkpoints/best_model.pt

# 7. Visualization
echo ""
echo "=== Generating Plots ==="
python scripts/visualize.py \
    --checkpoint outputs/celeba_baseline/checkpoints/best_model.pt \
    --pareto_results outputs/pareto/sweep_results.json \
    --output_dir outputs/plots

echo ""
echo "=== All experiments complete ==="
echo "Results in outputs/"
