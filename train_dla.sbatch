#!/bin/bash
# =============================================================================
# train_dla.sbatch — Phase 1: RT-DETR-L Document Layout Analysis training
#
# Submit:   sbatch train_dla.sbatch
# Resume:   Re-submitting this script is safe — it auto-detects the latest
#           checkpoint and passes --resume to train.py automatically.
#
# If training needs more than 8 hours, re-submit after each run.
# The checkpoint at runs/rtdetr-l-dla-1024/weights/last.pt will be picked up.
# =============================================================================

#SBATCH --job-name=docudecode-dla
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/katoch.aa/DocuDecode-UA/logs/dla_%j.out
#SBATCH --error=/scratch/katoch.aa/DocuDecode-UA/logs/dla_%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=katoch.aa@northeastern.edu

set -euo pipefail

PROJECT=/scratch/katoch.aa/DocuDecode-UA
ENV_DIR=/scratch/katoch.aa/envs/docudecode

echo "====== Phase 1: DLA Training ======"
echo "Node     : $(hostname)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Time     : $(date)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "===================================="

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load cuda/12.8.0
module load cuDNN/9.10.2
module load miniconda3/25.9.1
source activate "$ENV_DIR"

# ── CUDA / performance tuning ──────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
# Allow TF32 for matmuls and convolutions on H200
export TORCH_ALLOW_TF32=1
# Faster cuDNN kernels (auto-tunes on first run, cached thereafter)
export TORCH_CUDNN_V8_API_ENABLED=1

cd "$PROJECT"

# ── Auto-detect latest checkpoint for seamless resumption ────────────────────
RESUME_FLAG=""
LAST_CKPT="$PROJECT/runs/rtdetr-l-dla-1024/weights/last.pt"
if [ -f "$LAST_CKPT" ]; then
    echo "Found checkpoint: $LAST_CKPT — resuming training"
    RESUME_FLAG="--resume $LAST_CKPT"
else
    echo "No checkpoint found — starting fresh"
fi

# ── Run prepare_dataset.py if dataset/ doesn't exist yet ─────────────────────
if [ ! -f "$PROJECT/dataset/dataset.yaml" ]; then
    echo "dataset/dataset.yaml not found — running prepare_dataset.py first"
    python prepare_dataset.py
fi

# ── Train ─────────────────────────────────────────────────────────────────────
echo "Starting RT-DETR-L training at $(date)"
python train.py $RESUME_FLAG

echo ""
echo "====== Training finished: $(date) ======"
BEST="$PROJECT/runs/rtdetr-l-dla-1024/weights/best.pt"
if [ -f "$BEST" ]; then
    echo "Best weights: $BEST ($(du -sh $BEST | cut -f1))"
fi
