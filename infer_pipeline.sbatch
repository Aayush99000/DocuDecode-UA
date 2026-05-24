#!/bin/bash
#SBATCH --job-name=docudecode-infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/katoch.aa/DocuDecode-UA/logs/infer_%j.out
#SBATCH --error=/scratch/katoch.aa/DocuDecode-UA/logs/infer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=katoch.aa@northeastern.edu

set -euo pipefail

PROJECT=/scratch/katoch.aa/DocuDecode-UA
ENV_DIR=/scratch/katoch.aa/envs/docudecode

echo "====== DocuDecode-UA Inference ======"
echo "Node     : $(hostname)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Time     : $(date)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "======================================"

module purge
module load cuda/12.8.0
module load cuDNN/9.10.2
module load miniconda3/25.9.1
source activate "$ENV_DIR"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/katoch.aa/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$PROJECT"

python infer_pipeline.py \
    --test-dir    "$PROJECT/test" \
    --dla-weights "$PROJECT/runs/rtdetr-l-dla-1024/weights/best.pt" \
    --htr-model   "$PROJECT/runs/phase3-htr/final" \
    --output      "$PROJECT/submission.jsonl" \
    --conf        0.25 \
    --imgsz       1024 \
    --htr-batch   16

echo ""
echo "====== Inference finished: $(date) ======"
echo "Submission: $PROJECT/submission.jsonl"
wc -l "$PROJECT/submission.jsonl"
