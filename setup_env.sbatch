#!/bin/bash
# =============================================================================
# setup_env.sbatch — One-time environment setup for DocuDecode-UA
#
# Run ONCE before any training job:
#   sbatch setup_env.sbatch
#
# Creates conda env at /scratch/katoch.aa/envs/docudecode with:
#   Python 3.11, PyTorch 2.11.0 (CUDA bundled), flash-attn 2.8.3,
#   and all packages from requirements.txt.
#
# Uses the H100 node so flash-attn compiles against the right CUDA arch.
# Takes ~15-20 minutes.
# =============================================================================

#SBATCH --job-name=docudecode-setup
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/katoch.aa/DocuDecode-UA/logs/setup_%j.out
#SBATCH --error=/scratch/katoch.aa/DocuDecode-UA/logs/setup_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=katoch.aa@northeastern.edu

set -euo pipefail

PROJECT=/scratch/katoch.aa/DocuDecode-UA
ENV_DIR=/scratch/katoch.aa/envs/docudecode

echo "====== DocuDecode-UA Environment Setup ======"
echo "Node     : $(hostname)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Time     : $(date)"
echo "GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "=============================================="

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load cuda/12.8.0
module load cuDNN/9.10.2
module load miniconda3/25.9.1

echo "[1/5] CUDA $(nvcc --version | grep release | awk '{print $6}') loaded"

# ── Conda environment ─────────────────────────────────────────────────────────
if [ -d "$ENV_DIR" ]; then
    echo "[2/5] Env already exists at $ENV_DIR — skipping creation"
else
    echo "[2/5] Creating conda env (Python 3.11)"
    conda create -y -p "$ENV_DIR" python=3.11
fi
source activate "$ENV_DIR"
echo "      Python: $(python --version)"

# ── PyTorch — unified CUDA wheel from PyPI (torch 2.9+ packaging) ─────────────
echo "[3/5] Installing PyTorch 2.11.0"
pip install --upgrade pip setuptools wheel -q
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 -q

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'      torch {torch.__version__} — CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}')
"

# ── flash-attn — needs --no-build-isolation so it links against our torch ─────
echo "[4/5] Installing flash-attn 2.8.3 (compiling for H100 sm_90a)"
pip install flash-attn==2.8.3 --no-build-isolation -q
python -c "import flash_attn; print(f'      flash-attn {flash_attn.__version__}')"

# ── Everything else ───────────────────────────────────────────────────────────
echo "[5/5] Installing remaining dependencies"
pip install -r "$PROJECT/requirements.txt" -q

echo ""
echo "====== Setup complete: $(date) ======"
python - <<'PYEOF'
import torch, ultralytics, transformers, cv2
print(f"  torch        {torch.__version__}  CUDA={torch.cuda.is_available()}")
print(f"  ultralytics  {ultralytics.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  opencv       {cv2.__version__}")
PYEOF
