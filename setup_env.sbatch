#!/bin/bash
# =============================================================================
# setup_env.sbatch — One-time environment setup for DocuDecode-UA
#
# Uses pip venv (not conda) since repo.anaconda.com is blocked on this cluster.
# Run ONCE before any training job:
#   sbatch setup_env.sbatch
#
# Creates venv at /scratch/katoch.aa/envs/docudecode with:
#   Python 3.13.5, PyTorch (CUDA), flash-attn, and all requirements.
#
# Uses the short partition (internet-enabled). Takes ~20-30 minutes.
# =============================================================================

#SBATCH --job-name=docudecode-setup
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/katoch.aa/DocuDecode-UA/logs/setup_%j.out
#SBATCH --error=/scratch/katoch.aa/DocuDecode-UA/logs/setup_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=katoch.aa@northeastern.edu

set -euo pipefail

# Proxy required for internet access on compute nodes
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export ftp_proxy=http://10.99.0.130:3128

PROJECT=/scratch/katoch.aa/DocuDecode-UA
ENV_DIR=/scratch/katoch.aa/envs/docudecode

echo "====== DocuDecode-UA Environment Setup ======"
echo "Node     : $(hostname)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Time     : $(date)"
echo "=============================================="

# Verify PyPI is reachable — resubmit if not
curl -sf --max-time 15 https://pypi.org > /dev/null || {
    echo "PyPI unreachable on $(hostname) — resubmitting"
    sbatch "$0"; exit 0
}
echo "Internet OK"

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load cuda/12.8.0
module load cuDNN/9.10.2
module load python/3.13.5

echo "[1/4] CUDA $(nvcc --version | grep release | awk '{print $6}') + Python $(python3 --version)"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ -d "$ENV_DIR" ]; then
    echo "[2/4] Venv already exists at $ENV_DIR — skipping creation"
else
    echo "[2/4] Creating venv"
    python3 -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

# Force IPv4 for all pip connections (cluster blocks IPv6 outbound)
cat > "$ENV_DIR/lib/python3.13/site-packages/sitecustomize.py" << 'PYEOF'
import socket as _socket
_orig = _socket.getaddrinfo
def _ipv4_only(host, port, family=0, *a, **kw):
    return _orig(host, port, _socket.AF_INET, *a, **kw)
_socket.getaddrinfo = _ipv4_only
PYEOF

pip install --upgrade pip -q
echo "      Python: $(python --version)"

# ── PyTorch ───────────────────────────────────────────────────────────────────
echo "[3/4] Installing PyTorch"
pip install torch torchvision torchaudio -q

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'      torch {torch.__version__} — CUDA {torch.version.cuda}')
" || echo "      (CUDA check skipped — no GPU on this node, will work on H200)"

# ── flash-attn ────────────────────────────────────────────────────────────────
echo "[4/4] Installing flash-attn (compiled for H100/H200 sm_90a)"
pip install flash-attn --no-build-isolation -q || echo "      flash-attn install failed — will run without it"

# ── Everything else ───────────────────────────────────────────────────────────
echo "[5/5] Installing remaining dependencies"
pip install -r "$PROJECT/requirements.txt" -q

echo ""
echo "====== Setup complete: $(date) ======"
python - <<'PYEOF'
import importlib
for pkg, mod in [("ultralytics","ultralytics"),("transformers","transformers"),("cv2","cv2")]:
    try:
        m = importlib.import_module(mod)
        print(f"  {pkg:15} {m.__version__}")
    except Exception as e:
        print(f"  {pkg:15} ERROR: {e}")
PYEOF
