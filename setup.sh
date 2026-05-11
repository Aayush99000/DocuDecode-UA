#!/usr/bin/env bash
# setup.sh — Reproducible environment bootstrap for DocuDecode-UA on H100
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# What it does:
#   Step 0 — Preflight: verify CUDA driver and Python version
#   Step 1 — System packages required by OpenCV headless
#   Step 2 — Upgrade pip/setuptools/wheel
#   Step 3 — PyTorch stack (unified CUDA wheel from PyPI, no special index needed)
#   Step 4 — flash-attn (must see installed torch before compiling/linking)
#   Step 5 — Everything else in requirements.txt

set -euo pipefail

PYTHON=${PYTHON:-python3.11}
PIP="$PYTHON -m pip"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn] ${NC} $*"; }
die()   { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── Step 0: Preflight ─────────────────────────────────────────────────────────
info "── Step 0: Preflight checks"

# CUDA driver
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found. Install NVIDIA drivers before continuing."
fi
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' || echo "unknown")
info "  GPU driver: ${DRIVER_VER}  |  CUDA runtime: ${CUDA_VER}"

# Require CUDA 12.4+ for H100 sm_90a full support
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
if [[ "$CUDA_MAJOR" -lt 12 ]] || ( [[ "$CUDA_MAJOR" -eq 12 ]] && [[ "$CUDA_MINOR" -lt 4 ]] ); then
    warn "CUDA < 12.4 detected. H100 (sm_90a) needs CUDA ≥ 12.4 for full performance."
    warn "Consider upgrading drivers before proceeding."
fi

# Python version
if ! command -v "$PYTHON" &>/dev/null; then
    die "$PYTHON not found. Install Python 3.11 (recommended for this stack)."
fi
PY_VER=$($PYTHON --version 2>&1)
info "  Python: ${PY_VER}"

# ── Step 1: System packages ───────────────────────────────────────────────────
info "── Step 1: System packages (OpenCV headless runtime libs)"

sudo apt-get update -qq
sudo apt-get install -y -qq \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    ninja-build      # speeds up flash-attn compilation fallback

# ── Step 2: Upgrade pip toolchain ─────────────────────────────────────────────
info "── Step 2: Upgrading pip / setuptools / wheel"
$PIP install --upgrade pip setuptools wheel

# ── Step 3: PyTorch (CUDA-enabled unified wheel from PyPI) ───────────────────
info "── Step 3: Installing PyTorch stack"
#
# As of torch 2.9, Canonical Linux x86_64 wheels on PyPI already bundle CUDA
# (the wheel is ~900 MB vs ~200 MB for a CPU-only build).
# No --index-url https://download.pytorch.org/whl/cuXXX is required.
# The bundled CUDA version matches the driver requirement above.
#
$PIP install \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0

# Sanity-check: confirm CUDA is accessible from the installed torch
$PYTHON - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("WARNING: torch.cuda.is_available() == False. "
          "Check your driver / wheel installation.", file=sys.stderr)
else:
    print(f"  torch {torch.__version__} — CUDA {torch.version.cuda} — "
          f"{torch.cuda.get_device_name(0)}")
EOF

# ── Step 4: flash-attn ────────────────────────────────────────────────────────
info "── Step 4: Installing flash-attn"
#
# --no-build-isolation: prevents pip from creating an isolated venv for the
# build, which would pull in a *second* torch install and link against the
# wrong CUDA version. With this flag the build sees the torch we just installed.
#
# If a pre-built wheel for your exact torch/CUDA/Python combo is available at
#   https://github.com/Dao-AILab/flash-attention/releases
# pip will use it (fast, ~60 s). Otherwise it compiles from source (~15 min).
#
$PIP install flash-attn==2.8.3 --no-build-isolation

# ── Step 5: Remaining dependencies ───────────────────────────────────────────
info "── Step 5: Installing remaining dependencies from requirements.txt"
#
# torch/torchvision/torchaudio/flash-attn are already satisfied after steps 3-4.
# pip will skip re-downloading them.
#
$PIP install -r requirements.txt

# ── Done ──────────────────────────────────────────────────────────────────────
info "Environment setup complete."
$PYTHON - <<'EOF'
import torch, ultralytics, transformers, cv2
print(f"  torch        {torch.__version__}")
print(f"  ultralytics  {ultralytics.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  opencv       {cv2.__version__}")
try:
    import flash_attn
    print(f"  flash-attn   {flash_attn.__version__}")
except ImportError:
    print("  flash-attn   [not importable — check build log]")
EOF
