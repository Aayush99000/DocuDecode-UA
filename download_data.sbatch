#!/bin/bash
# =============================================================================
# download_data.sbatch — Download RUKOPYS dataset from HuggingFace to scratch
#
# Runs inside SLURM so the download survives SSH disconnects.
# snapshot_download resumes partial downloads automatically (HF Hub caching).
#
# Submit:  sbatch download_data.sbatch
# =============================================================================

#SBATCH --job-name=docudecode-download
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/katoch.aa/DocuDecode-UA/logs/download_%j.out
#SBATCH --error=/scratch/katoch.aa/DocuDecode-UA/logs/download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=katoch.aa@northeastern.edu

set -euo pipefail

PROJECT=/scratch/katoch.aa/DocuDecode-UA

echo "====== RUKOPYS Dataset Download ======"
echo "Node : $(hostname)"
echo "Time : $(date)"
echo "======================================"

module purge
module load python/3.13.5

python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
from pathlib import Path

local_dir = Path("/scratch/katoch.aa/DocuDecode-UA/rukopys_full")
local_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading UkrainianCatholicUniversity/rukopys → {local_dir}")
print("(Partial downloads resume automatically)")

path = snapshot_download(
    repo_id="UkrainianCatholicUniversity/rukopys",
    repo_type="dataset",
    local_dir=str(local_dir),
    resume_download=True,
)
print(f"Download complete: {path}")
PYEOF

echo ""
echo "====== Download finished: $(date) ======"
du -sh "$PROJECT/rukopys_full"
ls "$PROJECT/rukopys_full"
