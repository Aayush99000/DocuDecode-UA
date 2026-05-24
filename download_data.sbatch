#!/bin/bash
# =============================================================================
# download_data.sbatch — Download RUKOPYS using pre-generated file list
# No HF API calls needed — just wget each URL from download_filelist.txt
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

# Proxy required for internet access on compute nodes
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export ftp_proxy=http://10.99.0.130:3128

LOCAL_DIR=/scratch/katoch.aa/DocuDecode-UA/rukopys_full
FILELIST=/scratch/katoch.aa/DocuDecode-UA/download_filelist.txt

echo "====== RUKOPYS Dataset Download ======"
echo "Node : $(hostname)"
echo "Job  : $SLURM_JOB_ID"
echo "Time : $(date)"
echo "======================================"

# Basic connectivity check — resubmit if no internet
curl -sf --max-time 10 https://huggingface.co > /dev/null || {
    echo "No internet on $(hostname) — resubmitting"
    sbatch "$0"; exit 0
}
echo "Internet OK"

mkdir -p "$LOCAL_DIR"

TOTAL=$(wc -l < "$FILELIST")
COUNT=0
SKIP=0

while IFS=$'\t' read -r url relpath; do
    dest="$LOCAL_DIR/$relpath"
    mkdir -p "$(dirname "$dest")"
    COUNT=$((COUNT + 1))

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        SKIP=$((SKIP + 1))
        continue
    fi

    echo "[$COUNT/$TOTAL] $relpath"
    wget -q -c -O "$dest" "$url" || {
        echo "  WARN: failed $relpath — skipping"
        rm -f "$dest"
    }
done < "$FILELIST"

echo ""
echo "====== Download finished: $(date) ======"
echo "Downloaded: $((COUNT - SKIP)) new, skipped: $SKIP existing"
ls "$LOCAL_DIR"
du -sh "$LOCAL_DIR"/*/
