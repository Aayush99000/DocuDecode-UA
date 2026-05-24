#!/bin/bash
# =============================================================================
# train_htr.sbatch — Phase 3: TrOCR-style HTR fine-tuning
#
# Submit:   sbatch train_htr.sbatch
# Resume:   Re-submitting is safe — the script finds the latest HuggingFace
#           Trainer checkpoint under runs/phase3-htr/ and resumes from it.
#
# Dependency chain (run DLA first, then HTR):
#   JID=$(sbatch train_dla.sbatch | awk '{print $4}')
#   sbatch --dependency=afterok:$JID train_htr.sbatch
# =============================================================================

#SBATCH --job-name=docudecode-htr
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/katoch.aa/DocuDecode-UA/logs/htr_%j.out
#SBATCH --error=/scratch/katoch.aa/DocuDecode-UA/logs/htr_%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=katoch.aa@northeastern.edu

set -euo pipefail

# Proxy required for internet access on compute nodes
export http_proxy=http://10.99.0.130:3128
export https_proxy=http://10.99.0.130:3128
export ftp_proxy=http://10.99.0.130:3128

PROJECT=/scratch/katoch.aa/DocuDecode-UA
ENV_DIR=/scratch/katoch.aa/envs/docudecode

echo "====== Phase 3: HTR Training ======"
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
export TORCH_ALLOW_TF32=1
export TOKENIZERS_PARALLELISM=false
# Use pre-downloaded models from scratch — compute nodes have no internet
export HF_HOME=/scratch/katoch.aa/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$PROJECT"

# ── Auto-detect latest HuggingFace Trainer checkpoint ────────────────────────
# Trainer saves checkpoints as runs/phase3-htr/checkpoint-<step>/.
# We pass the latest one to trainer.train(resume_from_checkpoint=...).
RESUME_CKPT=$(ls -dt "$PROJECT/runs/phase3-htr/checkpoint-"* 2>/dev/null | head -1) || true
if [ -n "$RESUME_CKPT" ]; then
    echo "Resuming from checkpoint: $RESUME_CKPT"
    export HTR_RESUME_CHECKPOINT="$RESUME_CKPT"
else
    echo "No checkpoint found — starting fresh"
    export HTR_RESUME_CHECKPOINT=""
fi

python - <<'PYEOF'
import os, sys, signal, traceback as _tb

def _sigterm_handler(signum, frame):
    print(f"\n[SLURM] Received signal {signum} — likely OOM or time-limit kill", flush=True)
    sys.exit(signum)
signal.signal(signal.SIGTERM, _sigterm_handler)

sys.path.insert(0, os.environ["PROJECT"] if "PROJECT" in os.environ else ".")

from phase3_htr import (
    build_htr_pairs_from_metadata,
    split_pairs,
    init_model_and_tokenizer,
    build_compute_metrics,
    get_training_args,
    ImageTextDataCollator,
    RukopysDataset,
)
from pathlib import Path
from transformers import Seq2SeqTrainer
import random, logging, json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)

ROOT        = Path(os.environ.get("PROJECT", "."))
HTR_DATA    = ROOT / "htr_data"
OUTPUT_DIR  = ROOT / "runs" / "phase3-htr"
RESUME_CKPT = os.environ.get("HTR_RESUME_CHECKPOINT", "") or None

# ── Build / load OCR pairs ────────────────────────────────────────────────────
clean_pairs = build_htr_pairs_from_metadata(
    jsonl_path=ROOT / "train"  / "metadata.jsonl",
    img_root=ROOT / "train",
    out_dir=HTR_DATA / "crops" / "train",
)
silver_pairs = build_htr_pairs_from_metadata(
    jsonl_path=ROOT / "silver" / "metadata.jsonl",
    img_root=ROOT / "silver",
    out_dir=HTR_DATA / "crops" / "silver",
)

train_clean, val_pairs = split_pairs(clean_pairs, val_ratio=0.10)
train_pairs = train_clean + silver_pairs
random.seed(42)
random.shuffle(train_pairs)

# ── Model + tokenizer ─────────────────────────────────────────────────────────
model, tokenizer, image_processor = init_model_and_tokenizer()

# ── Datasets + collator ───────────────────────────────────────────────────────
train_dataset = RukopysDataset(train_pairs, image_processor, tokenizer)
eval_dataset  = RukopysDataset(val_pairs,   image_processor, tokenizer)
collator      = ImageTextDataCollator(tokenizer=tokenizer)

# ── Trainer ───────────────────────────────────────────────────────────────────
training_args = get_training_args(OUTPUT_DIR)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    compute_metrics=build_compute_metrics(tokenizer),
    processing_class=tokenizer,
)

try:
    print("Starting trainer.train() …", flush=True)
    trainer.train(resume_from_checkpoint=RESUME_CKPT)
except Exception as exc:
    print(f"\n[FATAL] trainer.train() raised: {exc}", flush=True)
    print(_tb.format_exc(), flush=True)
    sys.exit(1)

# ── Save final artefacts ──────────────────────────────────────────────────────
final_dir = OUTPUT_DIR / "final"
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))
image_processor.save_pretrained(str(final_dir))
print(f"Final model saved to {final_dir}")
PYEOF

echo ""
echo "====== HTR job finished: $(date) ======"

# Auto-resubmit if a checkpoint exists but no final model yet — keeps training
# across 1-hour sharing-partition slots without manual intervention.
FINAL_DIR="$PROJECT/runs/phase3-htr/final"
HAS_CKPT=$(ls -d "$PROJECT/runs/phase3-htr/checkpoint-"* 2>/dev/null | head -1)
if [ -n "$HAS_CKPT" ] && [ ! -d "$FINAL_DIR" ]; then
    echo "Checkpoint found; no final model yet — auto-resubmitting …"
    sbatch "$PROJECT/train_htr.sbatch"
else
    echo "Training complete (final model exists) or no checkpoint to resume from."
fi
