"""
train.py — RT-DETR-L training for RUKOPYS Document Layout Analysis.

Hardware target: single NVIDIA H100 80GB.
Run after prepare_dataset.py has created dataset/dataset.yaml.

Usage:
    python train.py
    python train.py --resume runs/rtdetr-l-dla-1024/weights/last.pt
"""

import argparse
from pathlib import Path

from ultralytics import RTDETR

ROOT = Path(__file__).parent
DATASET_YAML = ROOT / "dataset" / "dataset.yaml"


def main(resume=None):
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"{DATASET_YAML} not found. Run prepare_dataset.py first."
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if resume:
        # Resume from a checkpoint — Ultralytics picks up epoch, optimizer state, etc.
        model = RTDETR(resume)
    else:
        model = RTDETR("rtdetr-l.pt")  # pretrained COCO weights

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    results = model.train(
        data=str(DATASET_YAML),

        # ── Resolution & batch ────────────────────────────────────────
        # 1024 preserves fine detail in dense handwritten text.
        # RT-DETR-L at fp16 uses ~4.5 GB/image at this size;
        # batch=16 sits comfortably under 80 GB on H100.
        # Increase to batch=24 if utilisation allows.
        imgsz=1024,
        batch=16,

        # ── Duration ──────────────────────────────────────────────────
        epochs=120,
        patience=25,        # early-stop: no val-mAP50 improvement for 25 epochs

        # ── Optimiser — transformer models want low lr + AdamW ────────
        optimizer="AdamW",
        lr0=1e-4,           # initial learning rate
        lrf=0.01,           # final lr = lr0 * lrf  (cosine annealing)
        weight_decay=1e-4,
        momentum=0.9,       # used as β1 for AdamW
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=1e-6,

        # ── Augmentation (tuned for document images) ──────────────────
        # Colour shifts — simulate aged paper, ink fading, scanner noise
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.4,

        # Spatial — documents are mostly upright; tiny angles only
        degrees=3.0,
        translate=0.1,
        scale=0.5,          # multi-scale text sizes across pages
        shear=0.5,
        perspective=0.0003, # page curvature / flatbed scan warp

        # Flipping is disabled — Cyrillic text is not horizontally symmetric,
        # and flipped labels would teach the model nonsense.
        fliplr=0.0,
        flipud=0.0,

        # Mosaic packs 4 pages together, forcing the model to handle
        # objects at varied scales and positions — very effective for DLA.
        mosaic=1.0,
        # Disable mosaic for the final 15 epochs so the model fine-tunes
        # on clean, single-image samples before evaluation.
        close_mosaic=15,

        copy_paste=0.0,     # not meaningful for document regions

        # Mild label smoothing helps the highly-imbalanced class set
        # (138 k handwritten vs 63 graph instances).
        label_smoothing=0.05,

        # ── Hardware ──────────────────────────────────────────────────
        device=0,           # single H100
        workers=8,
        amp=True,           # fp16 mixed-precision — essential for H100 throughput

        # ── Checkpointing & logging ───────────────────────────────────
        project=str(ROOT / "runs"),
        name="rtdetr-l-dla-1024",
        save=True,
        save_period=10,     # checkpoint every N epochs (insurance against preemption)
        resume=bool(resume),

        # ── Evaluation ────────────────────────────────────────────────
        val=True,
        plots=True,
        verbose=True,
        seed=42,
        deterministic=False,  # keep False — determinism hurts throughput on H100
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights → {best}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CKPT",
        help="Path to a checkpoint .pt file to resume training from",
    )
    args = parser.parse_args()
    main(resume=args.resume)
