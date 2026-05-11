"""
prepare_dataset.py — Convert RUKOPYS metadata.jsonl to Ultralytics YOLO format.

Strategy:
  - Val split: 10% of the clean `train/` annotations (human-annotated, high quality).
  - Train split: remaining 90% of `train/` + all of `silver/` (larger, silver-quality set).
  - Images are symlinked (not copied) to save disk space.

Output layout:
    dataset/
        images/train/   ← symlinks to original images
        images/val/
        labels/train/   ← YOLO .txt files
        labels/val/
        dataset.yaml
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"

CLASS_MAP = {
    "handwritten": 0,
    "printed":     1,
    "formula":     2,
    "table":       3,
    "annotation":  4,
    "image":       5,
    "graph":       6,
}

VAL_RATIO = 0.10
SEED = 42

# Minimum box dimension (pixels) — discard degenerate annotations
MIN_BOX_PX = 4


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Convert [x1,y1,x2,y2] absolute pixel coords → YOLO normalised format.
    Returns (xc, yc, w, h) or None for degenerate boxes.
    """
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    # Clamp to image bounds
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))

    box_w, box_h = x2 - x1, y2 - y1
    if box_w < MIN_BOX_PX or box_h < MIN_BOX_PX:
        return None  # skip degenerate / out-of-bounds boxes

    return (
        (x1 + x2) / 2.0 / img_w,
        (y1 + y2) / 2.0 / img_h,
        box_w / img_w,
        box_h / img_h,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_label(label_path, regions, img_w, img_h):
    """Write one YOLO label .txt file. Returns number of valid boxes written."""
    lines = []
    for region in regions:
        cls_id = CLASS_MAP.get(region.get("type"))
        if cls_id is None:
            continue
        result = bbox_to_yolo(*region["bbox"], img_w, img_h)
        if result is None:
            continue
        xc, yc, w, h = result
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    label_path.write_text("\n".join(lines))
    return len(lines)


def safe_symlink(src, dst):
    """Create a symlink dst → src; skip if dst already exists."""
    if not dst.exists():
        dst.symlink_to(src.resolve())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_records(records_with_source, split_name, class_stats):
    images_dir = DATASET_DIR / "images" / split_name
    labels_dir = DATASET_DIR / "labels" / split_name

    skipped_missing = 0
    skipped_empty = 0
    total_boxes = 0

    for record, source_split in records_with_source:
        # e.g. "images/uuid.jpg"
        rel_file = record["file_name"]
        img_path = ROOT / source_split / rel_file

        if not img_path.exists():
            skipped_missing += 1
            continue

        regions = record.get("regions", [])
        if not regions:
            skipped_empty += 1
            continue

        img_w = record["image_width"]
        img_h = record["image_height"]

        # Symlink image
        safe_symlink(img_path, images_dir / img_path.name)

        # Write label
        label_path = labels_dir / f"{img_path.stem}.txt"
        n = write_label(label_path, regions, img_w, img_h)
        total_boxes += n

        # Accumulate class stats
        for region in regions:
            cls_id = CLASS_MAP.get(region.get("type"))
            if cls_id is not None:
                class_stats[split_name][cls_id] += 1

    if skipped_missing:
        print(f"  [{split_name}] {skipped_missing} records skipped — image file not found")
    if skipped_empty:
        print(f"  [{split_name}] {skipped_empty} records skipped — no regions")

    return total_boxes


def print_class_stats(class_stats):
    id_to_name = {v: k for k, v in CLASS_MAP.items()}
    print(f"\n{'Class':<15} {'Train':>10} {'Val':>10}")
    print("-" * 38)
    for cls_id in sorted(id_to_name):
        name = id_to_name[cls_id]
        tr = class_stats["train"].get(cls_id, 0)
        vl = class_stats["val"].get(cls_id, 0)
        print(f"  {name:<13} {tr:>10,} {vl:>10,}")


def main():
    random.seed(SEED)

    # ------------------------------------------------------------------
    # 1. Load records
    # ------------------------------------------------------------------
    print("Loading metadata...")
    train_records  = [(r, "train")  for r in load_jsonl(ROOT / "train"  / "metadata.jsonl")]
    silver_records = [(r, "silver") for r in load_jsonl(ROOT / "silver" / "metadata.jsonl")]
    print(f"  train:  {len(train_records):,} records")
    print(f"  silver: {len(silver_records):,} records")

    # ------------------------------------------------------------------
    # 2. Split: val from clean train only; silver goes entirely to train
    # ------------------------------------------------------------------
    random.shuffle(train_records)
    n_val = max(1, int(len(train_records) * VAL_RATIO))
    val_records      = train_records[:n_val]
    train_clean_part = train_records[n_val:]

    train_combined = train_clean_part + silver_records
    random.shuffle(train_combined)

    print(f"\nDataset split:")
    print(f"  train: {len(train_combined):,} images  ({len(train_clean_part):,} clean + {len(silver_records):,} silver)")
    print(f"  val:   {len(val_records):,} images  (clean only)")

    # ------------------------------------------------------------------
    # 3. Create directory structure
    # ------------------------------------------------------------------
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Write labels + symlink images
    # ------------------------------------------------------------------
    class_stats = {"train": defaultdict(int), "val": defaultdict(int)}

    print("\nProcessing train split...")
    n_train_boxes = process_records(train_combined, "train", class_stats)

    print("Processing val split...")
    n_val_boxes = process_records(val_records, "val", class_stats)

    print(f"\nTotal boxes written — train: {n_train_boxes:,}  val: {n_val_boxes:,}")
    print_class_stats(class_stats)

    # ------------------------------------------------------------------
    # 5. Write dataset.yaml
    # ------------------------------------------------------------------
    names_block = "\n".join(f"  {i}: {name}" for name, i in sorted(CLASS_MAP.items(), key=lambda x: x[1]))
    yaml_text = f"""# RUKOPYS Document Layout Analysis — Ultralytics dataset config
path: {DATASET_DIR.resolve()}
train: images/train
val:   images/val

nc: {len(CLASS_MAP)}
names:
{names_block}
"""
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_text)
    print(f"\ndataset.yaml → {yaml_path}")
    print("Done.")


if __name__ == "__main__":
    main()
