"""
infer_and_crop.py — RT-DETR inference + adaptive-padded crop pipeline.

Runs the trained DLA model on test images, crops each detected region with
adaptive padding (to preserve Cyrillic cursive ascenders/descenders), then
saves crops ready for a downstream OCR model.

Usage:
    # Default: runs on test/images/, saves to inference_output/
    python infer_and_crop.py

    # Custom paths / thresholds
    python infer_and_crop.py \
        --weights runs/rtdetr-l-dla-1024/weights/best.pt \
        --source  test/images \
        --output  inference_output \
        --conf 0.30 \
        --imgsz 1024

Output layout:
    inference_output/
        crops/
            handwritten/   ← padded crops for OCR
            printed/
            formula/
            ...
        viz/               ← annotated images for visual QC
        predictions.jsonl  ← machine-readable manifest of all detections
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import RTDETR

ROOT = Path(__file__).parent

CLASS_NAMES = {
    0: "handwritten",
    1: "printed",
    2: "formula",
    3: "table",
    4: "annotation",
    5: "image",
    6: "graph",
}

# Classes whose crops should be forwarded to OCR (contain readable text)
OCR_CLASSES = {0, 1, 4}  # handwritten, printed, annotation

# Colours for visualisation (BGR)
VIZ_COLOURS = {
    0: (  0, 200,   0),   # handwritten — green
    1: ( 30, 144, 255),   # printed     — blue
    2: (255, 165,   0),   # formula     — orange
    3: (138,  43, 226),   # table       — purple
    4: (  0, 200, 200),   # annotation  — cyan
    5: (200,   0,   0),   # image       — red
    6: (255, 215,   0),   # graph       — gold
}


# ---------------------------------------------------------------------------
# Padding logic
# ---------------------------------------------------------------------------

def adaptive_padding(box_w: int, box_h: int) -> int:
    """
    Compute how many pixels to expand a bounding box on each side.

    Design rationale
    ─────────────────
    Cyrillic cursive ascenders (б, д, й…) and descenders (у, р, ф…) can
    extend 15-25% beyond a tight AABB.  A fixed margin of 3-5 px is enough
    for small crops, but larger line-blocks need proportionally more room.

        pad = base  +  scale × shorter_side

    Values:
      • box_h = 40 px  → pad = 3 + 0.04 × 40  =  4 px
      • box_h = 80 px  → pad = 3 + 0.04 × 80  =  6 px
      • box_h = 200 px → pad = 3 + 0.04 × 200 = 11 px (capped at 12)

    The cap prevents runaway padding on very large multi-line regions where
    the annotations already include whitespace margins.
    """
    shorter = min(box_w, box_h)
    pad = int(3 + 0.04 * shorter)
    return min(pad, 12)


def padded_crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """
    Expand the AABB by adaptive_padding pixels on each side,
    clamp to image boundaries, and return (crop_array, padded_xyxy).
    """
    img_h, img_w = image.shape[:2]
    pad = adaptive_padding(x2 - x1, y2 - y1)
    cx1 = max(0,     x1 - pad)
    cy1 = max(0,     y1 - pad)
    cx2 = min(img_w, x2 + pad)
    cy2 = min(img_h, y2 + pad)
    return image[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2)


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model_path: str,
    source_dir: str,
    output_dir: str,
    conf: float = 0.25,
    iou: float = 0.50,
    imgsz: int = 1024,
):
    model = RTDETR(model_path)

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Create output directories
    for name in CLASS_NAMES.values():
        (output_dir / "crops" / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "viz").mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    )
    if not image_paths:
        print(f"No images found in {source_dir}")
        return []

    print(f"Running inference on {len(image_paths)} images  (conf≥{conf}, imgsz={imgsz})")

    all_predictions = []

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Cannot read {img_path.name} — skipping")
            continue

        # ── Predict ──────────────────────────────────────────────────
        # RT-DETR produces a fixed-size set of predictions directly from
        # its decoder; iou= here only gates a minimal post-processing step
        # and has almost no effect (no NMS cascade as in YOLO).
        result = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=0,
            verbose=False,
            save=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  {img_path.name}: 0 detections")
            all_predictions.append({"file_name": img_path.name, "detections": []})
            continue

        # ── Sort detections top-to-bottom (reading order) ────────────
        # Sorting by y-centre makes the downstream OCR pipeline easier:
        # crops arrive in roughly the order they appear on the page.
        xyxy_np   = boxes.xyxy.cpu().numpy()
        conf_np   = boxes.conf.cpu().numpy()
        cls_np    = boxes.cls.cpu().numpy().astype(int)
        y_centres = (xyxy_np[:, 1] + xyxy_np[:, 3]) / 2
        order     = np.argsort(y_centres)

        viz_image  = image.copy()
        detections = []

        for rank, idx in enumerate(order):
            x1, y1, x2, y2 = map(int, xyxy_np[idx])
            score           = float(conf_np[idx])
            cls_id          = int(cls_np[idx])
            class_name      = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
            colour          = VIZ_COLOURS.get(cls_id, (128, 128, 128))

            # ── Crop with adaptive padding ────────────────────────────
            crop, (cx1, cy1, cx2, cy2) = padded_crop(image, x1, y1, x2, y2)

            crop_fname = (
                f"{img_path.stem}"
                f"_r{rank:04d}"
                f"_{class_name}"
                f"_conf{score:.2f}.jpg"
            )
            crop_path = output_dir / "crops" / class_name / crop_fname
            cv2.imwrite(str(crop_path), crop)

            # ── Visualisation ─────────────────────────────────────────
            # Solid box = model prediction; dashed-style thin box = padded crop
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), colour, 2)
            cv2.rectangle(viz_image, (cx1, cy1), (cx2, cy2), (200, 200, 200), 1)

            label_text = f"{class_name} {score:.2f}"
            text_y     = max(y1 - 6, 14)
            # Opaque background for readability on busy pages
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(viz_image, (x1, text_y - th - 2), (x1 + tw + 2, text_y + 2), colour, -1)
            cv2.putText(
                viz_image, label_text, (x1 + 1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

            detections.append({
                "rank":         rank,
                "bbox":         [x1, y1, x2, y2],
                "padded_bbox":  [cx1, cy1, cx2, cy2],
                "class_id":     cls_id,
                "class_name":   class_name,
                "confidence":   round(score, 4),
                "crop_path":    str(crop_path.relative_to(output_dir)),
                "for_ocr":      cls_id in OCR_CLASSES,
            })

        cv2.imwrite(str(output_dir / "viz" / img_path.name), viz_image)
        print(f"  {img_path.name}: {len(detections)} detections")

        all_predictions.append({
            "file_name":  img_path.name,
            "detections": detections,
        })

    # ── Write manifest ────────────────────────────────────────────────
    manifest_path = output_dir / "predictions.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for pred in all_predictions:
            fh.write(json.dumps(pred, ensure_ascii=False) + "\n")

    total_dets = sum(len(p["detections"]) for p in all_predictions)
    print(f"\n{total_dets} total detections across {len(image_paths)} images")
    print(f"Crops  → {output_dir / 'crops'}")
    print(f"Viz    → {output_dir / 'viz'}")
    print(f"Manifest → {manifest_path}")

    return all_predictions


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    default_weights = str(ROOT / "runs" / "rtdetr-l-dla-1024" / "weights" / "best.pt")

    parser = argparse.ArgumentParser(
        description="RT-DETR inference + adaptive-padded crop for RUKOPYS DLA"
    )
    parser.add_argument("--weights", default=default_weights,
                        help="Path to trained .pt weights")
    parser.add_argument("--source",  default=str(ROOT / "test" / "images"),
                        help="Directory of test images")
    parser.add_argument("--output",  default=str(ROOT / "inference_output"),
                        help="Output directory for crops, viz, and manifest")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou",     type=float, default=0.50,
                        help="IoU threshold passed to predict() (default: 0.50)")
    parser.add_argument("--imgsz",   type=int,   default=1024,
                        help="Inference image size (default: 1024)")
    args = parser.parse_args()

    run_inference(
        model_path=args.weights,
        source_dir=args.source,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
