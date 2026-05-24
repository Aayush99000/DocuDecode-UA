"""
infer_pipeline.py — End-to-end DocuDecode-UA inference pipeline.

Chains:
    RT-DETR (DLA)  →  phase2_bridge (crop + deskew)  →  HTR (TrOCR)  →  submission JSONL

Usage:
    python infer_pipeline.py
    python infer_pipeline.py --test-dir test --dla-weights runs/rtdetr-l-dla-1024/weights/best.pt \
        --htr-model runs/phase3-htr/final --output submission.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
from ultralytics import RTDETR

from phase2_bridge import process_box

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent

# Classes that produce readable text → run HTR
TEXT_CLASSES = {"handwritten", "printed", "annotation"}

CLASS_NAMES = {
    0: "handwritten",
    1: "printed",
    2: "formula",
    3: "table",
    4: "annotation",
    5: "image",
    6: "graph",
}


def load_htr_model(model_dir: str, device: torch.device):
    log.info("Loading HTR model from %s", model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    image_processor = ViTImageProcessor.from_pretrained(model_dir)
    return model, tokenizer, image_processor


def run_htr_batch(
    crops: list[np.ndarray],
    model: VisionEncoderDecoderModel,
    tokenizer: AutoTokenizer,
    image_processor: ViTImageProcessor,
    device: torch.device,
    batch_size: int = 16,
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> list[str]:
    results = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        pil_imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in batch]
        pixel_values = image_processor(
            pil_imgs, return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
        results.extend([t.strip() for t in texts])
    return results


def run_pipeline(
    test_dir: str,
    dla_weights: str,
    htr_model_dir: str,
    output_path: str,
    conf: float = 0.25,
    imgsz: int = 1024,
    htr_batch_size: int = 16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load models ───────────────────────────────────────────────────────────
    log.info("Loading DLA model from %s", dla_weights)
    dla_model = RTDETR(dla_weights)

    htr_model, tokenizer, image_processor = load_htr_model(htr_model_dir, device)

    # ── Load test metadata ────────────────────────────────────────────────────
    test_dir = Path(test_dir)
    meta_path = test_dir / "metadata.jsonl"
    test_records = []
    with open(meta_path, encoding="utf-8") as fh:
        for line in fh:
            test_records.append(json.loads(line))
    log.info("Found %d test images", len(test_records))

    submission = []

    for record in tqdm(test_records, desc="Inference"):
        img_path = test_dir / record["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            log.warning("Cannot read %s — skipping", img_path)
            submission.append({"file_name": record["file_name"], "regions": []})
            continue

        # ── Phase 1: DLA ──────────────────────────────────────────────────────
        result = dla_model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=0.5,
            device=0 if device.type == "cuda" else "cpu",
            verbose=False,
            save=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            submission.append({"file_name": record["file_name"], "regions": []})
            continue

        xyxy_np = boxes.xyxy.cpu().numpy()
        conf_np = boxes.conf.cpu().numpy()
        cls_np  = boxes.cls.cpu().numpy().astype(int)

        # Sort top-to-bottom (reading order)
        y_centres = (xyxy_np[:, 1] + xyxy_np[:, 3]) / 2
        order = np.argsort(y_centres)

        regions_meta = []   # (bbox, class_name, needs_htr)
        ocr_crops    = []   # numpy BGR crops that need HTR
        ocr_indices  = []   # which region index each crop belongs to

        for idx in order:
            x1, y1, x2, y2 = map(int, xyxy_np[idx])
            cls_id     = int(cls_np[idx])
            class_name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
            needs_htr  = class_name in TEXT_CLASSES

            regions_meta.append({
                "bbox":       [x1, y1, x2, y2],
                "type":       class_name,
                "confidence": round(float(conf_np[idx]), 4),
                "text":       "",
            })

            if needs_htr:
                # Phase 2: pad + crop + deskew via phase2_bridge
                crop_result = process_box(image, [x1, y1, x2, y2])
                if crop_result.crop is not None:
                    ocr_crops.append(crop_result.crop)
                    ocr_indices.append(len(regions_meta) - 1)

        # ── Phase 3: HTR ──────────────────────────────────────────────────────
        if ocr_crops:
            texts = run_htr_batch(
                ocr_crops, htr_model, tokenizer, image_processor,
                device, batch_size=htr_batch_size,
            )
            for reg_idx, text in zip(ocr_indices, texts):
                regions_meta[reg_idx]["text"] = text

        # Build output regions (drop confidence, not in submission format)
        out_regions = [
            {
                "bbox":  r["bbox"],
                "type":  r["type"],
                "text":  r["text"],
            }
            for r in regions_meta
        ]

        submission.append({
            "file_name": record["file_name"],
            "regions":   out_regions,
        })

    # ── Write submission ──────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in submission:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_regions = sum(len(r["regions"]) for r in submission)
    log.info("Done. %d images, %d regions → %s", len(submission), total_regions, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir",    default=str(ROOT / "test"))
    parser.add_argument("--dla-weights", default=str(ROOT / "runs/rtdetr-l-dla-1024/weights/best.pt"))
    parser.add_argument("--htr-model",   default=str(ROOT / "runs/phase3-htr/final"))
    parser.add_argument("--output",      default=str(ROOT / "submission.jsonl"))
    parser.add_argument("--conf",        type=float, default=0.25)
    parser.add_argument("--imgsz",       type=int,   default=1024)
    parser.add_argument("--htr-batch",   type=int,   default=16)
    args = parser.parse_args()

    run_pipeline(
        test_dir=args.test_dir,
        dla_weights=args.dla_weights,
        htr_model_dir=args.htr_model,
        output_path=args.output,
        conf=args.conf,
        imgsz=args.imgsz,
        htr_batch_size=args.htr_batch,
    )


if __name__ == "__main__":
    main()
