# DocuDecode-UA Pipeline Summary
## Kaggle: Handwritten to Data (RUKOPYS Ukrainian Dataset)

---

## PIPELINE ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT                                        │
│  RUKOPYS Dataset (HuggingFace: osunlp/RUKOPYS)                      │
│  • Train:  1,330 full document images + ground-truth regions         │
│  • Silver: 8,202 weakly-labelled images (pseudo-labels)             │
│  • Test:   385 full document images (no labels)                      │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 1 — Document Layout Analysis (DLA)            │
│                                                                      │
│  Model:       RT-DETR-L  (Ultralytics 8.4.50)                       │
│  Pretrained:  COCO weights  (rtdetr-l.pt)                            │
│  Input size:  1024 × 1024 px                                         │
│  Batch size:  16                                                     │
│  Epochs:      120  (early stop patience=25)                          │
│  Optimizer:   AdamW  lr=1e-4 → 1e-6 (cosine), weight_decay=1e-4    │
│  Precision:   fp16 mixed (amp=True)                                  │
│  Augmentation: mosaic, HSV, rotation ±3°, scale 0.5, perspective   │
│  Label smooth: 0.05                                                  │
│                                                                      │
│  Output classes (7):                                                 │
│    0: handwritten  (136,800 train instances — dominant class)        │
│    1: printed      (19,761)                                          │
│    2: formula      (21,534)                                          │
│    3: table        (638)                                             │
│    4: annotation   (4,649)                                           │
│    5: image        (539)                                             │
│    6: graph        (61  — rarest class)                              │
│                                                                      │
│  Results:                                                            │
│    mAP50 (all)      = 0.636                                          │
│    mAP50-95 (all)   = 0.371                                          │
│    handwritten mAP50 = 0.949                                         │
│    formula mAP50    = 0.885                                          │
│    table mAP50      = 0.840                                          │
│    graph mAP50      = 0.745                                          │
│    annotation mAP50 = 0.213  (rare, needs improvement)              │
│    image mAP50      = 0.223  (rare, needs improvement)              │
│                                                                      │
│  Hardware: NVIDIA H200 (144GB), ~7 hours                            │
└─────────────────────┬───────────────────────────────────────────────┘
                      │  Detected bounding boxes (sorted top→bottom)
                      │  Only for text classes: handwritten, printed, annotation
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 2 — Crop + Deskew Bridge                      │
│                                                                      │
│  File: phase2_bridge.py                                              │
│  Padding strategy: Proportional adaptive padding                     │
│    • Vertical:   +15% of box height (top + bottom)                  │
│    • Horizontal: +5%  of box width  (left + right)                  │
│    • Rationale: preserve Cyrillic ascenders (б,д,й) and descenders  │
│                                                                      │
│  Deskew:  minAreaRect on binarized foreground pixels                 │
│    • Skew threshold: 1.5° (below → skip warpAffine)                 │
│    • Min ink pixels: 30 (degenerate crops skipped)                  │
│    • Fill: white background after rotation                           │
│                                                                      │
│  Output: padded + deskewed BGR numpy crop, ready for TrOCR          │
└─────────────────────┬───────────────────────────────────────────────┘
                      │  Cleaned text-region crops
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 3 — Handwritten Text Recognition (HTR)            │
│                                                                      │
│  ── ATTEMPT 1 (from scratch, deprecated) ──────────────────────    │
│  Encoder:  google/vit-base-patch16-384                               │
│  Decoder:  youscan/ukr-roberta-base  (Ukrainian RoBERTa)            │
│  Framework: VisionEncoderDecoderModel.from_encoder_decoder_pretrained│
│  Cross-attention: randomly initialized (cold start)                  │
│  Epochs:   30   LR: 5e-5   Batch: 16 + grad_accum=2 = eff. 32     │
│  Result:   CER = 0.278  (27.8% character error rate)                │
│                                                                      │
│  ── ATTEMPT 2 (current run) ───────────────────────────────────    │
│  Base:     microsoft/trocr-large-handwritten                         │
│            (ViT-Large encoder + RoBERTa-Large decoder,              │
│             pretrained on IAM + SROIE handwritten datasets)          │
│  Framework: VisionEncoderDecoderModel.from_pretrained (fine-tuning) │
│  Special tokens added: [illegible], ~~, {, }  (RUKOPYS markers)    │
│  Epochs:   50   LR: 2e-5   Batch: 16 + grad_accum=2 = eff. 32     │
│  Max label length: 256 tokens (GPT-2 BPE, byte-level Cyrillic)     │
│  Precision: bf16  (H200 native)                                      │
│  Eval metric: CER every 500 steps                                    │
│  Checkpointing: every 500 steps, keep best 3                         │
│  Hardware: NVIDIA H200 (144GB), ~8 hours × 2 runs                  │
│  Status:   RUNNING                                                   │
└─────────────────────┬───────────────────────────────────────────────┘
                      │  Predicted Ukrainian text per region
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  INFERENCE PIPELINE                                   │
│                                                                      │
│  File: infer_pipeline.py                                             │
│  Input:  385 test images                                             │
│  DLA conf threshold: 0.25,  IoU: 0.50,  imgsz: 1024                │
│  HTR batch size: 16,  num_beams: 4,  max_new_tokens: 128           │
│  Runtime: ~8 minutes on H200                                         │
│                                                                      │
│  Output: submission.csv                                              │
│    • 385 rows (one per image)                                        │
│    • 11,832 total regions detected                                   │
│    • avg 30.7 regions per image                                      │
│    • columns: image, regions (JSON array)                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## DATASET STATS

| Split  | Images | Avg regions/image | Notes                        |
|--------|--------|-------------------|------------------------------|
| Train  | 1,330  | ~22 regions       | human-annotated              |
| Silver | 8,202  | ~8 regions        | pseudo-labelled (noisier)    |
| Test   | 385    | unknown           | no ground truth              |

**Class imbalance:** `handwritten` (138k) vs `graph` (63) = 2190:1 ratio

---

## TRAINING HARDWARE & ENVIRONMENT

- Cluster: Northeastern Explorer HPC
- GPU: NVIDIA H200 SXM5 (144GB HBM3)
- CUDA: 12.8, PyTorch: 2.6.0+cu124
- Python: 3.11.15 (conda)
- Key libs: ultralytics 8.4.50, transformers 5.8.1, datasets 4.8.5

---

## CURRENT RESULTS vs LEADERBOARD

| Model                    | CER    | Estimated Kaggle Score |
|--------------------------|--------|------------------------|
| Attempt 1 (ViT-base)     | 0.278  | ~0.65–0.72             |
| Attempt 2 (TrOCR-Large)  | TBD    | TBD (running)          |
| Leaderboard #1           | ~0.05? | 0.906                  |
| Leaderboard #3           | ~0.08? | 0.900                  |

---

## WHAT WE HAVE NOT TRIED (potential improvements)

1. **DLA**: DINO, Co-DETR, or YOLOv10-X as detector backbone
2. **DLA**: Test-time augmentation (TTA) for better box recall
3. **DLA**: Separate small model for rare classes (graph, image, table)
4. **HTR**: Beam size > 4 at inference time
5. **HTR**: CTC-based decoder instead of autoregressive
6. **HTR**: Language model re-scoring (n-gram or small LM on top)
7. **HTR**: Data augmentation: elastic distortion, random ink simulation
8. **HTR**: Ensemble of multiple HTR models
9. **Pre/Post**: Line-level segmentation instead of bounding-box crops
10. **Post**: Spell-checker / Ukrainian language model post-correction
11. **Architecture**: Florence-2 or PaddleOCR as alternative backbone
12. **Architecture**: Donut (OCR-free document understanding)
13. **Data**: Additional Ukrainian text datasets for HTR pretraining
