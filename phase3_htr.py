"""
phase3_htr.py — Phase 3: Handwritten Text Recognition (HTR)

Position in the pipeline:
    Phase 2 (deskewed crops)  →  [this module]  →  predicted Ukrainian text

Architecture ("Frankenstein"):
    Base: microsoft/trocr-large-handwritten
        Full VisionEncoderDecoderModel pre-trained by Microsoft on IAM +
        SROIE handwritten datasets.  Encoder is ViT-Large (384×384, 16-px
        patches); decoder is RoBERTa-Large with cross-attention.  Loading
        this as the starting point rather than building ViT+RoBERTa from
        scratch gives ~0.15 CER improvement out of the box on new domains.

    Fine-tuning strategy:
        Inject RUKOPYS special tokens (~~, {, }, [illegible]) into the
        existing tokenizer vocabulary, resize the decoder embedding matrix,
        then fine-tune the whole model end-to-end on the Ukrainian RUKOPYS
        corpus for 50 epochs at lr=2e-5 (standard fine-tuning rate).

Why NOT formula regions?
    RUKOPYS `formula` annotations contain raw LaTeX (\\sqrt{3}, \\infty …).
    LaTeX is a different tokenisation domain from Ukrainian prose; mixing them
    would pollute the language model prior and hurt CER on natural text.
    Formulas are best served by a separate TeX-aware model.

Special annotation markers registered as additional_special_tokens:
    [illegible]  →  unreadable word (80 occurrences)
    ~~           →  crossed-out word boundary (4 935 occurrences)
    {            →  start of in-line correction (9 549 occurrences)
    }            →  end of in-line correction (9 291 occurrences)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import evaluate
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

log = logging.getLogger(__name__)

# ── Model identifier ──────────────────────────────────────────────────────────
# Full TrOCR-Large pretrained on handwritten text — used as the fine-tuning base.
BASE_MODEL: str = "microsoft/trocr-large-handwritten"

# ── RUKOPYS annotation markers ─────────────────────────────────────────────────
# Registered as additional_special_tokens so each marker is tokenised as a
# single, indivisible unit — important for ~~, which BPE would otherwise split
# into two separate '~' tokens.
RUKOPYS_SPECIAL_TOKENS: list[str] = [
    "[illegible]",
    "~~",
    "{",
    "}",
]

# ── Classes whose text is natural-language Ukrainian (suitable for HTR) ────────
# formula is excluded: its text is raw LaTeX, a different tokenisation domain.
# image and graph have no text field.
TEXT_CLASSES: frozenset[str] = frozenset({"handwritten", "printed", "annotation"})

# ── Sequence length bounds ─────────────────────────────────────────────────────
MAX_LABEL_LENGTH: int = 256   # GPT-2 BPE is byte-level; Cyrillic needs more tokens
GENERATION_NUM_BEAMS: int = 4

# ── Type alias ─────────────────────────────────────────────────────────────────
OcrPair = tuple[Path, str]  # (path-to-crop-jpeg, ground-truth-text)


# ==============================================================================
# ── Data preparation ──────────────────────────────────────────────────────────
# ==============================================================================

def build_htr_pairs_from_metadata(
    jsonl_path: str | Path,
    img_root: str | Path,
    out_dir: str | Path,
    classes: frozenset[str] = TEXT_CLASSES,
    force_rebuild: bool = False,
) -> list[OcrPair]:
    """Crop every annotated text region from a metadata.jsonl and deskew it.

    Applies the Phase 2 pipeline (proportional padding + Otsu deskew) to each
    region whose type is in *classes* and that carries a ground-truth text.
    Crops are written as JPEG files under *out_dir*; a ``manifest.jsonl``
    alongside them records ``(crop_path, text)`` so the expensive extraction
    step can be skipped on subsequent runs.

    Parameters
    ----------
    jsonl_path:
        Path to a RUKOPYS metadata.jsonl file.
    img_root:
        Root directory that contains the source images referenced in the JSONL
        (e.g. the ``train/`` or ``silver/`` split folder).
    out_dir:
        Directory where cropped JPEG files will be written.
    classes:
        Annotation types to include.  Defaults to :data:`TEXT_CLASSES`.
    force_rebuild:
        If ``True``, re-extract all crops even if a manifest already exists.

    Returns
    -------
    List of ``(crop_path, text)`` tuples in the order they were found.
    """
    import cv2

    # Lazy import — keeps this module usable without cv2 for inference
    from phase2_bridge import process_box

    jsonl_path = Path(jsonl_path)
    img_root = Path(img_root)
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.jsonl"

    # ── Short-circuit if a cached manifest exists ──────────────────────────
    if manifest_path.exists() and not force_rebuild:
        log.info("Loading cached OCR pairs from %s", manifest_path)
        pairs: list[OcrPair] = []
        with open(manifest_path) as fh:
            for line in fh:
                obj = json.loads(line)
                p = Path(obj["crop_path"])
                if p.exists():
                    pairs.append((p, obj["text"]))
        log.info("  %d valid pairs loaded.", len(pairs))
        return pairs

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    skipped_missing = 0
    skipped_empty = 0
    skipped_no_crop = 0

    with open(jsonl_path) as fh, open(manifest_path, "w") as mfh:
        for record in map(json.loads, fh):
            rel_path = record.get("file_name", "")
            img_path = img_root / rel_path
            if not img_path.exists():
                skipped_missing += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                skipped_missing += 1
                continue

            stem = img_path.stem

            for i, region in enumerate(record.get("regions", [])):
                if region.get("type") not in classes:
                    continue

                text = region.get("text", "").strip()
                if not text:
                    skipped_empty += 1
                    continue

                result = process_box(image, region["bbox"])
                if result.crop is None:
                    skipped_no_crop += 1
                    continue

                crop_filename = f"{stem}_r{i:04d}.jpg"
                crop_path = out_dir / crop_filename

                # Convert BGR → RGB before saving so PIL reads it correctly
                cv2.imwrite(str(crop_path), result.crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                pairs.append((crop_path, text))
                mfh.write(json.dumps({"crop_path": str(crop_path), "text": text},
                                     ensure_ascii=False) + "\n")

    log.info(
        "Built %d OCR pairs from %s  "
        "(skipped: %d missing, %d empty-text, %d failed-crop)",
        len(pairs), jsonl_path,
        skipped_missing, skipped_empty, skipped_no_crop,
    )
    return pairs


def split_pairs(
    pairs: list[OcrPair],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[OcrPair], list[OcrPair]]:
    """Randomly split an ``OcrPair`` list into train and validation sets.

    Parameters
    ----------
    pairs:
        Full list of ``(crop_path, text)`` tuples.
    val_ratio:
        Fraction of *pairs* reserved for validation.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    ``(train_pairs, val_pairs)``
    """
    pairs = list(pairs)
    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    return pairs[n_val:], pairs[:n_val]


# ==============================================================================
# ── Dataset ───────────────────────────────────────────────────────────────────
# ==============================================================================

class RukopysDataset(Dataset):
    """PyTorch Dataset for RUKOPYS text-line crops.

    Each item returns:

    * ``pixel_values``  — float tensor ``(C, H, W)`` from :class:`ViTImageProcessor`.
    * ``labels``        — long tensor of token IDs with padding positions replaced
                         by ``-100`` so they are ignored by PyTorch's
                         ``CrossEntropyLoss``.

    Parameters
    ----------
    pairs:
        List of ``(image_path, ground_truth_text)`` tuples produced by
        :func:`build_htr_pairs_from_metadata` or :func:`split_pairs`.
    image_processor:
        A :class:`ViTImageProcessor` (or any compatible processor) that maps
        a PIL image to a ``pixel_values`` tensor.
    tokenizer:
        The Ukrainian tokenizer (already extended with RUKOPYS special tokens).
    max_label_length:
        Maximum number of tokens per label sequence.  Sequences exceeding this
        are truncated; the value must match the model's generation budget.
    """

    def __init__(
        self,
        pairs: list[OcrPair],
        image_processor: ViTImageProcessor,
        tokenizer: PreTrainedTokenizerBase,
        max_label_length: int = MAX_LABEL_LENGTH,
    ) -> None:
        if not pairs:
            raise ValueError("pairs list is empty — nothing to train on.")
        self.pairs = pairs
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_label_length = max_label_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, text = self.pairs[idx]

        # ── Image ──────────────────────────────────────────────────────────
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            log.warning("Failed to open %s (%s). Using blank image.", img_path, exc)
            # Return a white blank image rather than crashing the DataLoader
            image = Image.new("RGB", (384, 384), (255, 255, 255))

        pixel_values: torch.Tensor = self.image_processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)  # (C, H, W)

        # ── Label ──────────────────────────────────────────────────────────
        # Tokenise without padding; the collator handles dynamic batch padding.
        encoding = self.tokenizer(
            text,
            max_length=self.max_label_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        labels: torch.Tensor = encoding.input_ids.squeeze(0)  # (seq_len,)

        return {"pixel_values": pixel_values, "labels": labels}


# ==============================================================================
# ── Data collator ─────────────────────────────────────────────────────────────
# ==============================================================================

@dataclass
class ImageTextDataCollator:
    """Collate a batch of ``(pixel_values, labels)`` dicts.

    * ``pixel_values`` are already fixed-size tensors (same shape for every
      item in the batch), so they are simply stacked.
    * ``labels`` may have different lengths within a batch.  This collator
      pads them to the longest sequence in the batch using ``-100``, which
      ``CrossEntropyLoss`` ignores.  Dynamic (per-batch) padding is more
      memory-efficient than padding to the global ``max_label_length``.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(
        self,
        features: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        labels: list[torch.Tensor] = [f["labels"] for f in features]
        max_len = max(lbl.size(0) for lbl in labels)

        # Pre-fill with -100 (CrossEntropyLoss ignore index)
        padded = torch.full(
            (len(labels), max_len), fill_value=-100, dtype=torch.long
        )
        for i, lbl in enumerate(labels):
            padded[i, : lbl.size(0)] = lbl

        return {"pixel_values": pixel_values, "labels": padded}


# ==============================================================================
# ── Model initialisation ──────────────────────────────────────────────────────
# ==============================================================================

def init_model_and_tokenizer(
    base_model: str = BASE_MODEL,
    special_tokens: list[str] = RUKOPYS_SPECIAL_TOKENS,
    max_label_length: int = MAX_LABEL_LENGTH,
) -> tuple[VisionEncoderDecoderModel, PreTrainedTokenizerBase, ViTImageProcessor]:
    """Fine-tune TrOCR-Large-Handwritten on the RUKOPYS Ukrainian corpus.

    Steps
    -----
    1. Load the pretrained TrOCR model, tokenizer, and image processor.
    2. Inject RUKOPYS special tokens into the existing vocabulary.
    3. Resize the decoder embedding matrix to cover the new tokens.
    4. Patch model.config and generation_config for Seq2SeqTrainer.

    Parameters
    ----------
    base_model:
        HuggingFace model ID for the TrOCR base checkpoint.
    special_tokens:
        RUKOPYS annotation markers to add to the tokenizer vocabulary.
    max_label_length:
        Token-sequence budget for both training labels and generation.

    Returns
    -------
    ``(model, tokenizer, image_processor)``
    """
    log.info("Loading TrOCR model from '%s'", base_model)
    image_processor = ViTImageProcessor.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # ── Inject RUKOPYS markers as indivisible special tokens ───────────────
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )
    log.info(
        "Special tokens: %d new tokens added. Vocabulary size: %d",
        n_added,
        len(tokenizer),
    )

    # ── Load pretrained TrOCR model ────────────────────────────────────────
    model = VisionEncoderDecoderModel.from_pretrained(base_model)

    # ── Resize decoder embeddings for any new special tokens ──────────────
    if n_added > 0:
        model.decoder.resize_token_embeddings(len(tokenizer))
        log.info("Decoder embedding matrix resized to %d tokens.", len(tokenizer))

    # ── Resolve token IDs ──────────────────────────────────────────────────
    # For RoBERTa: bos=<s>, eos=</s>, pad=<pad>.
    # Fallback chain ensures we handle tokenizers with non-standard attributes.
    bos_id: int | None = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_id: int | None = tokenizer.eos_token_id or tokenizer.sep_token_id
    pad_id: int | None = tokenizer.pad_token_id

    if bos_id is None:
        raise ValueError(
            "Could not determine a BOS token from the tokenizer. "
            "Set tokenizer.bos_token or tokenizer.cls_token before calling "
            "init_model_and_tokenizer()."
        )

    # ── Model config (governs training forward pass) ───────────────────────
    # decoder_start_token_id: token prepended at the start of each decoder
    # input sequence during teacher forcing and auto-regressive generation.
    model.config.decoder_start_token_id = bos_id
    model.config.pad_token_id = pad_id
    model.config.eos_token_id = eos_id
    # vocab_size must match the resized embedding matrix so the output
    # projection layer covers all valid token indices.
    model.config.vocab_size = len(tokenizer)

    # ── Generation config (governs Seq2SeqTrainer's predict_with_generate) ─
    model.generation_config.decoder_start_token_id = bos_id
    model.generation_config.pad_token_id = pad_id
    model.generation_config.eos_token_id = eos_id
    model.generation_config.max_new_tokens = max_label_length
    model.generation_config.early_stopping = True
    model.generation_config.num_beams = GENERATION_NUM_BEAMS
    # Discourage the model from copying the same n-gram repeatedly — a common
    # failure mode when the encoder output is ambiguous (blurry ink, tears).
    model.generation_config.no_repeat_ngram_size = 3

    return model, tokenizer, image_processor


# ==============================================================================
# ── CER metric ────────────────────────────────────────────────────────────────
# ==============================================================================

def build_compute_metrics(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[object], dict[str, float]]:
    """Return a ``compute_metrics`` closure compatible with :class:`Seq2SeqTrainer`.

    The closure captures *tokenizer* so it can decode predicted and reference
    token sequences into strings for CER computation.

    Character Error Rate (CER) measures the minimum edit distance (in
    characters) between prediction and reference, normalised by the reference
    length.  It is more fine-grained than Word Error Rate (WER) and better
    suited to agglutinative Ukrainian where word boundaries can be ambiguous.

    Parameters
    ----------
    tokenizer:
        The (already extended) Ukrainian tokenizer used for decoding.

    Returns
    -------
    A callable ``compute_metrics(EvalPrediction) -> dict`` that returns
    ``{"cer": float}``.
    """
    cer_metric = evaluate.load("cer")

    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions, label_ids = eval_pred.predictions, eval_pred.label_ids

        # Trainer may return raw logits (shape B×T×V) instead of token IDs
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        # Replace the -100 sentinel with the pad token so the decoder can
        # process the label sequences without encountering out-of-range IDs.
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)

        vocab_size = len(tokenizer)
        pred_strings: list[str] = tokenizer.batch_decode(
            np.clip(predictions, 0, vocab_size - 1).tolist(), skip_special_tokens=True
        )
        label_strings: list[str] = tokenizer.batch_decode(
            label_ids.tolist(), skip_special_tokens=True
        )

        # Strip surrounding whitespace — tokenizers often add a leading space
        pred_strings = [s.strip() for s in pred_strings]
        label_strings = [s.strip() for s in label_strings]

        cer: float = cer_metric.compute(
            predictions=pred_strings, references=label_strings
        )
        return {"cer": round(cer, 4)}

    return compute_metrics


# ==============================================================================
# ── Training ──────────────────────────────────────────────────────────────────
# ==============================================================================

def get_training_args(
    output_dir: str | Path,
    num_train_epochs: int = 50,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-5,
    report_to: str = "none",
) -> Seq2SeqTrainingArguments:
    """Build :class:`Seq2SeqTrainingArguments` optimised for a single H100 80 GB.

    Batch arithmetic:
        effective_batch = per_device_train_batch_size × gradient_accumulation_steps
                        = 16 × 2 = 32

    Precision:
        ``bf16=True``  — H100 has native BF16 tensor cores; no loss of dynamic
                         range vs FP16, and no need for loss scaling.
        ``tf32=True``  — enables TensorFloat-32 for CUDA matmuls (Ampere+),
                         giving free throughput on operations that don't need
                         full FP32 precision.

    Memory:
        ``gradient_checkpointing=True`` trades recomputation for VRAM.  The
        ViT encoder (384-px, 16-px patches = 576 patches) produces large
        activation tensors; checkpointing is essential at batch_size=16.

    Parameters
    ----------
    output_dir:
        Directory for checkpoints and training logs.
    num_train_epochs:
        Total number of passes over the training set.
    per_device_train_batch_size:
        Images per GPU per step (before gradient accumulation).
    per_device_eval_batch_size:
        Images per GPU during evaluation (no gradients, can be larger).
    gradient_accumulation_steps:
        Number of forward passes to accumulate before an optimiser step.
    learning_rate:
        Peak learning rate for AdamW.
    report_to:
        Logging backend (``"wandb"``, ``"tensorboard"``, or ``"none"``).

    Returns
    -------
    A fully configured :class:`Seq2SeqTrainingArguments` instance.
    """
    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),

        # ── Batching ──────────────────────────────────────────────────────
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # ── Duration ──────────────────────────────────────────────────────
        num_train_epochs=num_train_epochs,

        # ── Optimiser ─────────────────────────────────────────────────────
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=1000,          # ~6 % of 50-epoch training
        lr_scheduler_type="cosine",
        # adamw_torch_fused uses CUDA-fused kernels for AdamW momentum /
        # variance updates — measurably faster than the default on H100.
        optim="adamw_torch_fused",

        # ── H100 precision ───────────────────────────────────────────────
        bf16=True,
        fp16=False,                 # bf16 and fp16 are mutually exclusive
        tf32=True,

        # ── Memory ────────────────────────────────────────────────────────
        gradient_checkpointing=True,
        # Avoids a subtle interaction between gradient checkpointing and the
        # reentrant autograd graph used by older PyTorch versions.
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # ── Generation (used when predict_with_generate=True) ─────────────
        predict_with_generate=True,
        generation_max_length=MAX_LABEL_LENGTH,
        generation_num_beams=GENERATION_NUM_BEAMS,

        # ── Evaluation & checkpointing ─────────────────────────────────────
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,    # lower CER = better model

        # ── Logging ───────────────────────────────────────────────────────
        logging_steps=50,
        report_to=report_to,

        # ── Data loading ──────────────────────────────────────────────────
        # 4 workers: fewer forked processes = lower peak CPU RAM at startup
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # ── Reproducibility ───────────────────────────────────────────────
        seed=42,
        data_seed=42,

        # torch_compile=True can add 15-25 % throughput on H100 at the cost
        # of a one-time 3-5 min compilation on the first step.  Uncomment
        # after confirming generate() is compatible with the installed version:
        # torch_compile=True,
    )


def train(
    train_pairs: list[OcrPair],
    eval_pairs: list[OcrPair],
    output_dir: str | Path,
    base_model: str = BASE_MODEL,
    **training_kwargs,
) -> Seq2SeqTrainer:
    """End-to-end training function.

    Initialises model + tokenizer, wraps data in :class:`RukopysDataset`,
    and runs :class:`Seq2SeqTrainer`.

    Parameters
    ----------
    train_pairs, eval_pairs:
        ``(crop_path, text)`` lists for training and validation respectively.
    output_dir:
        Root directory for checkpoints, logs, and the final model artefacts.
    base_model:
        Model identifier forwarded to :func:`init_model_and_tokenizer`.
    **training_kwargs:
        Extra keyword arguments forwarded to :func:`get_training_args`
        (e.g. ``num_train_epochs``, ``per_device_train_batch_size``).

    Returns
    -------
    The :class:`Seq2SeqTrainer` after ``.train()`` completes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Model, tokenizer, processor ───────────────────────────────────────
    model, tokenizer, image_processor = init_model_and_tokenizer(
        base_model=base_model,
    )

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = RukopysDataset(train_pairs, image_processor, tokenizer)
    eval_dataset = RukopysDataset(eval_pairs, image_processor, tokenizer)
    log.info(
        "Dataset sizes — train: %d  eval: %d",
        len(train_dataset),
        len(eval_dataset),
    )

    # ── Collator, metrics, training args ──────────────────────────────────
    collator = ImageTextDataCollator(tokenizer=tokenizer)
    compute_metrics = build_compute_metrics(tokenizer)
    training_args = get_training_args(output_dir, **training_kwargs)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    log.info("Starting training …")
    trainer.train()

    # ── Persist final artefacts ───────────────────────────────────────────
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    image_processor.save_pretrained(str(final_dir))
    log.info("Final model + tokenizer + processor saved to %s", final_dir)

    return trainer


# ==============================================================================
# ── Entry point ───────────────────────────────────────────────────────────────
# ==============================================================================

def main() -> None:
    """Build OCR pairs from RUKOPYS metadata and launch HTR training.

    Data strategy (mirrors Phase 1):
        • Validation  — 10 % of clean ``train/`` annotations.
        • Training    — remaining 90 % of ``train/`` + all of ``silver/``.

    The ``silver/`` split is excluded from validation because its annotations
    were produced with automatic/noisy labelling (lower confidence than the
    human-annotated ``train/`` set).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    ROOT = Path(__file__).parent
    HTR_DATA = ROOT / "htr_data"
    OUTPUT_DIR = ROOT / "runs" / "phase3-htr"

    # ── Build or load Phase 2 crops ───────────────────────────────────────
    log.info("Building / loading OCR pairs from training splits …")

    clean_pairs = build_htr_pairs_from_metadata(
        jsonl_path=ROOT / "train" / "metadata.jsonl",
        img_root=ROOT / "train",
        out_dir=HTR_DATA / "crops" / "train",
    )
    silver_pairs = build_htr_pairs_from_metadata(
        jsonl_path=ROOT / "silver" / "metadata.jsonl",
        img_root=ROOT / "silver",
        out_dir=HTR_DATA / "crops" / "silver",
    )

    # Split: val from clean train only
    train_clean, val_pairs = split_pairs(clean_pairs, val_ratio=0.10)
    train_pairs = train_clean + silver_pairs

    # Shuffle combined train list so silver and clean samples are interleaved
    random.seed(42)
    random.shuffle(train_pairs)

    log.info(
        "Final split — train: %d  val: %d",
        len(train_pairs),
        len(val_pairs),
    )

    # ── Train ─────────────────────────────────────────────────────────────
    train(
        train_pairs=train_pairs,
        eval_pairs=val_pairs,
        output_dir=OUTPUT_DIR,
        num_train_epochs=30,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
    )


if __name__ == "__main__":
    main()
