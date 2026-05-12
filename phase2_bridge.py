"""
phase2_bridge.py — Phase 2 bridge module for the DocuDecode-UA OCR pipeline.

Position in the pipeline:
    Phase 1 (RT-DETR)  →  [this module]  →  Phase 3 (TrOCR)

Responsibilities:
    1. Apply proportional padding to raw AABB detections so cursive
       ascenders/descenders (б, д, у, з …) are not clipped.
    2. Extract the padded crop safely from the source image array.
    3. Deskew the crop so the text baseline is horizontal before the
       Vision Transformer encoder sees it. TrOCR's positional embeddings
       assume horizontal text; skewed input degrades CER significantly.

All functions are stateless and operate purely on NumPy arrays so the
module slots into any data-loading or batching strategy without side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

#: Proportional padding fractions applied to each detected box.
DEFAULT_H_PAD: Final[float] = 0.15   # 15% of box height → top and bottom
DEFAULT_W_PAD: Final[float] = 0.05   # 5%  of box width  → left and right

#: Minimum skew magnitude (degrees) before deskewing is applied.
#: Below this threshold the warpAffine cost is not justified.
DEFAULT_ANGLE_THRESHOLD: Final[float] = 1.5

#: Minimum number of foreground (ink) pixels required to estimate a reliable
#: skew angle via minAreaRect. Fewer pixels produce degenerate rectangles.
MIN_INK_PIXELS: Final[int] = 30

#: White fill colour used when warpAffine expands the canvas corners.
WHITE: Final[tuple[int, int, int]] = (255, 255, 255)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class CropResult:
    """Return value from :func:`process_box`.

    Attributes
    ----------
    crop:
        The processed crop (padded + deskewed) as a BGR NumPy array,
        ready to be passed to TrOCR's image processor.  ``None`` when
        the box is invalid or produces an empty crop.
    skew_angle:
        The detected skew angle in degrees before correction.
        Positive = counter-clockwise tilt, negative = clockwise tilt.
        ``0.0`` if deskewing was skipped or could not be computed.
    was_deskewed:
        ``True`` if an affine rotation was actually applied.
    padded_box:
        The clamped ``[x1, y1, x2, y2]`` pixel coordinates that were
        extracted from the source image (after padding and clamping).
    """
    crop: np.ndarray | None
    skew_angle: float = 0.0
    was_deskewed: bool = False
    padded_box: list[int] = field(default_factory=list)


# ── Step 1 — Dynamic, proportional padding ────────────────────────────────────

def get_padded_coords(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_h: int,
    img_w: int,
    h_pad_frac: float = DEFAULT_H_PAD,
    w_pad_frac: float = DEFAULT_W_PAD,
) -> tuple[int, int, int, int] | None:
    """Expand an AABB by proportional margins and clamp to image bounds.

    Padding is calculated relative to the box's own dimensions so that
    small text-line crops get a smaller absolute margin than large paragraph
    blocks — avoiding over-padding that would drown the text in whitespace.

    Parameters
    ----------
    x1, y1, x2, y2:
        Raw detector output in pixel coordinates (top-left origin).
    img_h, img_w:
        Height and width of the source image in pixels.
    h_pad_frac:
        Fraction of box height to add to **each** of the top and bottom edges.
    w_pad_frac:
        Fraction of box width to add to **each** of the left and right edges.

    Returns
    -------
    ``(cx1, cy1, cx2, cy2)`` — clamped padded coordinates, or ``None`` if the
    input box is degenerate (zero area or entirely outside the image).
    """
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        log.warning("Degenerate box [%d,%d,%d,%d] — skipping.", x1, y1, x2, y2)
        return None

    h_pad = int(round(box_h * h_pad_frac))
    w_pad = int(round(box_w * w_pad_frac))

    cx1 = max(0,     x1 - w_pad)
    cy1 = max(0,     y1 - h_pad)
    cx2 = min(img_w, x2 + w_pad)
    cy2 = min(img_h, y2 + h_pad)

    if cx2 <= cx1 or cy2 <= cy1:
        log.warning(
            "Box [%d,%d,%d,%d] is outside or on the image boundary after clamping.",
            x1, y1, x2, y2,
        )
        return None

    return cx1, cy1, cx2, cy2


# ── Step 2 — Crop extraction ───────────────────────────────────────────────────

def extract_crop(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray | None:
    """Slice a region from a NumPy image array.

    Parameters
    ----------
    image:
        Source image as a ``(H, W)`` or ``(H, W, C)`` uint8 array.
    x1, y1, x2, y2:
        Pixel coordinates (already validated and clamped).

    Returns
    -------
    A contiguous copy of the slice, or ``None`` if the slice is empty.
    """
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Return a copy so the caller can modify the crop freely without
    # mutating the source image array.
    return np.ascontiguousarray(crop)


# ── Step 3 — Algorithmic deskewing ────────────────────────────────────────────

def _estimate_skew_angle(binary: np.ndarray) -> float | None:
    """Estimate the skew angle of ink strokes using ``cv2.minAreaRect``.

    Parameters
    ----------
    binary:
        A binarized single-channel image where ink pixels are **255**
        (i.e., produced with ``THRESH_BINARY_INV``).

    Returns
    -------
    Skew angle in degrees (approximately in ``(-45, 45)``), or ``None``
    if there are too few ink pixels to compute a reliable estimate.

    Notes — OpenCV angle convention
    --------------------------------
    ``cv2.minAreaRect`` returns an angle in ``(-90, 0]`` degrees, defined
    as the angle between the x-axis and the first (shorter) side of the
    bounding rectangle.

    For a landscape rectangle (``w >= h``, i.e. a text line):
        The angle directly represents the tilt from horizontal.

    For a portrait rectangle (``w < h``, i.e. a very steep region):
        Adding 90° converts it to the equivalent landscape interpretation.

    After this correction the angle is in ``(-45, 45]``; positive values
    indicate a counter-clockwise tilt and negative a clockwise tilt.
    """
    pts = cv2.findNonZero(binary)
    if pts is None or len(pts) < MIN_INK_PIXELS:
        return None

    # minAreaRect requires float32 input
    _center, (w, h), angle = cv2.minAreaRect(pts.astype(np.float32))

    # Correct for OpenCV's portrait-rectangle convention
    if w < h:
        angle += 90.0

    return float(angle)


def _rotate_image(
    image: np.ndarray,
    angle_deg: float,
    fill_color: tuple[int, ...] = WHITE,
) -> np.ndarray:
    """Rotate *image* by *angle_deg* degrees (CCW) expanding the canvas to
    prevent corner clipping.

    Parameters
    ----------
    image:
        BGR or grayscale uint8 array.
    angle_deg:
        Rotation angle in degrees.  Positive = counter-clockwise.
    fill_color:
        Background fill for the newly exposed corner triangles.

    Returns
    -------
    Rotated image with expanded canvas.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # New canvas dimensions that fully contain the rotated rectangle
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(w * sin_a + h * cos_a)

    # Shift the rotation centre to the new canvas centre
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_color,
    )


def deskew_crop(
    crop: np.ndarray,
    angle_threshold: float = DEFAULT_ANGLE_THRESHOLD,
) -> tuple[np.ndarray, float, bool]:
    """Detect and correct the skew of a text-line crop.

    Algorithm
    ---------
    1. Convert to grayscale.
    2. Binarize with Otsu's method (``THRESH_BINARY_INV``) so ink = 255.
    3. Locate all ink-pixel coordinates with ``findNonZero``.
    4. Fit a minimum-area rectangle to them via ``minAreaRect`` to find the
       dominant orientation angle.
    5. If the angle exceeds *angle_threshold*, apply an affine rotation that
       brings the text baseline to horizontal, filling exposed corners white.

    Parameters
    ----------
    crop:
        BGR or grayscale uint8 array of the padded text-line region.
    angle_threshold:
        Minimum absolute angle (degrees) below which deskewing is skipped.
        Very small angles produce negligible improvement but carry the cost
        of bilinear interpolation artefacts.

    Returns
    -------
    ``(result_crop, skew_angle, was_deskewed)``

    - *result_crop*: the (possibly rotated) crop, same dtype as input.
    - *skew_angle*: detected tilt in degrees; ``0.0`` if undetermined.
    - *was_deskewed*: whether the affine rotation was applied.

    Edge cases
    ----------
    - Empty array → returned unchanged, angle 0.0, not deskewed.
    - All-white (no ink) after binarisation → returned unchanged.
    - Fewer than ``MIN_INK_PIXELS`` ink pixels → skew unreliable, skip.
    """
    if crop is None or crop.size == 0:
        return crop, 0.0, False

    # ── Grayscale conversion ───────────────────────────────────────────────
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()

    # ── Otsu binarisation — ink becomes 255, paper becomes 0 ──────────────
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if cv2.countNonZero(binary) < MIN_INK_PIXELS:
        log.debug("deskew_crop: fewer than %d ink pixels — skipping.", MIN_INK_PIXELS)
        return crop, 0.0, False

    # ── Skew estimation ────────────────────────────────────────────────────
    angle = _estimate_skew_angle(binary)
    if angle is None:
        return crop, 0.0, False

    if abs(angle) <= angle_threshold:
        log.debug("deskew_crop: angle %.2f° within threshold — skipping.", angle)
        return crop, angle, False

    # ── Apply corrective rotation ──────────────────────────────────────────
    # A positive angle from _estimate_skew_angle means CCW tilt.
    # Rotating by -angle (CW) cancels the tilt and returns the line horizontal.
    log.debug("deskew_crop: applying %.2f° correction.", -angle)
    rotated = _rotate_image(crop, -angle, fill_color=WHITE)

    return rotated, angle, True


# ── Orchestration ──────────────────────────────────────────────────────────────

def process_box(
    image: np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    h_pad_frac: float = DEFAULT_H_PAD,
    w_pad_frac: float = DEFAULT_W_PAD,
    angle_threshold: float = DEFAULT_ANGLE_THRESHOLD,
) -> CropResult:
    """Full Phase 2 pipeline for a single bounding box.

    Applies padding → extraction → deskewing in sequence and returns a
    :class:`CropResult` regardless of failure mode (failed stages produce
    ``crop=None`` with contextual log messages).

    Parameters
    ----------
    image:
        Full source document image as a BGR uint8 NumPy array.
    box:
        ``[x1, y1, x2, y2]`` in pixel coordinates (top-left origin),
        as output by the Phase 1 RT-DETR detector.
    h_pad_frac:
        Vertical padding fraction (see :func:`get_padded_coords`).
    w_pad_frac:
        Horizontal padding fraction (see :func:`get_padded_coords`).
    angle_threshold:
        Minimum skew to trigger deskewing (see :func:`deskew_crop`).

    Returns
    -------
    :class:`CropResult` with the processed crop ready for TrOCR, or
    ``crop=None`` if any stage failed.
    """
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)

    # Step 1 — pad
    padded = get_padded_coords(x1, y1, x2, y2, img_h, img_w, h_pad_frac, w_pad_frac)
    if padded is None:
        return CropResult(crop=None)
    cx1, cy1, cx2, cy2 = padded

    # Step 2 — extract
    raw_crop = extract_crop(image, cx1, cy1, cx2, cy2)
    if raw_crop is None:
        log.warning("Empty crop from box [%d,%d,%d,%d].", x1, y1, x2, y2)
        return CropResult(crop=None, padded_box=list(padded))

    # Step 3 — deskew
    final_crop, angle, was_deskewed = deskew_crop(raw_crop, angle_threshold)

    return CropResult(
        crop=final_crop,
        skew_angle=angle,
        was_deskewed=was_deskewed,
        padded_box=list(padded),
    )


def process_document(
    image: np.ndarray,
    boxes: list[list[int]],
    h_pad_frac: float = DEFAULT_H_PAD,
    w_pad_frac: float = DEFAULT_W_PAD,
    angle_threshold: float = DEFAULT_ANGLE_THRESHOLD,
) -> list[CropResult]:
    """Process all Phase 1 detections from a single document image.

    Calls :func:`process_box` for each detection and returns results in the
    same order as *boxes*.  Boxes that fail (degenerate, out-of-bounds, empty)
    produce ``CropResult(crop=None)`` entries so index alignment with the
    original detection list is preserved.

    Parameters
    ----------
    image:
        Full document image, BGR uint8.
    boxes:
        List of ``[x1, y1, x2, y2]`` detections from Phase 1, sorted
        top-to-bottom by the Phase 1 inference script (reading order).
    h_pad_frac, w_pad_frac, angle_threshold:
        Forwarded to :func:`process_box`.

    Returns
    -------
    List of :class:`CropResult` objects, one per input box.
    """
    results = []
    for i, box in enumerate(boxes):
        result = process_box(image, box, h_pad_frac, w_pad_frac, angle_threshold)
        if result.crop is None:
            log.warning("Box %d / %d produced no valid crop.", i + 1, len(boxes))
        results.append(result)

    n_ok = sum(1 for r in results if r.crop is not None)
    n_deskewed = sum(1 for r in results if r.was_deskewed)
    log.info(
        "process_document: %d / %d boxes OK, %d deskewed.",
        n_ok, len(boxes), n_deskewed,
    )
    return results


# ── Demo / smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)-8s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    # ── Build a synthetic test image ──────────────────────────────────────
    # 1200 × 900 white canvas with three fake text-line regions drawn as
    # dark diagonal strokes to simulate real skewed handwriting.

    rng = np.random.default_rng(seed=0)
    IMG_H, IMG_W = 900, 1200
    canvas = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)

    def draw_skewed_line(
        img: np.ndarray,
        y_center: int,
        skew_deg: float,
        thickness: int = 6,
    ) -> None:
        """Draw a dark line at *y_center* tilted by *skew_deg* degrees."""
        angle_rad = np.deg2rad(skew_deg)
        x0, x1 = 100, 1100
        dy = int((x1 - x0) * np.tan(angle_rad))
        p0 = (x0, y_center - dy // 2)
        p1 = (x1, y_center + dy // 2)
        cv2.line(img, p0, p1, (40, 40, 40), thickness)
        # Add some noise to simulate ink texture
        for _ in range(80):
            ox = int(rng.integers(x0, x1))
            frac = (ox - x0) / (x1 - x0)
            oy = int(p0[1] + frac * (p1[1] - p0[1]) + rng.integers(-8, 8))
            cv2.circle(img, (ox, oy), rng.integers(1, 4), (60, 60, 60), -1)

    draw_skewed_line(canvas, y_center=200, skew_deg=-8.0)   # clockwise slant
    draw_skewed_line(canvas, y_center=450, skew_deg=0.5)    # nearly horizontal
    draw_skewed_line(canvas, y_center=700, skew_deg=12.0)   # counter-clockwise

    # ── Corresponding Phase 1 bounding boxes ──────────────────────────────
    # Each box tightly wraps one of the three drawn lines (plus a bad box).
    boxes: list[list[int]] = [
        [100, 165, 1100, 235],   # line 1: CW slant  → should deskew
        [100, 430, 1100, 470],   # line 2: ~flat     → should skip deskew
        [100, 660, 1100, 740],   # line 3: CCW slant → should deskew
        [1100, 800, 1100, 900],  # bad box: zero width → should return None
    ]

    # ── Run the pipeline ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 Bridge — smoke test")
    print("=" * 60)

    results = process_document(canvas, boxes)

    for i, (box, res) in enumerate(zip(boxes, results)):
        if res.crop is None:
            print(f"\nBox {i}: FAILED  raw={box}")
            continue
        h, w = res.crop.shape[:2]
        print(
            f"\nBox {i}:  raw={box}"
            f"\n         padded={res.padded_box}"
            f"\n         crop shape=({h}, {w})"
            f"\n         skew={res.skew_angle:+.2f}°  deskewed={res.was_deskewed}"
        )

    # ── Optionally write crops to disk for visual inspection ──────────────
    WRITE_CROPS = False   # set True to save JPEGs for manual inspection
    if WRITE_CROPS:
        for i, res in enumerate(results):
            if res.crop is not None:
                cv2.imwrite(f"/tmp/phase2_crop_{i}.jpg", res.crop)
                print(f"Saved /tmp/phase2_crop_{i}.jpg")

    print("\nSmoke test complete.")
