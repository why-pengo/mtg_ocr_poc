"""Name-region cropping, image enhancement, and OCR text normalisation helpers."""

from __future__ import annotations

import re

import cv2
import numpy as np

# MTG standard card layout (portrait, after perspective correction).
# The card name text sits in a narrow strip at the top, left of the mana cost icons.
_NAME_TOP = 0.034
_NAME_BOTTOM = 0.115
_NAME_LEFT = 0.040
_NAME_RIGHT = 0.730

# Characters that are visually similar to others and commonly mis-read by OCR engines.
# Maps (wrong → correct). Applied in order — more specific patterns first.
_OCR_SUBSTITUTIONS: list[tuple[str, str]] = [
    # Semicolon mis-read as comma separator (e.g. "Auntie Ool; Cursewretch")
    (r";", ","),
    # Pipe or broken-bar mis-read as letter I or l
    (r"\|", "I"),
    # Backtick or grave mis-read as apostrophe
    (r"`", "'"),
    # Two or more spaces collapsed to one
    (r"  +", " "),
]


def crop_name_region(card_image: np.ndarray) -> np.ndarray:
    """Return the card name text strip from a rectified card image."""
    h, w = card_image.shape[:2]
    return card_image[
        int(h * _NAME_TOP) : int(h * _NAME_BOTTOM),
        int(w * _NAME_LEFT) : int(w * _NAME_RIGHT),
    ]


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply CLAHE + denoising for better OCR accuracy.

    Returns a single-channel (grayscale) uint8 array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    return cv2.fastNlMeansDenoising(enhanced, h=10)


def resize_for_ingestion(image: np.ndarray, max_long_edge: int = 1500) -> np.ndarray:
    """Downsample *image* so its longest edge is at most *max_long_edge* pixels.

    Preserves aspect ratio using area interpolation (best for downscaling).
    Returns the original array unchanged if it already fits within the limit.
    """
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_ocr_text(text: str) -> str:
    """Apply MTG-specific corrections to raw OCR output.

    Fixes common character confusions that OCR engines make on card name text.
    Should be called on the raw detected string before storing or displaying it.
    """
    for pattern, replacement in _OCR_SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    return text.strip()
