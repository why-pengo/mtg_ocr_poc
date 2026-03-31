"""Name-region cropping and image enhancement helpers."""
from __future__ import annotations

import cv2
import numpy as np

# MTG standard card layout (portrait, after perspective correction).
# The card name text sits in a narrow strip at the top, left of the mana cost icons.
_NAME_TOP = 0.034
_NAME_BOTTOM = 0.115
_NAME_LEFT = 0.040
_NAME_RIGHT = 0.730


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
