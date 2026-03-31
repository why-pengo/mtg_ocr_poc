"""TrOCR engine (HuggingFace transformer-based) — not yet implemented."""
from __future__ import annotations

import logging

import numpy as np

from engines.base import OCREngine, OCRResult

logger = logging.getLogger(__name__)


class TrOCREngine(OCREngine):
    """OCR engine backed by Microsoft TrOCR via HuggingFace Transformers.

    Install: pip install transformers torch torchvision
    First run downloads the model (~400 MB).
    """

    name = "TrOCR"

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def detect(self, image: np.ndarray) -> OCRResult:
        # TODO: implement TrOCR detection
        return OCRResult(
            engine_name=self.name,
            card_name=None,
            confidence=None,
            elapsed_ms=0.0,
            error="TrOCR engine not yet implemented",
        )
