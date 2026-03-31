"""PaddleOCR engine — not yet implemented."""
from __future__ import annotations

import logging
import time

import numpy as np

from engines.base import OCREngine, OCRResult

logger = logging.getLogger(__name__)


class PaddleOCREngine(OCREngine):
    """OCR engine backed by PaddleOCR.

    Install: pip install paddlepaddle paddleocr
    """

    name = "PaddleOCR"

    def is_available(self) -> bool:
        try:
            import paddleocr  # noqa: F401

            return True
        except ImportError:
            return False

    def detect(self, image: np.ndarray) -> OCRResult:
        # TODO: implement PaddleOCR detection
        elapsed_ms = 0.0
        return OCRResult(
            engine_name=self.name,
            card_name=None,
            confidence=None,
            elapsed_ms=elapsed_ms,
            error="PaddleOCR engine not yet implemented",
        )
