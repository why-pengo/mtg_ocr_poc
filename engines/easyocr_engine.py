"""EasyOCR engine — no system packages required."""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from engines.base import OCREngine, OCRResult
from preprocessing.image_utils import normalize_ocr_text

logger = logging.getLogger(__name__)


class EasyOCREngine(OCREngine):
    """OCR engine backed by EasyOCR.

    Models are downloaded on first use (~100 MB) and cached in ~/.EasyOCR/.
    Install: pip install easyocr  (already in requirements.txt)
    """

    name = "EasyOCR"

    def __init__(self) -> None:
        self._reader: Optional[object] = None

    def _get_reader(self) -> object:
        if self._reader is None:
            import easyocr  # lazy import — triggers model download on first use

            self._reader = easyocr.Reader(["en"], gpu=False)
        return self._reader

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401

            return True
        except ImportError:
            return False

    def detect(self, image: np.ndarray) -> OCRResult:
        start = time.perf_counter()
        try:
            reader = self._get_reader()
            results = reader.readtext(image)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if not results:
                return OCRResult(
                    engine_name=self.name,
                    card_name=None,
                    confidence=None,
                    elapsed_ms=elapsed_ms,
                    error="No text detected",
                )

            # results: list of (bbox, text, confidence) — join all fragments
            text = normalize_ocr_text(" ".join(r[1] for r in results))
            confidence = sum(r[2] for r in results) / len(results)

            return OCRResult(
                engine_name=self.name,
                card_name=text or None,
                confidence=round(confidence, 3),
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("EasyOCR detection failed")
            return OCRResult(
                engine_name=self.name,
                card_name=None,
                confidence=None,
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )
