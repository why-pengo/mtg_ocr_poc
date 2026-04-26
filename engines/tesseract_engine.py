"""Tesseract OCR engine — runs inside Docker to avoid system package installation."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from engines.base import OCREngine, OCRResult

logger = logging.getLogger(__name__)

_DOCKER_IMAGE = "mtg-ocr-tesseract"


class TesseractEngine(OCREngine):
    """Tesseract OCR wrapped in a Docker container.

    Build the image before first use:
        docker build -t mtg-ocr-tesseract -f docker/tesseract/Dockerfile .
    """

    name = "Tesseract"

    def is_available(self) -> bool:
        return shutil.which("docker") is not None and self._image_exists()

    def _image_exists(self) -> bool:
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", _DOCKER_IMAGE],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def detect(self, image: np.ndarray) -> OCRResult:
        start = time.perf_counter()
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            cv2.imwrite(str(tmp_path), image)

            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{tmp_path.parent}:/mnt/images:ro",
                    _DOCKER_IMAGE,
                    f"/mnt/images/{tmp_path.name}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            if result.returncode != 0:
                return OCRResult(
                    engine_name=self.name,
                    card_name=None,
                    confidence=None,
                    elapsed_ms=elapsed_ms,
                    error=result.stderr.strip() or "Docker run failed",
                )

            data = json.loads(result.stdout)
            return OCRResult(
                engine_name=self.name,
                card_name=data.get("card_name"),
                confidence=data.get("confidence"),
                elapsed_ms=elapsed_ms,
                error=data.get("error"),
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("Tesseract Docker engine failed")
            return OCRResult(
                engine_name=self.name,
                card_name=None,
                confidence=None,
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
