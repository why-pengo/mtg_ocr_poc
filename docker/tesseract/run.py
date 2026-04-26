"""Tesseract OCR script executed inside the Docker container.

Reads a preprocessed image, runs pytesseract, and writes a JSON result to stdout.

Usage (called by TesseractEngine via docker run):
    python run.py /mnt/images/<image_file>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import pytesseract


def main() -> None:
    if len(sys.argv) < 2:
        _fail("Usage: run.py <image_path>")
        return

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        _fail(f"File not found: {image_path}")
        return

    start = time.perf_counter()
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            _fail(f"Could not read image: {image_path}")
            return

        # PSM 7 = treat image as a single text line (ideal for the name strip)
        # OEM 1 = LSTM engine only
        config = "--psm 7 --oem 1"
        text = pytesseract.image_to_string(image, config=config).strip()

        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        confidences = [c for c in data["conf"] if isinstance(c, (int, float)) and c > 0]
        avg_conf = sum(confidences) / len(confidences) / 100.0 if confidences else None

        elapsed_ms = (time.perf_counter() - start) * 1000
        print(
            json.dumps(
                {
                    "card_name": text or None,
                    "confidence": avg_conf,
                    "elapsed_ms": elapsed_ms,
                }
            )
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(
            json.dumps(
                {
                    "card_name": None,
                    "confidence": None,
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc),
                }
            )
        )


def _fail(message: str) -> None:
    elapsed_ms = 0.0
    print(
        json.dumps(
            {"card_name": None, "confidence": None, "elapsed_ms": elapsed_ms, "error": message}
        )
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
