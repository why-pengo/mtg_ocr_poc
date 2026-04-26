"""MTG OCR POC — benchmark OCR engines and look up a card on Scryfall."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from benchmark.results import BenchmarkRow, print_results_table
from engines.easyocr_engine import EasyOCREngine
from engines.paddleocr_engine import PaddleOCREngine
from engines.tesseract_engine import TesseractEngine
from engines.trocr_engine import TrOCREngine
from preprocessing.card_detect import detect_and_rectify
from preprocessing.image_utils import crop_name_region, enhance_for_ocr
from scryfall.lookup import prompt_and_lookup

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

_ALL_ENGINES = [
    EasyOCREngine(),
    PaddleOCREngine(),
    TrOCREngine(),
    TesseractEngine(),
]


def _run(image_path: Path, ground_truth: str | None) -> None:
    print(f"📷  Loading {image_path.name}…")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: could not read image: {image_path}", file=sys.stderr)
        sys.exit(1)

    print("🔍  Detecting card boundaries and correcting perspective…")
    rectified = detect_and_rectify(image)
    name_crop = crop_name_region(rectified)
    enhanced = enhance_for_ocr(name_crop)

    available = [e for e in _ALL_ENGINES if e.is_available()]
    if not available:
        print(
            "No OCR engines are available.\n"
            "Install at least one — see requirements.txt for options.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"🧠  Running {len(available)} engine(s)…\n")
    rows: list[BenchmarkRow] = []
    for engine in available:
        print(f"  [{engine.name}]", end=" ", flush=True)
        result = engine.detect(enhanced)
        rows.append(BenchmarkRow(result=result, ground_truth=ground_truth))
        print(result.card_name or f"FAILED — {result.error}")

    print_results_table(rows)

    # Best guess: first successful result from the sorted table
    best = next((r for r in rows if r.result.card_name), None)
    if best is None:
        print("\nAll engines failed to detect a card name.", file=sys.stderr)
        sys.exit(1)

    prompt_and_lookup(best.result.card_name)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR an MTG card image and look it up on Scryfall."
    )
    parser.add_argument("image", type=Path, help="Path to card image file")
    parser.add_argument(
        "--ground-truth",
        "-g",
        metavar="NAME",
        help="Known card name for accuracy benchmarking",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.image.exists():
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    _run(args.image, args.ground_truth)


if __name__ == "__main__":
    main()
