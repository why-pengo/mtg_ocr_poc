"""Batch MTG card ingestion: Apple Photos → EasyOCR → Scryfall → JSON.

Usage:
    .venv/bin/python ingest.py [--album ALBUM] [--output OUTPUT] [--verbose]

Photos are read from a named Apple Photos album, OCR'd with EasyOCR, confirmed
interactively, looked up on Scryfall, and written to a JSON file compatible with
PaperCard.upsert_from_scryfall() in the mtgas app.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from engines.easyocr_engine import EasyOCREngine
from ingestion.export import write_json
from ingestion.osxphotos_source import iter_album_photos
from preprocessing.card_detect import detect_and_rectify
from preprocessing.image_utils import crop_name_region, enhance_for_ocr, resize_for_ingestion
from scryfall.lookup import lookup_for_ingestion

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

_DEFAULT_ALBUM = "MTG Cards to Scan"
_DEFAULT_OUTPUT = Path("paper_cards_import.json")

_DIVIDER = "─" * 60
_DOUBLE_DIVIDER = "═" * 60


def _process_photo(
    filename: str,
    image: np.ndarray,
    engine: EasyOCREngine,
) -> dict[str, Any] | None:
    """Run OCR + Scryfall lookup for one photo with interactive review.

    Returns a Scryfall card dict on confirmation, or None if the user skips.
    """
    print(f"\n{_DIVIDER}")
    print(f"📷  {filename}")

    resized = resize_for_ingestion(image)
    rectified = detect_and_rectify(resized)
    name_crop = crop_name_region(rectified)
    enhanced = enhance_for_ocr(name_crop)

    result = engine.detect(enhanced)
    detected = result.card_name or ""

    if detected:
        conf_str = (
            f"  (confidence: {result.confidence:.0%})" if result.confidence is not None else ""
        )
        print(f"🔍  Detected: \033[1m{detected}\033[0m{conf_str}")
    else:
        print(f"❌  OCR failed: {result.error or 'no text detected'}")

    response = input("  [enter] accept / type correction / [s]kip: ").strip()

    if response.lower() == "s":
        print("  Skipped.")
        return None

    name = response if response else detected
    if not name:
        print("  No name — skipped.")
        return None

    return lookup_for_ingestion(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-ingest MTG card photos from Apple Photos → Scryfall → JSON."
    )
    parser.add_argument(
        "--album",
        default=_DEFAULT_ALBUM,
        metavar="NAME",
        help=f"Apple Photos album to read from (default: '{_DEFAULT_ALBUM}')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="FILE",
        help=f"Output JSON file path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if sys.platform != "darwin":
        print("Error: ingest.py requires macOS (osxphotos is macOS-only).", file=sys.stderr)
        sys.exit(1)

    engine = EasyOCREngine()
    if not engine.is_available():
        print("Error: EasyOCR is not installed. Run: pip install easyocr", file=sys.stderr)
        sys.exit(1)

    print(f'📚  Reading album "{args.album}" from Apple Photos…')
    try:
        photo_source = iter_album_photos(args.album)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    confirmed: list[dict[str, Any]] = []
    total = 0

    for filename, image in photo_source:
        total += 1
        card_dict = _process_photo(filename, image, engine)
        if card_dict is not None:
            confirmed.append(card_dict)
            print(f"  ✓  Added: \033[1m{card_dict.get('name', '?')}\033[0m")

    print(f"\n{_DOUBLE_DIVIDER}")
    print(f"Processed {total} photo(s), confirmed {len(confirmed)} card(s).")

    if not confirmed:
        print("No cards confirmed — no output file written.")
        return

    write_json(confirmed, args.output)


if __name__ == "__main__":
    main()
