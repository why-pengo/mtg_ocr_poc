"""MTG OCR POC — benchmark OCR engines and look up a card on Scryfall."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tui.app import MTGOcrApp

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _run(image_path: Path, ground_truth: str | None) -> None:
    app = MTGOcrApp(image_path=image_path, ground_truth=ground_truth)
    app.run()


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
