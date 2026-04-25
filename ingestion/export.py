"""JSON export for batch-ingested card data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(cards: list[dict[str, Any]], output_path: Path) -> None:
    """Write *cards* as a JSON array to *output_path*.

    The format is an array of Scryfall card objects, compatible with
    ``PaperCard.upsert_from_scryfall()`` in the mtgas app.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(cards, fh, indent=2, ensure_ascii=False)
    print(f"\n✅  Wrote {len(cards)} card(s) to {output_path}")
