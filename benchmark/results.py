"""Accuracy tracking and ranked results summary table."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

from engines.base import OCRResult


@dataclass
class BenchmarkRow:
    """Wraps an OCRResult with optional ground-truth scoring."""

    result: OCRResult
    ground_truth: Optional[str] = None
    # Populated in __post_init__ when ground_truth is provided
    similarity: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.ground_truth and self.result.card_name:
            gt = self.ground_truth.strip()
            detected = self.result.card_name.strip()
            # fuzz.ratio returns 0–100; normalise to 0.0–1.0
            self.similarity = round(fuzz.ratio(gt, detected, processor=str.lower) / 100.0, 3)


def print_results_table(rows: list[BenchmarkRow]) -> None:
    """Print a ranked results table to stdout.

    Ranking order: successful detections first, then by similarity desc (if
    ground truth provided), then by confidence desc.
    """

    def sort_key(r: BenchmarkRow) -> tuple:
        failed = r.result.card_name is None
        sim = -(r.similarity or 0.0)
        conf = -(r.result.confidence or 0.0)
        return (failed, sim, conf)

    sorted_rows = sorted(rows, key=sort_key)

    col_engine = max(len("Engine"), max(len(r.result.engine_name) for r in rows))
    col_name = max(len("Detected Name"), max(len(r.result.card_name or "—") for r in rows))
    has_scores = any(r.similarity is not None for r in rows)

    header = (
        f"  {'Engine':<{col_engine}}  {'Detected Name':<{col_name}}"
        f"  {'Conf':>6}  {'ms':>7}" + (f"  {'Score':>6}" if has_scores else "") + "  Notes"
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for row in sorted_rows:
        r = row.result
        name = r.card_name or "—"
        conf = f"{r.confidence:.2f}" if r.confidence is not None else "   n/a"
        ms = f"{r.elapsed_ms:.0f}"
        score_col = (
            (f"  {row.similarity:.3f}" if row.similarity is not None else "     —")
            if has_scores
            else ""
        )
        notes = r.error or ""
        print(
            f"  {r.engine_name:<{col_engine}}  {name:<{col_name}}"
            f"  {conf:>6}  {ms:>7}"
            f"{score_col}  {notes}"
        )

    print(sep)
