"""Tests for the Textual TUI application."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from engines.base import OCRResult

# ---------------------------------------------------------------------------
# EngineResultsTable unit tests (no running app required)
# ---------------------------------------------------------------------------


class TestEngineResultsTable:
    def test_best_guess_returns_highest_confidence(self) -> None:
        from tui.widgets.engine_table import EngineResultsTable

        table = EngineResultsTable()
        table._results = {
            "Engine A": OCRResult("Engine A", "Lightning Bolt", 0.9, 100.0),
            "Engine B": OCRResult("Engine B", "Dark Ritual", 0.7, 80.0),
        }
        assert table.best_guess() == "Lightning Bolt"

    def test_best_guess_returns_none_when_no_results(self) -> None:
        from tui.widgets.engine_table import EngineResultsTable

        table = EngineResultsTable()
        table._results = {}
        assert table.best_guess() is None

    def test_best_guess_ignores_failed_results(self) -> None:
        from tui.widgets.engine_table import EngineResultsTable

        table = EngineResultsTable()
        table._results = {
            "Engine A": OCRResult("Engine A", None, None, 100.0, error="Failed"),
            "Engine B": OCRResult("Engine B", "Shock", 0.5, 80.0),
        }
        assert table.best_guess() == "Shock"

    def test_best_guess_handles_none_confidence(self) -> None:
        from tui.widgets.engine_table import EngineResultsTable

        table = EngineResultsTable()
        table._results = {
            "Engine A": OCRResult("Engine A", "Counterspell", None, 50.0),
        }
        assert table.best_guess() == "Counterspell"


# ---------------------------------------------------------------------------
# Integration smoke tests using Textual's async test harness
# ---------------------------------------------------------------------------

_DUMMY_IMAGE = np.zeros((880, 630, 3), dtype=np.uint8)
_MOCK_RESULT = OCRResult("EasyOCR", "Lightning Bolt", 0.9, 100.0)

_ENGINE_PATCHES = [
    patch("engines.easyocr_engine.EasyOCREngine.is_available", return_value=True),
    patch("engines.easyocr_engine.EasyOCREngine.detect", return_value=_MOCK_RESULT),
    patch("engines.paddleocr_engine.PaddleOCREngine.is_available", return_value=False),
    patch("engines.trocr_engine.TrOCREngine.is_available", return_value=False),
    patch("engines.tesseract_engine.TesseractEngine.is_available", return_value=False),
]

_PREPROCESS_PATCHES = [
    patch("cv2.imread", return_value=_DUMMY_IMAGE),
    patch("preprocessing.card_detect.detect_and_rectify", return_value=_DUMMY_IMAGE),
    patch("preprocessing.image_utils.crop_name_region", return_value=_DUMMY_IMAGE),
    patch("preprocessing.image_utils.enhance_for_ocr", return_value=_DUMMY_IMAGE),
]


@pytest.fixture()
def all_patches():
    """Context manager that activates all mocks needed for TUI smoke tests."""
    import contextlib

    patches = _PREPROCESS_PATCHES + _ENGINE_PATCHES

    @contextlib.contextmanager
    def _apply():
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield

    return _apply


async def test_app_composes_without_error(all_patches) -> None:
    """Smoke test: app mounts without crashing."""
    from tui.app import MTGOcrApp

    with all_patches():
        app = MTGOcrApp(image_path=Path("tests/fixtures/sample.jpg"))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            assert app.title == "MTG OCR"


async def test_app_has_engine_table_and_scryfall_panel(all_patches) -> None:
    """Verify the two main widgets are present in the DOM."""
    from tui.app import MTGOcrApp
    from tui.widgets.engine_table import EngineResultsTable
    from tui.widgets.scryfall_panel import ScryfallPanel

    with all_patches():
        app = MTGOcrApp(image_path=Path("tests/fixtures/sample.jpg"))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            assert app.screen.query_one(EngineResultsTable) is not None
            assert app.screen.query_one(ScryfallPanel) is not None


async def test_scryfall_panel_show_results_writes_to_log(all_patches) -> None:
    """ScryfallPanel.show_results() should write at least one line to the RichLog."""
    from textual.widgets import RichLog

    from tui.app import MTGOcrApp
    from tui.widgets.scryfall_panel import ScryfallPanel

    cards = [
        {
            "name": "Lightning Bolt",
            "type_line": "Instant",
            "set_name": "Limited Edition Alpha",
            "set": "lea",
            "scryfall_uri": "https://scryfall.com/card/lea/161",
        }
    ]

    with all_patches():
        app = MTGOcrApp(image_path=Path("tests/fixtures/sample.jpg"))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            panel = app.screen.query_one(ScryfallPanel)
            panel.show_results(cards)
            await pilot.pause(0.1)
            log = panel.query_one(RichLog)
            assert len(log.lines) > 0


async def test_scryfall_panel_show_error_writes_to_log(all_patches) -> None:
    """ScryfallPanel.show_error() should write to the RichLog."""
    from textual.widgets import RichLog

    from tui.app import MTGOcrApp
    from tui.widgets.scryfall_panel import ScryfallPanel

    with all_patches():
        app = MTGOcrApp(image_path=Path("tests/fixtures/sample.jpg"))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            panel = app.screen.query_one(ScryfallPanel)
            panel.show_error("Something went wrong")
            await pilot.pause(0.1)
            log = panel.query_one(RichLog)
            assert len(log.lines) > 0


async def test_scryfall_panel_prefill_sets_input_value(all_patches) -> None:
    """ScryfallPanel.prefill() should update the Input value."""
    from textual.widgets import Input

    from tui.app import MTGOcrApp
    from tui.widgets.scryfall_panel import ScryfallPanel

    with all_patches():
        app = MTGOcrApp(image_path=Path("tests/fixtures/sample.jpg"))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            panel = app.screen.query_one(ScryfallPanel)
            panel.prefill("Counterspell")
            await pilot.pause(0.1)
            assert panel.query_one("#name-input", Input).value == "Counterspell"
