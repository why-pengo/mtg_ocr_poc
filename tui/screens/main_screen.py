"""Main screen: OCR engine results table + Scryfall lookup panel."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from textual import work
from textual.app import ComposeResult
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input

from engines.base import OCRResult
from tui.widgets.engine_table import EngineResultsTable
from tui.widgets.scryfall_panel import ScryfallPanel


class EngineResultReady(Message):
    """Posted by a background worker when an OCR engine finishes."""

    def __init__(self, result: OCRResult) -> None:
        super().__init__()
        self.result = result


class ScryfallResultsReady(Message):
    """Posted by the Scryfall lookup worker when results are available."""

    def __init__(self, cards: list[dict]) -> None:
        super().__init__()
        self.cards = cards


class MainScreen(Screen):
    """Full-screen layout: engine results table above, Scryfall panel below."""

    def __init__(self, image_path: Path, ground_truth: str | None = None) -> None:
        super().__init__()
        self.image_path = image_path
        self.ground_truth = ground_truth

    def compose(self) -> ComposeResult:
        yield Header()
        yield EngineResultsTable(id="engine-table")
        yield ScryfallPanel(id="scryfall-section")
        yield Footer()

    def on_mount(self) -> None:
        self._preprocess_and_run()

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    @work(thread=True)
    def _preprocess_and_run(self) -> None:
        """Load the image, preprocess it, then run each available OCR engine."""
        # Lazy engine imports — avoids slow startup at module level
        from engines.easyocr_engine import EasyOCREngine
        from engines.paddleocr_engine import PaddleOCREngine
        from engines.tesseract_engine import TesseractEngine
        from engines.trocr_engine import TrOCREngine
        from preprocessing.card_detect import detect_and_rectify
        from preprocessing.image_utils import crop_name_region, enhance_for_ocr

        all_engines = [EasyOCREngine(), PaddleOCREngine(), TrOCREngine(), TesseractEngine()]
        available = [e for e in all_engines if e.is_available()]

        table = self.query_one(EngineResultsTable)
        for engine in available:
            self.app.call_from_thread(table.add_pending_row, engine.name)

        image: np.ndarray | None = cv2.imread(str(self.image_path))
        if image is None:
            panel = self.query_one(ScryfallPanel)
            self.app.call_from_thread(panel.show_error, f"Could not read image: {self.image_path}")
            return

        rectified = detect_and_rectify(image)
        name_crop = crop_name_region(rectified)
        enhanced = enhance_for_ocr(name_crop)

        for engine in available:
            result = engine.detect(enhanced)
            self.post_message(EngineResultReady(result))

    @work(thread=True)
    def _run_scryfall_lookup(self, name: str) -> None:
        """Query Scryfall for *name* and post ScryfallResultsReady."""
        from scryfall.lookup import scryfall_search

        cards = scryfall_search(name)
        self.post_message(ScryfallResultsReady(cards))

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def on_engine_result_ready(self, message: EngineResultReady) -> None:
        table = self.query_one(EngineResultsTable)
        table.update_row(message.result)

        best = table.best_guess()
        if best:
            self.query_one(ScryfallPanel).prefill(best)

    def on_scryfall_results_ready(self, message: ScryfallResultsReady) -> None:
        panel = self.query_one(ScryfallPanel)
        if message.cards:
            panel.show_results(message.cards)
        else:
            panel.show_error("No cards found.")

    # ------------------------------------------------------------------
    # UI event handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "lookup-btn":
            name = self.query_one("#name-input", Input).value.strip()
            self._do_lookup(name)
        elif event.button.id == "skip-btn":
            self.app.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "name-input":
            self._do_lookup(event.value.strip())

    def _do_lookup(self, name: str) -> None:
        if not name:
            return
        self.query_one(ScryfallPanel).show_searching()
        self._run_scryfall_lookup(name)
