"""Textual TUI application for MTG OCR."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from tui.screens.main_screen import MainScreen


class MTGOcrApp(App):
    """Textual TUI for the MTG OCR proof-of-concept."""

    TITLE = "MTG OCR"
    CSS_PATH = "app.tcss"
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+c", "quit", "Quit")]

    def __init__(self, image_path: Path, ground_truth: str | None = None) -> None:
        super().__init__()
        self.image_path = image_path
        self.ground_truth = ground_truth

    def on_mount(self) -> None:
        self.push_screen(MainScreen(self.image_path, self.ground_truth))
