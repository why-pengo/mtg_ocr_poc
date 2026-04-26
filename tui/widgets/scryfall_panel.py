"""Scryfall lookup panel: input, buttons, and scrollable results log."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, Input, Label, RichLog

from scryfall.lookup import _hyperlink


class ScryfallPanel(Widget):
    """Bottom-half panel for Scryfall card lookup and result display."""

    def compose(self) -> ComposeResult:
        yield Label("Scryfall", id="scryfall-label")
        with Horizontal(id="lookup-row"):
            yield Input(placeholder="Card name…", id="name-input")
            yield Button("Look up", id="lookup-btn", variant="primary")
            yield Button("Skip", id="skip-btn", variant="default")
        yield RichLog(id="results-log", highlight=True, markup=True)

    def prefill(self, name: str) -> None:
        """Pre-populate the name input with *name*."""
        self.query_one("#name-input", Input).value = name

    def show_searching(self) -> None:
        """Write a searching indicator to the log."""
        log = self.query_one(RichLog)
        log.clear()
        log.write("🔎 Searching…")

    def show_results(self, cards: list[dict[str, Any]]) -> None:
        """Write formatted card results to the log using Rich markup."""
        log = self.query_one(RichLog)
        log.clear()
        if not cards:
            log.write("[red]No cards found.[/red]")
            return

        log.write(f"[bold]Found {len(cards)} match(es):[/bold]")
        for i, card in enumerate(cards, 1):
            url = card.get("scryfall_uri", "")
            name = card.get("name", "Unknown")
            type_line = card.get("type_line", "")
            set_name = card.get("set_name", "")
            set_code = card.get("set", "").upper()
            link = _hyperlink(url, name)
            log.write(f"  {i}. {link}  [dim]{set_name} ({set_code})[/dim]  {type_line}")

    def show_error(self, msg: str) -> None:
        """Write an error message in red to the log."""
        log = self.query_one(RichLog)
        log.clear()
        log.write(f"[bold red]Error:[/bold red] {msg}")
