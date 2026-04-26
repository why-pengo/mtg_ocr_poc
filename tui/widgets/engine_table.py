"""DataTable widget showing per-engine OCR results."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable, Label

from engines.base import OCRResult


class EngineResultsTable(Widget):
    """Displays a DataTable of OCR engine results with live row updates."""

    def compose(self) -> ComposeResult:
        yield Label("OCR Results", id="engine-table-label")
        yield DataTable(id="data-table")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("Engine", key="engine", width=14)
        table.add_column("Detected Name", key="name", width=30)
        table.add_column("Conf", key="conf", width=6)
        table.add_column("ms", key="ms", width=8)

        # Track results keyed by engine name for best_guess()
        self._results: dict[str, OCRResult] = {}

    def add_pending_row(self, engine_name: str) -> None:
        """Add a placeholder row for *engine_name* while it is still running."""
        table = self.query_one(DataTable)
        table.add_row("⏳ running…", "—", "—", key=engine_name, label=engine_name)

    def update_row(self, result: OCRResult) -> None:
        """Update the row for *result.engine_name* with the finished result."""
        self._results[result.engine_name] = result

        conf_str = f"{result.confidence:.2f}" if result.confidence is not None else "n/a"
        ms_str = f"{result.elapsed_ms:.0f}"

        if result.card_name:
            name_str = result.card_name
        else:
            name_str = f"❌ {result.error or 'failed'}"

        table = self.query_one(DataTable)
        table.update_cell(result.engine_name, "name", name_str, update_width=True)
        table.update_cell(result.engine_name, "conf", conf_str)
        table.update_cell(result.engine_name, "ms", ms_str)

    def best_guess(self) -> str | None:
        """Return the card_name with the highest confidence among successful results."""
        successful = [r for r in self._results.values() if r.card_name is not None]
        if not successful:
            return None
        return max(successful, key=lambda r: r.confidence or 0.0).card_name
