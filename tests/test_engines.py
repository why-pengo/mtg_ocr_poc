"""Tests for OCR engine implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from engines.base import OCRResult
from engines.easyocr_engine import EasyOCREngine
from engines.paddleocr_engine import PaddleOCREngine
from engines.trocr_engine import TrOCREngine


class TestOCRResult:
    def test_defaults(self) -> None:
        r = OCRResult(engine_name="Test", card_name="Bolt", confidence=0.9, elapsed_ms=50.0)
        assert r.error is None

    def test_failed_result(self) -> None:
        r = OCRResult(
            engine_name="Test", card_name=None, confidence=None, elapsed_ms=5.0, error="boom"
        )
        assert r.card_name is None
        assert r.error == "boom"


class TestEasyOCREngine:
    def test_is_available_returns_bool(self) -> None:
        assert isinstance(EasyOCREngine().is_available(), bool)

    def test_detect_single_text_fragment(self) -> None:
        engine = EasyOCREngine()
        image = np.zeros((50, 400), dtype=np.uint8)
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [([0, 0, 100, 30], "Lightning Bolt", 0.95)]

        with patch.object(engine, "_get_reader", return_value=mock_reader):
            result = engine.detect(image)

        assert result.card_name == "Lightning Bolt"
        assert result.confidence == pytest.approx(0.95)
        assert result.elapsed_ms >= 0
        assert result.error is None

    def test_detect_joins_multiple_fragments(self) -> None:
        engine = EasyOCREngine()
        image = np.zeros((50, 400), dtype=np.uint8)
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([0, 0, 60, 30], "Lightning", 0.90),
            ([70, 0, 140, 30], "Bolt", 0.85),
        ]

        with patch.object(engine, "_get_reader", return_value=mock_reader):
            result = engine.detect(image)

        assert result.card_name == "Lightning Bolt"
        assert result.confidence == pytest.approx(0.875)

    def test_detect_returns_none_when_no_text(self) -> None:
        engine = EasyOCREngine()
        image = np.zeros((50, 400), dtype=np.uint8)
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []

        with patch.object(engine, "_get_reader", return_value=mock_reader):
            result = engine.detect(image)

        assert result.card_name is None
        assert result.error == "No text detected"

    def test_detect_handles_reader_exception(self) -> None:
        engine = EasyOCREngine()
        image = np.zeros((50, 400), dtype=np.uint8)
        mock_reader = MagicMock()
        mock_reader.readtext.side_effect = RuntimeError("CUDA out of memory")

        with patch.object(engine, "_get_reader", return_value=mock_reader):
            result = engine.detect(image)

        assert result.card_name is None
        assert result.error is not None
        assert "CUDA out of memory" in result.error


class TestStubEngines:
    """Stub engines should never crash and always return an OCRResult."""

    @pytest.mark.parametrize("engine_cls", [PaddleOCREngine, TrOCREngine])
    def test_is_available_returns_bool(self, engine_cls: type) -> None:
        assert isinstance(engine_cls().is_available(), bool)

    @pytest.mark.parametrize("engine_cls", [PaddleOCREngine, TrOCREngine])
    def test_detect_returns_ocr_result(self, engine_cls: type) -> None:
        engine = engine_cls()
        image = np.zeros((50, 400), dtype=np.uint8)
        result = engine.detect(image)
        assert isinstance(result, OCRResult)
        assert result.card_name is None
        assert result.error is not None
