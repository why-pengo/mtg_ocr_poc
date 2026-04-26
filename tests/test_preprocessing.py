"""Tests for the image preprocessing pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from preprocessing.card_detect import (
    _plausible_card_quad,
    _try_inner_detection,
    _try_outer_detection,
    detect_and_rectify,
)
from preprocessing.image_utils import crop_name_region, enhance_for_ocr, resize_for_ingestion


class TestDetectAndRectify:
    def test_returns_ndarray(self, sample_card_image: np.ndarray) -> None:
        result = detect_and_rectify(sample_card_image)
        assert isinstance(result, np.ndarray)

    def test_output_has_positive_dimensions(self, sample_card_image: np.ndarray) -> None:
        result = detect_and_rectify(sample_card_image)
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_falls_back_gracefully_when_no_contour(self, sample_card_image: np.ndarray) -> None:
        result = detect_and_rectify(sample_card_image)
        assert result.size > 0


class TestPlausibleCardQuad:
    def test_accepts_portrait_card_shape(self) -> None:
        pts = np.array([[0, 0], [630, 0], [630, 880], [0, 880]], dtype="float32")
        assert _plausible_card_quad(pts) is True

    def test_accepts_landscape_card_shape(self) -> None:
        pts = np.array([[0, 0], [880, 0], [880, 630], [0, 630]], dtype="float32")
        assert _plausible_card_quad(pts) is True

    def test_rejects_square(self) -> None:
        pts = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype="float32")
        assert _plausible_card_quad(pts) is False

    def test_rejects_tiny_quad(self) -> None:
        pts = np.array([[0, 0], [5, 0], [5, 7], [0, 7]], dtype="float32")
        assert _plausible_card_quad(pts) is False


class TestTryOuterDetection:
    def test_returns_none_or_ndarray_on_noise(self, sample_card_image: np.ndarray) -> None:
        result = _try_outer_detection(sample_card_image)
        assert result is None or isinstance(result, np.ndarray)

    def test_detects_card_shaped_rectangle(self) -> None:
        # White portrait card-shaped rect on a black background — should be detected.
        img = np.zeros((1200, 900, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (730, 980), (255, 255, 255), thickness=-1)
        result = _try_outer_detection(img)
        assert result is not None
        assert result.shape[0] > 0


class TestTryInnerDetection:
    def test_returns_none_or_ndarray_on_noise(self, sample_card_image: np.ndarray) -> None:
        result = _try_inner_detection(sample_card_image)
        assert result is None or isinstance(result, np.ndarray)

    def test_detects_type_line_in_synthetic_card(self) -> None:
        # Build a synthetic card with a clear horizontal type-line bar at the expected position.
        h, w = 880, 630
        img = np.full((h, w, 3), 200, dtype=np.uint8)
        type_line_y = int(h * 0.575)
        cv2.rectangle(img, (0, type_line_y - 5), (w, type_line_y + 18), (80, 70, 60), -1)
        result = _try_inner_detection(img)
        assert result is not None
        assert result.shape[0] > 0


class TestCropNameRegion:
    def test_crop_is_smaller_than_source(self, sample_card_image: np.ndarray) -> None:
        crop = crop_name_region(sample_card_image)
        assert crop.shape[0] < sample_card_image.shape[0]
        assert crop.shape[1] < sample_card_image.shape[1]

    def test_crop_is_not_empty(self, sample_card_image: np.ndarray) -> None:
        crop = crop_name_region(sample_card_image)
        assert crop.size > 0

    def test_crop_preserves_channels(self, sample_card_image: np.ndarray) -> None:
        crop = crop_name_region(sample_card_image)
        assert crop.ndim == sample_card_image.ndim


class TestEnhanceForOcr:
    def test_returns_grayscale_from_bgr(self, sample_name_crop: np.ndarray) -> None:
        result = enhance_for_ocr(sample_name_crop)
        assert result.ndim == 2

    def test_returns_grayscale_from_grayscale(self, sample_name_crop: np.ndarray) -> None:
        gray = np.mean(sample_name_crop, axis=2).astype(np.uint8)
        result = enhance_for_ocr(gray)
        assert result.ndim == 2

    def test_output_dtype_is_uint8(self, sample_name_crop: np.ndarray) -> None:
        result = enhance_for_ocr(sample_name_crop)
        assert result.dtype == np.uint8


class TestNormalizeOcrText:
    def test_semicolon_becomes_comma(self) -> None:
        from preprocessing.image_utils import normalize_ocr_text

        assert normalize_ocr_text("Auntie Ool; Cursewretch") == "Auntie Ool, Cursewretch"

    def test_pipe_becomes_capital_i(self) -> None:
        from preprocessing.image_utils import normalize_ocr_text

        assert normalize_ocr_text("|ron Will") == "Iron Will"

    def test_backtick_becomes_apostrophe(self) -> None:
        from preprocessing.image_utils import normalize_ocr_text

        assert normalize_ocr_text("Glen Elendra`s Answer") == "Glen Elendra's Answer"

    def test_collapses_extra_spaces(self) -> None:
        from preprocessing.image_utils import normalize_ocr_text

        assert normalize_ocr_text("Lightning  Bolt") == "Lightning Bolt"

    def test_strips_whitespace(self) -> None:
        from preprocessing.image_utils import normalize_ocr_text

        assert normalize_ocr_text("  Bolt  ") == "Bolt"

    def test_clean_input_unchanged(self) -> None:
        from preprocessing.image_utils import normalize_ocr_text

        assert normalize_ocr_text("Lightning Bolt") == "Lightning Bolt"


class TestResizeForIngestion:
    def test_no_op_when_image_fits(self) -> None:
        img = np.zeros((800, 600, 3), dtype=np.uint8)
        result = resize_for_ingestion(img, max_long_edge=1500)
        assert result.shape == img.shape

    def test_downsamples_landscape_image(self) -> None:
        img = np.zeros((1200, 3000, 3), dtype=np.uint8)
        result = resize_for_ingestion(img, max_long_edge=1500)
        assert max(result.shape[:2]) <= 1500

    def test_downsamples_portrait_image(self) -> None:
        img = np.zeros((3000, 1200, 3), dtype=np.uint8)
        result = resize_for_ingestion(img, max_long_edge=1500)
        assert max(result.shape[:2]) <= 1500

    def test_preserves_aspect_ratio(self) -> None:
        img = np.zeros((2000, 1000, 3), dtype=np.uint8)
        result = resize_for_ingestion(img, max_long_edge=1000)
        h, w = result.shape[:2]
        assert abs(h / w - 2.0) < 0.05  # original was 2:1

    def test_preserves_channel_count(self) -> None:
        img = np.zeros((3000, 2000, 3), dtype=np.uint8)
        result = resize_for_ingestion(img, max_long_edge=1500)
        assert result.ndim == 3

    def test_exactly_at_limit_unchanged(self) -> None:
        img = np.zeros((1500, 1000, 3), dtype=np.uint8)
        result = resize_for_ingestion(img, max_long_edge=1500)
        assert result.shape == img.shape
