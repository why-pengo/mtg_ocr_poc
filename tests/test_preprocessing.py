"""Tests for the image preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from preprocessing.card_detect import detect_and_rectify
from preprocessing.image_utils import crop_name_region, enhance_for_ocr


class TestDetectAndRectify:
    def test_returns_ndarray(self, sample_card_image: np.ndarray) -> None:
        result = detect_and_rectify(sample_card_image)
        assert isinstance(result, np.ndarray)

    def test_output_has_positive_dimensions(self, sample_card_image: np.ndarray) -> None:
        result = detect_and_rectify(sample_card_image)
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_falls_back_gracefully_when_no_contour(self, sample_card_image: np.ndarray) -> None:
        # A plain noise image has no clear card contour; should still return something usable
        result = detect_and_rectify(sample_card_image)
        assert result.size > 0


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
