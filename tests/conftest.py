"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_card_image() -> np.ndarray:
    """Synthetic card-like BGR image at a standard card aspect ratio (2.5×3.5 in)."""
    rng = np.random.default_rng(42)
    image = np.full((880, 630, 3), 230, dtype=np.uint8)
    noise = rng.integers(0, 15, image.shape, dtype=np.uint8)
    return np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def sample_name_crop() -> np.ndarray:
    """Synthetic BGR image representing just the name text region."""
    rng = np.random.default_rng(42)
    image = np.full((60, 460, 3), 200, dtype=np.uint8)
    noise = rng.integers(0, 25, image.shape, dtype=np.uint8)
    return np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)
