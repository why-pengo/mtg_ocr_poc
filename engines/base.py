"""Abstract base class and result dataclass for all OCR engine implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OCRResult:
    """Result produced by a single OCR engine run."""

    engine_name: str
    card_name: Optional[str]  # None if detection failed
    confidence: Optional[float]  # 0.0–1.0 if the engine provides it; None otherwise
    elapsed_ms: float
    error: Optional[str] = field(default=None)  # human-readable error if the engine failed


class OCREngine(ABC):
    """Base class that all OCR engine implementations must subclass."""

    #: Human-readable name shown in the results table.
    name: str = ""

    @abstractmethod
    def detect(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a pre-processed name-region crop.

        Args:
            image: Grayscale or BGR numpy array output of the preprocessing pipeline.

        Returns:
            OCRResult with the detected card name, or card_name=None on failure.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this engine's dependencies are installed and usable."""
        ...
