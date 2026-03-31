"""Card boundary detection and perspective correction using OpenCV."""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# A card contour must cover at least this fraction of the image area to be considered valid.
_MIN_AREA_FRACTION = 0.10


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest diff
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp *image* to a top-down view using the four corner points *pts*."""
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))


def detect_and_rectify(image: np.ndarray) -> np.ndarray:
    """Detect the card in *image* and return a perspective-corrected crop.

    Pipeline:
        1. Grayscale → Gaussian blur → Canny edge detection
        2. Dilate edges to close small gaps in card borders
        3. Find contours; pick the largest quadrilateral above the area threshold
        4. Apply a four-point perspective transform

    Falls back to returning the original image if no valid card contour is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = image.shape[0] * image.shape[1]

    for contour in contours[:5]:
        area = cv2.contourArea(contour)
        if area < img_area * _MIN_AREA_FRACTION:
            break  # remaining contours are even smaller

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            logger.debug("Card contour found (area=%.0f px²)", area)
            return _four_point_transform(image, approx.reshape(4, 2))

    logger.warning("No card contour found — using original image as-is")
    return image
