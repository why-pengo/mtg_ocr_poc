"""Card boundary detection and perspective correction using OpenCV.

Two detection strategies, tried in order:

Strategy A — Outer-edge detection (standard bordered cards):
    Canny edge detection → find largest quadrilateral contour → perspective warp.
    Retried with progressively lower thresholds before giving up.

Strategy B — Inner structure detection (borderless/extended-art cards):
    Find the type line (strong horizontal divider at ~58% card height) via Hough lines,
    then back-calculate the card boundaries from that anchor.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MIN_AREA_FRACTION = 0.10

# Canny threshold pairs to try in order (low, high). We widen the search progressively.
_CANNY_PARAMS = [(50, 150), (30, 100), (15, 60)]

# MTG card aspect ratio (63mm × 88mm portrait). Used to validate and reconstruct the card rect.
_CARD_ASPECT = 88.0 / 63.0  # ≈ 1.397  (height / width)
_ASPECT_TOLERANCE = 0.20  # allow ±20% deviation from ideal aspect ratio

# Expected y-position of the type line as a fraction of card height.
_TYPE_LINE_Y_FRAC = 0.575
_TYPE_LINE_SEARCH_BAND = 0.10  # search ±10% around the expected position


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_and_rectify(image: np.ndarray) -> np.ndarray:
    """Detect the card in *image* and return a perspective-corrected crop.

    Tries Strategy A (outer-edge detection) first across multiple Canny thresholds,
    then falls back to Strategy B (inner type-line detection) for borderless cards.
    Returns the original image if both strategies fail.
    """
    result = _try_outer_detection(image)
    if result is not None:
        return result

    logger.debug("Outer detection failed — trying inner structure (type-line) detection")
    result = _try_inner_detection(image)
    if result is not None:
        return result

    logger.warning("No card contour found — using original image as-is")
    return image


# ---------------------------------------------------------------------------
# Strategy A: outer-edge detection
# ---------------------------------------------------------------------------


def _try_outer_detection(image: np.ndarray) -> Optional[np.ndarray]:
    """Attempt to find the card's outer quadrilateral at multiple Canny thresholds."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bilateral filter preserves edges better than Gaussian on noisy/dark surfaces.
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    img_area = image.shape[0] * image.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for low, high in _CANNY_PARAMS:
        edges = cv2.Canny(filtered, low, high)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:
            if cv2.contourArea(contour) < img_area * _MIN_AREA_FRACTION:
                break
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                if _plausible_card_quad(pts):
                    logger.debug("Outer contour found (Canny %d/%d)", low, high)
                    return _four_point_transform(image, pts)

    return None


def _plausible_card_quad(pts: np.ndarray) -> bool:
    """Return True if the four points form a roughly card-shaped rectangle."""
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    w = float(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = float(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if w < 10 or h < 10:
        return False
    aspect = h / w
    # Accept both portrait and landscape orientations.
    return (
        abs(aspect - _CARD_ASPECT) < _ASPECT_TOLERANCE
        or abs(aspect - 1.0 / _CARD_ASPECT) < _ASPECT_TOLERANCE
    )


# ---------------------------------------------------------------------------
# Strategy B: inner structure detection via type-line anchor
# ---------------------------------------------------------------------------


def _try_inner_detection(image: np.ndarray) -> Optional[np.ndarray]:
    """Find the type line and use it to estimate the full card bounding box.

    The type line is a solid horizontal divider present on every MTG card frame,
    including borderless/extended-art variants.  We detect it with a horizontal
    Sobel gradient + probabilistic Hough lines, then reconstruct the card rect.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Horizontal Sobel highlights horizontal edges (top/bottom of the type-line bar).
    sobel_h = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_abs = cv2.convertScaleAbs(sobel_h)
    _, thresh = cv2.threshold(sobel_abs, 30, 255, cv2.THRESH_BINARY)

    # Restrict search to a horizontal band where the type line should appear.
    band_top = int(h * (_TYPE_LINE_Y_FRAC - _TYPE_LINE_SEARCH_BAND))
    band_bot = int(h * (_TYPE_LINE_Y_FRAC + _TYPE_LINE_SEARCH_BAND))
    mask = np.zeros_like(thresh)
    mask[band_top:band_bot, :] = 255
    search_region = cv2.bitwise_and(thresh, mask)

    lines = cv2.HoughLinesP(
        search_region,
        rho=1,
        theta=np.pi / 180,
        threshold=int(w * 0.30),  # line must span at least 30% of image width
        minLineLength=int(w * 0.30),
        maxLineGap=int(w * 0.05),
    )
    if lines is None:
        return None

    # Cluster lines by y-coordinate and pick the median y of the strongest cluster.
    ys = [int((line[0][1] + line[0][3]) / 2) for line in lines]
    if not ys:
        return None
    type_line_y = int(np.median(ys))
    logger.debug("Type line detected at y=%d (%.1f%% of height)", type_line_y, type_line_y / h * 100)

    return _crop_from_type_line(image, type_line_y)


def _crop_from_type_line(image: np.ndarray, type_line_y: int) -> Optional[np.ndarray]:
    """Given the y-coordinate of the type line, reconstruct the card crop.

    We know the type line sits at ~57.5% of card height, so we can back-calculate
    the card's top and bottom, then use the card's aspect ratio to get the width.
    """
    h, w = image.shape[:2]

    # Estimate full card height from the type-line position.
    card_h = int(type_line_y / _TYPE_LINE_Y_FRAC)
    card_w = int(card_h / _CARD_ASPECT)

    # Centre the estimated card rect on the image.
    cx = w // 2
    x1 = max(0, cx - card_w // 2)
    x2 = min(w, cx + card_w // 2)

    # Find the card top — assume type line y minus the proportional offset.
    card_top = max(0, type_line_y - int(card_h * _TYPE_LINE_Y_FRAC))
    card_bot = min(h, card_top + card_h)

    if (x2 - x1) < 20 or (card_bot - card_top) < 20:
        return None

    logger.debug(
        "Inner detection crop: x=%d–%d, y=%d–%d", x1, x2, card_top, card_bot
    )
    return image[card_top:card_bot, x1:x2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
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

