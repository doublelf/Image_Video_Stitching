"""Automatic alignment of neighbouring views.

The module estimates where a tile has to be placed relative to its left
neighbour so that the horizontal overlap can be fused without a visible seam.
Three independent estimators are used:

``template``
    Normalised cross correlation of a strip of the right frame against the
    left frame (``cv2.matchTemplate``). The strip width is tried in several
    sizes, so the estimator also works when the two frames share only a
    narrow band.

``features``
    SIFT key points matched with a brute force matcher and a RANSAC fit.
    Works across different cameras and small perspective changes.

``profile``
    Normalised cross correlation of the column and row intensity profiles.
    Cheap and very robust for a camera that pans sideways.

All estimators report the displacement ``(sx, sy)`` of the second frame
relative to the first one, that is ``b(x, y) == a(x - sx, y - sy)``. The
placement offset of the second tile is therefore ``(-sx, -sy)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

TEMPLATE_METHOD = "template"
FEATURE_METHOD = "features"
PROFILE_METHOD = "profile"

TEMPLATE_WIDTHS = (8, 16, 32, 64, 128)
TEMPLATE_MAX_HEIGHT = 200


@dataclass
class Alignment:
    """Where a tile has to be placed relative to its left neighbour."""

    dx: int
    dy: int
    overlap: int
    score: float
    method: str
    inliers: int = 0

    def as_dict(self) -> dict:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "overlap": self.overlap,
            "score": round(float(self.score), 4),
            "method": self.method,
            "inliers": int(self.inliers),
        }


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _resized(image: np.ndarray, scale: float) -> np.ndarray:
    """Scale an image down so the feature matching stays fast."""
    if scale >= 1.0:
        return image
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _shift_by_template(
    left: np.ndarray, right: np.ndarray
) -> Optional[Tuple[float, float, float, int]]:
    """Match a strip of the right frame inside the left frame.

    The strip is taken from the left edge of the right frame, the position
    where it is found is the horizontal offset of the two frames.
    """
    left_gray = _gray(left)
    right_gray = _gray(right)
    height = min(left_gray.shape[0], right_gray.shape[0])
    if height < 8:
        return None
    strip_height = min(height, TEMPLATE_MAX_HEIGHT)
    top = (height - strip_height) // 2

    best = None
    for width in TEMPLATE_WIDTHS:
        if width >= right_gray.shape[1] or width >= left_gray.shape[1]:
            continue
        template = right_gray[top : top + strip_height, :width]
        response = cv2.matchTemplate(
            left_gray[:height], template, cv2.TM_CCOEFF_NORMED
        )
        _, score, _, location = cv2.minMaxLoc(response)
        if best is None or score > best[2]:
            best = (float(location[0]), float(location[1] - top), float(score), width)
    if best is None:
        return None
    # the strip of the right frame was found at ``x`` inside the left frame,
    # so the right frame is displaced by -x
    return -best[0], -best[1], best[2], 0


def _shift_by_features(
    left: np.ndarray, right: np.ndarray
) -> Optional[Tuple[float, float, float, int]]:
    """SIFT based estimate of the displacement of ``right`` against ``left``."""
    scale = min(1.0, 480.0 / float(max(left.shape[0], right.shape[0])))
    left_gray = _gray(_resized(left, scale))
    right_gray = _gray(_resized(right, scale))
    sift = cv2.SIFT_create()
    key_left, desc_left = sift.detectAndCompute(left_gray, None)
    key_right, desc_right = sift.detectAndCompute(right_gray, None)
    if desc_left is None or desc_right is None:
        return None
    if len(key_left) < 8 or len(key_right) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_left, desc_right, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 8:
        return None

    points_left = np.float32(
        [key_left[match.queryIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    points_right = np.float32(
        [key_right[match.trainIdx].pt for match in good]
    ).reshape(-1, 1, 2)

    matrix, inliers = cv2.estimateAffinePartial2D(
        points_left, points_right, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if matrix is None or inliers is None:
        return None
    count = int(inliers.sum())
    if count < 8:
        return None
    ratio = count / float(len(good))
    # the median of the inlier displacements is the pure translation, the
    # affine fit would also absorb a small rotation or scale
    displacements = (points_right - points_left).reshape(-1, 2)
    displacements = displacements[inliers.ravel().astype(bool)]
    shift_x = float(np.median(displacements[:, 0]))
    shift_y = float(np.median(displacements[:, 1]))
    return shift_x / scale, shift_y / scale, ratio, count


def _profile(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float64) - float(values.mean())


def _best_shift(reference: np.ndarray, other: np.ndarray) -> Tuple[float, float]:
    """Shift ``s`` with ``other(x) == reference(x - s)`` and the best score."""
    size = min(len(reference), len(other))
    if size < 16:
        return 0.0, -2.0
    best_shift = 0
    best_score = -2.0
    for shift in range(-(size - 1), size):
        if shift >= 0:
            first = reference[: size - shift]
            second = other[shift : shift + size - shift]
        else:
            first = reference[-shift : -shift + size + shift]
            second = other[: size + shift]
        if len(first) < 16:
            continue
        left = _profile(first)
        right = _profile(second)
        denominator = float(np.sqrt((left * left).sum() * (right * right).sum()))
        if denominator < 1e-9:
            continue
        score = float((left * right).sum() / denominator)
        if score > best_score:
            best_score = score
            best_shift = shift
    return float(best_shift), best_score


def _shift_by_profile(
    left: np.ndarray, right: np.ndarray
) -> Optional[Tuple[float, float, float, int]]:
    """Profile correlation estimate of the displacement of ``right``."""
    left_gray = _gray(left).astype(np.float64)
    right_gray = _gray(right).astype(np.float64)

    columns = _best_shift(left_gray.mean(axis=0), right_gray.mean(axis=0))
    rows = _best_shift(left_gray.mean(axis=1), right_gray.mean(axis=1))
    score = 0.5 * (columns[1] + rows[1])
    return columns[0], rows[0], score, 0


def estimate_shift(
    left: np.ndarray,
    right: np.ndarray,
    methods: Sequence[str] = (TEMPLATE_METHOD, FEATURE_METHOD, PROFILE_METHOD),
) -> Optional[Tuple[float, float, float, str, int]]:
    """Estimate the displacement of ``right`` against ``left``.

    Returns ``(sx, sy, score, method, inliers)`` or ``None`` when no
    estimator is confident enough.
    """
    best = None
    for method in methods:
        if method == TEMPLATE_METHOD:
            estimate = _shift_by_template(left, right)
        elif method == FEATURE_METHOD:
            estimate = _shift_by_features(left, right)
        elif method == PROFILE_METHOD:
            estimate = _shift_by_profile(left, right)
        else:
            raise ValueError(f"unknown alignment method: {method!r}")
        if estimate is None:
            continue
        sx, sy, score, inliers = estimate
        if score < 0.35:
            continue
        if method == FEATURE_METHOD and inliers < 8:
            continue
        candidate = (sx, sy, score, method, inliers)
        if best is None or score > best[2]:
            best = candidate
    return best


def estimate_alignment(
    left: np.ndarray,
    right: np.ndarray,
    min_overlap: int = 8,
    max_shift_ratio: float = 0.98,
) -> Optional[Alignment]:
    """Estimate where ``right`` has to be placed relative to ``left``.

    The returned alignment carries the horizontal offset of the right tile,
    the vertical offset and the number of columns the two tiles share. When no
    overlap can be found the alignment is ``None`` and the caller falls back
    to a plain side by side layout.
    """
    estimate = estimate_shift(left, right)
    if estimate is None:
        return None
    sx, sy, score, method, inliers = estimate

    left_width = int(left.shape[1])
    right_width = int(right.shape[1])
    # the right tile has to start inside the left tile
    limit = int(left_width * max_shift_ratio)

    placement_x = int(round(-sx))
    placement_y = int(round(-sy))
    if placement_x <= 0 or placement_x >= limit:
        return None
    overlap = min(left_width - placement_x, right_width)
    if overlap < min_overlap:
        return None
    return Alignment(
        dx=placement_x,
        dy=placement_y,
        overlap=overlap,
        score=score,
        method=method,
        inliers=inliers,
    )


def median_alignment(
    alignments: Sequence[Optional[Alignment]],
    fallback: Optional[Alignment] = None,
) -> Optional[Alignment]:
    """Combine several estimates of the same pair into one."""
    usable = [item for item in alignments if item is not None]
    if not usable:
        return fallback
    dx = int(round(float(np.median([item.dx for item in usable]))))
    dy = int(round(float(np.median([item.dy for item in usable]))))
    overlap = int(round(float(np.median([item.overlap for item in usable]))))
    score = float(np.median([item.score for item in usable]))
    inliers = int(round(float(np.median([item.inliers for item in usable]))))
    method = usable[0].method
    return Alignment(
        dx=dx,
        dy=dy,
        overlap=overlap,
        score=score,
        method=method,
        inliers=inliers,
    )
