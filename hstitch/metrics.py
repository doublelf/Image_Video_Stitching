"""Quality metrics used by the tests and by the demo command."""

from __future__ import annotations

import cv2
import numpy as np


def psnr(reference: np.ndarray, candidate: np.ndarray, max_value: float = 255.0) -> float:
    """Peak signal to noise ratio between two images of the same size."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: {reference.shape} against {candidate.shape}"
        )
    difference = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(difference * difference))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((max_value * max_value) / mse))


def max_abs_difference(reference: np.ndarray, candidate: np.ndarray) -> int:
    """Largest absolute per channel difference between two images."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: {reference.shape} against {candidate.shape}"
        )
    difference = cv2.absdiff(reference, candidate)
    return int(difference.max())
