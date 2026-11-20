"""Height normalisation.

A horizontal strip has a single canvas height, so every tile is scaled to a
common height while its aspect ratio is preserved. Downscaling uses
``INTER_AREA`` (box filter, no aliasing), upscaling uses Lanczos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

HEIGHT_STRATEGIES = ("min", "max", "first")


@dataclass(frozen=True)
class ResizePlan:
    """The geometry of a single resize operation."""

    scale: float
    width: int
    height: int


def choose_target_height(
    images: Sequence[np.ndarray],
    strategy: str = "min",
    height: Optional[int] = None,
) -> int:
    """Pick the canvas height for a set of tiles.

    ``height`` wins over ``strategy``. The strategies are ``min`` (never
    upscale, the default), ``max`` (fill the canvas) and ``first`` (follow the
    first tile).
    """
    if height is not None:
        target = int(height)
        if target <= 0:
            raise ValueError("height must be positive")
        return target
    if not images:
        raise ValueError("no images given")
    if strategy not in HEIGHT_STRATEGIES:
        raise ValueError(
            f"unknown height strategy {strategy!r}, expected one of {HEIGHT_STRATEGIES}"
        )
    heights = [int(img.shape[0]) for img in images]
    if strategy == "min":
        return min(heights)
    if strategy == "max":
        return max(heights)
    return heights[0]


def plan_resize(image: np.ndarray, target_height: int) -> ResizePlan:
    """Compute the target size of a tile without resizing it."""
    height, width = image.shape[:2]
    scale = float(target_height) / float(height)
    return ResizePlan(
        scale=scale,
        width=max(1, int(round(width * scale))),
        height=int(target_height),
    )


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Scale a tile to ``target_height`` while keeping its aspect ratio."""
    plan = plan_resize(image, target_height)
    interpolation = cv2.INTER_AREA if plan.scale < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (plan.width, plan.height), interpolation=interpolation)


def normalize_heights(
    images: Sequence[np.ndarray],
    strategy: str = "min",
    height: Optional[int] = None,
) -> List[np.ndarray]:
    """Scale every tile to one common height."""
    target = choose_target_height(images, strategy, height)
    return [resize_to_height(image, target) for image in images]
