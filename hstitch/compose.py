"""Horizontal concatenation.

The tiles are placed from left to right on one canvas. A tile may be shorter
than the canvas, in which case it is aligned to the top, the centre or the
bottom and the remaining area keeps the background colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

ALIGNMENTS = ("top", "center", "bottom")


@dataclass(frozen=True)
class ComposeOptions:
    """Layout of the horizontal strip."""

    gap: int = 0
    align: str = "center"
    background: Tuple[int, int, int] = (0, 0, 0)

    def validate(self) -> None:
        if self.gap < 0:
            raise ValueError("gap must not be negative")
        if self.align not in ALIGNMENTS:
            raise ValueError(
                f"unknown alignment {self.align!r}, expected one of {ALIGNMENTS}"
            )


def tile_offsets(
    widths: Sequence[int],
    gap: int = 0,
    overlaps: Optional[Sequence[int]] = None,
) -> List[int]:
    """Left coordinate of every tile.

    ``overlaps`` gives the number of columns two neighbouring tiles share. The
    shared columns are counted once, so the canvas stays as wide as the source
    material when the tiles come from a split image.
    """
    offsets: List[int] = []
    x = 0
    for index, width in enumerate(widths):
        offsets.append(int(x))
        x += int(width)
        if index < len(widths) - 1:
            overlap = int(overlaps[index]) if overlaps else 0
            x -= overlap
            x += int(gap)
    return offsets


def canvas_size(
    images: Sequence[np.ndarray],
    offsets: Optional[Sequence[int]] = None,
) -> Tuple[int, int]:
    """Return the ``(width, height)`` of the canvas needed for the tiles."""
    if not images:
        raise ValueError("no images to compose")
    height = max(int(img.shape[0]) for img in images)
    if offsets is None:
        width = sum(int(img.shape[1]) for img in images)
    else:
        width = max(
            int(offset) + int(img.shape[1]) for offset, img in zip(offsets, images)
        )
    return int(width), int(height)


def concat_horizontal(
    images: Sequence[np.ndarray],
    options: Optional[ComposeOptions] = None,
) -> np.ndarray:
    """Place the images side by side on one canvas."""
    options = options or ComposeOptions()
    options.validate()
    if not images:
        raise ValueError("no images to concatenate")

    widths = [int(img.shape[1]) for img in images]
    offsets = tile_offsets(widths, options.gap)
    width, height = canvas_size(images, offsets)
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(options.background, dtype=np.uint8)

    for image, offset in zip(images, offsets):
        tile_h, tile_w = image.shape[:2]
        if options.align == "top":
            y = 0
        elif options.align == "bottom":
            y = height - tile_h
        else:
            y = (height - tile_h) // 2
        canvas[y : y + tile_h, offset : offset + tile_w] = image
    return canvas
