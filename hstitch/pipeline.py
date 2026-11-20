"""End to end stitching.

``stitch`` takes the tiles in left to right order and returns the finished
image. Two modes are available:

``concat``
    The tiles are scaled to a common height and placed side by side. The seam
    between two tiles is a hard edge.

``blend``
    The tiles are assumed to overlap horizontally. The overlap is fused with
    the OpenCV blenders, which removes the seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .blend import blend_horizontal
from .compose import ComposeOptions, concat_horizontal, tile_offsets
from .io import imread, list_images
from .resize import normalize_heights

MODES = ("concat", "blend")


@dataclass
class StitchOptions:
    """Everything that controls one stitching run."""

    mode: str = "concat"
    height: Optional[int] = None
    height_strategy: str = "min"
    gap: int = 0
    align: str = "center"
    background: Tuple[int, int, int] = (0, 0, 0)
    blend: str = "multiband"
    overlaps: Optional[Sequence[int]] = field(default=None)

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}, expected one of {MODES}")
        if self.mode == "blend" and self.gap:
            raise ValueError("blend mode needs the tiles to touch, gap must be 0")


def _prepare(images: Sequence[np.ndarray], options: StitchOptions) -> List[np.ndarray]:
    return normalize_heights(
        images, strategy=options.height_strategy, height=options.height
    )


def stitch(
    images: Sequence[np.ndarray],
    options: Optional[StitchOptions] = None,
) -> np.ndarray:
    """Stitch a list of tiles into one horizontal image."""
    options = options or StitchOptions()
    options.validate()
    tiles = list(images)
    if not tiles:
        raise ValueError("no images to stitch")
    for tile in tiles:
        if tile.ndim != 3 or tile.shape[2] != 3:
            raise ValueError("every tile must be a 3 channel BGR image")

    tiles = _prepare(tiles, options)

    if options.mode == "concat":
        compose = ComposeOptions(
            gap=options.gap, align=options.align, background=options.background
        )
        return concat_horizontal(tiles, compose)

    widths = [int(tile.shape[1]) for tile in tiles]
    overlaps = list(options.overlaps) if options.overlaps else [0] * (len(tiles) - 1)
    if len(overlaps) != len(tiles) - 1:
        raise ValueError("overlaps must have one entry less than the tile count")
    offsets = tile_offsets(widths, gap=0, overlaps=overlaps)
    return blend_horizontal(tiles, offsets, mode=options.blend)


def stitch_files(
    paths: Sequence[str],
    options: Optional[StitchOptions] = None,
) -> np.ndarray:
    """Read the tiles from disk and stitch them."""
    images = [imread(path) for path in paths]
    return stitch(images, options)


def resolve_inputs(patterns: Sequence[str], recursive: bool = False) -> List[Path]:
    """Turn the command line inputs into a sorted list of image files.

    Every pattern may be a file, a directory or a glob expression. The result
    is sorted by name so that the left to right order is predictable.
    """
    import glob

    collected: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            collected.extend(list_images(path, recursive=recursive))
            continue
        if path.is_file():
            collected.append(path)
            continue
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no input matched: {pattern}")
        for match in matches:
            candidate = Path(match)
            if candidate.is_dir():
                collected.extend(list_images(candidate, recursive=recursive))
            else:
                collected.append(candidate)
    unique: Dict[str, Path] = {}
    for item in collected:
        unique[str(item).lower()] = item
    return sorted(unique.values(), key=lambda item: str(item).lower())


def manifest_overlaps(manifest: Dict) -> List[int]:
    """Read the pairwise overlaps out of a split manifest."""
    overlaps = manifest.get("overlaps")
    if not overlaps:
        raise ValueError("the manifest does not contain any overlap information")
    return [int(value) for value in overlaps]
