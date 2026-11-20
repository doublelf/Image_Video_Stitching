"""Split an image into vertical slices.

The slices are the test material of this package: an image is cut into N
vertical strips, optionally with a horizontal overlap, and the strips are
stitched back together afterwards. Without overlap the strips tile the source
exactly, so the round trip is lossless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .io import imread, imwrite


@dataclass(frozen=True)
class SplitOptions:
    """How an image is cut into strips."""

    parts: int = 3
    overlap: float = 0.0

    def validate(self) -> None:
        if self.parts < 1:
            raise ValueError("parts must be at least 1")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")


def slice_bounds(width: int, parts: int, overlap_px: int = 0) -> List[Tuple[int, int]]:
    """Return the ``[x0, x1)`` span of every slice.

    With ``overlap_px == 0`` the spans tile ``[0, width)`` exactly. With a
    positive overlap every interior boundary is widened by half the overlap on
    each side, so two neighbours share exactly ``2 * (overlap_px // 2)``
    columns.
    """
    if parts < 1:
        raise ValueError("parts must be at least 1")
    if overlap_px < 0:
        raise ValueError("overlap_px must not be negative")
    edges = [int(round(index * width / parts)) for index in range(parts + 1)]
    edges[0] = 0
    edges[-1] = width
    half = overlap_px // 2
    bounds: List[Tuple[int, int]] = []
    for index in range(parts):
        x0 = max(0, edges[index] - half) if index > 0 else 0
        x1 = min(width, edges[index + 1] + half) if index < parts - 1 else width
        if x1 <= x0:
            x1 = min(width, x0 + 1)
        bounds.append((x0, x1))
    return bounds


def overlap_px_for(width: int, parts: int, overlap: float) -> int:
    """Convert an overlap ratio (relative to the slice width) into pixels."""
    if overlap <= 0:
        return 0
    slice_width = max(1, width // max(1, parts))
    return int(round(overlap * slice_width))


def split_image(
    image: np.ndarray,
    parts: int = 3,
    overlap: float = 0.0,
) -> List[np.ndarray]:
    """Cut an image into ``parts`` vertical slices."""
    options = SplitOptions(parts=parts, overlap=overlap)
    options.validate()
    height, width = image.shape[:2]
    overlap_px = overlap_px_for(width, options.parts, options.overlap)
    bounds = slice_bounds(width, options.parts, overlap_px)
    return [image[:, x0:x1] for x0, x1 in bounds]


def pair_overlaps(bounds: Sequence[Tuple[int, int]]) -> List[int]:
    """Number of columns two consecutive slices share."""
    overlaps: List[int] = []
    for index in range(len(bounds) - 1):
        overlaps.append(int(bounds[index][1] - bounds[index + 1][0]))
    return overlaps


def split_to_dir(
    source: str,
    out_dir: str,
    parts: int = 3,
    overlap: float = 0.0,
    prefix: str = "part",
    ext: str = ".png",
) -> Dict:
    """Split an image file into a directory and write a manifest next to it.

    The manifest records the slice bounds and the pairwise overlaps, so the
    stitching step can rebuild the exact same layout without guessing.
    """
    options = SplitOptions(parts=parts, overlap=overlap)
    options.validate()
    image = imread(source)
    height, width = image.shape[:2]
    overlap_px = overlap_px_for(width, options.parts, options.overlap)
    bounds = slice_bounds(width, options.parts, overlap_px)
    overlaps = pair_overlaps(bounds)

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    for index, (x0, x1) in enumerate(bounds):
        name = f"{prefix}{index}{ext}"
        imwrite(root / name, image[:, x0:x1])
        files.append(name)

    manifest = {
        "source": str(Path(source)),
        "size": [width, height],
        "parts": options.parts,
        "overlap_ratio": options.overlap,
        "overlap_px": overlap_px,
        "bounds": [list(bound) for bound in bounds],
        "overlaps": overlaps,
        "files": files,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def read_manifest(path: str) -> Dict:
    """Load a manifest written by :func:`split_to_dir`."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
