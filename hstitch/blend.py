"""Seam blending for tiles that overlap horizontally.

The fusion itself is done by the OpenCV stitching module
(``cv2.detail.MultiBandBlender`` and ``cv2.detail.FeatherBlender``), which
implements the multi band blender of Brown and Lowe and the feather blender
that OpenCV ships with its stitching pipeline.

Three details of the OpenCV 5.0 API are worth remembering:

* ``prepare`` takes the destination rectangle as a four element tuple
  ``(x, y, width, height)``.
* ``feed`` only accepts ``CV_16SC3`` images, so an 8 bit image is widened to
  16 bit. The values stay in the 0..255 range: the blenders build a Laplacian
  pyramid in ``int16`` and the upsampling step multiplies the values by four,
  which overflows as soon as the input is scaled up.
* ``blend`` writes its result into the destination array in the same 16 bit
  format, the destination therefore has to be an ``int16`` array as well. The
  result is scaled back to 8 bit with :func:`from_int16`.

This is the same convention the OpenCV stitcher itself uses.

A pure NumPy feather implementation is kept as a fallback so the package still
produces a result when the OpenCV blenders refuse the input.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

BLEND_MODES = ("multiband", "feather")

Offset = Union[int, Tuple[int, int], List[int]]


def to_int16(image: np.ndarray) -> np.ndarray:
    """Widen an 8 bit BGR image into the ``CV_16SC3`` form the blenders want.

    The pixel values stay in the 0..255 range, exactly like the images the
    OpenCV stitcher feeds into its blenders.
    """
    return image.astype(np.int16)


def from_int16(image: np.ndarray) -> np.ndarray:
    """Convert the 16 bit blender result back into an 8 bit BGR image."""
    return np.clip(image, 0, 255).astype(np.uint8)


def normalize_offsets(offsets: Sequence[Offset]) -> List[Tuple[int, int]]:
    """Turn the offset list into ``(x, y)`` pairs.

    A bare integer is read as the horizontal offset, a pair carries the
    vertical offset as well.
    """
    pairs: List[Tuple[int, int]] = []
    for item in offsets:
        if isinstance(item, (tuple, list)):
            pairs.append((int(item[0]), int(item[1])))
        else:
            pairs.append((int(item), 0))
    return pairs


def canvas_size(
    images: Sequence[np.ndarray],
    offsets: Sequence[Offset],
) -> Tuple[int, int]:
    """Canvas size needed for the tiles, ignoring negative vertical offsets."""
    pairs = normalize_offsets(offsets)
    width = max(
        x + int(image.shape[1]) for (x, _), image in zip(pairs, images)
    )
    top = min(y for (_, y), _ in zip(pairs, images))
    bottom = max(y + int(image.shape[0]) for (_, y), image in zip(pairs, images))
    return int(width), int(bottom - top)


def feather_numpy(
    images: Sequence[np.ndarray],
    offsets: Sequence[Offset],
    canvas: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Distance transform weighted average, used when OpenCV refuses the input."""
    pairs = normalize_offsets(offsets)
    if canvas is None:
        canvas = canvas_size(images, pairs)
    canvas_width, canvas_height = canvas
    top = min(y for (_, y) in pairs)

    accumulator = np.zeros((canvas_height, canvas_width, 3), dtype=np.float64)
    weights = np.zeros((canvas_height, canvas_width), dtype=np.float64)
    for image, (x, y) in zip(images, pairs):
        tile_h, tile_w = image.shape[:2]
        row = y - top
        mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        mask[row : row + tile_h, x : x + tile_w] = 255
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        weight = distance[row : row + tile_h, x : x + tile_w]
        accumulator[row : row + tile_h, x : x + tile_w] += (
            image.astype(np.float64) * weight[:, :, None]
        )
        weights[row : row + tile_h, x : x + tile_w] += weight
    result = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    valid = weights > 1e-9
    if valid.any():
        blended = accumulator[valid] / weights[valid][:, None]
        result[valid] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def blend_horizontal(
    images: Sequence[np.ndarray],
    offsets: Sequence[Offset],
    mode: str = "multiband",
    canvas: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Fuse overlapping tiles into one seamless image.

    ``offsets`` gives the position of every tile, either as ``(x, y)`` pairs or
    as plain horizontal offsets. ``mode`` selects the OpenCV blender,
    ``multiband`` removes both exposure and structural seams, ``feather`` only
    ramps the intensities.
    """
    if mode not in BLEND_MODES:
        raise ValueError(f"unknown blend mode {mode!r}, expected one of {BLEND_MODES}")
    if not images:
        raise ValueError("no images to blend")
    if len(images) != len(offsets):
        raise ValueError("images and offsets must have the same length")

    pairs = normalize_offsets(offsets)
    if canvas is None:
        canvas = canvas_size(images, pairs)
    canvas_width, canvas_height = canvas
    top = min(y for (_, y) in pairs)

    blender_type = (
        cv2.detail.Blender_MULTI_BAND
        if mode == "multiband"
        else cv2.detail.Blender_FEATHER
    )
    try:
        blender = cv2.detail.Blender_createDefault(blender_type)
        blender.prepare((0, 0, canvas_width, canvas_height))
        for image, (x, y) in zip(images, pairs):
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            blender.feed(to_int16(image), mask, (int(x), int(y) - top))
        destination = np.zeros((canvas_height, canvas_width, 3), dtype=np.int16)
        destination_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        blender.blend(destination, destination_mask)
        if destination_mask.any():
            return from_int16(destination)
    except cv2.error as error:  # pragma: no cover - depends on the input
        raise RuntimeError(f"the OpenCV blender failed: {error}") from error

    return feather_numpy(images, pairs, (canvas_width, canvas_height))


def seam_energy(
    image: np.ndarray,
    columns: Sequence[int],
    radius: int = 2,
) -> List[float]:
    """Mean absolute column difference around the given columns.

    The value is a cheap seam quality probe: a hard seam shows a spike, a
    blended seam stays close to the noise level of the scene.
    """
    scores: List[float] = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for column in columns:
        left = max(0, column - radius)
        right = min(gray.shape[1], column + radius + 1)
        if right - left < 2:
            scores.append(0.0)
            continue
        difference = np.abs(np.diff(gray[:, left:right], axis=1))
        scores.append(float(difference.mean()))
    return scores
