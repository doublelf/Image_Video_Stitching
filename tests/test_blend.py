from __future__ import annotations

import numpy as np
import pytest

from hstitch.blend import blend_horizontal, from_int16, seam_energy, to_int16
from hstitch.split import overlap_px_for, pair_overlaps, slice_bounds, split_image


def _split_pair(image: np.ndarray, overlap: float = 0.2):
    """Split into two slices and return the slices plus their overlap."""
    overlap_px = overlap_px_for(image.shape[1], 2, overlap)
    bounds = slice_bounds(image.shape[1], 2, overlap_px)
    overlaps = pair_overlaps(bounds)
    slices = [image[:, x0:x1] for x0, x1 in bounds]
    return slices, overlaps[0]


def test_to_int16_keeps_the_pixel_range():
    image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    converted = to_int16(image)
    assert converted.dtype == np.int16
    assert converted[0, 0, 0] == 0
    assert converted[0, 1, 0] == 255


def test_from_int16_round_trips():
    image = np.array([[[0, 64, 128], [255, 200, 100]]], dtype=np.uint8)
    assert np.array_equal(from_int16(to_int16(image)), image)


def test_blend_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        blend_horizontal([np.zeros((4, 4, 3), np.uint8)], [0], mode="unknown")


def test_blend_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        blend_horizontal([np.zeros((4, 4, 3), np.uint8)], [0, 10])


def test_blend_keeps_the_canvas_size(source_image):
    slices, overlap = _split_pair(source_image)
    offsets = [0, slices[0].shape[1] - overlap]
    result = blend_horizontal(slices, offsets, mode="multiband")
    assert result.shape == source_image.shape


def test_blend_removes_the_seam(source_image):
    slices, overlap = _split_pair(source_image)
    offsets = [0, slices[0].shape[1] - overlap]
    blended = blend_horizontal(slices, offsets, mode="multiband")

    seam = offsets[1]
    hard = np.hstack(slices)
    blended_score = seam_energy(blended, [seam])[0]
    hard_score = seam_energy(hard, [slices[0].shape[1]])[0]
    assert blended_score < hard_score


def test_feather_blender_also_works(source_image):
    slices, overlap = _split_pair(source_image)
    offsets = [0, slices[0].shape[1] - overlap]
    feather = blend_horizontal(slices, offsets, mode="feather")
    assert feather.shape == source_image.shape
    assert feather.dtype == np.uint8


def test_the_numpy_fallback_reproduces_the_seam(source_image):
    from hstitch.blend import feather_numpy

    slices, overlap = _split_pair(source_image)
    offsets = [0, slices[0].shape[1] - overlap]
    fallback = feather_numpy(slices, offsets, source_image.shape[1::-1])
    reference = blend_horizontal(slices, offsets, mode="feather")
    assert fallback.shape == reference.shape
    assert np.abs(fallback.astype(int) - reference.astype(int)).max() <= 2
