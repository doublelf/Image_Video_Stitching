from __future__ import annotations

from pathlib import Path

import numpy as np

from hstitch.split import (
    overlap_px_for,
    pair_overlaps,
    read_manifest,
    slice_bounds,
    split_image,
    split_to_dir,
)


def test_bounds_tile_the_width_without_overlap():
    bounds = slice_bounds(860, 3, 0)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 860
    for index in range(len(bounds) - 1):
        assert bounds[index][1] == bounds[index + 1][0]
    assert sum(end - start for start, end in bounds) == 860


def test_bounds_share_the_requested_columns():
    bounds = slice_bounds(860, 3, 40)
    overlaps = pair_overlaps(bounds)
    assert overlaps == [40, 40]
    assert sum(end - start for start, end in bounds) == 860 + 80


def test_split_image_keeps_the_pixel_content(source_image):
    slices = split_image(source_image, parts=3, overlap=0.0)
    assert len(slices) == 3
    assert slices[0].shape[0] == source_image.shape[0]
    assert sum(slice.shape[1] for slice in slices) == source_image.shape[1]
    rebuilt = np.hstack(slices)
    assert np.array_equal(rebuilt, source_image)


def test_split_image_with_overlap_shares_columns(source_image):
    slices = split_image(source_image, parts=3, overlap=0.15)
    overlap_px = overlap_px_for(source_image.shape[1], 3, 0.15)
    overlaps = pair_overlaps(
        slice_bounds(source_image.shape[1], 3, overlap_px)
    )
    for index, overlap in enumerate(overlaps):
        assert overlap > 0
        left = slices[index][:, -overlap:]
        right = slices[index + 1][:, :overlap]
        assert np.array_equal(left, right)


def test_split_to_dir_writes_the_slices_and_a_manifest(tmp_path: Path, source_image):
    from hstitch.io import imwrite

    source = tmp_path / "source.png"
    imwrite(source, source_image)
    manifest = split_to_dir(str(source), str(tmp_path / "parts"), parts=3, overlap=0.15)

    assert len(manifest["files"]) == 3
    for name in manifest["files"]:
        assert (tmp_path / "parts" / name).is_file()

    stored = read_manifest(str(tmp_path / "parts"))
    assert stored["overlaps"] == manifest["overlaps"]
    assert stored["size"] == [source_image.shape[1], source_image.shape[0]]
