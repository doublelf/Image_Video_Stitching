from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hstitch.metrics import psnr
from hstitch.pipeline import (
    StitchOptions,
    manifest_overlaps,
    resolve_inputs,
    stitch,
    stitch_files,
)
from hstitch.split import pair_overlaps, slice_bounds, split_image, split_to_dir


def test_concat_round_trip_is_lossless(source_image):
    slices = split_image(source_image, parts=3, overlap=0.0)
    result = stitch(slices, StitchOptions(mode="concat"))
    assert result.shape == source_image.shape
    assert np.array_equal(result, source_image)


def test_concat_scales_the_tiles_to_one_height():
    tiles = [
        np.zeros((100, 50, 3), np.uint8),
        np.zeros((200, 100, 3), np.uint8),
    ]
    result = stitch(tiles, StitchOptions(mode="concat"))
    # the shortest tile defines the canvas height, nothing is upscaled
    assert result.shape[0] == 100
    assert result.shape[1] == 50 + 50


def test_concat_honours_an_explicit_height():
    tiles = [np.zeros((100, 50, 3), np.uint8)]
    result = stitch(tiles, StitchOptions(mode="concat", height=200))
    assert result.shape[0] == 200
    assert result.shape[1] == 100


def test_blend_reconstructs_the_source(source_image):
    parts = 3
    slices = split_image(source_image, parts=parts, overlap=0.15)
    overlap_px = int(round(0.15 * max(1, source_image.shape[1] // parts)))
    overlaps = pair_overlaps(slice_bounds(source_image.shape[1], parts, overlap_px))
    result = stitch(
        slices, StitchOptions(mode="blend", blend="multiband", overlaps=overlaps)
    )
    assert result.shape == source_image.shape
    assert psnr(source_image, result) > 30.0


def test_feather_blend_reconstructs_the_source(source_image):
    parts = 3
    slices = split_image(source_image, parts=parts, overlap=0.15)
    overlap_px = int(round(0.15 * max(1, source_image.shape[1] // parts)))
    overlaps = pair_overlaps(slice_bounds(source_image.shape[1], parts, overlap_px))
    result = stitch(
        slices, StitchOptions(mode="blend", blend="feather", overlaps=overlaps)
    )
    assert result.shape == source_image.shape
    assert psnr(source_image, result) > 40.0


def test_blend_rejects_a_gap():
    with pytest.raises(ValueError):
        stitch(
            [np.zeros((4, 4, 3), np.uint8)],
            StitchOptions(mode="blend", gap=3),
        )


def test_stitch_rejects_a_bad_tile():
    with pytest.raises(ValueError):
        stitch([np.zeros((4, 4), np.uint8)], StitchOptions())


def test_stitch_files_reads_from_disk(tmp_path: Path, source_image):
    from hstitch.io import imwrite

    slices = split_image(source_image, parts=3, overlap=0.0)
    paths = []
    for index, tile in enumerate(slices):
        path = tmp_path / f"tile{index}.png"
        imwrite(path, tile)
        paths.append(str(path))
    result = stitch_files(paths, StitchOptions(mode="concat"))
    assert np.array_equal(result, source_image)


def test_resolve_inputs_handles_directories_and_globs(tmp_path: Path, source_image):
    from hstitch.io import imwrite

    for index in range(2):
        imwrite(tmp_path / f"part{index}.png", source_image)
    files = resolve_inputs([str(tmp_path)])
    assert [item.name for item in files] == ["part0.png", "part1.png"]

    matched = resolve_inputs([str(tmp_path / "*.png")])
    assert len(matched) == 2


def test_resolve_inputs_reports_a_missing_pattern(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_inputs([str(tmp_path / "*.nothing")])


def test_manifest_overlaps_reads_the_split_metadata(tmp_path: Path, source_image):
    from hstitch.io import imwrite

    source = tmp_path / "source.png"
    imwrite(source, source_image)
    manifest = split_to_dir(str(source), str(tmp_path / "parts"), parts=3, overlap=0.15)
    overlaps = manifest_overlaps(manifest)
    assert len(overlaps) == 2
    assert all(value > 0 for value in overlaps)
