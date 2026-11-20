from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hstitch.io import imread, imwrite, list_images


def test_imread_reads_the_project_image(source_path: Path):
    if not source_path.is_file():
        pytest.skip("test.png is not available")
    image = imread(source_path)
    assert image.ndim == 3
    assert image.shape[2] == 3


def test_imread_reports_a_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        imread(tmp_path / "does_not_exist.png")


def test_imread_rejects_a_broken_file(tmp_path: Path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image at all")
    with pytest.raises(ValueError):
        imread(broken)


def test_imwrite_creates_the_parent_directory(tmp_path: Path):
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    target = tmp_path / "nested" / "folder" / "out.png"
    written = imwrite(target, image)
    assert written.is_file()
    assert imread(written).shape == image.shape


def test_round_trip_keeps_the_pixels(tmp_path: Path):
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)
    target = tmp_path / "random.png"
    imwrite(target, image)
    assert np.array_equal(imread(target), image)


def test_unicode_path_round_trip(tmp_path: Path):
    image = np.full((10, 10, 3), (10, 20, 30), dtype=np.uint8)
    target = tmp_path / "测试目录" / "图片_001.png"
    imwrite(target, image)
    assert target.is_file()
    assert np.array_equal(imread(target), image)


def test_list_images_is_sorted_and_filtered(tmp_path: Path):
    for name in ("b.png", "a.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    files = list_images(tmp_path)
    assert [item.name for item in files] == ["a.jpg", "b.png"]


def test_imwrite_rejects_an_unknown_extension(tmp_path: Path):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        imwrite(tmp_path / "out.weird", image)
