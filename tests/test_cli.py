from __future__ import annotations

from pathlib import Path

import numpy as np

from hstitch.cli import main
from hstitch.io import imread, imwrite


def test_split_command_writes_slices_and_a_manifest(tmp_path: Path, source_image):
    source = tmp_path / "source.png"
    imwrite(source, source_image)
    out_dir = tmp_path / "parts"

    assert main(["split", "--input", str(source), "--parts", "3",
                 "--overlap", "0.15", "--out-dir", str(out_dir)]) == 0
    assert len(list(out_dir.glob("part*.png"))) == 3
    assert (out_dir / "manifest.json").is_file()


def test_stitch_command_builds_the_strip(tmp_path: Path, source_image):
    slices = split_three(source_image)
    paths = []
    for index, tile in enumerate(slices):
        path = tmp_path / f"part{index}.png"
        imwrite(path, tile)
        paths.append(str(path))
    out = tmp_path / "out.png"

    assert main(["stitch", "--input", *paths, "--out", str(out)]) == 0
    result = imread(out)
    assert result.shape == source_image.shape


def test_stitch_command_blends_with_a_manifest(tmp_path: Path, source_image):
    source = tmp_path / "source.png"
    imwrite(source, source_image)
    parts_dir = tmp_path / "parts"
    assert main(["split", "--input", str(source), "--parts", "3",
                 "--overlap", "0.15", "--out-dir", str(parts_dir)]) == 0

    out = tmp_path / "blend.png"
    assert main(["stitch", "--input", str(parts_dir), "--out", str(out),
                 "--mode", "blend",
                 "--manifest", str(parts_dir / "manifest.json")]) == 0
    result = imread(out)
    assert result.shape == source_image.shape


def test_demo_command_writes_both_results(tmp_path: Path, source_image):
    source = tmp_path / "source.png"
    imwrite(source, source_image)
    out = tmp_path / "demo.png"

    assert main(["demo", "--input", str(source), "--out", str(out)]) == 0
    assert out.is_file()
    assert (tmp_path / "demo_concat.png").is_file()
    assert imread(out).shape == source_image.shape


def split_three(source_image: np.ndarray):
    from hstitch.split import split_image

    return split_image(source_image, parts=3, overlap=0.0)
