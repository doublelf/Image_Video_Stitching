"""Shared fixtures.

The tests run against the checked in ``test.png`` when it is available and
fall back to a synthetic image so that the suite stays self contained.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_IMAGE = PROJECT_ROOT / "test.png"


def _synthetic(width: int = 480, height: int = 320) -> np.ndarray:
    """A deterministic image with structure, gradients and noise."""
    rng = np.random.default_rng(20240521)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack(
        [
            128.0 + 90.0 * np.sin(x / 37.0),
            128.0 + 90.0 * np.cos(y / 23.0),
            128.0 + 60.0 * np.sin((x + y) / 51.0),
        ],
        axis=2,
    )
    for shift in range(0, width, 80):
        base[:, shift : shift + 24] += 40.0
    noise = rng.normal(0.0, 6.0, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def source_image() -> np.ndarray:
    """The demo image of the project."""
    if SOURCE_IMAGE.is_file():
        return cv2.imread(str(SOURCE_IMAGE), cv2.IMREAD_COLOR)
    return _synthetic()


@pytest.fixture(scope="session")
def synthetic_image() -> np.ndarray:
    return _synthetic()


@pytest.fixture(scope="session")
def source_path() -> Path:
    return SOURCE_IMAGE


def _write_video(path, frames, fps: float = 25.0, size=None) -> Path:
    """Write a list of frames into an mp4 file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if size is None:
        height, width = frames[0].shape[:2]
        size = (width + width % 2, height + height % 2)
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size
    )
    assert writer.isOpened(), f"cannot open a writer for {target}"
    for frame in frames:
        if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
            padded = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            padded[: frame.shape[0], : frame.shape[1]] = frame
            frame = padded
        writer.write(frame)
    writer.release()
    return target


def _panning_videos(target_dir, source: np.ndarray, offset: int = 287, frames: int = 24):
    """Two videos that show the same scene from a slightly shifted viewpoint."""
    tile = 330
    left = []
    right = []
    for index in range(frames):
        shift = index
        left.append(source[:, shift : shift + tile])
        right.append(source[:, shift + offset : shift + offset + tile])
    return (
        _write_video(Path(target_dir) / "left.mp4", left),
        _write_video(Path(target_dir) / "right.mp4", right),
    )


@pytest.fixture()
def video_writer():
    """Helper that writes a list of frames into an mp4 file."""
    return _write_video


@pytest.fixture()
def video_pair(tmp_path, source_image):
    """Two short videos that overlap by 287 pixels."""
    return _panning_videos(tmp_path, source_image)
