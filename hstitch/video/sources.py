"""Video sources that expose frame timestamps."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Static properties of a video file."""

    path: str
    width: int
    height: int
    fps: float
    frames: int

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": self.frames,
        }


def _needs_ascii_copy(path: Path) -> bool:
    return any(ord(character) > 127 for character in str(path))


class VideoStream:
    """Sequential reader that hands out ``(timestamp, frame)`` pairs.

    ``cv2.VideoCapture`` cannot open a path with non ASCII characters on
    Windows, so such a file is copied to a temporary ASCII name first.
    """

    def __init__(self, path):
        self.source = Path(path)
        self._temporary: Optional[str] = None
        open_path = str(self.source)
        if _needs_ascii_copy(self.source):
            handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            handle.close()
            shutil.copyfile(self.source, handle.name)
            self._temporary = handle.name
            open_path = handle.name
        self._capture = cv2.VideoCapture(open_path)
        if not self._capture.isOpened():
            self.close()
            raise ValueError(f"cannot open video: {self.source}")
        self.info = VideoInfo(
            path=str(self.source),
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)) or 25.0,
            frames=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        self._index = 0
        self._last_timestamp = 0.0

    def next(self) -> Optional[Tuple[float, np.ndarray]]:
        """Read the next frame and return it with its timestamp in seconds."""
        stamp_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        if stamp_ms and stamp_ms > 0:
            timestamp = stamp_ms / 1000.0
        else:
            timestamp = self._index / max(self.info.fps, 1e-6)
        if timestamp < self._last_timestamp:
            timestamp = self._last_timestamp
        self._last_timestamp = timestamp
        self._index += 1
        return timestamp, frame

    @property
    def index(self) -> int:
        """Number of frames that were read so far."""
        return self._index

    def close(self) -> None:
        if getattr(self, "_capture", None) is not None:
            self._capture.release()
            self._capture = None
        if self._temporary:
            try:
                Path(self._temporary).unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
            self._temporary = None

    def __enter__(self) -> "VideoStream":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def probe(path) -> VideoInfo:
    """Read the properties of a video without decoding it."""
    with VideoStream(path) as stream:
        return stream.info
