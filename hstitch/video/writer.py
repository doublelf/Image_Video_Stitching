"""Video writer that also copes with non ASCII paths."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def _needs_ascii_copy(path: Path) -> bool:
    return any(ord(character) > 127 for character in str(path))


class VideoWriter:
    """Writes frames into a video file.

    ``cv2.VideoWriter`` cannot write to a path with non ASCII characters on
    Windows, so the frames go into a temporary file that is moved to the final
    destination when the writer is closed.
    """

    def __init__(
        self,
        path,
        fps: float,
        size: Tuple[int, int],
        fourcc: str = "mp4v",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # the common codecs only handle even dimensions
        self.size = (int(size[0]) + int(size[0]) % 2, int(size[1]) + int(size[1]) % 2)
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self._temporary: Optional[str] = None

        target = str(self.path)
        if _needs_ascii_copy(self.path):
            handle = tempfile.NamedTemporaryFile(suffix=self.path.suffix, delete=False)
            handle.close()
            self._temporary = handle.name
            target = handle.name

        self._writer = cv2.VideoWriter(
            target, cv2.VideoWriter_fourcc(*fourcc), self.fps, self.size
        )
        if not self._writer.isOpened():
            self.close()
            raise RuntimeError(
                f"cannot open the video writer for {self.path} (fourcc={fourcc})"
            )

    def write(self, frame: np.ndarray) -> None:
        """Write one frame, padding it to the codec size when needed."""
        height, width = frame.shape[:2]
        target_width, target_height = self.size
        if width > target_width or height > target_height:
            frame = frame[:target_height, :target_width]
        elif width != target_width or height != target_height:
            padded = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
            padded[:height, :width] = frame
            frame = padded
        self._writer.write(frame)

    def close(self) -> None:
        if getattr(self, "_writer", None) is not None:
            self._writer.release()
            self._writer = None
        if self._temporary:
            shutil.move(self._temporary, str(self.path))
            self._temporary = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
