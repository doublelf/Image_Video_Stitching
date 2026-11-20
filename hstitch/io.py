"""Unicode safe image input and output.

``cv2.imread`` and ``cv2.imwrite`` go through the C runtime, which mangles
non ASCII paths on Windows. The helpers below read the file into a buffer with
the Python IO layer and hand the buffer to OpenCV, so a path such as
``D:\\测试\\图片.png`` works everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm")


def imread(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from disk and return it as a BGR array.

    Raises ``FileNotFoundError`` when the file is missing and ``ValueError``
    when OpenCV cannot decode it.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"image not found: {target}")
    buffer = np.fromfile(str(target), dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError(f"image file is empty: {target}")
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ValueError(f"unsupported or corrupted image file: {target}")
    return image


def imwrite(path: PathLike, image: np.ndarray, ext: Optional[str] = None) -> Path:
    """Encode an image and write it to disk, returning the written path.

    The parent directory is created when it does not exist.
    """
    target = Path(path)
    suffix = ext or target.suffix or ".png"
    if not suffix.startswith("."):
        suffix = "." + suffix
    try:
        ok, buffer = cv2.imencode(suffix, image)
    except cv2.error as error:
        raise ValueError(f"failed to encode image as {suffix}: {error}") from error
    if not ok:
        raise ValueError(f"failed to encode image as {suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(target))
    return target


def list_images(directory: PathLike, recursive: bool = False) -> list:
    """List the image files of a directory, sorted by name."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    pattern = "**/*" if recursive else "*"
    files = [
        item
        for item in root.glob(pattern)
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files, key=lambda item: str(item).lower())
