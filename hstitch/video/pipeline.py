"""Stitch several videos into one horizontal video.

The pipeline is:

1. open every source and align the frames by timestamp,
2. estimate where the neighbouring views overlap (optional, ``auto_overlap``),
3. scale every frame to a common height,
4. stitch the frames with the image pipeline and write them into an mp4 file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..blend import blend_horizontal
from ..compose import ComposeOptions, concat_horizontal
from .align import Alignment, estimate_alignment, median_alignment
from .sources import VideoStream
from .sync import aligned_groups
from .writer import VideoWriter

MODES = ("concat", "blend")


@dataclass
class VideoStitchOptions:
    """Everything that controls one video stitching run."""

    mode: str = "blend"
    sync: str = "timestamp"
    fps: Optional[float] = None
    height: Optional[int] = None
    height_strategy: str = "min"
    align: str = "center"
    background: Tuple[int, int, int] = (0, 0, 0)
    blend: str = "multiband"
    auto_overlap: bool = True
    overlap_px: Optional[Sequence[int]] = None
    estimate_frames: int = 3
    max_frames: Optional[int] = None
    fourcc: str = "mp4v"

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}, expected one of {MODES}")
        if self.mode == "blend" and self.overlap_px is None and not self.auto_overlap:
            raise ValueError("blend mode needs overlap_px or auto_overlap")


ProgressHook = Callable[[str, int, int], None]


def _notify(hook: Optional[ProgressHook], stage: str, done: int, total: int) -> None:
    if hook is not None:
        hook(stage, done, total)


def _target_height(heights: Sequence[int], options: VideoStitchOptions) -> int:
    if options.height is not None:
        return int(options.height)
    if options.height_strategy == "max":
        return max(heights)
    if options.height_strategy == "first":
        return heights[0]
    return min(heights)


def _normalize(frame: np.ndarray, target_height: int) -> np.ndarray:
    """Scale a frame to the common height, keeping its aspect ratio."""
    if frame.shape[0] == target_height:
        return frame
    scale = target_height / float(frame.shape[0])
    return cv2.resize(
        frame,
        (max(1, int(round(frame.shape[1] * scale))), target_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4,
    )


def _layout(
    widths: Sequence[int],
    tile_height: int,
    alignments: Sequence[Optional[Alignment]],
    mode: str,
) -> Tuple[List[Tuple[int, int]], int, int]:
    """Compute the tile positions and the canvas size."""
    offsets: List[Tuple[int, int]] = [(0, 0)]
    for index, alignment in enumerate(alignments):
        if mode == "blend" and alignment is not None:
            dx = alignment.dx
            dy = alignment.dy
        else:
            dx = widths[index]
            dy = 0
        x, y = offsets[-1]
        offsets.append((x + dx, y + dy))
    top = min(y for _, y in offsets)
    bottom = max(y + tile_height for _, y in offsets)
    width = max(x + widths[index] for index, (x, _) in enumerate(offsets))
    return offsets, int(width), int(bottom - top)


def _compose(
    frames: Sequence[np.ndarray],
    offsets: Sequence[Tuple[int, int]],
    options: VideoStitchOptions,
    canvas: Tuple[int, int],
) -> np.ndarray:
    if options.mode == "concat":
        compose = ComposeOptions(
            gap=0, align=options.align, background=options.background
        )
        return concat_horizontal(frames, compose)
    shifted = [(x, y - min(item[1] for item in offsets)) for x, y in offsets]
    return blend_horizontal(frames, shifted, mode=options.blend, canvas=canvas)


def stitch_videos(
    paths: Sequence[str],
    out_path: str,
    options: Optional[VideoStitchOptions] = None,
    progress: Optional[ProgressHook] = None,
) -> Dict:
    """Stitch the videos and write the result to ``out_path``.

    Returns a report with the sources, the estimated alignments, the number of
    written frames and the geometry of the result.
    """
    options = options or VideoStitchOptions()
    options.validate()
    if len(paths) < 2:
        raise ValueError("at least two videos are needed")

    streams = [VideoStream(path) for path in paths]
    try:
        target_height = _target_height(
            [stream.info.height for stream in streams], options
        )

        groups = aligned_groups(streams, options.sync)
        buffered: List[List[np.ndarray]] = []
        for _ in range(max(1, options.estimate_frames)):
            group = next(groups, None)
            if group is None:
                break
            buffered.append(group)

        alignments: List[Optional[Alignment]] = [None] * (len(paths) - 1)
        if options.mode == "blend":
            if options.overlap_px:
                for index, value in enumerate(options.overlap_px[: len(paths) - 1]):
                    alignments[index] = Alignment(
                        dx=0,
                        dy=0,
                        overlap=int(value),
                        score=1.0,
                        method="manual",
                    )
            else:
                for index in range(len(paths) - 1):
                    estimates = []
                    for group in buffered[1:] or buffered:
                        if len(group) <= index + 1:
                            continue
                        left = _normalize(group[index], target_height)
                        right = _normalize(group[index + 1], target_height)
                        estimates.append(estimate_alignment(left, right))
                    alignments[index] = median_alignment(estimates)

        widths = [
            max(1, int(round(stream.info.width * target_height / stream.info.height)))
            for stream in streams
        ]
        # the overlap follows from the horizontal offset, keep both consistent
        for index, alignment in enumerate(alignments):
            if alignment is None:
                continue
            if alignment.method == "manual":
                dx = widths[index] - alignment.overlap
                overlap = alignment.overlap
            else:
                dx = alignment.dx
                overlap = min(widths[index] - dx, widths[index + 1])
            alignments[index] = Alignment(
                dx=dx,
                dy=alignment.dy,
                overlap=overlap,
                score=alignment.score,
                method=alignment.method,
                inliers=alignment.inliers,
            )
        if options.mode == "blend" and options.overlap_px:
            # a manual overlap means the next tile starts after the shared columns
            offsets: List[Tuple[int, int]] = [(0, 0)]
            for index, alignment in enumerate(alignments):
                dx = widths[index] - int(alignment.overlap)
                x, y = offsets[-1]
                offsets.append((x + dx, y + int(alignment.dy)))
            top = min(y for _, y in offsets)
            bottom = max(y + target_height for _, y in offsets)
            canvas_width = max(x + widths[index] for index, (x, _) in enumerate(offsets))
            offsets = [(x, y - top) for x, y in offsets]
            canvas = (int(canvas_width), int(bottom - top))
        else:
            offsets, canvas_width, canvas_height = _layout(
                widths, target_height, alignments, options.mode
            )
            canvas = (canvas_width, canvas_height)

        output_fps = options.fps or min(stream.info.fps for stream in streams)

        written = 0
        with VideoWriter(
            out_path, output_fps, canvas, options.fourcc
        ) as writer:
            pending = list(buffered)
            while True:
                if pending:
                    group = pending.pop(0)
                else:
                    group = next(groups, None)
                    if group is None:
                        break
                frames = [_normalize(frame, target_height) for frame in group]
                writer.write(_compose(frames, offsets, options, canvas))
                written += 1
                _notify(progress, "stitch", written, options.max_frames or 0)
                if options.max_frames and written >= options.max_frames:
                    break

        return {
            "sources": [stream.info.as_dict() for stream in streams],
            "output": str(Path(out_path)),
            "size": [canvas[0], canvas[1]],
            "fps": float(output_fps),
            "frames": written,
            "mode": options.mode,
            "sync": options.sync,
            "alignments": [item.as_dict() if item else None for item in alignments],
        }
    finally:
        for stream in streams:
            stream.close()
