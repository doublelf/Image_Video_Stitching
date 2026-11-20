"""Video stitching: read several videos, align them and write one mp4."""

from .align import Alignment, estimate_alignment, estimate_shift, median_alignment
from .pipeline import VideoStitchOptions, stitch_videos
from .sources import VideoInfo, VideoStream, probe
from .sync import FrameCursor, aligned_groups
from .writer import VideoWriter

__all__ = [
    "Alignment",
    "FrameCursor",
    "VideoInfo",
    "VideoStream",
    "VideoStitchOptions",
    "VideoWriter",
    "aligned_groups",
    "estimate_alignment",
    "estimate_shift",
    "median_alignment",
    "probe",
    "stitch_videos",
]
