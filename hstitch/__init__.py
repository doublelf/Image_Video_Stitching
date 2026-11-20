"""hstitch - horizontal image and video stitching.

The package does two things:

* ``concat`` places a list of images side by side on one canvas. The tiles are
  scaled to a common height, aligned vertically and separated by an optional
  gap.
* ``blend`` does the same but the neighbouring tiles are assumed to overlap
  horizontally. The overlap is fused with the OpenCV stitching blenders
  (``cv2.detail.MultiBandBlender`` / ``cv2.detail.FeatherBlender``) so the
  seam disappears.

The same two modes are available for videos: :func:`stitch_videos` aligns the
frames of several sources by timestamp, estimates the horizontal overlap of the
neighbouring views on its own and writes one mp4 file.

No geometric registration is performed, the tiles are expected to be already
aligned by the caller.
"""

from .blend import blend_horizontal, feather_numpy
from .compose import ComposeOptions, concat_horizontal, tile_offsets
from .io import imread, imwrite
from .pipeline import StitchOptions, stitch, stitch_files
from .resize import normalize_heights, resize_to_height
from .split import SplitOptions, split_image, split_to_dir
from .video import (
    Alignment,
    VideoStitchOptions,
    VideoStream,
    estimate_alignment,
    probe,
    stitch_videos,
)

__version__ = "0.2.0"

__all__ = [
    "Alignment",
    "ComposeOptions",
    "SplitOptions",
    "StitchOptions",
    "VideoStitchOptions",
    "VideoStream",
    "blend_horizontal",
    "feather_numpy",
    "concat_horizontal",
    "estimate_alignment",
    "imread",
    "imwrite",
    "normalize_heights",
    "probe",
    "resize_to_height",
    "split_image",
    "split_to_dir",
    "stitch",
    "stitch_files",
    "stitch_videos",
    "tile_offsets",
]
