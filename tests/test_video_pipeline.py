from __future__ import annotations

import cv2
import numpy as np
import pytest

from hstitch.metrics import psnr
from hstitch.video.pipeline import VideoStitchOptions, stitch_videos


def _read_first_frame(path):
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    assert ok
    return frame


def _expected(source_image, shape):
    """The source padded to the size the video writer produced."""
    expected = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    height = min(shape[0], source_image.shape[0])
    width = min(shape[1], source_image.shape[1])
    expected[:height, :width] = source_image[:height, :width]
    return expected


def test_stitch_videos_builds_a_panorama(tmp_path, video_pair, source_image):
    left, right = video_pair
    output = tmp_path / "panorama.mp4"
    report = stitch_videos(
        [str(left), str(right)],
        str(output),
        VideoStitchOptions(mode="blend", estimate_frames=3),
    )
    assert output.is_file()
    assert report["frames"] == 24
    assert report["alignments"][0] is not None
    assert abs(report["alignments"][0]["dx"] - 287) <= 3
    # the canvas is as wide as the two tiles minus their overlap
    assert report["size"][0] == 330 + 330 - report["alignments"][0]["overlap"]

    frame = _read_first_frame(output)
    expected = _expected(source_image, frame.shape)
    assert psnr(expected, frame) > 18.0


def test_the_blend_removes_the_seam(tmp_path, video_pair, source_image):
    left, right = video_pair
    blended = tmp_path / "blend.mp4"
    concat = tmp_path / "concat.mp4"
    stitch_videos([str(left), str(right)], str(blended), VideoStitchOptions(mode="blend"))
    stitch_videos([str(left), str(right)], str(concat), VideoStitchOptions(mode="concat"))

    blend_frame = _read_first_frame(blended)
    concat_frame = _read_first_frame(concat)

    def column_jumps(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        return np.abs(np.diff(gray, axis=1)).mean(axis=0)

    blend_jumps = column_jumps(blend_frame)
    concat_jumps = column_jumps(concat_frame)
    # the overlap of the blend mode sits at the columns 286 to 330, the concat
    # mode cuts the picture at column 330
    blend_seam = float(blend_jumps[280:332].max())
    concat_seam = float(concat_jumps[326:336].max())
    assert blend_seam < concat_seam * 0.5


def test_concat_mode_places_the_views_side_by_side(tmp_path, video_pair):
    left, right = video_pair
    output = tmp_path / "concat.mp4"
    report = stitch_videos(
        [str(left), str(right)], str(output), VideoStitchOptions(mode="concat")
    )
    assert report["size"][0] == 660
    assert report["frames"] == 24
    assert _read_first_frame(output).shape[1] == 660


def test_a_manual_overlap_is_used(tmp_path, video_pair):
    left, right = video_pair
    output = tmp_path / "manual.mp4"
    report = stitch_videos(
        [str(left), str(right)],
        str(output),
        VideoStitchOptions(mode="blend", overlap_px=[44], auto_overlap=False),
    )
    assert report["alignments"][0]["overlap"] == 44
    assert report["size"][0] == 330 + 330 - 44


def test_max_frames_limits_the_output(tmp_path, video_pair):
    left, right = video_pair
    output = tmp_path / "short.mp4"
    report = stitch_videos(
        [str(left), str(right)],
        str(output),
        VideoStitchOptions(mode="blend", max_frames=5),
    )
    assert report["frames"] == 5


def test_two_videos_are_required(tmp_path, video_pair):
    left, _ = video_pair
    with pytest.raises(ValueError):
        stitch_videos([str(left)], str(tmp_path / "x.mp4"), VideoStitchOptions())
