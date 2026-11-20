from __future__ import annotations

import cv2
import numpy as np

from hstitch.video.sources import VideoStream
from hstitch.video.sync import FrameCursor, aligned_groups

BLOCK = 8
BITS = 8


def _counter_frames(count: int, start: int = 0, size=(64, 48)):
    """Frames that carry their index as a black and white barcode.

    A barcode survives the video compression, a plain grey value does not.
    """
    frames = []
    for index in range(count):
        frame = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
        value = start + index
        for bit in range(BITS):
            if value & (1 << bit):
                frame[:, bit * BLOCK : (bit + 1) * BLOCK] = 0
        frames.append(frame)
    return frames


def _decode(frame) -> int:
    """Read the barcode back."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    value = 0
    for bit in range(BITS):
        block = gray[:, bit * BLOCK : (bit + 1) * BLOCK]
        if float(block.mean()) < 128.0:
            value |= 1 << bit
    return value


def test_video_stream_reports_the_metadata(tmp_path, video_writer):
    path = video_writer(tmp_path / "a.mp4", _counter_frames(12))
    with VideoStream(path) as stream:
        assert stream.info.width == 64
        assert stream.info.height == 48
        assert abs(stream.info.fps - 25.0) < 1e-6
        assert stream.info.frames == 12


def test_video_stream_yields_increasing_timestamps(tmp_path, video_writer):
    path = video_writer(tmp_path / "a.mp4", _counter_frames(10))
    stamps = []
    with VideoStream(path) as stream:
        while True:
            item = stream.next()
            if item is None:
                break
            stamps.append(item[0])
    assert len(stamps) == 10
    assert all(later >= earlier for earlier, later in zip(stamps, stamps[1:]))
    assert stamps[-1] > 0.0


def test_index_sync_pairs_the_frames_by_position(tmp_path, video_writer):
    first = video_writer(tmp_path / "a.mp4", _counter_frames(8))
    second = video_writer(tmp_path / "b.mp4", _counter_frames(8, start=100))
    with VideoStream(first) as one, VideoStream(second) as two:
        groups = list(aligned_groups([one, two], sync="index"))
    assert len(groups) == 8
    for index, group in enumerate(groups):
        assert _decode(group[0]) == index
        assert _decode(group[1]) == 100 + index


def test_timestamp_sync_follows_the_time(tmp_path, video_writer):
    first = video_writer(tmp_path / "a.mp4", _counter_frames(12))
    # the second video starts three frames later
    second = video_writer(tmp_path / "b.mp4", _counter_frames(9, start=3))
    with VideoStream(first) as one, VideoStream(second) as two:
        groups = list(aligned_groups([one, two], sync="timestamp"))
    assert groups
    for index, group in enumerate(groups):
        left = _decode(group[0])
        right = _decode(group[1])
        if index < 9:
            # the second video started three frames later, so it always shows
            # the content the first one shows three frames later
            assert abs((left - right) + 3) <= 1
        else:
            # the second video is exhausted and holds its last frame
            assert right == 11


def test_frame_cursor_picks_the_nearest_frame(tmp_path, video_writer):
    path = video_writer(tmp_path / "a.mp4", _counter_frames(10))
    with VideoStream(path) as stream:
        cursor = FrameCursor(stream)
        first = cursor.nearest(0.0)
        assert first is not None
        assert _decode(first) == 0
        second = cursor.nearest(10.0)
        assert second is not None
        assert _decode(second) > 0
