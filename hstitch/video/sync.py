"""Frame alignment between several video sources.

Two strategies are available:

``timestamp``
    The first source drives the timeline. For every reference timestamp the
    frame of every other source that lies closest in time is picked.

``index``
    The n-th frame of every source belongs to the n-th group. This is the
    simple variant and requires strictly synchronised recordings.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import numpy as np

from .sources import VideoStream

SYNC_MODES = ("timestamp", "index")


class FrameCursor:
    """Keeps one frame of lookahead so the nearest frame can be picked."""

    def __init__(self, stream: VideoStream):
        self.stream = stream
        self._previous: Optional[tuple] = None
        self._current = stream.next()

    @property
    def current(self) -> Optional[tuple]:
        return self._current

    def advance(self) -> None:
        following = self.stream.next()
        self._previous = self._current
        self._current = following

    def nearest(self, timestamp: float) -> Optional[np.ndarray]:
        """Return the frame whose timestamp is closest to ``timestamp``."""
        while self._current is not None and self._current[0] < timestamp:
            self.advance()
        if self._current is None:
            return self._previous[1] if self._previous is not None else None
        if self._previous is None:
            return self._current[1]
        if abs(self._current[0] - timestamp) <= abs(self._previous[0] - timestamp):
            return self._current[1]
        return self._previous[1]


def aligned_groups(
    streams: Sequence[VideoStream],
    sync: str = "timestamp",
) -> Iterator[List[np.ndarray]]:
    """Yield one frame per source for every point of the shared timeline."""
    if sync not in SYNC_MODES:
        raise ValueError(f"unknown sync mode {sync!r}, expected one of {SYNC_MODES}")
    if not streams:
        return

    if sync == "index":
        while True:
            group: List[np.ndarray] = []
            for stream in streams:
                item = stream.next()
                if item is None:
                    return
                group.append(item[1])
            yield group
        return

    cursors = [FrameCursor(stream) for stream in streams]
    master = cursors[0]
    while master.current is not None:
        timestamp = master.current[0]
        group = []
        for cursor in cursors:
            frame = cursor.nearest(timestamp)
            if frame is None:
                return
            group.append(frame)
        yield group
        master.advance()
