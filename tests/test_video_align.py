from __future__ import annotations

import cv2
import numpy as np

from hstitch.video.align import (
    estimate_alignment,
    estimate_shift,
    median_alignment,
)


def test_alignment_finds_the_known_shift(source_image):
    left = source_image[:, 0:330]
    right = source_image[:, 287:573]
    alignment = estimate_alignment(left, right)
    assert alignment is not None
    assert abs(alignment.dx - 287) <= 2
    assert alignment.dy == 0
    assert alignment.overlap == 330 - alignment.dx


def test_alignment_works_with_a_wide_overlap(source_image):
    left = source_image[:, 0:430]
    right = source_image[:, 287:573]
    alignment = estimate_alignment(left, right)
    assert alignment is not None
    assert abs(alignment.dx - 287) <= 2
    assert alignment.overlap == 430 - alignment.dx


def test_alignment_finds_a_vertical_offset(source_image):
    # the right view shows the scene ten rows higher, so it has to be drawn
    # ten rows above the left one
    left = source_image[:, 0:430]
    right = np.zeros_like(source_image[:, 287:717])
    right[10:, :] = source_image[:-10, 287:717]
    alignment = estimate_alignment(left, right)
    assert alignment is not None
    assert abs(alignment.dx - 287) <= 2
    assert abs(alignment.dy + 10) <= 2


def test_alignment_needs_at_least_two_views(source_image):
    left = source_image[:, 0:330]
    right = np.zeros_like(left)
    right[:] = 255
    assert estimate_alignment(left, right) is None


def test_estimate_shift_reports_the_method(source_image):
    left = source_image[:, 0:330]
    right = source_image[:, 287:573]
    estimate = estimate_shift(left, right)
    assert estimate is not None
    shift_x, shift_y, score, method, inliers = estimate
    assert method in ("template", "features", "profile")
    assert score > 0.35
    assert abs(-shift_x - 287) <= 2


def test_median_alignment_uses_the_median():
    from hstitch.video.align import Alignment

    alignments = [
        Alignment(dx=100, dy=0, overlap=30, score=0.9, method="template"),
        Alignment(dx=104, dy=2, overlap=26, score=0.8, method="template"),
        Alignment(dx=102, dy=1, overlap=28, score=0.85, method="template"),
    ]
    combined = median_alignment(alignments)
    assert combined is not None
    assert combined.dx == 102
    assert combined.dy == 1
    assert combined.overlap == 28


def test_median_alignment_falls_back():
    from hstitch.video.align import Alignment

    fallback = Alignment(dx=10, dy=0, overlap=5, score=0.5, method="template")
    assert median_alignment([None, None], fallback) is fallback
    assert median_alignment([None, None]) is None
