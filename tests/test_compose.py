from __future__ import annotations

import numpy as np

from hstitch.compose import ComposeOptions, concat_horizontal, tile_offsets


def _tile(width: int, height: int, value: int = 255) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_tile_offsets_accumulates_the_widths():
    assert tile_offsets([100, 200, 50]) == [0, 100, 300]


def test_tile_offsets_accounts_for_the_gap():
    assert tile_offsets([100, 200], gap=10) == [0, 110]


def test_tile_offsets_accounts_for_the_overlap():
    assert tile_offsets([100, 100], overlaps=[40]) == [0, 60]


def test_concat_places_the_tiles_side_by_side():
    result = concat_horizontal([_tile(10, 20, 10), _tile(30, 20, 200)])
    assert result.shape == (20, 40, 3)
    assert np.all(result[:, :10] == 10)
    assert np.all(result[:, 10:] == 200)


def test_concat_adds_the_gap():
    result = concat_horizontal(
        [_tile(10, 20, 10), _tile(10, 20, 200)], ComposeOptions(gap=5)
    )
    assert result.shape == (20, 25, 3)
    assert np.all(result[:, 10:15] == 0)


def test_concat_uses_the_background_for_the_gap():
    result = concat_horizontal(
        [_tile(10, 20, 10), _tile(10, 20, 200)],
        ComposeOptions(gap=4, background=(7, 8, 9)),
    )
    assert tuple(result[0, 10]) == (7, 8, 9)


def test_concat_aligns_the_shorter_tile():
    tall = _tile(10, 40, 10)
    short = _tile(10, 20, 200)
    for align, expected_top in (
        ("top", 0),
        ("center", 10),
        ("bottom", 20),
    ):
        result = concat_horizontal([tall, short], ComposeOptions(align=align))
        assert result.shape == (40, 20, 3)
        column = result[:, 10, 0]
        first = int(np.argmax(column == 200))
        assert first == expected_top


def test_concat_rejects_an_unknown_alignment():
    try:
        concat_horizontal([_tile(4, 4)], ComposeOptions(align="diagonal"))
    except ValueError as error:
        assert "alignment" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")
