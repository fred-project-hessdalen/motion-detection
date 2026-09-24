"""What the marks drawn over a detection leave visible."""

import numpy as np

from hessdalen.dashboard.panels import TRAIL_LAG, draw_box, draw_trail

COLOR = (0, 255, 0)
HEAD = (100, 100)
CLEAR = 20
STROKE = 2
"""How wide a drawn line is, which is how far over its own end it reaches."""


def test_the_trail_runs_into_the_box_and_stops_behind_the_head() -> None:
    """The path says where the thing came from, and the last frames of it are
    the thing itself, which the box is there to show."""
    canvas = _canvas()

    draw_trail(canvas, polyline=_walk(HEAD[0] - 40, HEAD[0]), color=COLOR)

    assert _reaches(canvas) <= CLEAR
    assert _reaches(canvas) >= TRAIL_LAG - STROKE
    assert canvas[100, HEAD[0] - CLEAR - 1].any()


def test_a_trail_shorter_than_the_lag_is_not_drawn() -> None:
    canvas = _canvas()

    draw_trail(canvas, polyline=_walk(HEAD[0] - TRAIL_LAG + 1, HEAD[0]), color=COLOR)

    assert not canvas.any()


def test_a_track_that_stays_where_it_is_draws_no_trail() -> None:
    canvas = _canvas()

    draw_trail(canvas, polyline=_path([(100, 100)] * (TRAIL_LAG + 3)), color=COLOR)

    assert not canvas.any()


def test_the_box_is_drawn_around_the_head_it_marks() -> None:
    canvas = _canvas()

    draw_box(canvas, point=np.array(HEAD), color=COLOR, size=CLEAR, label="7")

    assert canvas[HEAD[1] - CLEAR, HEAD[0]].any()
    assert not canvas[HEAD[1], HEAD[0]].any()


def _reaches(canvas: np.ndarray) -> int:
    """How near the head of the path the drawing comes, in pixels."""
    ys, xs = np.nonzero(canvas[:, :, 1])
    return int(np.maximum(np.abs(xs - HEAD[0]), np.abs(ys - HEAD[1])).min())


def _walk(first: int, last: int) -> np.ndarray:
    """A path of one point per frame, running along a row to the head."""
    return _path([(x, HEAD[1]) for x in range(first, last + 1)])


def _path(points: list[tuple[int, int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32)


def _canvas() -> np.ndarray:
    return np.zeros((200, 200, 3), dtype=np.uint8)
