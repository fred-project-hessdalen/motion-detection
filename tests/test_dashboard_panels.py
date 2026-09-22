"""What the marks drawn over a detection leave visible."""

import numpy as np

from hessdalen.dashboard.panels import draw_box, draw_trail

COLOR = (0, 255, 0)
HEAD = (100, 100)
CLEAR = 20
STROKE = 2
"""How wide a drawn line is, which is how far over its own end it reaches."""


def test_the_trail_stops_at_the_box_around_its_head() -> None:
    """The box marks where the detection is, and a trail drawn through it
    covers the very pixels it is there to show."""
    canvas = _canvas()

    draw_trail(canvas, polyline=_path([(10, 100), HEAD]), color=COLOR, clear=CLEAR)

    assert _reaches(canvas) >= CLEAR - STROKE
    assert canvas[100, HEAD[0] - CLEAR - 1].any()


def test_a_trail_that_comes_back_leaves_the_box_clear() -> None:
    """A track that returns to where it has been crosses the square around its
    last position on a step that neither starts nor ends inside it."""
    canvas = _canvas()

    draw_trail(canvas, polyline=_path([(10, 100), (190, 100), HEAD]), color=COLOR, clear=CLEAR)

    assert _reaches(canvas) >= CLEAR - STROKE
    assert canvas[100, 50].any()
    assert canvas[100, 150].any()


def test_a_track_that_stays_where_it_is_draws_no_trail() -> None:
    canvas = _canvas()

    draw_trail(canvas, polyline=_path([(98, 100), (99, 100), HEAD]), color=COLOR, clear=CLEAR)

    assert not canvas.any()


def test_the_box_is_drawn_around_the_head_the_trail_keeps_clear() -> None:
    canvas = _canvas()

    draw_box(canvas, point=np.array(HEAD), color=COLOR, size=CLEAR, label="7")

    assert canvas[HEAD[1] - CLEAR, HEAD[0]].any()
    assert not canvas[HEAD[1], HEAD[0]].any()


def _reaches(canvas: np.ndarray) -> int:
    """How near the head of the path the drawing comes, in pixels."""
    ys, xs = np.nonzero(canvas[:, :, 1])
    return int(np.maximum(np.abs(xs - HEAD[0]), np.abs(ys - HEAD[1])).min())


def _path(points: list[tuple[int, int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32)


def _canvas() -> np.ndarray:
    return np.zeros((200, 200, 3), dtype=np.uint8)
