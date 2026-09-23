"""How a track's stored path is fitted and drawn without its video."""

import base64
import re

import pandas as pd
import pytest

from hessdalen.dashboard.track_preview import (
    CACHED_MARK,
    JUMP_MARK,
    PLAYING_COLOUR,
    PLAYING_MARK,
    VALIDATED_MARK,
    GalleryEntry,
    close_up,
    fitted,
    frame_view,
    gallery_html,
    light_curve,
)


def test_a_fitted_track_keeps_its_proportions() -> None:
    """A track ten times wider than tall stays ten times wider than tall, so a
    straight streak does not come out as a square."""
    drawn = fitted(_track("a", xs=[100.0, 150.0, 200.0], ys=[500.0, 505.0, 510.0]))

    assert drawn["u"].max() - drawn["u"].min() == pytest.approx(1.0)
    assert drawn["v"].max() - drawn["v"].min() == pytest.approx(0.1)


def test_every_track_is_fitted_to_its_own_box() -> None:
    drawn = fitted(
        pd.concat(
            [
                _track("small", xs=[10.0, 12.0], ys=[10.0, 10.0]),
                _track("large", xs=[100.0, 900.0], ys=[400.0, 400.0]),
            ]
        )
    )

    for key in ("small", "large"):
        held = drawn[drawn["key"] == key]
        assert held["u"].min() == pytest.approx(-0.5)
        assert held["u"].max() == pytest.approx(0.5)


def test_the_phase_runs_from_the_first_frame_to_the_last() -> None:
    drawn = fitted(_track("a", xs=[0.0, 1.0, 2.0], ys=[0.0, 0.0, 0.0], frames=[10, 15, 20]))

    assert drawn["phase"].tolist() == [0.0, 0.5, 1.0]


def test_a_track_that_never_moves_does_not_divide_by_nothing() -> None:
    drawn = fitted(_track("still", xs=[50.0, 50.0], ys=[60.0, 60.0]))

    assert drawn["u"].tolist() == [0.0, 0.0]


def test_every_chart_is_drawn_from_the_path_alone() -> None:
    points = _track("a", xs=[100.0, 150.0, 200.0], ys=[500.0, 505.0, 510.0])

    for chart in (frame_view(points), close_up(points), light_curve(points)):
        assert chart.to_dict()


def test_a_gallery_draws_each_track_under_its_caption_in_the_order_given() -> None:
    drawn = gallery_html(_paths(), entries=[_entry("b", "2. planes"), _entry("a", "1. birds <&>")], frame="#e6007e")

    figures = drawn.split("<figure")[1:]
    assert len(figures) == 2
    assert all("#e6007e" in figure for figure in figures)
    assert "2. planes" in figures[0]
    assert "1. birds &lt;&amp;&gt;" in figures[1]
    assert _svg(figures[0]).count("<circle") == 2
    assert _svg(figures[1]).count("<circle") == 3


def test_a_panel_carries_the_key_of_the_track_it_draws() -> None:
    """A click on the gallery is read back to a track from the panel it landed
    in."""
    drawn = gallery_html(_paths(), entries=[_entry("a", "1. birds")], frame="#e6007e")

    assert 'data-track="a"' in drawn


def test_every_panel_carries_a_button_that_names_its_track() -> None:
    """A press of the button selects the track, which the panel itself no
    longer does."""
    figures = gallery_html(
        _paths(), entries=[_entry("a", "1. birds"), _entry("b", "2. planes")], frame="#e6007e"
    ).split("<figure")[1:]

    assert JUMP_MARK in figures[0]
    assert 'data-jump="a"' in figures[0]
    assert 'data-jump="b"' in figures[1]


def test_a_panel_is_marked_by_what_is_known_of_its_track() -> None:
    entries = [
        _entry("a", "1. on disk", cached=True),
        _entry("b", "2. confirmed", validated=True),
    ]

    figures = gallery_html(_paths(), entries=entries, frame="#e6007e").split("<figure")[1:]

    assert CACHED_MARK in figures[0]
    assert VALIDATED_MARK not in figures[0]
    assert VALIDATED_MARK in figures[1]
    assert CACHED_MARK not in figures[1]


def test_a_panel_of_a_track_nothing_is_known_of_carries_no_mark() -> None:
    drawn = gallery_html(_paths(), entries=[_entry("a", "1. birds")], frame="#e6007e")

    assert CACHED_MARK not in drawn
    assert VALIDATED_MARK not in drawn
    assert PLAYING_MARK not in drawn


def test_the_panel_whose_video_plays_stands_out_from_the_gallery() -> None:
    """A click plays the track's video, and the panel clicked has to be the one
    the eye goes back to."""
    entries = [_entry("a", "1. playing", playing=True), _entry("b", "2. birds")]

    figures = gallery_html(_paths(), entries=entries, frame="#e6007e").split("<figure")[1:]

    assert PLAYING_MARK in figures[0]
    assert PLAYING_COLOUR in figures[0]
    assert "#e6007e" not in figures[0]
    assert PLAYING_MARK not in figures[1]
    assert "#e6007e" in figures[1]


def _entry(
    key: str, caption: str, *, cached: bool = False, validated: bool = False, playing: bool = False
) -> GalleryEntry:
    return GalleryEntry(key=key, caption=caption, cached=cached, validated=validated, playing=playing)


def _paths() -> pd.DataFrame:
    return pd.concat([_track("a", xs=[1.0, 2.0, 3.0], ys=[1.0, 2.0, 3.0]), _track("b", xs=[5.0, 9.0], ys=[1.0, 1.0])])


def _svg(figure: str) -> str:
    encoded = re.search(r"data:image/svg\+xml;base64,([^\"]+)", figure)
    assert encoded is not None
    return base64.b64decode(encoded.group(1)).decode()


def _track(key: str, *, xs: list[float], ys: list[float], frames: list[int] | None = None) -> pd.DataFrame:
    count = len(xs)
    return pd.DataFrame(
        {
            "key": [key] * count,
            "frame_number": frames or list(range(count)),
            "centre_x": xs,
            "centre_y": ys,
            "brightness": [900.0] * count,
            "pixel_count": [20] * count,
            "frame_width": [1920] * count,
            "frame_height": [1080] * count,
        }
    )
