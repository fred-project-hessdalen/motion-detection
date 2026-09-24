"""How a track's stored path is fitted and drawn without its video."""

import base64
import re

import cv2
import numpy as np
import pandas as pd
import pytest

from hessdalen.analysis.spectra import BRIGHTNESS, PRESENCE, RATE_COUNT, WOBBLE
from hessdalen.dashboard.track_preview import (
    CACHED_MARK,
    JUMP_MARK,
    PANEL_PIXELS,
    PLAYING_COLOUR,
    PLAYING_MARK,
    VALIDATED_MARK,
    GalleryEntry,
    close_up,
    fitted,
    frame_view,
    gallery_html,
    light_curve,
    rhythm,
    rhythm_chart,
    spectra_html,
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


def test_a_picked_panel_is_boxed_and_its_neighbour_is_not() -> None:
    entries = [_entry("a", "1. picked", picked=True), _entry("b", "2. birds")]

    figures = gallery_html(_paths(), entries=entries, frame="#e6007e").split("<figure")[1:]

    assert "outline" in figures[0]
    assert "outline" not in figures[1]


def test_a_rhythm_gallery_draws_one_panel_per_entry_at_the_panel_size() -> None:
    entries = [_entry("a", "1. birds"), _entry("b", "2. planes")]

    figures = spectra_html(_beating(), entries=entries, frame="#123456", signal=BRIGHTNESS).split("<figure")[1:]

    assert len(figures) == 2
    assert 'data-track="a"' in figures[0]
    assert "1. birds" in figures[0]
    assert _png(figures[0]).shape == (PANEL_PIXELS, PANEL_PIXELS, 3)


def test_a_rhythm_panel_carries_the_same_marks_as_a_path_panel() -> None:
    """A gallery answers a click the same way whichever drawing it holds, so
    both drawings need the mark that jumps to the track."""
    entries = [_entry("a", "1. birds", cached=True, validated=True, playing=True)]

    drawn = spectra_html(_beating(), entries=entries, frame="#123456", signal=BRIGHTNESS).split("<figure")[1]

    assert JUMP_MARK in drawn
    assert CACHED_MARK in drawn
    assert VALIDATED_MARK in drawn
    assert PLAYING_COLOUR in drawn


def test_a_track_that_repeats_is_drawn_brighter_than_one_that_does_not() -> None:
    """The panels of a gallery are there to be told apart at a glance."""
    entries = [_entry("a", "1. birds")]
    beating = spectra_html(_beating(), entries=entries, frame="#123", signal=BRIGHTNESS)
    steady = spectra_html(_steady(), entries=entries, frame="#123", signal=BRIGHTNESS)

    assert _png(beating.split("<figure")[1]).max() > _png(steady.split("<figure")[1]).max()


def test_a_rhythm_is_read_over_every_frame_a_track_spans() -> None:
    found = rhythm(_track("a", xs=[0.0, 1.0, 2.0], ys=[0.0, 0.0, 0.0], frames=[4, 6, 7]), signal=PRESENCE)

    assert found.frame_number.tolist() == [4, 5, 6, 7]
    assert found.image.power.shape == (RATE_COUNT, 4)


def test_a_rhythm_chart_is_built_over_the_frames_the_track_ran() -> None:
    drawn = rhythm_chart(_beating()[lambda held: held["key"] == "a"], signal=WOBBLE).to_dict()

    assert drawn["encoding"]["x"]["scale"]["domain"] == [0, 120]


def _beating() -> pd.DataFrame:
    """Two tracks whose brightness and path both repeat every five frames."""
    steps = np.arange(120)
    beat = np.sin(2.0 * np.pi * steps / 5.0)
    return pd.concat(
        [
            _track(key, xs=list(steps.astype(float)), ys=list(4.0 * beat), brightness=list(900.0 + 400.0 * beat))
            for key in ("a", "b")
        ]
    )


def _steady() -> pd.DataFrame:
    steps = np.arange(120)
    return _track("a", xs=list(steps.astype(float)), ys=[0.0] * 120, brightness=[900.0] * 120)


def _png(figure: str) -> np.ndarray:
    encoded = re.search(r"data:image/png;base64,([^\"]+)", figure)
    assert encoded is not None
    return cv2.imdecode(np.frombuffer(base64.b64decode(encoded.group(1)), dtype=np.uint8), cv2.IMREAD_COLOR)


def _entry(
    key: str,
    caption: str,
    *,
    cached: bool = False,
    validated: bool = False,
    playing: bool = False,
    picked: bool = False,
) -> GalleryEntry:
    return GalleryEntry(key=key, caption=caption, cached=cached, validated=validated, playing=playing, picked=picked)


def _paths() -> pd.DataFrame:
    return pd.concat([_track("a", xs=[1.0, 2.0, 3.0], ys=[1.0, 2.0, 3.0]), _track("b", xs=[5.0, 9.0], ys=[1.0, 1.0])])


def _svg(figure: str) -> str:
    encoded = re.search(r"data:image/svg\+xml;base64,([^\"]+)", figure)
    assert encoded is not None
    return base64.b64decode(encoded.group(1)).decode()


def _track(
    key: str,
    *,
    xs: list[float],
    ys: list[float],
    frames: list[int] | None = None,
    brightness: list[float] | None = None,
) -> pd.DataFrame:
    count = len(xs)
    return pd.DataFrame(
        {
            "key": [key] * count,
            "frame_number": frames or list(range(count)),
            "centre_x": xs,
            "centre_y": ys,
            "brightness": brightness or [900.0] * count,
            "pixel_count": [20] * count,
            "frame_width": [1920] * count,
            "frame_height": [1080] * count,
        }
    )
