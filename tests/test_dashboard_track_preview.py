"""How a track's stored path is fitted and drawn without its video."""

import pandas as pd
import pytest

from hessdalen.dashboard.track_preview import close_up, fitted, frame_view, gallery, light_curve


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
    panels = pd.concat([_track("a", xs=[1.0, 2.0], ys=[1.0, 2.0]), _track("b", xs=[5.0, 9.0], ys=[1.0, 1.0])])
    panels["panel"] = panels["key"].map({"a": "1. birds", "b": "2. planes"})

    for chart in (frame_view(points), close_up(points), light_curve(points), gallery(panels)):
        assert chart.to_dict()


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
