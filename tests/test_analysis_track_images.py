"""What a track's picture keeps, and what it must not depend on."""

import numpy as np
import pytest

from hessdalen.analysis.spectra import track_signals
from hessdalen.analysis.track_images import (
    CANDIDATES,
    SIDE,
    path_drawn,
    path_similarity,
    path_transform,
    state_similarity,
)

REACH = 1920.0


@pytest.mark.parametrize("name", sorted(CANDIDATES))
@pytest.mark.parametrize("frames", [10, 97, 2680])
def test_every_picture_comes_out_the_same_size_at_any_length(name: str, frames: int) -> None:
    """A corpus whose tracks run from ten frames to two thousand has to become
    a corpus of one size, or nothing can be put beside anything."""
    drawn = CANDIDATES[name](_wandering(frames))

    assert drawn.shape[:2] == (SIDE, SIDE)
    assert drawn.min() >= 0.0
    assert drawn.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("name", sorted(CANDIDATES))
def test_a_track_that_moves_is_not_drawn_blank(name: str) -> None:
    drawn = CANDIDATES[name](_wandering(120))

    assert drawn.max() > drawn.min()


@pytest.mark.parametrize("name", sorted(CANDIDATES))
def test_a_track_looks_more_like_its_other_half_than_like_another_track(name: str) -> None:
    """Whether a picture holds the track or the noise is judged by cutting a
    track in two and seeing whether the halves still find each other, so a
    candidate that cannot do it on a track built to be found has nothing to
    find on a real one."""
    drawn = CANDIDATES[name]
    first = drawn(_wandering(240, over=slice(0, 120)))
    second = drawn(_wandering(240, over=slice(120, 240)))
    other = drawn(_straight(120))

    assert float(np.abs(first - second).mean()) < float(np.abs(first - other).mean())


def test_a_track_that_comes_back_on_itself_is_drawn_unlike_one_that_does_not() -> None:
    """A path against itself is read by how far it strays from the diagonal,
    which is where a track returning to where it was parts from a passage."""
    passing = path_similarity(_straight(120))
    circling = path_similarity(_circling(120))

    assert float(np.abs(passing - circling).max()) > 0.4


def test_a_track_run_backwards_is_drawn_reflected() -> None:
    """A self-similarity picture is a track against itself, so reversing the
    track can only turn the picture about its other diagonal."""
    forwards = state_similarity(_wandering(120))
    backwards = state_similarity(_wandering(120, backwards=True))

    assert float(np.abs(forwards - backwards[::-1, ::-1]).mean()) < 0.02


def test_a_track_is_drawn_the_same_whichever_way_it_travelled() -> None:
    """Which way a thing crossed the frame is where the camera was pointed."""
    east = path_drawn(_straight(120))
    north = path_drawn(_straight(120, heading=(0.0, 1.0)))

    assert float(np.abs(east - north).mean()) < 0.02


def test_a_wave_along_a_track_carries_the_transform_off_the_cross() -> None:
    """A straight line answers along one line through the middle and nowhere
    else. A wave laid along it adds marks away from that line, which is what
    the transform is read by."""
    middle = SIDE // 2
    steps = np.arange(SIDE)
    cross = (np.abs(steps - middle)[:, None] < 3) | (np.abs(steps - middle)[None, :] < 3)

    waved = path_transform(_wandering(120))
    straight = path_transform(_straight(120))

    assert float(waved[~cross].mean()) > float(straight[~cross].mean()) * 1.5


def _wandering(frames: int, *, backwards: bool = False, over: slice = slice(None)) -> object:
    """A track travelling across the frame while it weaves and pulses."""
    steps = np.arange(frames)
    along = steps / max(frames - 1, 1)
    across = 30.0 * np.sin(2.0 * np.pi * 6.0 * along)
    bright = 900.0 + 400.0 * np.sin(2.0 * np.pi * 9.0 * along)
    order = slice(None, None, -1) if backwards else slice(None)
    return track_signals(
        frame_number=steps[over],
        centre_x=(600.0 * along)[order][over],
        centre_y=(540.0 + across)[order][over],
        brightness=bright[order][over],
        pixel_count=np.full(frames, 30)[over],
        reach=REACH,
    )


def _circling(frames: int) -> object:
    """A track that comes back to where it started."""
    turn = 2.0 * np.pi * np.arange(frames) / frames
    return track_signals(
        frame_number=np.arange(frames),
        centre_x=600.0 + 200.0 * np.cos(turn),
        centre_y=540.0 + 200.0 * np.sin(turn),
        brightness=np.full(frames, 900.0),
        pixel_count=np.full(frames, 30),
        reach=REACH,
    )


def _straight(frames: int, *, heading: tuple[float, float] = (1.0, 0.0)) -> object:
    steps = np.arange(frames)
    along = steps / max(frames - 1, 1) * 600.0
    return track_signals(
        frame_number=steps,
        centre_x=200.0 + heading[0] * along,
        centre_y=200.0 + heading[1] * along,
        brightness=np.full(frames, 900.0),
        pixel_count=np.full(frames, 30),
        reach=REACH,
    )
