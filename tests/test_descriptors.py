"""What the description of a track says about the track it came from."""

import math

import numpy as np
import pytest

from hessdalen.analysis.descriptors import Frame, TrackRows, describe

FRAME = Frame(height=1080, width=1920)
REACH = 1920.0


def test_a_straight_track_reads_as_straight() -> None:
    rows = _rows(x=np.arange(20) * 10.0 + 100.0, y=np.full(20, 500.0))

    described = describe(rows, frame=FRAME)

    assert described.straightness == pytest.approx(1.0)
    assert described.turn_max == pytest.approx(0.0, abs=1e-9)
    assert described.line_residual == pytest.approx(0.0, abs=1e-9)
    assert described.velocity_residual == pytest.approx(0.0, abs=1e-9)


def test_a_track_that_turns_back_on_itself_covers_no_ground() -> None:
    there = np.arange(10) * 10.0 + 100.0
    rows = _rows(x=np.concatenate([there, there[::-1]]), y=np.full(20, 500.0))

    described = describe(rows, frame=FRAME)

    assert described.displacement == pytest.approx(0.0)
    assert described.straightness == pytest.approx(0.0)
    assert described.path_length == pytest.approx(2.0 * 90.0 / REACH)


def test_a_slowing_track_fits_an_acceleration_and_not_a_speed() -> None:
    """A meteor slows as it burns, so a fit holding speed steady misses it."""
    elapsed = np.arange(20, dtype=np.float64)
    rows = _rows(x=100.0 + 40.0 * elapsed - 0.8 * elapsed**2, y=np.full(20, 500.0))

    described = describe(rows, frame=FRAME)

    assert described.acceleration_residual == pytest.approx(0.0, abs=1e-9)
    assert described.velocity_residual > 0.01


def test_a_beating_brightness_reads_as_one_rhythm() -> None:
    """A wingbeat puts the brightness's variation at one rate, and the rate
    reported is the one it was drawn at."""
    elapsed = np.arange(40, dtype=np.float64)
    beating = 900.0 + 200.0 * np.sin(2.0 * math.pi * elapsed / 5.0)
    rows = _rows(x=100.0 + elapsed * 5.0, y=np.full(40, 500.0), brightness=beating)

    described = describe(rows, frame=FRAME)

    assert described.flicker > 0.8
    assert described.flicker_rate == pytest.approx(1.0 / 5.0, abs=0.02)


def test_a_track_of_one_rise_and_decay_reads_as_no_rhythm() -> None:
    elapsed = np.arange(40, dtype=np.float64)
    once = 900.0 + 400.0 * np.exp(-(((elapsed - 20.0) / 4.0) ** 2))
    rows = _rows(x=100.0 + elapsed * 5.0, y=np.full(40, 500.0), brightness=once)

    described = describe(rows, frame=FRAME)

    assert described.flicker < 0.5


def test_a_crossing_track_starts_and_ends_on_an_edge() -> None:
    rows = _rows(x=np.linspace(0.0, 1919.0, 30), y=np.full(30, 540.0))

    described = describe(rows, frame=FRAME)

    assert described.start_edge == pytest.approx(0.0)
    assert described.end_edge == pytest.approx(0.0)


def test_a_track_inside_the_frame_touches_no_edge() -> None:
    rows = _rows(x=np.linspace(800.0, 1000.0, 30), y=np.full(30, 540.0))

    described = describe(rows, frame=FRAME)

    assert described.start_edge > 0.2
    assert described.end_edge > 0.2


def test_missed_frames_are_counted_and_speed_is_per_frame() -> None:
    frames = np.array([0, 1, 2, 5, 6], dtype=np.int64)
    rows = _rows(x=np.array([0.0, 10.0, 20.0, 50.0, 60.0]), y=np.full(5, 500.0), frame_number=frames)

    described = describe(rows, frame=FRAME)

    assert described.frames == 5
    assert described.missed_frames == 2
    assert described.speed_max == pytest.approx(10.0 / REACH)


def test_an_evenly_moving_track_reads_as_smooth_whether_straight_or_curving() -> None:
    elapsed = np.arange(40, dtype=np.float64)
    straight = describe(_rows(x=100.0 + elapsed * 6.0, y=np.full(40, 500.0)), frame=FRAME)
    curving = describe(
        _rows(x=500.0 + 200.0 * np.cos(elapsed / 20.0), y=500.0 + 200.0 * np.sin(elapsed / 20.0)), frame=FRAME
    )

    assert straight.roughness == pytest.approx(0.0, abs=1e-9)
    assert curving.roughness < 0.1
    assert straight.jump == pytest.approx(1.0)


def test_a_track_hopping_about_reads_as_rough() -> None:
    rng = np.random.default_rng(0)
    rows = _rows(x=500.0 + rng.normal(0.0, 8.0, 40), y=500.0 + rng.normal(0.0, 8.0, 40))

    assert describe(rows, frame=FRAME).roughness > 0.6


def test_a_slow_track_reads_as_smooth_once_its_motion_outgrows_its_jitter() -> None:
    """A slow object moves less per frame than its blob's centre jitters, so
    frame by frame it looks as rough as clutter, and every few frames it
    does not."""
    rng = np.random.default_rng(1)
    elapsed = np.arange(64, dtype=np.float64)
    rows = _rows(x=100.0 + elapsed * 1.0 + rng.normal(0.0, 0.6, 64), y=500.0 + rng.normal(0.0, 0.6, 64))

    assert describe(rows, frame=FRAME).roughness < 0.6


def test_one_far_step_reads_as_a_jump() -> None:
    """A clean track the tracker linked to a noise blob far away shows one
    step many times its usual length."""
    x = np.concatenate([100.0 + np.arange(30) * 4.0, [100.0 + 29 * 4.0 + 120.0]])

    assert describe(_rows(x=x, y=np.full(31, 500.0)), frame=FRAME).jump == pytest.approx(30.0)


def _rows(*, x, y, frame_number=None, brightness=None) -> TrackRows:
    count = x.size
    frames = np.arange(count, dtype=np.int64) if frame_number is None else frame_number
    return TrackRows(
        recording="Cam1_2025-02-20__11-40-00_018.mkv",
        track_id=1,
        frame_number=frames,
        centre_x=x,
        centre_y=y,
        pixel_count=np.full(count, 20, dtype=np.int32),
        peak_deviation=np.full(count, 18.0),
        brightness=np.full(count, 900.0) if brightness is None else brightness,
        major_axis=np.full(count, 6.0),
        minor_axis=np.full(count, 3.0),
    )
