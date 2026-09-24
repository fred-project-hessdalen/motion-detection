"""What a track's rhythm reads as, and what its gaps do not make it read."""

import numpy as np
import pytest

from hessdalen.analysis.spectra import (
    BRIGHTNESS,
    DYNAMIC_RANGE_DB,
    PRESENCE,
    RATE_COUNT,
    SIGNALS,
    SIZE,
    WOBBLE,
    Spectrogram,
    spectrogram,
    track_signals,
)

FRAMES = 200
REACH = 1920.0


def test_a_repeating_signal_bands_at_the_rate_it_repeats() -> None:
    drawn = spectrogram(_beating(rate=0.08))

    assert _loudest(drawn) == pytest.approx(0.08, rel=0.12)


def test_a_signal_that_only_drifts_reads_far_below_one_that_repeats() -> None:
    """A track's slow drift is far stronger than anything that repeats, so an
    image that did not weigh it away would show every track as a rhythm."""
    steps = np.arange(FRAMES, dtype=np.float64)

    drifting = spectrogram(np.square(steps / FRAMES))

    assert _held(drifting).max() < _held(spectrogram(_beating(rate=0.08))).max() / 2.0


def test_a_rhythm_that_comes_and_goes_stands_only_over_the_frames_it_ran() -> None:
    """Telling when a track repeated is the whole of what the second axis is
    for; a spectrum alone says only that it did."""
    passing = np.zeros(FRAMES)
    passing[80:140] = np.sin(2.0 * np.pi * 0.2 * np.arange(60))

    drawn = spectrogram(passing)
    over = _held(drawn)[int(np.abs(drawn.rates - 0.2).argmin())]

    assert over[80:140].mean() > DYNAMIC_RANGE_DB / 2.0
    assert over[:60].max() < DYNAMIC_RANGE_DB / 2.0
    assert over[160:].max() < DYNAMIC_RANGE_DB / 2.0


def test_a_track_missing_frames_still_reads_the_rate_it_repeats_at() -> None:
    """Half the frames a corpus track spans are filled in, so a rhythm has to
    survive the filling and the filling must not pass for one."""
    kept = np.arange(FRAMES)[np.arange(FRAMES) % 3 != 0]
    signals = track_signals(
        frame_number=kept,
        centre_x=np.zeros(kept.size),
        centre_y=np.zeros(kept.size),
        brightness=np.expm1(_beating(rate=0.08)[kept] + 6.0),
        pixel_count=np.full(kept.size, 30),
        reach=REACH,
    )

    drawn = spectrogram(signals.values[BRIGHTNESS])

    assert _loudest(drawn) == pytest.approx(0.08, rel=0.15)


def test_the_rate_frames_go_missing_at_reads_as_presence_and_not_as_brightness() -> None:
    """The detector drops a track and picks it up again in a rhythm of its own,
    which is the detector's and not the object's."""
    kept = np.arange(FRAMES)[np.arange(FRAMES) % 4 != 0]
    signals = track_signals(
        frame_number=kept,
        centre_x=np.zeros(kept.size),
        centre_y=np.zeros(kept.size),
        brightness=np.full(kept.size, 1000.0),
        pixel_count=np.full(kept.size, 30),
        reach=REACH,
    )

    assert _loudest(spectrogram(signals.values[PRESENCE])) == pytest.approx(0.25, rel=0.15)
    assert _held(spectrogram(signals.values[BRIGHTNESS])).max() == 0.0


def test_a_frame_reported_twice_is_taken_once() -> None:
    """The corpus repeats a frame on a couple of thousand rows, and reading a
    grid by interpolation needs the frames it is given to climb."""
    signals = track_signals(
        frame_number=np.array([10, 11, 11, 12, 14]),
        centre_x=np.array([0.0, 1.0, 9.0, 2.0, 4.0]),
        centre_y=np.zeros(5),
        brightness=np.array([100.0, 200.0, 900.0, 300.0, 400.0]),
        pixel_count=np.array([3, 4, 9, 5, 6]),
        reach=REACH,
    )

    assert list(signals.frame_number) == [10, 11, 12, 13, 14]
    assert signals.values[BRIGHTNESS][1] == pytest.approx(np.log1p(200.0))
    assert list(signals.values[PRESENCE]) == [1.0, 1.0, 1.0, 0.0, 1.0]


def test_every_signal_covers_the_frames_the_track_spans() -> None:
    signals = track_signals(
        frame_number=np.array([4, 6, 7]),
        centre_x=np.array([0.0, 30.0, 45.0]),
        centre_y=np.array([0.0, 4.0, 0.0]),
        brightness=np.array([100.0, 200.0, 300.0]),
        pixel_count=np.array([3, 4, 5]),
        reach=REACH,
    )

    assert sorted(signals.values) == sorted(SIGNALS)
    assert all(signals.values[name].size == 4 for name in SIGNALS)
    assert signals.values[WOBBLE].max() < 1.0


def test_a_wandering_track_wobbles_and_a_straight_one_does_not() -> None:
    steps = np.arange(FRAMES)
    straight = _wobble(steps, across=np.zeros(FRAMES))

    assert np.abs(straight).max() == pytest.approx(0.0, abs=1e-9)
    assert np.abs(_wobble(steps, across=20.0 * np.sin(steps))).max() > 10.0 / REACH


def test_a_track_too_short_to_carry_a_rate_reads_none() -> None:
    drawn = spectrogram(np.array([1.0, 3.0]))

    assert drawn.power.shape == (RATE_COUNT, 2)
    assert not drawn.measurable.any()
    assert not drawn.power.any()


def test_the_slow_rates_of_a_short_track_are_not_measurable() -> None:
    """A wavelet slower than the track hangs over its ends and reads them, so
    the image says where it read the track and where it read the edge."""
    drawn = spectrogram(_beating(rate=0.25)[:24])

    assert not drawn.measurable[0].any()
    assert drawn.measurable[-1].any()


def _beating(*, rate: float) -> np.ndarray:
    return np.sin(2.0 * np.pi * rate * np.arange(FRAMES))


def _wobble(steps: np.ndarray, *, across: np.ndarray) -> np.ndarray:
    signals = track_signals(
        frame_number=steps,
        centre_x=steps.astype(np.float64),
        centre_y=across,
        brightness=np.full(steps.size, 100.0),
        pixel_count=np.full(steps.size, 4),
        reach=REACH,
    )
    return signals.values[WOBBLE]


def _held(drawn: Spectrogram) -> np.ndarray:
    """The power where the wavelet lay inside the track."""
    return np.where(drawn.measurable, drawn.power, 0.0)


def _loudest(drawn: Spectrogram) -> float:
    """The rate carrying the most power anywhere it could be measured."""
    return float(drawn.rates[int(_held(drawn).max(axis=1).argmax())])


def test_size_follows_the_pixel_count_it_is_read_from() -> None:
    signals = track_signals(
        frame_number=np.array([0, 1]),
        centre_x=np.zeros(2),
        centre_y=np.zeros(2),
        brightness=np.array([10.0, 20.0]),
        pixel_count=np.array([7, 70]),
        reach=REACH,
    )

    assert list(signals.values[SIZE]) == [pytest.approx(np.log1p(7)), pytest.approx(np.log1p(70))]
