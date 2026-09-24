"""A track read for rhythm, as an image of rate against time.

A wingbeat, a strobe and a rotor are all one thing: a signal that
repeats. The descriptors carry a single number for it, the strongest
rhythm in the brightness, and a number cannot say whether the rhythm
held for the whole track or passed through it, nor which of the things
a track does is the one that repeats. This draws the whole of it.

Four signals are read, because a rhythm shows in whichever of them the
object varies: how bright it was, how large, how far it strayed from
its own straight line, and whether it was found at all. The last is the
detector's own, not the object's, and it is here to be told apart from
the other three rather than mistaken for them.

Rates are in cycles per frame, as the flicker descriptor already
reports them, so nothing here has to know a recording's frame rate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

BRIGHTNESS = "brightness"
SIZE = "size"
WOBBLE = "wobble"
PRESENCE = "presence"
SIGNALS = (BRIGHTNESS, SIZE, WOBBLE, PRESENCE)

RATE_COUNT = 32
LOWEST_RATE = 0.02
HIGHEST_RATE = 0.5
"""The rates an image spans, in cycles per frame.

Every track is read at the same rates, whatever its length, because two
images are only worth putting side by side if a height means the same
in both. The top is the fastest rhythm any sampling can carry, one
cycle every two frames, which is 12.5 Hz on a 25 frame a second
recording; the bottom is 0.5 Hz there.
"""

WAVELET_CYCLES = 6.0
"""Cycles the wavelet holds under its envelope.

It sets what the image trades: a wider wavelet separates two near rates
but smears when they start and stop. Six is the usual choice and puts
the trade in the middle.
"""

DYNAMIC_RANGE_DB = 20.0
"""The reading a drawing gives its full colour to, in decibels.

Held the same for every track so that the brightness of a band means
the same from one panel to the next. A reading runs past it, and what
is drawn stops there.
"""

TAPER_SHARE = 0.1
"""How much of each end of a track is brought down to nothing."""

SETTLING_WIDTHS = 2.0
"""Wavelet widths a rate's reading is averaged over along the track."""

QUIET_DB = 40.0
"""How far below the average rate's power a rate stops being read.

At a rate a signal holds nothing at, what is left is the arithmetic's
own dust, and dust weighed against the dust beside it is a ratio like
any other, so such a rate is left dark over the whole track rather than
weighed at all. Weighing it frame by frame is not enough: a rate whose
whole reading is dust still has frames standing over the dust of the
frames around them.
"""

BACKGROUND_RATES = 9
"""Rates a rate's background is taken over, itself included.

Wider than the band one rhythm covers at WAVELET_CYCLES, which is three
or four of these rows, so that a band cannot become its own background.
"""

MIN_FRAMES = 4
"""Frames a track needs before any rate can be read from it."""

STILL = 1e-9
"""How much of its own size a signal has to keep once its drift is taken
out before any rate is read from it.

A track whose brightness never moved keeps only what the arithmetic
left behind, and one rate's worth of that over another's is a ratio
like any other.
"""


@dataclass(frozen=True, slots=True)
class TrackSignals:
    """What a track did on every frame it spanned, gaps filled in.

    The frame numbers run one by one from the track's first to its last,
    whether or not the detector reported it on each of them, because a
    rhythm is read against even time. Half the frames a track spans are
    typically filled this way.

    Where the blob was on each of those frames is carried alongside the
    signals, because a reading that takes the track's shape needs the
    path on the same grid and would otherwise fill the gaps a second
    time and differently.
    """

    frame_number: np.ndarray
    centre_x: np.ndarray
    centre_y: np.ndarray
    values: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class Spectrogram:
    """How strongly a signal repeated, by rate and by frame.

    The power is in decibels over what the rates around it carry, and
    nothing where it carries no more than they do. Where measurable is
    false the wavelet at that rate reached past the end of the track and
    the reading there is the edge, not the track.
    """

    power: np.ndarray
    rates: np.ndarray
    measurable: np.ndarray


def track_signals(
    *,
    frame_number: np.ndarray,
    centre_x: np.ndarray,
    centre_y: np.ndarray,
    brightness: np.ndarray,
    pixel_count: np.ndarray,
    reach: float,
) -> TrackSignals:
    """The four signals of one track, each on the track's own frame grid.

    Brightness and size are read as logarithms, because a rhythm in
    either is a ratio: the same doubling is the same rhythm whether the
    object summed a hundred grey levels or a million, and the corpus
    spans that whole range.
    """
    grid, level, taken = _on_even_frames(frame_number)
    along_x = np.interp(grid, level, centre_x[taken])
    along_y = np.interp(grid, level, centre_y[taken])
    across = _across_the_line(along_x, along_y)

    found = np.zeros(grid.size)
    found[level.astype(np.int64)] = 1.0

    return TrackSignals(
        frame_number=grid.astype(np.int64) + int(frame_number[0]),
        centre_x=along_x,
        centre_y=along_y,
        values={
            BRIGHTNESS: np.interp(grid, level, np.log1p(brightness[taken].astype(np.float64))),
            SIZE: np.interp(grid, level, np.log1p(pixel_count[taken].astype(np.float64))),
            WOBBLE: across / reach,
            PRESENCE: found,
        },
    )


def spectrogram(signal: np.ndarray) -> Spectrogram:
    """What the signal repeated at, frame by frame.

    A track carries a slow drift far stronger than anything that
    repeats, and it would be the whole image, so each rate is weighed
    against the falling background the track's own rates trace out. What
    is left standing is a rhythm.
    """
    rates = np.geomspace(LOWEST_RATE, HIGHEST_RATE, RATE_COUNT)
    if signal.size < MIN_FRAMES:
        return _nothing(rates, count=signal.size)

    steps = np.arange(signal.size, dtype=np.float64)
    level = signal - np.polynomial.Polynomial.fit(steps, signal, 1)(steps)
    if np.abs(level).max() <= STILL * max(float(np.abs(signal).max()), 1.0):
        return _nothing(rates, count=signal.size)

    scales = _scales(rates)
    measurable = _measurable(scales, count=signal.size)
    return Spectrogram(
        power=_above_background(_wavelet_power(level, scales=scales), measurable=measurable),
        rates=rates,
        measurable=measurable,
    )


def _nothing(rates: np.ndarray, *, count: int) -> Spectrogram:
    """The image of a track that holds no rate to read."""
    flat = np.zeros((rates.size, count))
    return Spectrogram(power=flat, rates=rates, measurable=flat.astype(bool))


def _on_even_frames(frame_number: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grid a track's signals are read on, where its own frames sit on it,
    and which of its rows those are.

    A frame the detector reported twice keeps its first row only,
    because the grid is read by interpolation and that needs the frames
    it is given to climb. The rows come back so that every signal drops
    the same ones.
    """
    level, taken = np.unique((frame_number - frame_number[0]).astype(np.float64), return_index=True)
    return np.arange(level[-1] + 1.0), level, taken


def _across_the_line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """How far the track stood to the side of its own straight line, on each
    frame.

    A path that flaps strays from side to side of the line it travels
    along; one that is flown or driven does not. Taken about the line
    the track itself fits, so a heading of any direction reads the same.
    """
    points = np.column_stack((x, y))
    centred = points - points.mean(axis=0)
    _, _, directions = np.linalg.svd(centred, full_matrices=False)
    return np.asarray(centred @ directions[1])


def _scales(rates: np.ndarray) -> np.ndarray:
    """The width of the wavelet that answers each rate, in frames."""
    return (WAVELET_CYCLES + np.sqrt(2.0 + WAVELET_CYCLES**2)) / (4.0 * np.pi * rates)


def _measurable(scales: np.ndarray, *, count: int) -> np.ndarray:
    """Where a wavelet lies wholly inside the track.

    Near either end the wavelet hangs over the edge and reads the
    track's beginning or end, which is a step and answers at every rate.
    A short track is outside this at its slow rates throughout, and a
    median track of the corpus spans 45 frames, so most of it is.
    """
    reaches = np.sqrt(2.0) * scales
    steps = np.arange(count)
    return (steps >= reaches[:, None]) & (steps <= count - 1.0 - reaches[:, None])


def _wavelet_power(level: np.ndarray, *, scales: np.ndarray) -> np.ndarray:
    """The strength of each rate on each frame, by Morlet wavelet.

    The wavelet is applied as a multiplication over the signal's own
    spectrum, which answers every frame of one rate at once.
    """
    spectrum = np.fft.fft(_tapered(level))

    angular = 2.0 * np.pi * np.fft.fftfreq(level.size)
    ahead = angular > 0.0

    power = np.empty((scales.size, level.size))
    for row, scale in enumerate(scales):
        wavelet = np.where(ahead, np.exp(-0.5 * np.square(scale * angular - WAVELET_CYCLES)), 0.0)
        answer = np.square(np.abs(np.fft.ifft(spectrum * wavelet))) * scale
        power[row] = _settled(answer, scale=scale)
    return power


def _tapered(level: np.ndarray) -> np.ndarray:
    """The signal brought to nothing at both ends.

    A wavelet is laid over the signal through its spectrum, which joins
    the last frame to the first. A track rarely ends where it began, and
    that join is a step: it answers at every rate at once and stands at
    the ends of the image as the brightest thing in it, on tracks that
    hold no rhythm at all. Brought down to nothing at both ends there is
    no step to answer.
    """
    ramp = max(1, int(round(TAPER_SHARE * level.size)))
    rise = 0.5 * (1.0 - np.cos(np.pi * (np.arange(ramp) + 0.5) / ramp))
    window = np.ones(level.size)
    window[:ramp] = rise
    window[level.size - ramp :] = rise[::-1]
    return level * window


def _settled(power: np.ndarray, *, scale: float) -> np.ndarray:
    """One rate's reading averaged over the frames its own wavelet covers.

    A reading taken frame by frame scatters by about ten decibels either
    way whatever the track did, which against a colour scale of twenty
    would be speckle over every image and a rhythm in none of them. The
    average is taken over the wavelet's own width, so nothing the
    wavelet could tell apart is lost.
    """
    window = np.ones(min(max(3, int(round(SETTLING_WIDTHS * scale))), power.size))
    return np.convolve(power, window, mode="same") / np.convolve(np.ones_like(power), window, mode="same")


def _above_background(power: np.ndarray, *, measurable: np.ndarray) -> np.ndarray:
    """The power in decibels over what the rates around it carry, from nothing
    to DYNAMIC_RANGE_DB.

    Every one of these signals falls away with rate, so an image drawn
    as it stands is a bright floor at the bottom and dark everywhere
    else, the same for a track that repeats and one that does not. Each
    rate is divided by what its neighbours carry, which leaves a rhythm
    standing as a band and a track without one flat.
    """
    counted = measurable.sum(axis=1)
    totals = np.where(measurable, power, 0.0).sum(axis=1)
    level = np.divide(totals, counted, where=counted > 0, out=np.zeros(power.shape[0]))
    held = (counted > 0) & (level > 0.0)
    if held.sum() < 2:
        return np.zeros_like(power)

    carrying = held & (level >= level[held].mean() * 10.0 ** (-QUIET_DB / 10.0))
    if carrying.sum() < 2:
        return np.zeros_like(power)

    quiet = np.full(power.shape[0], np.log10(level[carrying]).max())
    quiet[carrying] = _neighbouring_level(np.log10(level[carrying]))
    excess = 10.0 * np.log10(np.maximum(power / np.power(10.0, quiet)[:, None], 1.0))
    return np.where(carrying[:, None], excess, 0.0)


def _neighbouring_level(level: np.ndarray) -> np.ndarray:
    """What the rates around each one carry, without whatever stands over them,
    in the logarithm the level is given in.

    The lowest reading within reach of each rate is taken first, which
    leaves nothing of a band narrower than that reach, and the highest
    of those is taken back, which returns the steep fall from the
    slowest rate to the fastest as it was. What comes out is the level
    the band interrupted.

    Neither a fitted line nor a middle reading can stand in for this. A
    line is pulled up by the band it is meant to ignore, and a middle
    reading follows the band's own flanks all the way to its top,
    because a reading in the middle of rates that only climb is the
    climbing one itself.
    """
    edge = BACKGROUND_RATES // 2
    return _reaching(_reaching(level, edge=edge, take=np.min), edge=edge, take=np.max)


def _reaching(level: np.ndarray, *, edge: int, take: Callable[[np.ndarray], np.floating]) -> np.ndarray:
    """One reading per rate, taken over the rates within edge of it."""
    return np.array([take(level[max(at - edge, 0) : at + edge + 1]) for at in range(level.size)])
