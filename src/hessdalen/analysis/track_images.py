"""A track carried into a square picture, one way per candidate.

A track is a short run of numbers, and a run of numbers is a poor thing
to hand a network trained on pictures. Each of these turns one track
into a fixed square whatever the track's length, so that a corpus of
tracks of every length becomes a corpus of images of one size.

They divide into two families. One measures the track against itself at
every pair of moments, which is where a repeating motion shows up as
stripes running parallel to the diagonal. The other lays the track out
as a picture first and reads it as a picture.

Every one of them fixes the length, and the length is thrown away in
doing so. A track of twenty frames and a track of two thousand come out
the same size and, if they did the same thing, looking the same. What
separates them has to be carried beside the image.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from hessdalen.analysis.spectra import (
    BRIGHTNESS,
    DYNAMIC_RANGE_DB,
    SIZE,
    WOBBLE,
    TrackSignals,
    spectrogram,
)

SIDE = 64
"""How wide and tall every candidate's picture comes out."""

STEPS = 64
"""Moments a track is read at, however many frames it spans.

A track is stretched or squeezed onto this many, so that the picture of
half a track can be put beside the picture of another whole one.
"""

LEVELS = 8
"""Bands a signal's range is cut into before its steps are counted."""


def path_similarity(signals: TrackSignals) -> np.ndarray:
    """How far the blob stood from where it stood, at every pair of moments.

    A track that returns to where it was darkens away from the diagonal,
    so circling and hovering leave a pattern of squares and a straight
    passage leaves a smooth gradient.
    """
    return _apart(_course(signals))


def state_similarity(signals: TrackSignals) -> np.ndarray:
    """How unlike the blob was to itself, at every pair of moments, over how
    fast it moved, how far it strayed, how bright it was and how large.

    Each is put on its own scale first, so that the picture is of the
    track's manner and not of how bright the night was.
    """
    course = _course(signals)
    steps = np.gradient(course, axis=0)
    state = np.column_stack(
        [
            _levelled(np.hypot(steps[:, 0], steps[:, 1])),
            _levelled(_stretched(signals.values[WOBBLE])),
            _levelled(_stretched(signals.values[BRIGHTNESS])),
            _levelled(_stretched(signals.values[SIZE])),
        ]
    )
    return _apart(state)


def wobble_fields(signals: TrackSignals) -> np.ndarray:
    """The track's straying laid out as three pictures at once."""
    return _angular_fields(signals.values[WOBBLE])


def brightness_fields(signals: TrackSignals) -> np.ndarray:
    """The track's brightness laid out as three pictures at once."""
    return _angular_fields(signals.values[BRIGHTNESS])


def path_drawn(signals: TrackSignals) -> np.ndarray:
    """The track drawn as a line, turned to lie along the picture.

    Turning it means a track is drawn the same way whichever direction
    it happened to travel in, which is a property of the camera's aim
    and not of the thing.
    """
    course = _course(signals)
    canvas = np.zeros((SIDE, SIDE), dtype=np.float32)
    corners = np.rint((_turned(course) + 1.0) / 2.0 * (SIDE - 1)).astype(np.int32)
    cv2.polylines(canvas, [corners], isClosed=False, color=1.0, thickness=1)
    return canvas


def path_transform(signals: TrackSignals) -> np.ndarray:
    """What the drawn track holds at every direction and fineness.

    A straight line answers across one direction alone, and every wave
    laid along the line adds a pair of marks either side of the middle,
    further out the tighter the wave.
    """
    spread = np.abs(np.fft.fftshift(np.fft.fft2(path_drawn(signals))))
    return _topped(np.log1p(spread))


def wobble_scalogram(signals: TrackSignals) -> np.ndarray:
    """How strongly the track's straying repeated, by rate and by moment."""
    drawn = spectrogram(signals.values[WOBBLE])
    held = np.where(drawn.measurable, np.clip(drawn.power / DYNAMIC_RANGE_DB, 0.0, 1.0), 0.0)
    return _square(np.flipud(held))


CANDIDATES: dict[str, Callable[[TrackSignals], np.ndarray]] = {
    "path similarity": path_similarity,
    "state similarity": state_similarity,
    "wobble fields": wobble_fields,
    "brightness fields": brightness_fields,
    "path transform": path_transform,
    "wobble scalogram": wobble_scalogram,
    "path drawn": path_drawn,
}
"""Every way a track is turned into a picture, by the name it is reported
under."""


def _course(signals: TrackSignals) -> np.ndarray:
    """Where the blob was at each of STEPS moments, about its own middle and
    in units of its own extent, so that a distant track and a near one of the
    same shape come out alike."""
    points = np.column_stack((_stretched(signals.centre_x), _stretched(signals.centre_y)))
    points = points - points.mean(axis=0)
    return points / max(float(np.abs(points).max()), 1e-9)


def _turned(course: np.ndarray) -> np.ndarray:
    """The course laid along the width of the picture and filling it."""
    _, _, directions = np.linalg.svd(course, full_matrices=False)
    laid = course @ directions.T
    return laid / max(float(np.abs(laid).max()), 1e-9)


def _apart(state: np.ndarray) -> np.ndarray:
    """How far every moment stood from every other, at its largest 1."""
    gaps = np.linalg.norm(state[:, None, :] - state[None, :, :], axis=-1)
    return _topped(gaps)


def _angular_fields(signal: np.ndarray) -> np.ndarray:
    """One signal as three pictures: what its moments come to when added,
    what they come to when taken apart, and how often a reading of one band
    is followed by a reading of another.

    The signal is put on a fixed range and read as an angle, so that
    adding two moments says where both stood and neither how bright the
    night was nor how far the thing was.
    """
    scaled = _ranged(_stretched(signal))
    angle = np.arccos(np.clip(scaled, -1.0, 1.0))
    summed = np.cos(angle[:, None] + angle[None, :])
    parted = np.sin(angle[:, None] - angle[None, :])
    return np.dstack(((summed + 1.0) / 2.0, (parted + 1.0) / 2.0, _following(scaled)))


def _following(scaled: np.ndarray) -> np.ndarray:
    """How often a moment in one band is followed by a moment in another,
    read off for every pair of moments.

    Two moments far apart in time whose bands rarely follow one another
    stand out, which is what carries a signal that moves between levels
    in an order rather than at random.
    """
    edges = np.quantile(scaled, np.linspace(0.0, 1.0, LEVELS + 1)[1:-1])
    bands = np.digitize(scaled, edges)

    counted = np.zeros((LEVELS, LEVELS))
    np.add.at(counted, (bands[:-1], bands[1:]), 1.0)
    leaving = counted.sum(axis=1, keepdims=True)
    share = np.divide(counted, leaving, where=leaving > 0.0, out=np.zeros_like(counted))
    return np.asarray(share[bands[:, None], bands[None, :]])


def _stretched(values: np.ndarray) -> np.ndarray:
    """One signal read at STEPS moments, whatever its length."""
    if values.size == STEPS:
        return values
    if values.size < 2:
        return np.full(STEPS, float(values[0]) if values.size else 0.0)
    return np.interp(np.linspace(0.0, values.size - 1.0, STEPS), np.arange(values.size, dtype=np.float64), values)


def _levelled(values: np.ndarray) -> np.ndarray:
    """One signal about its own middle and in units of its own spread."""
    spread = float(values.std())
    return (values - values.mean()) / spread if spread > 0.0 else np.zeros_like(values)


def _ranged(values: np.ndarray) -> np.ndarray:
    """One signal laid across the range an angle can be read from."""
    low, high = float(values.min()), float(values.max())
    return 2.0 * (values - low) / (high - low) - 1.0 if high > low else np.zeros_like(values)


def _topped(picture: np.ndarray) -> np.ndarray:
    """A picture at its largest 1."""
    top = float(picture.max())
    return picture / top if top > 0.0 else picture


def _square(picture: np.ndarray) -> np.ndarray:
    return np.asarray(cv2.resize(picture.astype(np.float32), (SIDE, SIDE), interpolation=cv2.INTER_AREA))
