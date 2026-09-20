"""One fixed-length description per track, read off a written track file.

Tracks run for different numbers of frames, so nothing compares them
until each is reduced to the same set of numbers. Every number here is
one a reader can name, which is what lets a flagged track be argued
about. Distances are ratios of the frame's larger side, so a track
described at one frame height reads the same as the same track described
at another.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FLICKER_FRAMES = 8
"""Frames a track needs before its brightness is read for a rhythm."""

FLICKER_CYCLES = 3
"""Cycles a rhythm has to complete within the track before it counts.

A meteor's single rise and decay is one narrow lobe at the low end of
the spectrum, and it holds over half the power in its strongest bin. Not
counting anything slower than three cycles is what keeps it apart from a
wingbeat.
"""

VELOCITY_TERMS = 2
"""Coefficients a steady-heading fit solves for, per axis."""

ACCELERATION_TERMS = 3
"""Coefficients a steady-acceleration fit solves for, per axis."""

TRACK_COLUMNS = (
    "frame_number",
    "centre_x",
    "centre_y",
    "pixel_count",
    "peak_deviation",
    "brightness",
    "major_axis",
    "minor_axis",
)
"""What a description is built from.

The blob's centre stands in for the reported position, because the
brightest pixel of a streak hops along it from frame to frame and lands
on whole pixels, which turns a straight path into a jagged one.
"""


@dataclass(frozen=True, slots=True)
class TrackDescriptor:
    """What one track looks like, in numbers that hold across recordings."""

    recording: str
    track_id: int

    frames: int
    missed_frames: int
    path_length: float
    displacement: float
    straightness: float
    speed_mean: float
    speed_max: float
    speed_deviation: float

    turn_mean: float
    turn_max: float
    line_residual: float
    velocity_residual: float
    acceleration_residual: float

    start_edge: float
    end_edge: float
    elevation: float

    area_mean: float
    area_variation: float
    elongation_mean: float
    brightness_mean: float
    brightness_variation: float
    peak_deviation_mean: float
    peak_deviation_max: float
    flicker: float
    flicker_rate: float


@dataclass(frozen=True, slots=True)
class Frame:
    """The frame the positions were measured in."""

    height: int
    width: int

    @property
    def reach(self) -> float:
        """The side distances are given as a ratio of, as the tracker gives
        them."""
        return float(max(self.height, self.width))


@dataclass(frozen=True, slots=True)
class TrackRows:
    """The frames of one track, in frame order."""

    recording: str
    track_id: int
    frame_number: np.ndarray
    centre_x: np.ndarray
    centre_y: np.ndarray
    pixel_count: np.ndarray
    peak_deviation: np.ndarray
    brightness: np.ndarray
    major_axis: np.ndarray
    minor_axis: np.ndarray


ARROW_TYPES = {"str": pa.string(), "int": pa.int32(), "float": pa.float64()}

SCHEMA = pa.schema([(field.name, ARROW_TYPES[str(field.type)]) for field in fields(TrackDescriptor)])
"""One row per track, with a column per field of TrackDescriptor."""


def describe_file(path: Path) -> list[TrackDescriptor]:
    """Describe every track a written track file holds."""
    table = pq.read_table(path)
    frame = frame_of(table)
    return [describe(rows, frame=frame) for rows in tracks_of(table)]


def describe(rows: TrackRows, *, frame: Frame) -> TrackDescriptor:
    """Reduce one track to the numbers that compare it with any other."""
    steps = _steps(rows, reach=frame.reach)
    turns = _turn_statistics(steps)
    path_length = float(steps.distance.sum())
    displacement = float(np.hypot(rows.centre_x[-1] - rows.centre_x[0], rows.centre_y[-1] - rows.centre_y[0]))
    displacement /= frame.reach
    flicker = _flicker(rows)
    area = rows.pixel_count.astype(np.float64)

    return TrackDescriptor(
        recording=rows.recording,
        track_id=rows.track_id,
        frames=int(rows.frame_number.size),
        missed_frames=int(rows.frame_number[-1] - rows.frame_number[0] + 1 - rows.frame_number.size),
        path_length=path_length,
        displacement=displacement,
        straightness=displacement / path_length if path_length > 0.0 else 0.0,
        speed_mean=_mean(steps.speed),
        speed_max=_max(steps.speed),
        speed_deviation=_deviation(steps.speed),
        turn_mean=turns.mean,
        turn_max=turns.maximum,
        line_residual=_line_residual(rows, reach=frame.reach),
        velocity_residual=_curve_residual(rows, reach=frame.reach, terms=VELOCITY_TERMS),
        acceleration_residual=_curve_residual(rows, reach=frame.reach, terms=ACCELERATION_TERMS),
        start_edge=_edge_distance(rows.centre_x[0], rows.centre_y[0], frame=frame),
        end_edge=_edge_distance(rows.centre_x[-1], rows.centre_y[-1], frame=frame),
        elevation=float(rows.centre_y.mean() / frame.height),
        area_mean=_mean(area),
        area_variation=_variation(area),
        elongation_mean=_mean(rows.major_axis / np.maximum(rows.minor_axis, 1.0)),
        brightness_mean=_mean(rows.brightness),
        brightness_variation=_variation(rows.brightness),
        peak_deviation_mean=_mean(rows.peak_deviation),
        peak_deviation_max=_max(rows.peak_deviation),
        flicker=flicker.strength,
        flicker_rate=flicker.rate,
    )


def tracks_of(table: pa.Table) -> list[TrackRows]:
    """Split a track file's rows into one TrackRows per track."""
    columns = {name: np.asarray(table.column(name).to_numpy()) for name in TRACK_COLUMNS}
    track_ids = np.asarray(table.column("track_id").to_numpy())
    recordings = table.column("recording").to_pylist()

    found = []
    for track_id in sorted(set(track_ids.tolist())):
        held = np.flatnonzero(track_ids == track_id)
        found.append(
            TrackRows(
                recording=recordings[int(held[0])],
                track_id=int(track_id),
                **{name: values[held] for name, values in columns.items()},
            )
        )
    return found


def frame_of(table: pa.Table) -> Frame:
    """The frame the file says its positions were measured in."""
    metadata = table.schema.metadata or {}
    return Frame(
        height=int(metadata[b"hessdalen_frame_height"]),
        width=int(metadata[b"hessdalen_frame_width"]),
    )


def write_descriptors(path: Path, descriptors: list[TrackDescriptor]) -> int:
    """Write the descriptions out, and return how many rows."""
    table = pa.Table.from_pylist([asdict(descriptor) for descriptor in descriptors], schema=SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return table.num_rows


@dataclass(frozen=True, slots=True)
class Steps:
    """What the track did between one reported frame and the next."""

    distance: np.ndarray
    speed: np.ndarray
    heading: np.ndarray


def _steps(rows: TrackRows, *, reach: float) -> Steps:
    """The move from each reported frame to the one after it.

    A track may miss frames and carry on, so a step is divided by the
    frames it spans and speed stays per frame.
    """
    across = np.diff(rows.centre_x)
    down = np.diff(rows.centre_y)
    elapsed = np.maximum(np.diff(rows.frame_number).astype(np.float64), 1.0)
    distance = np.hypot(across, down) / reach

    return Steps(distance=distance, speed=distance / elapsed, heading=np.arctan2(down, across))


@dataclass(frozen=True, slots=True)
class Turns:
    """How sharply a track changed heading, over the steps that had one."""

    mean: float
    maximum: float


def _turn_statistics(steps: Steps) -> Turns:
    """The heading changes, weighed by how far the track moved to make them.

    A heading measured across half a pixel says nothing, and a track
    that comes to rest and glows in place spends most of its steps
    there. Weighing each turn by the shorter of the two steps that meet
    at it keeps those from setting what the track looks like, and the
    largest turn is taken over the steps that carry at least the median
    weight.
    """
    if steps.heading.size < 2:
        return Turns(mean=0.0, maximum=0.0)

    turned = np.abs((np.diff(steps.heading) + math.pi) % (2.0 * math.pi) - math.pi)
    weight = np.minimum(steps.distance[:-1], steps.distance[1:])
    if weight.sum() <= 0.0:
        return Turns(mean=0.0, maximum=0.0)

    carrying = turned[weight >= np.median(weight)]
    return Turns(mean=float((turned * weight).sum() / weight.sum()), maximum=_max(carrying))


def _line_residual(rows: TrackRows, *, reach: float) -> float:
    """How far the track sits off the straight line through it.

    The line is the one that minimises the perpendicular distance, so a
    track running down the frame reads the same as one running across
    it.
    """
    if rows.centre_x.size < 3:
        return 0.0

    centred = np.stack([rows.centre_x - rows.centre_x.mean(), rows.centre_y - rows.centre_y.mean()])
    across = float(np.linalg.svd(centred, compute_uv=False)[-1])
    return across / math.sqrt(float(rows.centre_x.size)) / reach


def _curve_residual(rows: TrackRows, *, reach: float, terms: int) -> float:
    """How far the track sits off the smoothest path of the given order.

    Two terms is a steady heading and speed, which a satellite and a
    distant aircraft hold. Three adds a steady acceleration, which a
    meteor slowing in the air holds. A track that fits neither is
    manoeuvring or is noise.
    """
    if rows.centre_x.size <= terms:
        return 0.0

    elapsed = (rows.frame_number - rows.frame_number[0]).astype(np.float64)
    powers = np.stack([elapsed**power for power in range(terms)], axis=1)
    squared = sum(_fit_error(powers, axis.astype(np.float64)) for axis in (rows.centre_x, rows.centre_y))
    return math.sqrt(squared / float(rows.centre_x.size)) / reach


def _fit_error(powers: np.ndarray, axis: np.ndarray) -> float:
    """The squared distance from one axis of the track to its best fit."""
    coefficients = np.linalg.lstsq(powers, axis, rcond=None)[0]
    return float(np.square(axis - powers @ coefficients).sum())


def _edge_distance(x: float, y: float, *, frame: Frame) -> float:
    """How far a point sits from the nearest edge of the frame.

    A track that crosses the frame starts and ends on an edge. One that
    appears inside the frame and vanishes there does not, and the two
    are different populations.
    """
    nearest = min(float(x), float(y), float(frame.width - 1) - float(x), float(frame.height - 1) - float(y))
    return max(nearest, 0.0) / frame.reach


@dataclass(frozen=True, slots=True)
class Flicker:
    """The strongest rhythm in a track's brightness, and how fast it is."""

    strength: float
    rate: float


def _flicker(rows: TrackRows) -> Flicker:
    """The share of the brightness's variation that one rhythm carries.

    A wingbeat and an aircraft's strobe put their variation at one rate.
    A meteor rises and decays once and a cloud edge wanders, and both
    leave their variation below the rate a rhythm has to reach.

    The strongest rhythmic bin is weighed against the whole of the
    variation, so a track carrying most of its variation below that rate
    reports a weak rhythm however peaked the little above it is.

    The brightness is read on an even frame grid, because a track that
    missed frames would otherwise report a rhythm its gaps made.
    """
    if rows.frame_number.size < FLICKER_FRAMES:
        return Flicker(strength=0.0, rate=0.0)

    elapsed = (rows.frame_number - rows.frame_number[0]).astype(np.float64)
    even = np.arange(elapsed[-1] + 1.0)
    level = np.interp(even, elapsed, rows.brightness.astype(np.float64))
    drift = np.polynomial.Polynomial.fit(even, level, 1)(even)

    power = np.square(np.abs(np.fft.rfft(level - drift)))[1:]
    rhythmic = power[FLICKER_CYCLES - 1 :]
    total = float(power.sum())
    if total <= 0.0 or rhythmic.size == 0:
        return Flicker(strength=0.0, rate=0.0)

    strongest = int(np.argmax(rhythmic)) + FLICKER_CYCLES
    return Flicker(strength=float(rhythmic.max()) / total, rate=float(strongest) / float(even.size))


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _max(values: np.ndarray) -> float:
    return float(values.max()) if values.size else 0.0


def _deviation(values: np.ndarray) -> float:
    return float(values.std()) if values.size else 0.0


def _variation(values: np.ndarray) -> float:
    """How far a series swings about its own level, which lets a faint track
    and a bright one read the same."""
    average = _mean(values)
    return _deviation(values) / average if average > 0.0 else 0.0
