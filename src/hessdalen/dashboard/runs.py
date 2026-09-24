"""Detection runs kept on disk as an annotated video and a record of the
tracks.

A run is addressed by the recording it came from and a digest of the
settings and the detector source it ran with, so the dashboard finds an
earlier result again.
"""

from __future__ import annotations

import json
import time
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hessdalen.dashboard.encoder import STORED, encode, even, playback_rate
from hessdalen.dashboard.panels import (
    DEVIATION,
    RECORDING,
    Panels,
    box_size,
    draw_box,
    draw_frame_number,
    draw_trail,
    layout,
    recording_frames,
)
from hessdalen.dashboard.store import digest
from hessdalen.domain.models import MovementEvent, VideoFrame
from hessdalen.io.video import masked_stream
from hessdalen.processing.debug import track_color
from hessdalen.processing.devices import detection_stage
from hessdalen.processing.movement import MovementDetector, MovementSettings

PHASES = 2
MODULES = ("runs.py", "panels.py", "encoder.py")
"""The dashboard modules that decide what a stored run holds."""


@dataclass(frozen=True, slots=True)
class VideoProbe:
    frame_count: int
    frames_per_second: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class TrackPoint:
    frame_number: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Trajectory:
    track_id: int
    points: tuple[TrackPoint, ...]

    @property
    def first_frame(self) -> int:
        return self.points[0].frame_number

    @property
    def last_frame(self) -> int:
        return self.points[-1].frame_number

    @property
    def span(self) -> float:
        """Larger side of the box the track's points fit into, in pixels."""
        x_coords = [point.x for point in self.points]
        y_coords = [point.y for point in self.points]
        return max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))


@dataclass(frozen=True, slots=True)
class DetectionRun:
    video: Path
    annotated_video: Path
    frame_count: int
    detection_seconds: float
    trajectories: tuple[Trajectory, ...]


def run_detection(
    video: Path,
    *,
    settings: MovementSettings,
    target_height: int,
    panels: Panels,
    output_dir: Path,
    on_progress: Callable[[float], None],
) -> DetectionRun:
    """Detect movement in a recording and write the annotated video and the
    track record under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _run_paths(video, settings=settings, target_height=target_height, panels=panels, output_dir=output_dir)
    details = probe(video)
    phase = _phase_progress(on_progress, expected_frames=max(1, details.frame_count))

    started = time.perf_counter()
    trajectories = _detect(video, settings=settings, target_height=target_height, on_frame=phase(0))
    detection_seconds = time.perf_counter() - started

    frame_count = _render(
        video,
        settings=settings,
        target_height=target_height,
        panels=panels,
        frames_per_second=details.frames_per_second,
        trajectories=trajectories,
        output=paths.video,
        on_frame=phase(1),
    )

    run = DetectionRun(
        video=video,
        annotated_video=paths.video,
        frame_count=frame_count,
        detection_seconds=detection_seconds,
        trajectories=trajectories,
    )
    paths.record.write_text(json.dumps(_as_record(run), indent=2))
    return run


def _phase_progress(
    on_progress: Callable[[float], None],
    *,
    expected_frames: int,
) -> Callable[[int], Callable[[int], None]]:
    """A per-frame reporter for each phase, covering its own share of the
    bar."""

    def phase(index: int) -> Callable[[int], None]:
        return lambda frame_index: on_progress((index + min(1.0, (frame_index + 1) / expected_frames)) / PHASES)

    return phase


def load_run(
    video: Path,
    *,
    settings: MovementSettings,
    target_height: int,
    panels: Panels,
    output_dir: Path,
) -> DetectionRun | None:
    """The stored run for these settings, or None when the recording has not
    been run with them."""
    paths = _run_paths(video, settings=settings, target_height=target_height, panels=panels, output_dir=output_dir)
    if not (paths.record.is_file() and paths.video.is_file()):
        return None

    record = json.loads(paths.record.read_text())
    return DetectionRun(
        video=video,
        annotated_video=paths.video,
        frame_count=int(record["frame_count"]),
        detection_seconds=float(record["detection_seconds"]),
        trajectories=tuple(_trajectory_from_record(entry) for entry in record["trajectories"]),
    )


def probe(video: Path) -> VideoProbe:
    """Frame count, rate and size the container reports, without decoding."""
    capture = cv2.VideoCapture(str(video))
    try:
        return VideoProbe(
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            frames_per_second=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def trajectories_from_events(events: Iterable[MovementEvent]) -> tuple[Trajectory, ...]:
    """Gather the centroids a detector reported into one trajectory per track.

    A track is confirmed some frames after it starts and the detector
    then replays the centroids it held back, so the events of one track
    do not arrive in frame order.
    """
    points: dict[int, list[TrackPoint]] = {}
    for event in events:
        if event.track_id is None or event.centroid is None:
            continue
        point = TrackPoint(frame_number=event.frame_number, x=event.centroid[0], y=event.centroid[1])
        points.setdefault(event.track_id, []).append(point)

    return tuple(
        Trajectory(track_id=track_id, points=tuple(sorted(track_points, key=lambda point: point.frame_number)))
        for track_id, track_points in sorted(points.items())
    )


def _detect(
    video: Path,
    *,
    settings: MovementSettings,
    target_height: int,
    on_frame: Callable[[int], None],
) -> tuple[Trajectory, ...]:
    detector = MovementDetector(stream=masked_stream(video, target_height=target_height), settings=settings)
    return trajectories_from_events(_reporting_progress(detector.detect(), on_frame))


def _reporting_progress(events: Iterable[MovementEvent], on_frame: Callable[[int], None]) -> Iterator[MovementEvent]:
    for event in events:
        on_frame(event.frame_number)
        yield event


def _render(
    video: Path,
    *,
    settings: MovementSettings,
    target_height: int,
    panels: Panels,
    frames_per_second: float,
    trajectories: tuple[Trajectory, ...],
    output: Path,
    on_frame: Callable[[int], None],
) -> int:
    """Draw every track onto the panels asked for and encode them as one frame.

    The deviation panel is measured again here from the same grayscale
    the detector read, which is what makes it the picture the detections
    were taken from. A panel costs a decode of the recording and its
    share of the encode, and a panel left out costs neither, so asking
    for one is about half the work of asking for both. Returns the
    number of frames written. Some containers declare a frame count
    their stream does not hold, so this is the count that was decoded.
    """
    names = layout(panels)
    height, width = masked_stream(video, target_height=target_height).frame_shape
    panel_height, panel_width = even(height), even(width)

    overlays = _overlays(trajectories)
    size = box_size(panel_height, panel_width)

    stacked = np.empty((panel_height * len(names), panel_width, 3), dtype=np.uint8)
    views = {name: stacked[index * panel_height : (index + 1) * panel_height] for index, name in enumerate(names)}
    planar = np.empty((panel_height * len(names) * 3 // 2, panel_width), dtype=np.uint8)

    colour_frames = recording_frames(video, target_height=target_height, wanted=RECORDING in names, first_frame=0)
    gray_stream = masked_stream(video, target_height=target_height)
    gray_frames: Iterator[VideoFrame | None] = gray_stream.stream_gray_frames() if DEVIATION in names else repeat(None)
    stage = detection_stage(
        device=settings.device,
        background=settings.background,
        detection=settings.detection,
        timestamp_mask=gray_stream.mask,
    )

    def planar_frames() -> Iterator[np.ndarray]:
        for frame_number, (colour, gray) in enumerate(zip(colour_frames, gray_frames)):
            if colour is not None:
                np.copyto(views[RECORDING], colour.frame[:panel_height, :panel_width])
            if gray is not None:
                stage.detections(gray.frame)
                view = views[DEVIATION]
                cv2.cvtColor(stage.deviation_image()[:panel_height, :panel_width], cv2.COLOR_GRAY2BGR, dst=view)

            for panel in views.values():
                _draw_overlays(panel, overlays=overlays, frame_number=frame_number, size=size)
            draw_frame_number(views[names[0]], frame_number)

            cv2.cvtColor(stacked, cv2.COLOR_BGR2YUV_I420, dst=planar)
            yield planar
            on_frame(frame_number)

    return encode(
        planar_frames(),
        output=output,
        width=panel_width,
        height=panel_height * len(names),
        frames_per_second=playback_rate(frames_per_second),
        encoding=STORED,
    )


@dataclass(frozen=True, slots=True)
class _Overlay:
    track_id: int
    frames: tuple[int, ...]
    polyline: np.ndarray


def _overlays(trajectories: tuple[Trajectory, ...]) -> tuple[_Overlay, ...]:
    return tuple(
        _Overlay(
            track_id=trajectory.track_id,
            frames=tuple(point.frame_number for point in trajectory.points),
            polyline=np.array([(int(point.x), int(point.y)) for point in trajectory.points], dtype=np.int32),
        )
        for trajectory in trajectories
    )


def _draw_overlays(canvas: np.ndarray, *, overlays: tuple[_Overlay, ...], frame_number: int, size: int) -> None:
    for overlay in overlays:
        reached = bisect_right(overlay.frames, frame_number)
        if reached == 0:
            continue

        color = track_color(overlay.track_id)
        draw_trail(canvas, polyline=overlay.polyline[:reached], color=color)
        if overlay.frames[reached - 1] == frame_number:
            draw_box(canvas, point=overlay.polyline[reached - 1], color=color, size=size, label=str(overlay.track_id))


@dataclass(frozen=True, slots=True)
class _RunPaths:
    video: Path
    record: Path


def _run_paths(
    video: Path,
    *,
    settings: MovementSettings,
    target_height: int,
    panels: Panels,
    output_dir: Path,
) -> _RunPaths:
    name = digest(
        {"settings": asdict(settings), "target_height": target_height, "panels": panels},
        modules=MODULES,
    )
    stem = f"{video.stem}__{name}"
    return _RunPaths(video=output_dir / f"{stem}.mp4", record=output_dir / f"{stem}.json")


def _as_record(run: DetectionRun) -> dict[str, Any]:
    return {
        "video": str(run.video),
        "frame_count": run.frame_count,
        "detection_seconds": run.detection_seconds,
        "trajectories": [
            {
                "track_id": trajectory.track_id,
                "points": [[point.frame_number, point.x, point.y] for point in trajectory.points],
            }
            for trajectory in run.trajectories
        ],
    }


def _trajectory_from_record(entry: dict[str, Any]) -> Trajectory:
    return Trajectory(
        track_id=int(entry["track_id"]),
        points=tuple(TrackPoint(frame_number=int(frame), x=float(x), y=float(y)) for frame, x, y in entry["points"]),
    )
