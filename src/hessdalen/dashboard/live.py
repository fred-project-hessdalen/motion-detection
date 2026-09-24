"""A segment of a recording, built into a clip with its detections drawn on it.

The dashboard plays one segment over and over while a setting is being
found, and builds it again under each new value. A built clip loops in
the page at the rate it was written at. A frame pushed at a time cannot,
because nothing on the far side holds it to a rate, so each frame
arrives when its fetch and its decode happen to finish.

Only the segment is detected. The background model therefore opens on
the segment's first frame and takes its scene from the frames after it,
so a segment beginning in the middle of a recording reads high for its
first frames where a run over the whole recording would not.

Reaching the segment still costs a decode of everything before it,
because neither decoder lands on the frame a linear decode calls by that
number on the cameras' files. Those frames are passed by as cheaply as
each decoder allows.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hessdalen.dashboard.encoder import LIVE, encode, even, playback_rate
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
from hessdalen.domain.models import MovementEvent
from hessdalen.io.video import masked_stream
from hessdalen.processing.debug import track_color
from hessdalen.processing.movement import MovementDetector, MovementSettings

MODULES = ("live.py", "panels.py", "encoder.py")
"""The dashboard modules that decide what a built segment holds."""


@dataclass(frozen=True, slots=True)
class Segment:
    """The stretch of a recording the live view draws, in frames."""

    begin_frame: int
    end_frame: int

    @property
    def drawn_frames(self) -> int:
        return self.end_frame - self.begin_frame + 1


@dataclass(frozen=True, slots=True)
class Progress:
    """How many of the segment's frames a build has drawn."""

    frame_number: int
    frame_count: int

    @property
    def fraction(self) -> float:
        return min(1.0, (self.frame_number + 1) / max(1, self.frame_count))


@dataclass(frozen=True, slots=True)
class BuiltSegment:
    """A clip a build produced.

    A frame count of zero means the recording ended before the segment
    started, and no clip was written.
    """

    video: Path
    frame_count: int
    track_count: int
    build_seconds: float


def build_segment(
    video: Path,
    *,
    settings: MovementSettings,
    target_height: int,
    panels: Panels,
    segment: Segment,
    frames_per_second: float,
    output_dir: Path,
    on_progress: Callable[[Progress], None],
) -> BuiltSegment:
    """Draw the segment of a recording into a clip under output_dir.

    Reporting progress puts the build at the mercy of the caller, which
    is what lets a setting moved while it runs stop it at the frame it
    has reached. A build stopped that way leaves a part of a clip behind
    and no record of it, and the record is what a later read looks for,
    so the part is never taken for a clip and the next build of the same
    segment writes over it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _segment_paths(
        video, settings=settings, target_height=target_height, panels=panels, segment=segment, output_dir=output_dir
    )

    names = layout(panels)
    gray_stream = masked_stream(video, target_height=target_height, first_frame=segment.begin_frame)
    height, width = gray_stream.frame_shape
    panel_height, panel_width = even(height), even(width)
    size = box_size(panel_height, panel_width)

    stacked = np.empty((panel_height * len(names), panel_width, 3), dtype=np.uint8)
    views = {name: stacked[index * panel_height : (index + 1) * panel_height] for index, name in enumerate(names)}
    planar = np.empty((panel_height * len(names) * 3 // 2, panel_width), dtype=np.uint8)

    colour_frames = recording_frames(
        video, target_height=target_height, wanted=RECORDING in names, first_frame=segment.begin_frame
    )
    detector = MovementDetector(stream=gray_stream, settings=settings)
    gray_frames = gray_stream.stream_gray_frames()
    trails: dict[int, Trail] = {}

    def planar_frames() -> Iterator[np.ndarray]:
        for offset, (colour, gray) in enumerate(zip(colour_frames, gray_frames)):
            frame_number = segment.begin_frame + offset
            if frame_number > segment.end_frame:
                return

            extend_trails(trails, detector.process_frame(gray))
            if colour is not None:
                np.copyto(views[RECORDING], colour.frame[:panel_height, :panel_width])
            if DEVIATION in views:
                deviation = detector.stage.deviation_image()[:panel_height, :panel_width]
                cv2.cvtColor(deviation, cv2.COLOR_GRAY2BGR, dst=views[DEVIATION])

            for panel in views.values():
                _draw_trails(panel, trails=trails, frame_number=frame_number, size=size)
            draw_frame_number(views[names[0]], frame_number)

            cv2.cvtColor(stacked, cv2.COLOR_BGR2YUV_I420, dst=planar)
            yield planar
            on_progress(Progress(frame_number=offset, frame_count=segment.drawn_frames))

    started = time.perf_counter()
    frame_count = encode(
        planar_frames(),
        output=paths.video,
        width=panel_width,
        height=panel_height * len(names),
        frames_per_second=playback_rate(frames_per_second),
        encoding=LIVE,
    )
    built = BuiltSegment(
        video=paths.video,
        frame_count=frame_count,
        track_count=len(trails),
        build_seconds=time.perf_counter() - started,
    )
    if frame_count == 0:
        return built

    paths.record.write_text(json.dumps(_as_record(built), indent=2))
    return built


def load_segment(
    video: Path,
    *,
    settings: MovementSettings,
    target_height: int,
    panels: Panels,
    segment: Segment,
    output_dir: Path,
) -> BuiltSegment | None:
    """The clip built for these settings, or None when there is none."""
    paths = _segment_paths(
        video, settings=settings, target_height=target_height, panels=panels, segment=segment, output_dir=output_dir
    )
    if not (paths.record.is_file() and paths.video.is_file()):
        return None

    record = json.loads(paths.record.read_text())
    return BuiltSegment(
        video=paths.video,
        frame_count=int(record["frame_count"]),
        track_count=int(record["track_count"]),
        build_seconds=float(record["build_seconds"]),
    )


@dataclass(slots=True)
class Trail:
    """Where one track has been, in the order the detector reported it."""

    points: list[tuple[int, int]] = field(default_factory=list)
    last_frame: int = -1


def extend_trails(trails: dict[int, Trail], events: Iterable[MovementEvent]) -> None:
    """Add the centroid each event carries to the trail of its track.

    A track is confirmed some frames after it starts, and the detector
    reports the centroids it held back at that moment, so a trail
    reaches its full length in the frame that confirms it.
    """
    for event in events:
        if event.track_id is None or event.centroid is None:
            continue
        trail = trails.setdefault(event.track_id, Trail())
        trail.points.append((int(event.centroid[0]), int(event.centroid[1])))
        trail.last_frame = max(trail.last_frame, event.frame_number)


def _draw_trails(canvas: np.ndarray, *, trails: dict[int, Trail], frame_number: int, size: int) -> None:
    for track_id, trail in trails.items():
        color = track_color(track_id)
        draw_trail(canvas, polyline=np.array(trail.points, dtype=np.int32), color=color)
        if trail.last_frame == frame_number:
            draw_box(canvas, point=np.array(trail.points[-1]), color=color, size=size, label=str(track_id))


@dataclass(frozen=True, slots=True)
class _SegmentPaths:
    video: Path
    record: Path


def _segment_paths(
    video: Path,
    *,
    settings: MovementSettings,
    target_height: int,
    panels: Panels,
    segment: Segment,
    output_dir: Path,
) -> _SegmentPaths:
    name = digest(
        {
            "settings": asdict(settings),
            "target_height": target_height,
            "panels": panels,
            "begin_frame": segment.begin_frame,
            "end_frame": segment.end_frame,
        },
        modules=MODULES,
    )
    stem = f"{video.stem}__{name}"
    return _SegmentPaths(video=output_dir / f"{stem}.mp4", record=output_dir / f"{stem}.json")


def _as_record(built: BuiltSegment) -> dict[str, Any]:
    return {
        "frame_count": built.frame_count,
        "track_count": built.track_count,
        "build_seconds": built.build_seconds,
    }
