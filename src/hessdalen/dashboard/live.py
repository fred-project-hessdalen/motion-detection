"""A detection that hands every frame back as it is drawn.

The dashboard plays a segment of a recording while the detector runs on
it, so a setting can be moved and its effect watched without waiting for
a run to finish. The frames ahead of the segment are measured as well
and not drawn, which leaves the background model where a run over the
whole recording would leave it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

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
from hessdalen.domain.models import MovementEvent
from hessdalen.io.video import masked_stream
from hessdalen.processing.debug import track_color
from hessdalen.processing.movement import MovementDetector, MovementSettings

DISPLAY_WIDTH = 1460
"""Widest a frame is sent at.

Streamlit reopens and re-encodes any image wider than this before it
sends it, which costs more than the detection of the frame does.
"""

JPEG_QUALITY = 80


@dataclass(frozen=True, slots=True)
class Segment:
    """The stretch of a recording the live view draws, in frames."""

    begin_frame: int
    end_frame: int


@dataclass(frozen=True, slots=True)
class LiveFrame:
    frame_number: int
    image: bytes
    track_count: int


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


def live_frames(
    video: Path,
    *,
    settings: MovementSettings,
    target_height: int,
    panels: Panels,
    segment: Segment,
    frames_per_second: float,
    on_lead_in: Callable[[int], None],
) -> Iterator[LiveFrame]:
    """Detect movement in a recording and yield the drawn frames of the
    segment.

    Every frame is yielded as a JPEG, so the caller only has to put it
    on the page. The pace is held to the rate the recording was taken
    at, measured after the caller has drawn the frame, and a caller that
    cannot keep up sets the pace itself.
    """
    names = layout(panels)
    gray_stream = masked_stream(video, target_height=target_height)
    height, width = gray_stream.frame_shape
    size = box_size(height, width)

    stacked = np.empty((height * len(names), width, 3), dtype=np.uint8)
    views = {name: stacked[index * height : (index + 1) * height] for index, name in enumerate(names)}

    colour_frames = recording_frames(video, target_height=target_height, wanted=RECORDING in names)
    detector = MovementDetector(stream=gray_stream, settings=settings)
    gray_frames = gray_stream.stream_gray_frames()

    trails: dict[int, Trail] = {}
    interval = 1.0 / frames_per_second if frames_per_second > 0 else 0.0
    deadline = time.perf_counter()

    for frame_number, (colour, gray) in enumerate(zip(colour_frames, gray_frames)):
        if frame_number > segment.end_frame:
            return

        extend_trails(trails, detector.process_frame(gray))
        if frame_number < segment.begin_frame:
            on_lead_in(frame_number)
            continue

        if colour is not None:
            np.copyto(views[RECORDING], colour.frame)
        if DEVIATION in views:
            cv2.cvtColor(detector.stage.deviation_image(), cv2.COLOR_GRAY2BGR, dst=views[DEVIATION])

        for panel in views.values():
            _draw_trails(panel, trails=trails, frame_number=frame_number, size=size)
        draw_frame_number(views[names[0]], frame_number)

        yield LiveFrame(frame_number=frame_number, image=_encode(stacked), track_count=len(trails))

        deadline = max(deadline + interval, time.perf_counter())
        time.sleep(max(0.0, deadline - time.perf_counter()))


def _draw_trails(canvas: np.ndarray, *, trails: dict[int, Trail], frame_number: int, size: int) -> None:
    for track_id, trail in trails.items():
        color = track_color(track_id)
        draw_trail(canvas, polyline=np.array(trail.points, dtype=np.int32), color=color)
        if trail.last_frame == frame_number:
            draw_box(canvas, point=np.array(trail.points[-1]), color=color, size=size, label=str(track_id))


def _encode(canvas: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", _fitted(canvas), (cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY))
    if not ok:
        raise RuntimeError("OpenCV could not encode a live frame as JPEG.")
    return buffer.tobytes()


def _fitted(canvas: np.ndarray) -> np.ndarray:
    height, width = canvas.shape[:2]
    if width <= DISPLAY_WIDTH:
        return canvas

    scale = DISPLAY_WIDTH / width
    return cv2.resize(canvas, (DISPLAY_WIDTH, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
