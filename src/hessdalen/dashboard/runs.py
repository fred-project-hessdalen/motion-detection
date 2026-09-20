"""Detection runs kept on disk as an annotated video and a record of the
tracks.

A run is addressed by the recording it came from and a digest of the
settings and the detector source it ran with, so the dashboard finds an
earlier result again.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hessdalen.domain.models import MovementEvent
from hessdalen.io.video import TIMESTAMP_MASK_COORDS, FileFrameSource, VideoStream
from hessdalen.processing.debug import track_color
from hessdalen.processing.movement import MovementDetector, MovementSettings

BOX_RATIO = 0.025
MIN_BOX_SIZE = 8
FALLBACK_FPS = 25.0
DETECTION_SHARE = 0.5
DETECTOR_PACKAGES = ("domain", "io", "processing")
ENCODER_PIXEL_FORMAT = "yuv420p"


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
    output_dir: Path,
    on_progress: Callable[[float], None],
) -> DetectionRun:
    """Detect movement in a recording and write the annotated video and the
    track record under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _run_paths(video, settings=settings, target_height=target_height, output_dir=output_dir)
    details = probe(video)
    expected_frames = max(1, details.frame_count)

    started = time.perf_counter()
    trajectories = _detect(
        video,
        settings=settings,
        target_height=target_height,
        on_frame=lambda index: on_progress(DETECTION_SHARE * min(1.0, (index + 1) / expected_frames)),
    )
    detection_seconds = time.perf_counter() - started

    frame_count = _render(
        video,
        target_height=target_height,
        frames_per_second=details.frames_per_second,
        trajectories=trajectories,
        output=paths.video,
        on_frame=lambda index: on_progress(
            DETECTION_SHARE + (1.0 - DETECTION_SHARE) * min(1.0, (index + 1) / expected_frames)
        ),
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


def load_run(
    video: Path,
    *,
    settings: MovementSettings,
    target_height: int,
    output_dir: Path,
) -> DetectionRun | None:
    """The stored run for these settings, or None when the recording has not
    been run with them."""
    paths = _run_paths(video, settings=settings, target_height=target_height, output_dir=output_dir)
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
    stream = VideoStream(
        FileFrameSource(video),
        mask_coords=TIMESTAMP_MASK_COORDS,
        target_height=target_height,
    )
    detector = MovementDetector(stream=stream, settings=settings)
    return trajectories_from_events(_reporting_progress(detector.detect(), on_frame))


def _reporting_progress(events: Iterable[MovementEvent], on_frame: Callable[[int], None]) -> Iterator[MovementEvent]:
    for event in events:
        on_frame(event.frame_number)
        yield event


def _render(
    video: Path,
    *,
    target_height: int,
    frames_per_second: float,
    trajectories: tuple[Trajectory, ...],
    output: Path,
    on_frame: Callable[[int], None],
) -> int:
    """Draw every track onto the frames the detector saw and encode them.

    Returns the number of frames written. Some containers declare a
    frame count their stream does not hold, so this is the count that
    was decoded.
    """
    stream = VideoStream(
        FileFrameSource(video),
        mask_coords=TIMESTAMP_MASK_COORDS,
        target_height=target_height,
    )
    height, width = stream.frame_shape
    overlays = _overlays(trajectories)
    box_size = max(MIN_BOX_SIZE, int(BOX_RATIO * max(height, width)))

    encoder = _open_encoder(
        output,
        width=_even(width),
        height=_even(height),
        frames_per_second=frames_per_second if frames_per_second > 0 else FALLBACK_FPS,
    )
    stdin = encoder.stdin
    if stdin is None:
        raise RuntimeError("ffmpeg was started without an input pipe.")

    frames_written = 0
    try:
        for frame in stream.stream_frames():
            canvas = frame.frame
            _draw_overlays(canvas, overlays=overlays, frame_number=frame.frame_number, box_size=box_size)
            _draw_frame_number(canvas, frame.frame_number)
            stdin.write(_planar_bytes(canvas))
            frames_written += 1
            on_frame(frame.frame_number)
    finally:
        stdin.close()
        encoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with status {encoder.returncode}.")
    return frames_written


def _planar_bytes(canvas: np.ndarray) -> bytes:
    """The frame in the planar format the encoder reads.

    Handing the encoder colour frames makes it convert them itself,
    which costs ten times what OpenCV charges and sends twice the bytes
    down the pipe.
    """
    height, width = canvas.shape[:2]
    even = canvas[: _even(height), : _even(width)]
    planar: np.ndarray = cv2.cvtColor(even, cv2.COLOR_BGR2YUV_I420)
    return planar.tobytes()


def _even(size: int) -> int:
    """The largest even size at or below this one.

    Colour is stored for every second row and column, so a frame with an
    odd side loses that side's last line.
    """
    return size - size % 2


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


def _draw_overlays(canvas: np.ndarray, *, overlays: tuple[_Overlay, ...], frame_number: int, box_size: int) -> None:
    for overlay in overlays:
        reached = bisect_right(overlay.frames, frame_number)
        if reached == 0:
            continue

        color = track_color(overlay.track_id)
        cv2.polylines(canvas, [overlay.polyline[:reached]], isClosed=False, color=color, thickness=2)
        if overlay.frames[reached - 1] == frame_number:
            _draw_box(
                canvas, point=overlay.polyline[reached - 1], color=color, box_size=box_size, label=str(overlay.track_id)
            )


def _draw_box(
    canvas: np.ndarray,
    *,
    point: np.ndarray,
    color: tuple[int, int, int],
    box_size: int,
    label: str,
) -> None:
    x, y = int(point[0]), int(point[1])
    cv2.rectangle(canvas, (x - box_size, y - box_size), (x + box_size, y + box_size), color, 2)
    cv2.putText(canvas, label, (x - box_size, y - box_size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _draw_frame_number(canvas: np.ndarray, frame_number: int) -> None:
    position = (12, 36)
    cv2.putText(canvas, f"frame {frame_number}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(canvas, f"frame {frame_number}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def _open_encoder(
    output: Path,
    *,
    width: int,
    height: int,
    frames_per_second: float,
) -> subprocess.Popen[bytes]:
    """Start an ffmpeg process that turns raw planar frames into a video a
    browser can play."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        ENCODER_PIXEL_FORMAT,
        "-s",
        f"{width}x{height}",
        "-r",
        f"{frames_per_second:.6f}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


@dataclass(frozen=True, slots=True)
class _RunPaths:
    video: Path
    record: Path


def _run_paths(video: Path, *, settings: MovementSettings, target_height: int, output_dir: Path) -> _RunPaths:
    digest = _digest(settings=settings, target_height=target_height)
    stem = f"{video.stem}__{digest}"
    return _RunPaths(video=output_dir / f"{stem}.mp4", record=output_dir / f"{stem}.json")


def _digest(*, settings: MovementSettings, target_height: int) -> str:
    payload = json.dumps(
        {"settings": asdict(settings), "target_height": target_height, "code": _detector_digest()},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


@cache
def _detector_digest() -> str:
    """Digest of the modules the detector is built from.

    The dashboard is used while the detector is being changed, so a
    stored run has to stop matching once its code has been edited.
    """
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha1()
    for name in DETECTOR_PACKAGES:
        for source in sorted((package / name).rglob("*.py")):
            digest.update(source.read_bytes())
    return digest.hexdigest()[:10]


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
