"""One stored track drawn on the stretch of its recording it was found in.

Nothing is detected. The track file already holds where the track was on
every frame it was matched, and the frames are reached by grabbing,
which lands on exactly the frame a linear decode numbers the same way.
The drawn path therefore sits where the detector saw the object, at the
cost of passing every frame ahead of the stretch and none of the
detection.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hessdalen.dashboard.encoder import LIVE, encode, even, playback_rate
from hessdalen.dashboard.panels import box_size, draw_box, draw_frame_number, draw_trail
from hessdalen.dashboard.store import digest
from hessdalen.io.video import masked_stream
from hessdalen.processing.debug import track_color

MODULES = ("track_clip.py", "panels.py", "encoder.py")
"""The dashboard modules that decide what a built track clip holds."""

LEAD_SECONDS = 1.0
"""How much of the recording is shown before the track starts and after it
ends."""


@dataclass(frozen=True, slots=True)
class StoredTrack:
    """A track as its track file holds it, frame by frame in frame order."""

    track_id: int
    frame_numbers: np.ndarray
    x: np.ndarray
    y: np.ndarray
    frame_height: int

    @property
    def first_frame(self) -> int:
        return int(self.frame_numbers[0])

    @property
    def last_frame(self) -> int:
        return int(self.frame_numbers[-1])


@dataclass(frozen=True, slots=True)
class Stretch:
    """The frames of a recording a track clip shows."""

    begin_frame: int
    end_frame: int


def stretch_around(track: StoredTrack, *, frames_per_second: float, frame_count: int) -> Stretch:
    """The track's own frames, and a second of the recording on either side."""
    lead = int(round(LEAD_SECONDS * frames_per_second))
    last = frame_count - 1 if frame_count > 0 else track.last_frame + lead
    return Stretch(begin_frame=max(0, track.first_frame - lead), end_frame=min(last, track.last_frame + lead))


def track_clip_path(video: Path, *, track: StoredTrack, stretch: Stretch, output_dir: Path) -> Path:
    """Where the clip of this track is kept, named for everything that decides
    it."""
    name = digest(
        {
            "video": video.name,
            "track_id": track.track_id,
            "begin_frame": stretch.begin_frame,
            "end_frame": stretch.end_frame,
            "frame_height": track.frame_height,
        },
        modules=MODULES,
    )
    return output_dir / f"{video.stem}__track{track.track_id}__{name}.mp4"


def build_track_clip(
    video: Path,
    *,
    track: StoredTrack,
    stretch: Stretch,
    frames_per_second: float,
    output: Path,
) -> int:
    """Draw the track on its stretch of the recording, and return the frames
    written.

    The frames are resized to the height the track was detected at,
    because that is the frame its positions are pixels of.
    """
    stream = masked_stream(video, target_height=track.frame_height, first_frame=stretch.begin_frame)
    height, width = stream.frame_shape
    frame_height, frame_width = even(height), even(width)
    size = box_size(frame_height, frame_width)
    color = track_color(track.track_id)
    planar = np.empty((frame_height * 3 // 2, frame_width), dtype=np.uint8)

    def planar_frames() -> Iterator[np.ndarray]:
        for offset, colour in enumerate(stream.stream_frames()):
            frame_number = stretch.begin_frame + offset
            if frame_number > stretch.end_frame:
                return

            canvas = np.ascontiguousarray(colour.frame[:frame_height, :frame_width])
            draw_stored_track(canvas, track=track, frame_number=frame_number, color=color, size=size)
            draw_frame_number(canvas, frame_number)
            cv2.cvtColor(canvas, cv2.COLOR_BGR2YUV_I420, dst=planar)
            yield planar

    output.parent.mkdir(parents=True, exist_ok=True)
    return encode(
        planar_frames(),
        output=output,
        width=frame_width,
        height=frame_height,
        frames_per_second=playback_rate(frames_per_second),
        encoding=LIVE,
    )


def draw_stored_track(
    canvas: np.ndarray, *, track: StoredTrack, frame_number: int, color: tuple[int, int, int], size: int
) -> None:
    """The path the track has taken up to this frame, and a box where it is on
    a frame it was matched on."""
    reached = track.frame_numbers <= frame_number
    if not reached.any():
        return

    points = np.column_stack([track.x[reached], track.y[reached]]).astype(np.int32)
    draw_trail(canvas, polyline=points, color=color)
    if int(track.frame_numbers[reached][-1]) == frame_number:
        draw_box(canvas, point=points[-1], color=color, size=size, label=str(track.track_id))
