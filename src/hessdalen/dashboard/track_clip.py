"""One stored track drawn on the stretch of its recording it was found in.

Nothing is detected. The track file already holds where the track was on
every frame it was matched, so the drawn path sits where the detector
saw the object as long as the frames carry the numbers the detector
counted. A recording whose stamps rise with its frames is reached by
seeking to the keyframe before the stretch and reading the numbers off
those stamps. Any other recording is reached by passing every frame
ahead of the stretch.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from hessdalen.dashboard.encoder import LIVE, encode, even, playback_rate
from hessdalen.dashboard.panels import box_size, draw_box, draw_frame_number, draw_trail
from hessdalen.dashboard.store import digest
from hessdalen.io.video import TIMESTAMP_MASK_COORDS, FrameSource, VideoStream
from hessdalen.processing.debug import track_color

MODULES = ("track_clip.py", "panels.py", "encoder.py")
"""The dashboard modules that decide what a built track clip holds."""

LEAD_SECONDS = 1.0
"""How much of the recording is shown before the track starts and after it
ends."""

REPORT_EVERY = 25
"""Frames passed between two reports of how far the build has got."""

INDEXED_FROM = 500
"""Frames a stretch has to start past before the recording is indexed.

Reading the index costs about 40 microseconds a frame over the whole
recording, and passing a frame costs about 2.5 milliseconds at the size
the cameras record at, so a stretch nearer the start than this is
reached faster by passing the frames ahead of it.
"""


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

    @property
    def drawn_frames(self) -> int:
        return self.end_frame - self.begin_frame + 1


@dataclass(frozen=True, slots=True)
class ClipProgress:
    """How far a build has got through the frames it has to handle.

    Every frame of the recording up to the end of the stretch is
    handled, the ones ahead of it by being passed and the stretch by
    being drawn.
    """

    frames_done: int
    stretch: Stretch

    @property
    def frames_total(self) -> int:
        return self.stretch.end_frame + 1

    @property
    def drawing(self) -> bool:
        return self.frames_done > self.stretch.begin_frame

    @property
    def fraction(self) -> float:
        return min(1.0, self.frames_done / max(1, self.frames_total))


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
    on_progress: Callable[[ClipProgress], None],
) -> int:
    """Draw the track on its stretch of the recording, and return the frames
    written.

    The frames are resized to the height the track was detected at,
    because that is the frame its positions are pixels of.
    """
    stream = VideoStream(
        stretch_source(video, stretch=stretch, on_progress=on_progress),
        mask_coords=TIMESTAMP_MASK_COORDS,
        target_height=track.frame_height,
    )
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
            if (offset + 1) % REPORT_EVERY == 0 or frame_number == stretch.end_frame:
                on_progress(ClipProgress(frames_done=frame_number + 1, stretch=stretch))

    output.parent.mkdir(parents=True, exist_ok=True)
    return encode(
        planar_frames(),
        output=output,
        width=frame_width,
        height=frame_height,
        frames_per_second=playback_rate(frames_per_second),
        encoding=LIVE,
    )


def stretch_source(video: Path, *, stretch: Stretch, on_progress: Callable[[ClipProgress], None]) -> FrameSource:
    """The source a build reads its stretch of the recording from.

    A recording whose stamps say which frame is which is seeked into.
    Any other is read from its first frame, which is the only way left
    to give a frame the number the detector counted it under.
    """
    index = frame_index(video) if stretch.begin_frame >= INDEXED_FROM else None
    if index is None:
        return PassingFrameSource(video, stretch=stretch, on_progress=on_progress)
    return SeekingFrameSource(video, stretch=stretch, index=index, on_progress=on_progress)


@dataclass(frozen=True, slots=True)
class FrameIndex:
    """Every frame of a recording by its stamp, and which of them are
    keyframes, both in the order a decode from the first frame reaches
    them."""

    stamps: tuple[int, ...]
    keyframes: tuple[int, ...]

    def keyframe_before(self, frame_number: int) -> int:
        """The last keyframe at or before the frame, which is the nearest place
        a decode can start from."""
        return self.keyframes[bisect.bisect_right(self.keyframes, frame_number) - 1]


def frame_index(video: Path) -> FrameIndex | None:
    """Every frame's stamp, or None when the stamps do not say which frame is
    which.

    Read by demuxing the packets and decoding none of them, which costs
    about a second on a recording of 30000 frames. One packet holds one
    frame, and a decode hands the frames over in the order of their
    stamps, so a frame's number is the place of its stamp among them
    all. A packet without a stamp, or two packets sharing one, leave a
    frame that cannot be told from another.
    """
    packets: list[tuple[int, bool]] = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.size == 0:
                continue
            if packet.pts is None:
                return None
            packets.append((int(packet.pts), bool(packet.is_keyframe)))

    ordered = sorted(packets)
    stamps = tuple(stamp for stamp, _key in ordered)
    keyframes = tuple(number for number, (_stamp, key) in enumerate(ordered) if key)
    if len(set(stamps)) != len(stamps) or not keyframes or keyframes[0] != 0:
        return None
    return FrameIndex(stamps=stamps, keyframes=keyframes)


class SeekingFrameSource:
    """A recording read from the start of a stretch, reached by seeking to the
    last keyframe before it.

    Every frame handed on carries the stamp the index holds for its
    number, so the path is drawn on the picture the detector saw under
    that number. A decode that hands back anything else ends the
    stretch there, and one that hands back nothing leaves the recording
    to be read from its first frame.
    """

    def __init__(
        self,
        video: Path,
        *,
        stretch: Stretch,
        index: FrameIndex,
        on_progress: Callable[[ClipProgress], None],
    ) -> None:
        self.video = video
        self.stretch = stretch
        self.index = index
        self.on_progress = on_progress
        self.first_frame = stretch.begin_frame

    def colour_frames(self) -> Generator[np.ndarray, None, None]:
        stamps = self.index.stamps
        wanted = self.first_frame
        with av.open(str(self.video)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(stamps[self.index.keyframe_before(wanted)], stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None or wanted >= len(stamps) or frame.pts > stamps[wanted]:
                    break
                if frame.pts < stamps[wanted]:
                    continue
                yield frame.to_ndarray(format="bgr24")
                wanted += 1

        if wanted == self.first_frame:
            yield from self._passing().colour_frames()

    def gray_frames(self) -> Generator[np.ndarray, None, None]:
        for frame in self.colour_frames():
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def sample_frame(self) -> np.ndarray | None:
        return self._passing().sample_frame()

    def _passing(self) -> PassingFrameSource:
        return PassingFrameSource(self.video, stretch=self.stretch, on_progress=self.on_progress)


class PassingFrameSource:
    """A recording read from the start of a stretch, saying how many of the
    frames ahead of it it has passed.

    Passing them is most of a build when the track sits deep in a long
    recording, and the frames are passed as the detector's own source
    passes them, by grabbing.
    """

    def __init__(self, video: Path, *, stretch: Stretch, on_progress: Callable[[ClipProgress], None]) -> None:
        self.video = video
        self.stretch = stretch
        self.on_progress = on_progress
        self.first_frame = stretch.begin_frame

    def colour_frames(self) -> Generator[np.ndarray, None, None]:
        capture = cv2.VideoCapture(str(self.video))
        try:
            for passed in range(self.first_frame):
                if not capture.grab():
                    return
                if (passed + 1) % REPORT_EVERY == 0:
                    self.on_progress(ClipProgress(frames_done=passed + 1, stretch=self.stretch))
            while True:
                ret, frame = capture.read()
                if not ret:
                    return
                yield frame
        finally:
            capture.release()

    def gray_frames(self) -> Generator[np.ndarray, None, None]:
        for frame in self.colour_frames():
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def sample_frame(self) -> np.ndarray | None:
        """The first frame of the recording, read without passing anything."""
        capture = cv2.VideoCapture(str(self.video))
        try:
            ret, frame = capture.read()
            return frame if ret else None
        finally:
            capture.release()


def draw_stored_track(
    canvas: np.ndarray, *, track: StoredTrack, frame_number: int, color: tuple[int, int, int], size: int
) -> None:
    """The path the track has taken up to this frame, and a box where it is on
    a frame it was matched on."""
    reached = track.frame_numbers <= frame_number
    if not reached.any():
        return

    points = np.column_stack([track.x[reached], track.y[reached]]).astype(np.int32)
    draw_trail(canvas, polyline=points, color=color, clear=size)
    if int(track.frame_numbers[reached][-1]) == frame_number:
        draw_box(canvas, point=points[-1], color=color, size=size, label=str(track.track_id))
