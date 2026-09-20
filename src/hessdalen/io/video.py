import queue
import threading
from pathlib import Path
from typing import Generator, Iterator, Protocol, TypeVar

import av
import cv2
import numpy as np

from hessdalen.domain.models import VideoFile, VideoFrame

TIMESTAMP_MASK_COORDS = (0.8, 0.8, 1.0, 1.0)
"""Corner the cameras burn their timestamp into, in relative coordinates."""

PREFETCH_DEPTH = 4
"""Frames a worker may run ahead of the caller."""

LUMA_FIRST_FORMATS = frozenset({"gray", "yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "nv12"})
"""Decoded formats whose first plane already holds the grayscale."""

Frame = TypeVar("Frame")


class FrameSource(Protocol):
    """Protocol for frame sources."""

    def colour_frames(self) -> Generator[np.ndarray, None, None]:
        """The frames in colour, at the size they were recorded."""
        ...

    def gray_frames(self) -> Generator[np.ndarray, None, None]:
        """The frames in grayscale, at the size they were recorded.

        A frame stays readable only until the next one is pulled, so a
        caller that keeps one has to copy it.
        """
        ...


class FileFrameSource:
    """Frame source from video file."""

    def __init__(self, video_path: Path):
        self.video_file = VideoFile(path=video_path)

    def colour_frames(self) -> Generator[np.ndarray, None, None]:
        capture = cv2.VideoCapture(str(self.video_file.path))
        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    return
                yield frame
        finally:
            capture.release()

    def gray_frames(self) -> Generator[np.ndarray, None, None]:
        """The luma plane of each frame, read where the decoder wrote it.

        Grayscale is what the decoder produced before it built a colour
        frame, so taking that plane costs neither the colour conversion
        nor a second pass to undo it.
        """
        container = av.open(str(self.video_file.path))
        container.streams.video[0].thread_type = "AUTO"
        try:
            for frame in container.decode(video=0):
                yield luma_plane(frame)
        finally:
            container.close()


class CameraFrameSource:
    """Frame source from camera device."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

    def colour_frames(self) -> Generator[np.ndarray, None, None]:
        capture = cv2.VideoCapture(self.camera_index)
        try:
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


class VideoStream:
    def __init__(
        self,
        source: FrameSource,
        mask_coords: tuple[float, float, float, float] | None = None,
        target_height: int | None = None,
    ):
        self.source = source
        self.mask_coords = mask_coords
        self.target_height = target_height

        if self.target_height is not None and self.target_height <= 0:
            raise ValueError("target_height must be a positive integer")

        self._frame_shape: tuple[int, int] | None = None
        self.mask: np.ndarray | None = self._build_mask(mask_coords) if mask_coords is not None else None

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Height and width of the frames this stream yields, after
        resizing."""
        if self._frame_shape is None:
            self._frame_shape = self._read_frame_shape()
        return self._frame_shape

    def stream_frames(self) -> Generator[VideoFrame, None, None]:
        """Stream video frames."""
        yield from prefetched(self._colour_frames())

    def stream_gray_frames(self) -> Generator[VideoFrame, None, None]:
        """Stream the frames as the grayscale the detector measures.

        The worker carries the resize as well as the decode, which keeps
        the frames the decoder still owns off the queue and leaves the
        caller free for the detection.
        """
        yield from prefetched(self._gray_frames())

    def _read_frame_shape(self) -> tuple[int, int]:
        frames = self.source.colour_frames()
        try:
            frame = next(frames, None)
        finally:
            frames.close()

        if frame is None:
            raise ValueError("Video source yielded no frames.")

        height, width = self._resize_frame(frame).shape[:2]
        return int(height), int(width)

    def _build_mask(self, mask_coords: tuple[float, float, float, float]) -> np.ndarray:
        height, width = self.frame_shape
        mask = np.full((height, width), 255, dtype=np.uint8)
        x1 = int(width * mask_coords[0])
        y1 = int(height * mask_coords[1])
        x2 = int(width * mask_coords[2])
        y2 = int(height * mask_coords[3])
        mask[y1:y2, x1:x2] = 0
        return mask

    def _colour_frames(self) -> Iterator[VideoFrame]:
        for frame_number, frame in enumerate(self.source.colour_frames()):
            frame = self._resize_frame(frame)
            if self.mask is not None:
                frame = cv2.bitwise_and(frame, frame, mask=self.mask)
            yield VideoFrame(frame_number=frame_number, frame=frame)

    def _gray_frames(self) -> Iterator[VideoFrame]:
        for frame_number, gray in enumerate(self.source.gray_frames()):
            gray = self._resize_frame(gray)
            if self.mask is not None:
                gray = cv2.bitwise_and(gray, self.mask)
            yield VideoFrame(frame_number=frame_number, frame=gray)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.target_height is None:
            return frame

        height, width = frame.shape[:2]
        if height == self.target_height:
            return frame

        scale = float(self.target_height) / float(height)
        new_width = max(1, int(round(width * scale)))

        interpolation = cv2.INTER_AREA if self.target_height < height else cv2.INTER_LINEAR
        return cv2.resize(
            frame,
            (new_width, int(self.target_height)),
            interpolation=interpolation,
        )


def stream_frames_from_file(video_path: Path, *, target_height: int | None = None) -> Generator[VideoFrame, None, None]:
    return VideoStream(FileFrameSource(video_path), target_height=target_height).stream_frames()


def luma_plane(frame: av.VideoFrame) -> np.ndarray:
    """The decoded frame's grayscale, as a view on the decoder's own memory.

    Planar YUV keeps the grayscale in its first plane. Anything else is
    converted, which costs a frame of its own and is why the cameras'
    format is the one worth reading directly.
    """
    if frame.format.name not in LUMA_FIRST_FORMATS:
        frame = frame.reformat(format="gray")

    plane = frame.planes[0]
    rows: np.ndarray = np.frombuffer(plane, dtype=np.uint8, count=plane.line_size * frame.height).reshape(
        frame.height, plane.line_size
    )
    return rows[:, : frame.width]


def prefetched(frames: Iterator[Frame]) -> Generator[Frame, None, None]:
    """Yield frames a worker thread has prepared ahead of the caller.

    Decoding, resizing and colour conversion all release the interpreter
    lock, so the worker keeps the queue filled while the caller works on
    the frame it already holds.
    """
    pending: queue.Queue[Frame | None] = queue.Queue(maxsize=PREFETCH_DEPTH)
    stop = threading.Event()

    def produce() -> None:
        for frame in frames:
            if stop.is_set():
                break
            pending.put(frame)
        pending.put(None)

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    try:
        while True:
            frame = pending.get()
            if frame is None:
                return
            yield frame
    finally:
        stop.set()
        while worker.is_alive():
            try:
                pending.get(timeout=0.1)
            except queue.Empty:
                continue
