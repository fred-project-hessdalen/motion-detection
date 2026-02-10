from pathlib import Path
from typing import Generator, Protocol
from itertools import count

import cv2
import numpy as np

from hessdalen.domain.models import VideoFile, VideoFrame


class FrameSource(Protocol):
    """Protocol for frame sources."""

    def open(self) -> cv2.VideoCapture:
        """Open and return a VideoCapture object."""
        ...


class FileFrameSource:
    """Frame source from video file."""

    def __init__(self, video_path: Path):
        self.video_file = VideoFile(path=video_path)

    def open(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(str(self.video_file.path))


class CameraFrameSource:
    """Frame source from camera device."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

    def open(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self.camera_index)


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
        self.mask: np.ndarray | None = None

        if self.target_height is not None and self.target_height <= 0:
            raise ValueError("target_height must be a positive integer")

        if mask_coords is not None:
            cap = source.open()
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = self._resize_frame(frame)
                height, width = frame.shape[:2]
                self.mask = np.ones((height, width), dtype=np.uint8) * 255
                x1 = int(width * mask_coords[0])
                y1 = int(height * mask_coords[1])
                x2 = int(width * mask_coords[2])
                y2 = int(height * mask_coords[3])
                self.mask[y1:y2, x1:x2] = 0

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

    def stream_frames(self) -> Generator[VideoFrame, None, None]:
        """Stream video frames."""
        cap = self.source.open()
        frame_number = count()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self._resize_frame(frame)

                if self.mask is not None:
                    frame = cv2.bitwise_and(frame, frame, mask=self.mask)

                yield VideoFrame(frame_number=next(frame_number), frame=frame)
        finally:
            cap.release()


def stream_frames_from_file(video_path: Path, *, target_height: int | None = None) -> Generator[VideoFrame, None, None]:
    return VideoStream(FileFrameSource(video_path), target_height=target_height).stream_frames()
