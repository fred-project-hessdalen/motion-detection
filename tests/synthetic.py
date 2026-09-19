"""In-memory video sources and generated scenes for the detector tests."""

from dataclasses import dataclass

import numpy as np


class MockVideoCapture:
    """Mock cv2.VideoCapture for testing."""

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.index = 0

    def read(self) -> tuple[bool, np.ndarray]:
        if self.index >= len(self.frames):
            return False, np.zeros((100, 100, 3), dtype=np.uint8)
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        pass


class MockFrameSource:
    """Mock frame source for testing."""

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames

    def open(self):
        return MockVideoCapture(self.frames)


@dataclass(frozen=True, slots=True)
class Scene:
    """A background to generate frames of, named for the sky it stands in
    for."""

    name: str
    level: int
    noise: float


def scene_frames(
    scene: Scene,
    *,
    count: int,
    width: int,
    height: int,
    seed: int,
) -> list[np.ndarray]:
    """Frames of nothing but this scene's background and its noise."""
    rng = np.random.default_rng(seed)
    return [
        np.clip(rng.normal(scene.level, scene.noise, (height, width, 1)), 0, 255).astype(np.uint8).repeat(3, axis=2)
        for _ in range(count)
    ]


def draw_object(frame: np.ndarray, x: int, y: int, contrast: float, radius: int = 2) -> None:
    """Put a square object of the given contrast into the frame, in place.

    The contrast is signed, so a scene can hold something brighter than
    its background or darker than it, the way a meteor and a bird do.
    """
    patch = frame[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float32)
    frame[y - radius : y + radius + 1, x - radius : x + radius + 1] = np.clip(patch + contrast, 0, 255).astype(np.uint8)
