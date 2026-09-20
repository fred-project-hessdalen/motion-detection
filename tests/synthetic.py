"""In-memory video sources and generated scenes for the detector tests."""

from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np


class MockFrameSource:
    """Mock frame source for testing."""

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.first_frame = 0

    def colour_frames(self) -> Generator[np.ndarray, None, None]:
        yield from self.frames

    def gray_frames(self) -> Generator[np.ndarray, None, None]:
        for frame in self.frames:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def sample_frame(self) -> np.ndarray | None:
        return self.frames[0] if self.frames else None


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
