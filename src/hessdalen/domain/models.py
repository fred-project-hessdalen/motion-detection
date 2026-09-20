from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydantic


class VideoFrame(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    frame_number: int
    frame: np.ndarray


@dataclass(frozen=True, slots=True)
class BlobMeasurement:
    """What a frame measured about a blob, beyond where it sat.

    A track carries these as a series over its frames, and that series
    is what separates a wingbeat from a meteor's decay.
    """

    pixel_count: int
    peak_deviation: float
    brightness: float
    major_axis: float
    minor_axis: float


class MovementEvent(pydantic.BaseModel):
    frame_number: int
    track_id: int | None = None
    centroid: tuple[float, float] | None
    blob: BlobMeasurement | None = None


class DetectedMovement(MovementEvent):
    track_id: int
    centroid: tuple[float, float]
    blob: BlobMeasurement


class VideoFile(pydantic.BaseModel):
    path: Path

    @pydantic.field_validator("path")
    @classmethod
    def _validate_path(cls, video_path: Path) -> Path:
        if not video_path.is_file():
            raise ValueError(f"File {video_path} does not exist.")
        if video_path.suffix not in {".mkv", ".mp4"}:
            raise ValueError(f"File {video_path} is not a video file.")
        return video_path
