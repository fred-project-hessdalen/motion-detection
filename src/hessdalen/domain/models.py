from pathlib import Path

import numpy as np
import pydantic


class VideoFrame(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    frame_number: int
    frame: np.ndarray


class MovementEvent(pydantic.BaseModel):
    frame_number: int
    track_id: int | None = None
    centroid: tuple[float, float] | None


class DetectedMovement(MovementEvent):
    track_id: int
    centroid: tuple[float, float]


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
