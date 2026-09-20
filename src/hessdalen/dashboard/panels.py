"""The panels a detection is drawn onto, and the marks drawn on them.

A stored run and the live view draw the same picture, one into a video
file and the other into the page, so the choice of panels and the marks
themselves live here.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import repeat
from pathlib import Path
from typing import Literal, get_args

import cv2
import numpy as np

from hessdalen.domain.models import VideoFrame
from hessdalen.io.video import masked_stream

BOX_RATIO = 0.025
MIN_BOX_SIZE = 8

RECORDING = "recording"
DEVIATION = "deviation"
Panels = Literal["recording", "deviation", "both"]
PANEL_CHOICES: tuple[Panels, ...] = get_args(Panels)


def layout(panels: Panels) -> tuple[str, ...]:
    """The panels to stack, from top to bottom."""
    return (RECORDING, DEVIATION) if panels == "both" else (panels,)


def box_size(height: int, width: int) -> int:
    """Half-width of the box a detection is marked with, in pixels."""
    return max(MIN_BOX_SIZE, int(BOX_RATIO * max(height, width)))


def recording_frames(
    video: Path,
    *,
    target_height: int,
    wanted: bool,
    first_frame: int,
) -> Iterator[VideoFrame | None]:
    """The frames the recording panel draws from first_frame on, or nothing in
    place of each of them when that panel is not drawn.

    A panel left out opens no decoder of its own, and the caller can
    still step through the recording one frame at a time.
    """
    if not wanted:
        return repeat(None)
    return masked_stream(video, target_height=target_height, first_frame=first_frame).stream_frames()


def draw_trail(canvas: np.ndarray, *, polyline: np.ndarray, color: tuple[int, int, int]) -> None:
    cv2.polylines(canvas, [polyline], isClosed=False, color=color, thickness=2)


def draw_box(
    canvas: np.ndarray,
    *,
    point: np.ndarray,
    color: tuple[int, int, int],
    size: int,
    label: str,
) -> None:
    x, y = int(point[0]), int(point[1])
    cv2.rectangle(canvas, (x - size, y - size), (x + size, y + size), color, 2)
    cv2.putText(canvas, label, (x - size, y - size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_frame_number(canvas: np.ndarray, frame_number: int) -> None:
    position = (12, 36)
    cv2.putText(canvas, f"frame {frame_number}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(canvas, f"frame {frame_number}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
