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

STILL = 1e-9
"""Movement along one axis put in place of none.

A step that stands still along an axis crosses neither of that axis's
edges. Dividing by a movement of almost nothing sends both of those
crossings off towards infinity, with the signs that say whether the step
runs between the edges or outside them.
"""

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


def draw_trail(canvas: np.ndarray, *, polyline: np.ndarray, color: tuple[int, int, int], clear: int) -> None:
    """The path a track has taken, drawn everywhere but inside the square of
    this half-width around its last point.

    That square is where the box goes, and a path drawn into it covers
    the thing the box is there to show.
    """
    cv2.polylines(canvas, _outside_head(polyline, half=clear), isClosed=False, color=color, thickness=2)


def _outside_head(polyline: np.ndarray, *, half: int) -> list[np.ndarray]:
    """Every part of the path that lies outside the square around its last
    point, each as the two points it runs between.

    A straight step crosses a square over one run of its length, so what
    lies outside it is the part before the step enters and the part after
    it leaves.
    """
    if len(polyline) < 2:
        return []

    starts = polyline[:-1].astype(np.float64)
    ends = polyline[1:].astype(np.float64)
    delta = ends - starts
    step = np.where(delta == 0.0, STILL, delta)

    near = (polyline[-1] - half - starts) / step
    far = (polyline[-1] + half - starts) / step
    enters = np.clip(np.minimum(near, far).max(axis=1), 0.0, 1.0)
    leaves = np.clip(np.maximum(near, far).min(axis=1), 0.0, 1.0)
    missed = enters >= leaves

    enter = np.where(missed, 1.0, enters)[:, None]
    leave = np.where(missed, 1.0, leaves)[:, None]
    before = np.stack([starts, starts + enter * delta], axis=1)[enter[:, 0] > 0.0]
    after = np.stack([starts + leave * delta, ends], axis=1)[leave[:, 0] < 1.0]
    return list(np.concatenate([before, after]).round().astype(np.int32))


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
