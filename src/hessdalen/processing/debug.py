from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from hessdalen.domain.models import MovementEvent


@dataclass(frozen=True, slots=True)
class MovementDebugFrame:
    frame_number: int
    filtered: np.ndarray
    diff: np.ndarray
    centroids: dict[int, tuple[float, float]]


class MovementDebugSink(Protocol):
    def emit(self, frame: MovementDebugFrame) -> None: ...

    def close(self) -> None: ...


class NullMovementDebugSink:
    def emit(self, frame: MovementDebugFrame) -> None:
        return

    def close(self) -> None:
        return


NULL_MOVEMENT_DEBUG_SINK = NullMovementDebugSink()


class TwoPanelVideoDebugSink:
    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: int = 30,
        *,
        box_size: int = 50,
        buffer_size: int = 0,
    ) -> None:
        self.box_size = int(box_size)
        self._buffer_size = max(0, int(buffer_size))
        self._buffer_order: deque[int] = deque()
        self._buffered: dict[int, tuple[np.ndarray, np.ndarray, dict[int, tuple[float, float]]]] = {}
        self._trajectories: dict[int, list[tuple[float, float]]] = {}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self._writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height * 2))

    def emit(self, frame: MovementDebugFrame) -> None:
        self._update_trajectories(frame.centroids)

        if self._buffer_size <= 1:
            self._write(frame.filtered, frame.diff, frame.centroids, frame.frame_number)
            return

        frame_number = int(frame.frame_number)
        self._buffer_order.append(frame_number)
        self._buffered[frame_number] = (
            frame.filtered,
            frame.diff,
            frame.centroids.copy(),
        )
        self._flush_oldest_if_needed()

    def record_event(self, event: MovementEvent) -> None:
        if self._buffer_size <= 1:
            return
        if event.centroid is None or event.track_id is None:
            return

        frame_number = int(event.frame_number)
        buffered = self._buffered.get(frame_number)
        if buffered is None:
            return

        filtered, diff, centroids = buffered
        updated_centroids = {**centroids, event.track_id: event.centroid}
        self._buffered[frame_number] = (filtered, diff, updated_centroids)

    def _flush_oldest_if_needed(self) -> None:
        if len(self._buffer_order) <= self._buffer_size:
            return
        oldest_frame_number = self._buffer_order.popleft()
        filtered, diff, centroids = self._buffered.pop(oldest_frame_number)
        self._write(filtered, diff, centroids, oldest_frame_number)

    def _flush_all(self) -> None:
        while self._buffer_order:
            frame_number = self._buffer_order.popleft()
            filtered, diff, centroids = self._buffered.pop(frame_number)
            self._write(filtered, diff, centroids, frame_number)

    def _write(
        self,
        filtered: np.ndarray,
        diff: np.ndarray,
        centroids: dict[int, tuple[float, float]],
        frame_number: int,
    ) -> None:
        filtered_bgr = filtered
        diff_bgr = diff

        if filtered_bgr.ndim == 2:
            filtered_bgr = cv2.cvtColor(filtered_bgr, cv2.COLOR_GRAY2BGR)
        if diff_bgr.ndim == 2:
            diff_bgr = cv2.cvtColor(diff_bgr, cv2.COLOR_GRAY2BGR)

        for track_id, trajectory in self._trajectories.items():
            if len(trajectory) > 1:
                color = self._get_track_color(track_id)
                self._draw_trajectory(filtered_bgr, trajectory, color, 2)
                self._draw_trajectory(diff_bgr, trajectory, color, 2)

        for track_id, centroid in centroids.items():
            print(f"Frame {frame_number}: Track {track_id} at {centroid}")
            color = self._get_track_color(track_id)
            self.draw_bbox(diff_bgr, centroid, color, 2)
            self.draw_bbox(filtered_bgr, centroid, color, 2)

        debug_frame = np.vstack([filtered_bgr, diff_bgr])
        self._writer.write(debug_frame)

    def close(self) -> None:
        if self._buffer_size > 1:
            self._flush_all()
        self._writer.release()

    def draw_bbox(
        self,
        frame: np.ndarray,
        centroid: tuple[float, float],
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        cx, cy = int(centroid[0]), int(centroid[1])
        cv2.rectangle(
            frame,
            (cx - self.box_size, cy - self.box_size),
            (cx + self.box_size, cy + self.box_size),
            color,
            thickness,
        )

    def _update_trajectories(self, centroids: dict[int, tuple[float, float]]) -> None:
        for track_id, centroid in centroids.items():
            if track_id not in self._trajectories:
                self._trajectories[track_id] = []
            self._trajectories[track_id].append(centroid)

    def _draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: list[tuple[float, float]],
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        if len(trajectory) < 2:
            return
        points = np.array([(int(x), int(y)) for x, y in trajectory], dtype=np.int32)
        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=thickness)

    def _get_track_color(self, track_id: int) -> tuple[int, int, int]:
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 255),
            (255, 128, 0),
        ]
        return colors[track_id % len(colors)]
