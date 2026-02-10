from dataclasses import dataclass, field
from functools import reduce
from typing import Generator

import cv2
import numpy as np
from funcy import lmapcat, mapcat, first

from hessdalen.io.video import VideoStream
from hessdalen.domain.models import DetectedMovement, MovementEvent, VideoFrame
from hessdalen.processing.debug import (
    MovementDebugFrame,
    MovementDebugSink,
    NULL_MOVEMENT_DEBUG_SINK,
)

Centroid = tuple[float, float]


@dataclass(slots=True)
class Track:
    track_id: int
    centroid: Centroid
    consecutive_hits: int = 1
    consecutive_misses: int = 0
    area: float = 0.0
    confirmed: bool = False
    pending: list[tuple[int, Centroid]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    events: list[MovementEvent]
    centroids: dict[int, Centroid]
    filtered: np.ndarray
    diff: np.ndarray


class MovementDetector:
    def __init__(
        self,
        stream: VideoStream,
        alpha: float = 0.1,
        adaptive_temporal_filter: bool = True,
        adaptive_change_threshold: int = 10,
        alpha_small_change: float | None = None,
        alpha_large_change: float | None = None,
        diff_threshold: int = 25,
        min_area: int = 5,
        kernel_size: int = 5,
        min_consecutive_frames: int = 3,
        max_movement_distance: float = 100.0,
        min_movement_distance: float = 5.0,
        max_missed_frames: int = 2,
        min_trajectory_span_ratio: float = 0.05,
    ):
        self.stream = stream
        self.alpha = alpha
        self.adaptive_temporal_filter = bool(adaptive_temporal_filter)
        self.adaptive_change_threshold = int(adaptive_change_threshold)
        self.alpha_small_change = float(alpha_small_change) if alpha_small_change is not None else float(alpha)
        default_alpha_large = float(alpha) * 0.2
        self.alpha_large_change = (
            float(alpha_large_change) if alpha_large_change is not None else float(default_alpha_large)
        )
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.kernel_size = kernel_size
        self.min_consecutive_frames = min_consecutive_frames
        self.max_movement_distance = max_movement_distance
        self.min_movement_distance = min_movement_distance
        self.max_missed_frames = max_missed_frames
        self.min_trajectory_span_ratio = min_trajectory_span_ratio
        self.filtered_frame: np.ndarray | None = None
        self._tracks: dict[int, Track] = {}
        self._next_track_id: int = 1
        self._debug_sink: MovementDebugSink = NULL_MOVEMENT_DEBUG_SINK
        self._frame_max_dimension: float | None = None

    def detect(
        self,
        debug_sink: MovementDebugSink = NULL_MOVEMENT_DEBUG_SINK,
    ) -> Generator[MovementEvent, None, None]:
        previous_debug_sink = self._debug_sink
        self._debug_sink = debug_sink
        try:
            yield from mapcat(self._process_frame, self.stream.stream_frames())
        finally:
            self._debug_sink = previous_debug_sink

    def _process_frame(self, frame: VideoFrame) -> Generator[MovementEvent, None, None]:
        frame_number = int(frame.frame_number)
        analysis = self._analyze_frame(frame_number, frame)

        self._debug_sink.emit(
            MovementDebugFrame(
                frame_number=frame_number,
                filtered=frame.frame,
                diff=analysis.diff,
                centroids=analysis.centroids,
            )
        )

        if analysis.events:
            yield from analysis.events
            return

        yield MovementEvent(frame_number=frame_number, track_id=None, centroid=None)

    def _analyze_frame(self, frame_number: int, frame: VideoFrame) -> FrameAnalysis:
        gray = self._to_grayscale(frame.frame)
        if self._frame_max_dimension is None:
            height, width = gray.shape
            self._frame_max_dimension = float(max(height, width))
        filtered = self._apply_temporal_filter(gray)

        spatial_filtered = self._compute_spatial_filtered(gray, filtered)
        detections = self._extract_detections(spatial_filtered)
        events = self._update_tracks(frame_number, detections)

        active_confirmed_tracks = [
            track for track in self._tracks.values() if track.confirmed and track.consecutive_misses == 0
        ]
        centroids = {track.track_id: track.centroid for track in active_confirmed_tracks}
        return FrameAnalysis(
            events=events,
            centroids=centroids,
            filtered=filtered,
            diff=spatial_filtered,
        )

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _apply_temporal_filter(self, gray_frame: np.ndarray) -> np.ndarray:
        if self.alpha >= 1.0 and not self.adaptive_temporal_filter:
            return gray_frame
        if self.filtered_frame is None:
            self.filtered_frame = gray_frame.astype(np.float32)
            return self.filtered_frame.astype(np.uint8)

        if not self.adaptive_temporal_filter:
            cv2.accumulateWeighted(gray_frame, self.filtered_frame, self.alpha)
            return self.filtered_frame.astype(np.uint8)

        self._apply_adaptive_temporal_filter(gray_frame)
        return self.filtered_frame.astype(np.uint8)

    def _apply_adaptive_temporal_filter(self, gray_frame: np.ndarray) -> None:
        if self.filtered_frame is None:
            raise ValueError("filtered_frame must be initialized")

        current_f = gray_frame.astype(np.float32)
        diff = np.abs(current_f - self.filtered_frame)
        small_change_mask = diff <= float(self.adaptive_change_threshold)
        alpha_small, alpha_large = self._adaptive_alphas()

        self._masked_accumulate(mask=small_change_mask, current=current_f, alpha=alpha_small)
        self._masked_accumulate(mask=~small_change_mask, current=current_f, alpha=alpha_large)

    def _adaptive_alphas(self) -> tuple[float, float]:
        alpha_small = float(np.clip(self.alpha_small_change, 0.0, 1.0))
        alpha_large = float(np.clip(self.alpha_large_change, 0.0, 1.0))
        return (
            (alpha_small, alpha_large)
            if alpha_small >= alpha_large
            else (
                alpha_large,
                alpha_small,
            )
        )

    def _masked_accumulate(self, *, mask: np.ndarray, current: np.ndarray, alpha: float) -> None:
        if self.filtered_frame is None:
            raise ValueError("filtered_frame must be initialized")

        if alpha >= 1.0:
            self.filtered_frame[mask] = current[mask]
            return

        self.filtered_frame[mask] = (1.0 - alpha) * self.filtered_frame[mask] + alpha * current[mask]

    def _compute_spatial_filtered(self, current: np.ndarray, filtered: np.ndarray) -> np.ndarray:
        diff = self._compute_frame_diff(current, filtered)
        return self._apply_spatial_filter(diff)

    def _compute_frame_diff(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(current, previous)
        if self.stream.mask is not None:
            diff = cv2.bitwise_and(diff, diff, mask=self.stream.mask)
        return diff

    def _apply_spatial_filter(self, diff_frame: np.ndarray) -> np.ndarray:
        kernel_size = int(self.kernel_size)
        if kernel_size <= 1:
            return diff_frame

        open_kernel_size = max(1, kernel_size // 3)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        filtered = diff_frame
        if open_kernel_size > 1:
            filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel_open)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel_close)
        return cv2.dilate(filtered, kernel_close, iterations=2)

    def _extract_detections(self, spatial_filtered: np.ndarray) -> list[tuple[Centroid, float]]:
        _, thresh = cv2.threshold(spatial_filtered, self.diff_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        min_contour_area = float(self.min_area)
        return [
            detection
            for contour in contours
            if (detection := self._contour_to_detection(contour, min_contour_area, spatial_filtered)) is not None
        ]

    def _contour_to_detection(
        self,
        contour: np.ndarray,
        min_contour_area: float,
        spatial_filtered: np.ndarray,
    ) -> tuple[Centroid, float] | None:
        area = float(cv2.contourArea(contour))
        if area < min_contour_area:
            return None

        centroid = self._contour_peak(spatial_filtered, contour)
        if centroid is None:
            centroid = self._contour_centroid(contour)
        if centroid is None:
            return None

        return (centroid, area)

    def _contour_peak(self, spatial_filtered: np.ndarray, contour: np.ndarray) -> Centroid | None:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return None

        roi = spatial_filtered[y : y + h, x : x + w]
        if roi.size == 0:
            return None

        local_contour = contour.copy()
        local_contour[:, 0, 0] -= x
        local_contour[:, 0, 1] -= y

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [local_contour], -1, 255, -1)

        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(roi, mask=mask)
        if max_val <= 0:
            return None

        peak_x = float(x + int(max_loc[0]))
        peak_y = float(y + int(max_loc[1]))
        return (peak_x, peak_y)

    def _contour_centroid(self, contour: np.ndarray) -> Centroid | None:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return (cx, cy)

    def _update_tracks(self, frame_number: int, detections: list[tuple[Centroid, float]]) -> list[MovementEvent]:
        if not detections:
            self._mark_all_tracks_missed()
            self._delete_expired_tracks()
            return []

        matches, assigned_detection_indexes = self._match_tracks_to_detections(detections)
        updated_track_events = lmapcat(
            lambda track: self._update_track_from_match(
                frame_number=frame_number, detections=detections, matches=matches, track=track
            ),
            self._tracks.values(),
        )

        self._delete_expired_tracks()
        unassigned_detections = (
            detection
            for detection_index, detection in enumerate(detections)
            if detection_index not in assigned_detection_indexes
        )
        new_track_events = lmapcat(
            lambda detection: self._create_track_for_detection_data(frame_number=frame_number, detection=detection),
            unassigned_detections,
        )
        return [*updated_track_events, *new_track_events]

    def _mark_all_tracks_missed(self) -> None:
        for track in self._tracks.values():
            self._mark_track_missed(track)

    def _mark_track_missed(self, track: Track) -> None:
        track.consecutive_hits = 0
        track.consecutive_misses += 1

    def _delete_expired_tracks(self) -> None:
        max_misses_for_deletion = int(self.max_missed_frames) * 2
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.consecutive_misses <= max_misses_for_deletion
        }

    def _match_tracks_to_detections(self, detections: list[tuple[Centroid, float]]) -> tuple[dict[int, int], set[int]]:
        max_dist_sq = float(self.max_movement_distance) ** 2
        candidate_pairs = self._candidate_track_detection_pairs(detections=detections, max_dist_sq=max_dist_sq)
        ordered_pairs = sorted(candidate_pairs, key=lambda pair: pair[0])
        matches, assigned_detection_indexes = self._greedy_assign_pairs(ordered_pairs)
        return matches, assigned_detection_indexes

    def _candidate_track_detection_pairs(
        self,
        *,
        detections: list[tuple[Centroid, float]],
        max_dist_sq: float,
    ) -> list[tuple[float, int, int]]:
        return lmapcat(
            lambda track: self._pairs_for_track(track=track, detections=detections, max_dist_sq=max_dist_sq),
            self._tracks.values(),
        )

    def _pairs_for_track(
        self,
        *,
        track: Track,
        detections: list[tuple[Centroid, float]],
        max_dist_sq: float,
    ) -> list[tuple[float, int, int]]:
        return [
            (dist_sq, track.track_id, detection_index)
            for detection_index, (centroid, _area) in enumerate(detections)
            if (dist_sq := self._distance_sq(track.centroid, centroid)) <= max_dist_sq
        ]

    def _distance_sq(self, a: Centroid, b: Centroid) -> float:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return float(dx * dx + dy * dy)

    def _greedy_assign_pairs(
        self,
        pairs: list[tuple[float, int, int]],
    ) -> tuple[dict[int, int], set[int]]:
        def step(
            acc: tuple[dict[int, int], set[int], set[int]],
            pair: tuple[float, int, int],
        ) -> tuple[dict[int, int], set[int], set[int]]:
            matches, assigned_tracks, assigned_detections = acc
            _dist_sq, track_id, detection_index = pair
            if track_id in assigned_tracks or detection_index in assigned_detections:
                return acc
            return (
                {**matches, track_id: detection_index},
                assigned_tracks | {track_id},
                assigned_detections | {detection_index},
            )

        initial: tuple[dict[int, int], set[int], set[int]] = ({}, set(), set())
        matches, _assigned_tracks, assigned_detections = reduce(step, pairs, initial)
        return matches, assigned_detections

    def _update_track_from_match(
        self,
        *,
        frame_number: int,
        detections: list[tuple[Centroid, float]],
        matches: dict[int, int],
        track: Track,
    ) -> list[MovementEvent]:
        detection_index = matches.get(track.track_id)
        if detection_index is None:
            self._mark_track_missed(track)
            return []

        centroid, area = detections[detection_index]
        distance = float(np.sqrt(self._distance_sq(track.centroid, centroid)))
        if distance < float(self.min_movement_distance):
            self._mark_track_missed(track)
            return []

        track.centroid = centroid
        track.area = area
        track.consecutive_hits += 1
        track.consecutive_misses = 0
        return self._events_for_track_hit(frame_number=frame_number, track=track)

    def _events_for_track_hit(self, *, frame_number: int, track: Track) -> list[MovementEvent]:
        if track.confirmed:
            return [
                self._detected_movement(frame_number=frame_number, track_id=track.track_id, centroid=track.centroid)
            ]

        track.pending = [
            *track.pending,
            (frame_number, track.centroid),
        ]
        if track.consecutive_hits < int(self.min_consecutive_frames):
            return []

        if not self._validate_trajectory(track.pending):
            return []

        first_frame, first_centroid = track.pending[0]
        mergeable_track_id = self._find_mergeable_track(first_centroid, track.track_id)
        if mergeable_track_id is not None:
            self._merge_track_with_existing(track, mergeable_track_id)

        track.confirmed = True
        replay_events: list[MovementEvent] = [
            self._detected_movement(
                frame_number=int(pending_frame_number), track_id=track.track_id, centroid=pending_centroid
            )
            for pending_frame_number, pending_centroid in track.pending
        ]
        track.pending = []
        return replay_events

    def _detected_movement(self, *, frame_number: int, track_id: int, centroid: Centroid) -> DetectedMovement:
        return DetectedMovement(frame_number=frame_number, track_id=track_id, centroid=centroid)

    def _validate_trajectory(self, pending: list[tuple[int, Centroid]]) -> bool:
        if len(pending) < 2:
            return True

        centroids = [centroid for _frame_number, centroid in pending]
        x_coords = [c[0] for c in centroids]
        y_coords = [c[1] for c in centroids]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        bbox_span = max(bbox_width, bbox_height)

        if self._frame_max_dimension is None:
            return True

        min_span = self.min_trajectory_span_ratio * self._frame_max_dimension
        return bbox_span >= min_span

    def _find_mergeable_track(self, first_centroid: Centroid, exclude_track_id: int) -> int | None:
        max_dist_sq = float(self.max_movement_distance)
        max_misses_for_merge = int(self.max_missed_frames)

        def is_mergeable(track_id: int, track: Track) -> int | None:
            if track_id == exclude_track_id:
                return None
            if not track.confirmed:
                return None
            if track.consecutive_misses == 0:
                return None
            if track.consecutive_misses > max_misses_for_merge:
                return None

            dist_sq = self._distance_sq(track.centroid, first_centroid)
            if dist_sq <= max_dist_sq:
                return track_id

            return None

        return first((is_mergeable(track_id, track) for track_id, track in self._tracks.items()))

    def _merge_track_with_existing(self, track: Track, target_track_id: int) -> None:
        old_track_id = track.track_id
        track.track_id = target_track_id

        if old_track_id in self._tracks:
            del self._tracks[old_track_id]
        self._tracks[target_track_id] = track

    def _create_track_for_detection_data(
        self,
        *,
        frame_number: int,
        detection: tuple[Centroid, float],
    ) -> list[MovementEvent]:
        centroid, area = detection
        return self._create_track_for_detection(frame_number=frame_number, centroid=centroid, area=area)

    def _create_track_for_detection(
        self,
        *,
        frame_number: int,
        centroid: Centroid,
        area: float,
    ) -> list[MovementEvent]:
        track = self._new_track(centroid=centroid, area=area)
        self._tracks = {
            **self._tracks,
            track.track_id: track,
        }
        track.pending = [
            *track.pending,
            (frame_number, track.centroid),
        ]
        if int(self.min_consecutive_frames) > 1:
            return []

        track.confirmed = True
        track.pending = []
        return [self._detected_movement(frame_number=frame_number, track_id=track.track_id, centroid=track.centroid)]

    def _new_track(self, *, centroid: Centroid, area: float) -> Track:
        track_id = int(self._next_track_id)
        self._next_track_id += 1
        return Track(track_id=track_id, centroid=centroid, consecutive_hits=1, consecutive_misses=0, area=area)
