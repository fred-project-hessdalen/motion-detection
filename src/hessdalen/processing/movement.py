from dataclasses import dataclass, field
from functools import reduce
from typing import Generator

import cv2
import numpy as np
from funcy import lmapcat, mapcat, first

from hessdalen.io.video import VideoStream
from hessdalen.domain.models import DetectedMovement, MovementEvent, VideoFrame
from hessdalen.processing.background import BackgroundModel, BackgroundSettings
from hessdalen.processing.debug import (
    MovementDebugFrame,
    MovementDebugSink,
    NULL_MOVEMENT_DEBUG_SINK,
)

Centroid = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    centroid: Centroid
    pixel_count: int
    peak_deviation: float


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    foreground_sigma: float = 5.0
    detection_sigma: float = 15.0
    min_pixels: int = 3
    close_size: int = 5


@dataclass(frozen=True, slots=True)
class TrackingSettings:
    min_consecutive_frames: int = 6
    max_movement_ratio: float = 0.02
    min_movement_ratio: float = 0.001
    max_missed_frames: int = 8
    min_trajectory_span_ratio: float = 0.02


@dataclass(frozen=True, slots=True)
class MovementSettings:
    background: BackgroundSettings = field(default_factory=BackgroundSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)


@dataclass(slots=True)
class Track:
    track_id: int
    centroid: Centroid
    consecutive_hits: int = 1
    consecutive_misses: int = 0
    confirmed: bool = False
    pending: list[tuple[int, Centroid]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    events: list[MovementEvent]
    centroids: dict[int, Centroid]
    deviation: np.ndarray


class MovementDetector:
    def __init__(self, stream: VideoStream, settings: MovementSettings):
        self.stream = stream
        self.settings = settings
        self.background = BackgroundModel(settings.background)

        height, width = stream.frame_shape
        frame_max_dimension = float(max(height, width))
        tracking = settings.tracking
        self._max_movement_distance = tracking.max_movement_ratio * frame_max_dimension
        self._min_movement_distance = tracking.min_movement_ratio * frame_max_dimension
        self._min_trajectory_span = tracking.min_trajectory_span_ratio * frame_max_dimension

        close_size = settings.detection.close_size
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

        self._tracks: dict[int, Track] = {}
        self._next_track_id: int = 1
        self._debug_sink: MovementDebugSink = NULL_MOVEMENT_DEBUG_SINK

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
                deviation=analysis.deviation,
                centroids=analysis.centroids,
            )
        )

        if analysis.events:
            yield from analysis.events
            return

        yield MovementEvent(frame_number=frame_number, track_id=None, centroid=None)

    def _analyze_frame(self, frame_number: int, frame: VideoFrame) -> FrameAnalysis:
        gray = self._to_grayscale(frame.frame)
        deviation = self.background.deviation(gray)
        events = self._update_tracks(frame_number, self._extract_detections(deviation))

        active_confirmed_tracks = [
            track for track in self._tracks.values() if track.confirmed and track.consecutive_misses == 0
        ]
        centroids = {track.track_id: track.centroid for track in active_confirmed_tracks}
        return FrameAnalysis(
            events=events,
            centroids=centroids,
            deviation=self._deviation_image(deviation),
        )

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _deviation_image(self, deviation: np.ndarray) -> np.ndarray:
        """Scale the deviation for the debug video, with the detection
        threshold at full white."""
        scaled = deviation * (255.0 / self.settings.detection.detection_sigma)
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    def _extract_detections(self, deviation: np.ndarray) -> list[Detection]:
        foreground = self._foreground_mask(deviation)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        return [
            detection
            for component in range(1, component_count)
            if (detection := self._component_to_detection(component, labels, stats, deviation)) is not None
        ]

    def _foreground_mask(self, deviation: np.ndarray) -> np.ndarray:
        mask: np.ndarray = (deviation > self.settings.detection.foreground_sigma).astype(np.uint8) * 255
        if self.stream.mask is not None:
            mask = cv2.bitwise_and(mask, self.stream.mask)
        if self.settings.detection.close_size <= 1:
            return mask
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)

    def _component_to_detection(
        self,
        component: int,
        labels: np.ndarray,
        stats: np.ndarray,
        deviation: np.ndarray,
    ) -> Detection | None:
        x, y, width, height, pixel_count = (int(value) for value in stats[component])
        if pixel_count < self.settings.detection.min_pixels:
            return None

        window = np.where(
            labels[y : y + height, x : x + width] == component, deviation[y : y + height, x : x + width], 0.0
        )
        peak_deviation = float(window.max())
        if peak_deviation < self.settings.detection.detection_sigma:
            return None

        peak_y, peak_x = np.unravel_index(int(np.argmax(window)), window.shape)
        return Detection(
            centroid=(float(x + int(peak_x)), float(y + int(peak_y))),
            pixel_count=pixel_count,
            peak_deviation=peak_deviation,
        )

    def _update_tracks(self, frame_number: int, detections: list[Detection]) -> list[MovementEvent]:
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
            lambda detection: self._create_track_for_detection(frame_number=frame_number, detection=detection),
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
        max_misses_for_deletion = int(self.settings.tracking.max_missed_frames) * 2
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.consecutive_misses <= max_misses_for_deletion
        }

    def _match_tracks_to_detections(self, detections: list[Detection]) -> tuple[dict[int, int], set[int]]:
        max_dist_sq = self._max_movement_distance**2
        candidate_pairs = self._candidate_track_detection_pairs(detections=detections, max_dist_sq=max_dist_sq)
        ordered_pairs = sorted(candidate_pairs, key=lambda pair: pair[0])
        matches, assigned_detection_indexes = self._greedy_assign_pairs(ordered_pairs)
        return matches, assigned_detection_indexes

    def _candidate_track_detection_pairs(
        self,
        *,
        detections: list[Detection],
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
        detections: list[Detection],
        max_dist_sq: float,
    ) -> list[tuple[float, int, int]]:
        return [
            (dist_sq, track.track_id, detection_index)
            for detection_index, detection in enumerate(detections)
            if (dist_sq := self._distance_sq(track.centroid, detection.centroid)) <= max_dist_sq
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
        detections: list[Detection],
        matches: dict[int, int],
        track: Track,
    ) -> list[MovementEvent]:
        detection_index = matches.get(track.track_id)
        if detection_index is None:
            self._mark_track_missed(track)
            return []

        centroid = detections[detection_index].centroid
        distance = float(np.sqrt(self._distance_sq(track.centroid, centroid)))
        if distance < self._min_movement_distance:
            self._mark_track_missed(track)
            return []

        track.centroid = centroid
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
        if track.consecutive_hits < int(self.settings.tracking.min_consecutive_frames):
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

        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        return max(bbox_width, bbox_height) >= self._min_trajectory_span

    def _find_mergeable_track(self, first_centroid: Centroid, exclude_track_id: int) -> int | None:
        max_dist_sq = self._max_movement_distance**2
        max_misses_for_merge = int(self.settings.tracking.max_missed_frames)

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

    def _create_track_for_detection(self, *, frame_number: int, detection: Detection) -> list[MovementEvent]:
        track = self._new_track(centroid=detection.centroid)
        self._tracks = {
            **self._tracks,
            track.track_id: track,
        }
        track.pending = [
            *track.pending,
            (frame_number, track.centroid),
        ]
        if int(self.settings.tracking.min_consecutive_frames) > 1:
            return []

        track.confirmed = True
        track.pending = []
        return [self._detected_movement(frame_number=frame_number, track_id=track.track_id, centroid=track.centroid)]

    def _new_track(self, *, centroid: Centroid) -> Track:
        track_id = int(self._next_track_id)
        self._next_track_id += 1
        return Track(track_id=track_id, centroid=centroid, consecutive_hits=1, consecutive_misses=0)
