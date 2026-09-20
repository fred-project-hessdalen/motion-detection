from dataclasses import dataclass, field
from functools import reduce
from typing import Generator

import cv2
import numpy as np
from funcy import lmapcat, mapcat, first

from hessdalen.io.video import VideoStream
from hessdalen.domain.models import DetectedMovement, MovementEvent, VideoFrame
from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.debug import (
    MovementDebugFrame,
    MovementDebugSink,
    NULL_MOVEMENT_DEBUG_SINK,
)
from hessdalen.processing.detection import Centroid, Detection, DetectionSettings
from hessdalen.processing.devices import Device, detection_stage

__all__ = [
    "Centroid",
    "Detection",
    "DetectionSettings",
    "MovementDetector",
    "MovementSettings",
    "TrackingSettings",
]


@dataclass(frozen=True, slots=True)
class TrackingSettings:
    """What a run of detections has to look like before it is a track.

    Every one of these is set from the dashboard and comes from the
    config, so they are asked of the caller.
    """

    min_consecutive_frames: int
    max_movement_ratio: float
    min_movement_ratio: float
    max_missed_frames: int
    min_trajectory_span_ratio: float


@dataclass(frozen=True, slots=True)
class MovementSettings:
    """Everything a run of the detector is told.

    These come from the config rather than from values written here, so
    the numbers a run starts from sit in one file that can be edited by
    hand and written back from the dashboard.
    """

    background: BackgroundSettings
    detection: DetectionSettings
    tracking: TrackingSettings
    device: Device


@dataclass(frozen=True, slots=True)
class TrackHit:
    """One frame of a track, as the detection stage measured it."""

    frame_number: int
    detection: Detection


@dataclass(slots=True)
class Track:
    track_id: int
    detection: Detection
    consecutive_hits: int = 1
    consecutive_misses: int = 0
    confirmed: bool = False
    pending: list[TrackHit] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    events: list[MovementEvent]
    centroids: dict[int, Centroid]


class MovementDetector:
    def __init__(self, stream: VideoStream, settings: MovementSettings):
        self.stream = stream
        self.settings = settings
        self.stage = detection_stage(
            device=settings.device,
            background=settings.background,
            detection=settings.detection,
            timestamp_mask=stream.mask,
        )

        height, width = stream.frame_shape
        frame_max_dimension = float(max(height, width))
        tracking = settings.tracking
        self._max_movement_distance = tracking.max_movement_ratio * frame_max_dimension
        self._min_movement_distance = tracking.min_movement_ratio * frame_max_dimension
        self._min_trajectory_span = tracking.min_trajectory_span_ratio * frame_max_dimension

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
            yield from mapcat(self.process_frame, self._frames())
        finally:
            self._debug_sink = previous_debug_sink

    def process_frame(self, frame: VideoFrame) -> list[MovementEvent]:
        """The events one frame produces, for a caller that owns the frame
        loop.

        A caller that drives the detector frame by frame can show each
        frame as it is measured, which detect cannot do because it holds
        the loop itself.
        """
        frame_number = int(frame.frame_number)
        analysis = self._analyze_frame(frame_number, frame)

        if self._debug_sink.wants_frames:
            self._debug_sink.emit(
                MovementDebugFrame(
                    frame_number=frame_number,
                    filtered=frame.frame,
                    deviation=self.stage.deviation_image(),
                    centroids=analysis.centroids,
                )
            )

        if analysis.events:
            return analysis.events

        return [MovementEvent(frame_number=frame_number, track_id=None, centroid=None)]

    def _frames(self) -> Generator[VideoFrame, None, None]:
        """Colour frames when the sink draws on them, grayscale otherwise."""
        if self._debug_sink.wants_frames:
            return self.stream.stream_frames()
        return self.stream.stream_gray_frames()

    def _analyze_frame(self, frame_number: int, frame: VideoFrame) -> FrameAnalysis:
        gray = self._to_grayscale(frame.frame)
        events = self._update_tracks(frame_number, self.stage.detections(gray))

        active_confirmed_tracks = [
            track for track in self._tracks.values() if track.confirmed and track.consecutive_misses == 0
        ]
        centroids = {track.track_id: track.detection.centroid for track in active_confirmed_tracks}
        return FrameAnalysis(events=events, centroids=centroids)

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _update_tracks(self, frame_number: int, detections: list[Detection]) -> list[MovementEvent]:
        if not detections:
            self._mark_all_tracks_missed()
            self._delete_expired_tracks()
            return []

        matches, assigned_detection_indexes = self._match_tracks_to_detections(detections)
        open_tracks = list(self._tracks.values())
        updated_track_events = lmapcat(
            lambda track: self._update_track_from_match(
                frame_number=frame_number, detections=detections, matches=matches, track=track
            ),
            open_tracks,
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
            if (dist_sq := self._distance_sq(track.detection.centroid, detection.centroid)) <= max_dist_sq
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
        """Take this track to the detection it was matched with.

        Confirming a track earlier in the round can merge it into this
        one's id, which leaves this one holding an id that now names the
        track it was merged with. Reporting it here would report that id
        twice for the one frame.
        """
        if self._tracks.get(track.track_id) is not track:
            return []

        detection_index = matches.get(track.track_id)
        if detection_index is None:
            self._mark_track_missed(track)
            return []

        detection = detections[detection_index]
        distance = float(np.sqrt(self._distance_sq(track.detection.centroid, detection.centroid)))
        if distance < self._min_movement_distance:
            self._mark_track_missed(track)
            return []

        track.detection = detection
        track.consecutive_hits += 1
        track.consecutive_misses = 0
        return self._events_for_track_hit(frame_number=frame_number, track=track)

    def _events_for_track_hit(self, *, frame_number: int, track: Track) -> list[MovementEvent]:
        if track.confirmed:
            return [
                self._detected_movement(frame_number=frame_number, track_id=track.track_id, detection=track.detection)
            ]

        track.pending = [
            *track.pending,
            TrackHit(frame_number=frame_number, detection=track.detection),
        ]
        if track.consecutive_hits < int(self.settings.tracking.min_consecutive_frames):
            return []

        if not self._validate_trajectory(track.pending):
            return []

        mergeable_track_id = self._find_mergeable_track(track.pending[0].detection.centroid, track.track_id)
        if mergeable_track_id is not None:
            self._merge_track_with_existing(track, mergeable_track_id)

        track.confirmed = True
        replay_events: list[MovementEvent] = [
            self._detected_movement(
                frame_number=int(hit.frame_number), track_id=track.track_id, detection=hit.detection
            )
            for hit in track.pending
        ]
        track.pending = []
        return replay_events

    def _detected_movement(self, *, frame_number: int, track_id: int, detection: Detection) -> DetectedMovement:
        return DetectedMovement(
            frame_number=frame_number, track_id=track_id, centroid=detection.centroid, blob=detection.blob
        )

    def _validate_trajectory(self, pending: list[TrackHit]) -> bool:
        if len(pending) < 2:
            return True

        centroids = [hit.detection.centroid for hit in pending]
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

            dist_sq = self._distance_sq(track.detection.centroid, first_centroid)
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
        track = self._new_track(detection=detection)
        self._tracks = {
            **self._tracks,
            track.track_id: track,
        }
        track.pending = [
            *track.pending,
            TrackHit(frame_number=frame_number, detection=track.detection),
        ]
        if int(self.settings.tracking.min_consecutive_frames) > 1:
            return []

        track.confirmed = True
        track.pending = []
        return [self._detected_movement(frame_number=frame_number, track_id=track.track_id, detection=track.detection)]

    def _new_track(self, *, detection: Detection) -> Track:
        track_id = int(self._next_track_id)
        self._next_track_id += 1
        return Track(track_id=track_id, detection=detection, consecutive_hits=1, consecutive_misses=0)
