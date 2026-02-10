"""Test trajectory merging across detection gaps."""

import numpy as np
import pytest

from hessdalen.io.video import VideoStream
from hessdalen.processing.movement import MovementDetector


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


def create_frame_with_dot(width: int, height: int, x: int, y: int, radius: int = 5) -> np.ndarray:
    """Create a black frame with a white dot at (x, y)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    y_min = max(0, y - radius)
    y_max = min(height, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(width, x + radius + 1)
    frame[y_min:y_max, x_min:x_max] = 255
    return frame


def test_trajectory_merging_with_gap():
    """Test that trajectories merge across detection gaps."""
    width, height = 200, 200

    # Create frames with a moving dot that has a gap in the middle
    frames = []

    # First trajectory: dot moves from (50, 50) to (70, 50) over 5 frames
    for i in range(5):
        x = 50 + i * 4
        frames.append(create_frame_with_dot(width, height, x, 50))

    # Gap: 3 frames with no dot (black frames)
    for _ in range(3):
        frames.append(np.zeros((height, width, 3), dtype=np.uint8))

    # Second trajectory: dot continues from (74, 50) to (94, 50) over 5 frames
    for i in range(5):
        x = 74 + i * 4
        frames.append(create_frame_with_dot(width, height, x, 50))

    # Create detector
    mock_source = MockFrameSource(frames)
    stream = VideoStream(mock_source, target_height=height)

    detector = MovementDetector(
        stream=stream,
        alpha=0.3,
        diff_threshold=25,
        min_area=10,
        kernel_size=3,
        min_consecutive_frames=3,
        max_movement_distance=50.0,
        min_movement_distance=2.0,
        max_missed_frames=3,
        min_trajectory_span_ratio=0.0,
    )

    # Collect all events
    events = [event for event in detector.detect() if event.centroid is not None]

    # Get unique track IDs
    track_ids = set(event.track_id for event in events if event.track_id is not None)

    print(f"\nTotal events: {len(events)}")
    print(f"Unique track IDs: {track_ids}")
    print(f"Number of unique tracks: {len(track_ids)}")

    # Print events by track
    for track_id in sorted(track_ids):
        track_events = [e for e in events if e.track_id == track_id]
        print(f"\nTrack {track_id}: {len(track_events)} events")
        print(f"  Frames: {[e.frame_number for e in track_events[:5]]}{'...' if len(track_events) > 5 else ''}")
        if track_events:
            first_centroid = track_events[0].centroid
            last_centroid = track_events[-1].centroid
            print(f"  First centroid: {first_centroid}")
            print(f"  Last centroid: {last_centroid}")

    # With merging, we should have only 1 track ID
    # Without merging, we would have 2 track IDs
    assert len(track_ids) == 1, (
        f"Expected 1 track (merged), but got {len(track_ids)} tracks. "
        f"Track IDs: {track_ids}. This suggests trajectory merging is not working."
    )

    # Verify we got detections from both trajectory segments
    assert len(events) >= 8, f"Expected at least 8 events, got {len(events)}"


def test_trajectory_no_merge_when_too_far():
    """Test that trajectories don't merge when spatially too far apart."""
    width, height = 200, 200

    frames = []

    # First trajectory: dot moves from (50, 50) to (70, 50)
    for i in range(5):
        x = 50 + i * 4
        frames.append(create_frame_with_dot(width, height, x, 50))

    # Gap: 5 frames to let temporal filter adapt
    for _ in range(5):
        frames.append(np.zeros((height, width, 3), dtype=np.uint8))

    # Second trajectory: dot moves from (150, 150) to (170, 150) - far away
    for i in range(5):
        x = 150 + i * 4
        frames.append(create_frame_with_dot(width, height, x, 150))

    mock_source = MockFrameSource(frames)
    stream = VideoStream(mock_source, target_height=height)

    detector = MovementDetector(
        stream=stream,
        alpha=0.3,
        diff_threshold=25,
        min_area=10,
        kernel_size=3,
        min_consecutive_frames=3,
        max_movement_distance=50.0,
        min_movement_distance=2.0,
        max_missed_frames=3,
        min_trajectory_span_ratio=0.0,
    )

    events = [event for event in detector.detect() if event.centroid is not None]
    track_ids = set(event.track_id for event in events if event.track_id is not None)

    print(f"\nNo-merge test - Unique track IDs: {track_ids}")

    # Should have 2 separate tracks since they're too far apart
    assert len(track_ids) == 2, f"Expected 2 separate tracks (too far to merge), but got {len(track_ids)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
