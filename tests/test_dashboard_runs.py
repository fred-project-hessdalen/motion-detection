"""Test how detector events are gathered into the trajectories the dashboard
draws."""

from hessdalen.dashboard.runs import trajectories_from_events
from hessdalen.domain.models import DetectedMovement, MovementEvent
from synthetic import blob_measurement


def movement(frame_number: int, track_id: int, x: float) -> DetectedMovement:
    return DetectedMovement(frame_number=frame_number, track_id=track_id, centroid=(x, 10.0), blob=blob_measurement())


def test_frames_without_movement_open_no_trajectory():
    events = [MovementEvent(frame_number=frame, track_id=None, centroid=None) for frame in range(4)]

    assert trajectories_from_events(events) == ()


def test_each_track_becomes_one_trajectory_ordered_by_track_id():
    events = [movement(0, 2, 10.0), movement(0, 1, 50.0), movement(1, 2, 12.0)]

    trajectories = trajectories_from_events(events)

    assert [trajectory.track_id for trajectory in trajectories] == [1, 2]


def test_replayed_events_are_put_back_into_frame_order():
    """A track is confirmed some frames after it starts, and the detector then
    replays the centroids it held back."""
    events = [movement(5, 1, 30.0), movement(3, 1, 10.0), movement(4, 1, 20.0)]

    trajectory = trajectories_from_events(events)[0]

    assert [point.frame_number for point in trajectory.points] == [3, 4, 5]
    assert trajectory.first_frame == 3
    assert trajectory.last_frame == 5


def test_the_span_is_the_larger_side_of_the_box_the_points_fit_into():
    events = [movement(0, 1, 10.0), movement(1, 1, 90.0)]

    assert trajectories_from_events(events)[0].span == 80.0
