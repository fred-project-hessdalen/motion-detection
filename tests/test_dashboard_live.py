"""Test what the live view draws and how far along it reports a build to be."""

from hessdalen.dashboard.live import Progress, Segment, Trail, extend_trails
from hessdalen.dashboard.panels import DEVIATION, RECORDING, layout
from hessdalen.domain.models import DetectedMovement, MovementEvent
from synthetic import blob_measurement


def movement(frame_number: int, track_id: int, x: float) -> DetectedMovement:
    return DetectedMovement(frame_number=frame_number, track_id=track_id, centroid=(x, 10.0), blob=blob_measurement())


def test_frames_without_movement_open_no_trail():
    trails: dict[int, Trail] = {}

    extend_trails(trails, [MovementEvent(frame_number=frame, track_id=None, centroid=None) for frame in range(4)])

    assert trails == {}


def test_each_track_keeps_its_own_trail():
    trails: dict[int, Trail] = {}

    extend_trails(trails, [movement(0, 1, 10.0), movement(0, 2, 50.0)])
    extend_trails(trails, [movement(1, 1, 14.0)])

    assert trails[1].points == [(10, 10), (14, 10)]
    assert trails[2].points == [(50, 10)]


def test_a_trail_is_drawn_on_the_frame_that_confirms_its_track():
    """The detector reports the centroids it held back when it confirms a
    track, so the whole trail arrives in one frame."""
    trails: dict[int, Trail] = {}

    extend_trails(trails, [movement(3, 1, 10.0), movement(4, 1, 20.0), movement(5, 1, 30.0)])

    assert trails[1].last_frame == 5
    assert len(trails[1].points) == 3


def test_a_track_that_goes_unmatched_keeps_the_frame_it_was_last_seen_in():
    trails: dict[int, Trail] = {}

    extend_trails(trails, [movement(7, 1, 10.0)])
    extend_trails(trails, [MovementEvent(frame_number=8, track_id=None, centroid=None)])

    assert trails[1].last_frame == 7


def test_one_panel_is_stacked_alone_and_both_are_stacked_in_order():
    assert layout("recording") == (RECORDING,)
    assert layout("deviation") == (DEVIATION,)
    assert layout("both") == (RECORDING, DEVIATION)


def test_a_segment_counts_the_frames_between_its_ends():
    assert Segment(begin_frame=500, end_frame=600).drawn_frames == 101
    assert Segment(begin_frame=0, end_frame=0).drawn_frames == 1


def test_the_last_frame_of_the_segment_fills_the_bar():
    assert Progress(frame_number=100, frame_count=101).fraction == 1.0


def test_the_first_frame_of_the_segment_opens_the_bar():
    point = Progress(frame_number=0, frame_count=101)

    assert 0.0 < point.fraction < 0.05
