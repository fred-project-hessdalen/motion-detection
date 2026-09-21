"""What a track clip shows of its recording, and what it draws on each
frame."""

import numpy as np

from hessdalen.dashboard.track_clip import StoredTrack, Stretch, draw_stored_track, stretch_around

RATE = 25.0
SIZE = 8
COLOR = (0, 255, 0)


def test_a_clip_shows_a_second_either_side_of_its_track() -> None:
    stretch = stretch_around(_track(frames=[100, 125, 150]), frames_per_second=RATE, frame_count=1500)

    assert stretch == Stretch(begin_frame=75, end_frame=175)


def test_a_clip_stops_at_either_end_of_its_recording() -> None:
    early = stretch_around(_track(frames=[10, 20]), frames_per_second=RATE, frame_count=1500)
    late = stretch_around(_track(frames=[1480, 1495]), frames_per_second=RATE, frame_count=1500)

    assert early.begin_frame == 0
    assert late.end_frame == 1499


def test_nothing_is_drawn_before_the_track_starts() -> None:
    canvas = _canvas()

    draw_stored_track(canvas, track=_track(frames=[5, 6, 9]), frame_number=4, color=COLOR, size=SIZE)

    assert not canvas.any()


def test_the_box_marks_the_track_on_a_frame_it_was_matched_on() -> None:
    canvas = _canvas()

    draw_stored_track(canvas, track=_track(frames=[5, 6, 9]), frame_number=6, color=COLOR, size=SIZE)

    x, y = _position(6)
    assert canvas[y - SIZE, x - SIZE].any()


def test_between_matches_the_path_is_drawn_and_no_box() -> None:
    """A track may go unmatched for a few frames and carry on, and on those
    frames there is no position to box."""
    canvas = _canvas()

    draw_stored_track(canvas, track=_track(frames=[5, 6, 9]), frame_number=7, color=COLOR, size=SIZE)

    x, y = _position(6)
    assert canvas[y, x].any()
    assert not canvas[y - SIZE, x - SIZE].any()


def _track(*, frames: list[int]) -> StoredTrack:
    numbers = np.array(frames, dtype=np.int64)
    return StoredTrack(
        track_id=1,
        frame_numbers=numbers,
        x=np.array([_position(frame)[0] for frame in frames], dtype=np.float32),
        y=np.array([_position(frame)[1] for frame in frames], dtype=np.float32),
        frame_height=240,
    )


def _position(frame: int) -> tuple[int, int]:
    return 40 + frame * 10, 120


def _canvas() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)
