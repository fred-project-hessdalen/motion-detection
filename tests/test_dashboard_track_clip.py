"""What a track clip shows of its recording, and what it draws on each
frame."""

import cv2
import numpy as np
import pytest

from hessdalen.dashboard.track_clip import (
    REPORT_EVERY,
    ClipProgress,
    PassingFrameSource,
    StoredTrack,
    Stretch,
    draw_stored_track,
    stretch_around,
)

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


def test_a_build_counts_every_frame_up_to_the_end_of_its_stretch() -> None:
    """The frames ahead of the stretch are passed and the stretch is drawn, and
    a build handles both."""
    stretch = Stretch(begin_frame=10_000, end_frame=10_102)

    reaching = ClipProgress(frames_done=5_000, stretch=stretch)
    drawing = ClipProgress(frames_done=10_050, stretch=stretch)

    assert reaching.frames_total == 10_103
    assert not reaching.drawing
    assert drawing.drawing
    assert reaching.fraction == pytest.approx(5_000 / 10_103)
    assert stretch.drawn_frames == 103


def test_passing_the_frames_ahead_of_a_stretch_reports_as_it_goes(tmp_path) -> None:
    video = _write_video(tmp_path / "recording.avi", frames=4 * REPORT_EVERY + 10)
    stretch = Stretch(begin_frame=4 * REPORT_EVERY, end_frame=4 * REPORT_EVERY + 5)
    reports: list[ClipProgress] = []

    frames = list(PassingFrameSource(video, stretch=stretch, on_progress=reports.append).colour_frames())

    assert [report.frames_done for report in reports] == [REPORT_EVERY * n for n in range(1, 5)]
    assert len(frames) == 10


def test_the_shape_is_read_from_the_start_without_passing_anything(tmp_path) -> None:
    video = _write_video(tmp_path / "recording.avi", frames=40)
    reports: list[ClipProgress] = []

    sample = PassingFrameSource(video, stretch=Stretch(30, 35), on_progress=reports.append).sample_frame()

    assert sample is not None
    assert reports == []


def _write_video(path, *, frames: int):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
    writer.release()
    return path


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
