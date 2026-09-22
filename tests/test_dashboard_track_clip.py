"""What a track clip shows of its recording, and what it draws on each
frame."""

from fractions import Fraction

import av
import cv2
import numpy as np
import pytest

from hessdalen.dashboard.track_clip import (
    INDEXED_FROM,
    REPORT_EVERY,
    ClipProgress,
    PassingFrameSource,
    SeekingFrameSource,
    StoredTrack,
    Stretch,
    draw_stored_track,
    frame_index,
    stretch_around,
    stretch_source,
)

RATE = 25.0
SIZE = 8
COLOR = (0, 255, 0)
GOP = 10
CODED_FRAMES = INDEXED_FROM + 20


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


def test_a_stretch_deep_in_a_recording_is_seeked_into(tmp_path) -> None:
    video = _write_coded_video(tmp_path / "recording.mp4", frames=CODED_FRAMES)

    deep = stretch_source(video, stretch=Stretch(INDEXED_FROM, INDEXED_FROM + 4), on_progress=_nothing)
    near = stretch_source(video, stretch=Stretch(INDEXED_FROM - 1, INDEXED_FROM + 3), on_progress=_nothing)

    assert isinstance(deep, SeekingFrameSource)
    assert isinstance(near, PassingFrameSource)


def test_seeking_into_a_stretch_lands_on_the_frames_passing_reaches(tmp_path) -> None:
    """The stored track's frames are numbered by a decode from the first frame,
    so a seek that lands anywhere else draws the path on the wrong
    picture."""
    video = _write_coded_video(tmp_path / "recording.mp4", frames=CODED_FRAMES)
    stretch = Stretch(begin_frame=INDEXED_FROM, end_frame=INDEXED_FROM + 4)
    index = frame_index(video)
    assert index is not None

    passed = list(PassingFrameSource(video, stretch=stretch, on_progress=_nothing).colour_frames())
    seeked = list(SeekingFrameSource(video, stretch=stretch, index=index, on_progress=_nothing).colour_frames())

    assert len(seeked) == len(passed) == CODED_FRAMES - stretch.begin_frame
    for reached, grabbed in zip(seeked, passed):
        assert np.abs(reached.astype(np.int32) - grabbed.astype(np.int32)).mean() < 1.0


def test_a_recording_whose_frames_carry_no_stamp_is_read_from_its_first_frame(tmp_path) -> None:
    """A stream written without a container carries no stamps to tell its
    frames apart by."""
    video = _write_coded_video(tmp_path / "recording.h264", frames=40, container="h264")

    assert frame_index(video) is None
    assert isinstance(stretch_source(video, stretch=Stretch(30, 35), on_progress=_nothing), PassingFrameSource)


def _nothing(progress: ClipProgress) -> None:
    pass


def _write_coded_video(path, *, frames: int, container: str | None = None):
    """A recording coded the way the archive's are, with a keyframe every GOP
    frames, written without a container when one is named to leave its frames
    unstamped."""
    step = Fraction(1, int(RATE))
    with av.open(str(path), mode="w", format=container) as held:
        stream = held.add_stream("libx264", rate=int(RATE))
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = GOP
        stream.codec_context.time_base = step
        for index in range(frames):
            picture = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index % 256, dtype=np.uint8), format="bgr24")
            picture.pts = index
            picture.time_base = step
            for packet in stream.encode(picture):
                held.mux(packet)
        for packet in stream.encode():
            held.mux(packet)
    return path


def _write_video(path, *, frames: int):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (64, 48))
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
