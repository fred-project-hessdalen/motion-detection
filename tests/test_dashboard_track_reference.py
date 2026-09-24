"""What a track's reference says to someone who has only the link."""

from hessdalen.dashboard.track_reference import NOMINAL_RATE, clock, reference, reference_line, references_csv

URL = "https://drive.google.com/file/d/abc123/view"


def test_a_reference_turns_the_frames_into_seconds_of_the_recording() -> None:
    held = _reference(first_frame=9257, last_frame=9349, frames_per_second=25.0)

    assert (round(held.begin_s, 1), round(held.end_s, 1)) == (370.3, 374.0)
    assert held.measured_rate


def test_a_reference_falls_back_on_the_nominal_rate_and_says_so() -> None:
    held = _reference(first_frame=250, last_frame=500, frames_per_second=0.0)

    assert (held.begin_s, held.end_s) == (250 / NOMINAL_RATE, 500 / NOMINAL_RATE)
    assert not held.measured_rate


def test_a_reference_reads_as_a_link_and_a_stretch_of_the_clock() -> None:
    held = _reference(first_frame=9257, last_frame=9349, frames_per_second=25.0)

    assert reference_line(held) == f"{URL} 6:10.3-6:14.0"


def test_a_reference_without_a_link_names_the_recording_instead() -> None:
    held = _reference(first_frame=0, last_frame=25, frames_per_second=25.0, url="")

    assert reference_line(held) == "Cam5_2026-07-25__11-40-00_UTC.mkv 0:00.0-0:01.0"


def test_the_clock_carries_hours_only_where_there_are_any() -> None:
    assert clock(0.0) == "0:00.0"
    assert clock(95.4) == "1:35.4"
    assert clock(3725.6) == "1:02:05.6"


def test_every_tagged_track_is_a_row_carrying_its_tags_and_its_link() -> None:
    rows = references_csv([_reference(first_frame=9257, last_frame=9349, frames_per_second=25.0)]).splitlines()

    assert rows[0].startswith("recording,track,tags,")
    assert rows[1] == f"Cam5_2026-07-25__11-40-00_UTC.mkv,48491,bird far away,370.3,374.0,6:10.3,6:14.0,measured,{URL}"


def test_a_row_says_where_the_seconds_stand_on_the_nominal_rate() -> None:
    rows = references_csv([_reference(first_frame=25, last_frame=50, frames_per_second=0.0)]).splitlines()

    assert f"{NOMINAL_RATE:.0f} fps assumed" in rows[1]


def _reference(*, first_frame: int, last_frame: int, frames_per_second: float, url: str = URL):
    return reference(
        recording="Cam5_2026-07-25__11-40-00_UTC.mkv",
        track_id=48491,
        url=url,
        first_frame=first_frame,
        last_frame=last_frame,
        frames_per_second=frames_per_second,
        tags=("bird", "far away"),
    )
