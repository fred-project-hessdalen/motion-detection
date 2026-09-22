"""How a search names one track of the corpus."""

import pandas as pd

from hessdalen.dashboard.map_view import searched

INSECT = "Cam1_2025-06-03__12-40-00_noInsect"
ROD = "Cam2_2025-01-09__23-20-00_rod"


def test_a_search_takes_the_heading_over_a_selected_track() -> None:
    """The heading names a track the way the page gives it, so it can be copied
    from one page and pasted into another."""
    found = searched(_tracks(), wanted=f"Track 7484 in {INSECT}")

    assert found["key"].tolist() == [f"2025-06-03/{INSECT}/7484"]


def test_a_number_on_its_own_finds_that_track_on_every_recording() -> None:
    found = searched(_tracks(), wanted="7484")

    assert found["clip"].tolist() == [INSECT, ROD]


def test_a_recording_on_its_own_finds_the_tracks_it_holds() -> None:
    found = searched(_tracks(), wanted="cam2")

    assert found["track_id"].tolist() == [7484]


def test_a_search_of_nothing_names_no_track() -> None:
    assert searched(_tracks(), wanted="   ").empty
    assert searched(_tracks(), wanted="track in").empty


def test_a_search_no_track_answers_names_none() -> None:
    assert searched(_tracks(), wanted=f"Track 12 in {ROD}").empty


def _tracks() -> pd.DataFrame:
    clips = [INSECT, INSECT, ROD]
    track_ids = [7484, 12, 7484]
    return pd.DataFrame(
        {
            "key": [f"{clip[5:15]}/{clip}/{track_id}" for clip, track_id in zip(clips, track_ids)],
            "clip": clips,
            "track_id": track_ids,
        }
    )
