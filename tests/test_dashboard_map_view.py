"""How a search names one track of the corpus, and where a cluster's name
goes on the map."""

import pandas as pd

from hessdalen.dashboard.map_view import cluster_names, searched

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


def test_a_name_stands_over_the_middle_of_its_cluster() -> None:
    placed = cluster_names(_mapped(), labels={"bird": ["a/1", "a/2", "a/3"]})

    assert placed.loc["4", "name"] == "bird"
    assert (placed.loc["4", "x"], placed.loc["4", "y"]) == (2.0, 20.0)


def test_a_cluster_of_more_than_one_name_takes_the_name_most_of_it_is_under() -> None:
    placed = cluster_names(_mapped(), labels={"bird": ["a/1", "a/2"], "insect": ["a/3"]})

    assert placed.loc["4", "name"] == "bird"


def test_a_cluster_no_name_was_given_to_carries_none() -> None:
    placed = cluster_names(_mapped(), labels={"bird": ["a/9"]})

    assert placed.empty
    assert placed.columns.tolist() == ["x", "y", "name"]


def test_the_tracks_of_no_cluster_carry_no_name() -> None:
    """A point the clustering left out is one track, and a name is about what a
    cluster holds."""
    placed = cluster_names(_mapped(), labels={"bird": ["a/4"]})

    assert placed.empty


def _mapped() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": ["a/1", "a/2", "a/3", "a/4"],
            "cluster": ["4", "4", "4", "none"],
            "x": [1.0, 2.0, 9.0, 50.0],
            "y": [10.0, 20.0, 90.0, 50.0],
        }
    )


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
