"""How a search names one track of the corpus, and where a cluster's name
goes on the map."""

import pandas as pd

from hessdalen.dashboard.map_view import (
    ANY_VALIDATION,
    BY_CACHED,
    BY_CLUSTER_NAME,
    BY_DISTANCE,
    CACHED_COLUMN,
    NAME_COLUMN,
    TRACK_NAME_COLUMN,
    UNNAMED,
    UNVALIDATED,
    VALIDATED,
    VALIDATED_COLUMN,
    by_validation,
    cluster_names,
    in_order,
    labelled,
    named_group,
    searched,
)

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


def test_every_track_carries_the_name_of_the_cluster_it_is_in() -> None:
    carried = labelled(_mapped(), clusters={"bird": ["a/1", "a/3"]}, own={})

    assert carried[NAME_COLUMN].tolist() == ["bird", UNNAMED, "bird", UNNAMED]


def test_a_track_stands_under_its_cluster_name_until_it_is_given_one() -> None:
    carried = labelled(_mapped(), clusters={"bird": ["a/1", "a/2"]}, own={})

    assert carried[TRACK_NAME_COLUMN].tolist() == ["bird", "bird", UNNAMED, UNNAMED]


def test_a_name_given_to_one_track_stands_over_its_cluster_name() -> None:
    """A cluster holds what its descriptors group, and one track of it can be
    something else."""
    carried = labelled(_mapped(), clusters={"bird": ["a/1", "a/2"]}, own={"plane": ["a/2"]})

    assert carried[TRACK_NAME_COLUMN].tolist() == ["bird", "plane", UNNAMED, UNNAMED]
    assert carried[NAME_COLUMN].tolist() == ["bird", "bird", UNNAMED, UNNAMED]


def test_every_track_is_on_the_map_whichever_of_them_was_confirmed() -> None:
    assert by_validation(_gone_through(), choice=ANY_VALIDATION)["key"].tolist() == ["a/1", "a/2", "a/3"]


def test_the_map_holds_to_the_tracks_someone_confirmed() -> None:
    assert by_validation(_gone_through(), choice=VALIDATED)["key"].tolist() == ["a/2"]


def test_the_map_holds_to_the_tracks_still_to_go_through() -> None:
    assert by_validation(_gone_through(), choice=UNVALIDATED)["key"].tolist() == ["a/1", "a/3"]


def test_a_gallery_stands_nearest_first_until_another_order_is_picked() -> None:
    assert in_order(_gathered(), order=BY_DISTANCE)["key"].tolist() == ["a/1", "a/2", "a/3", "a/4"]


def test_the_tracks_whose_recording_is_on_disk_come_first() -> None:
    assert in_order(_gathered(), order=BY_CACHED)["key"].tolist() == ["a/2", "a/4", "a/1", "a/3"]


def test_the_tracks_of_one_name_stand_together_with_the_unnamed_last() -> None:
    assert in_order(_gathered(), order=BY_CLUSTER_NAME)["key"].tolist() == ["a/2", "a/3", "a/1", "a/4"]


def _gathered() -> pd.DataFrame:
    """Four tracks as a gallery hands them over, nearest first."""
    return pd.DataFrame(
        {
            "key": ["a/1", "a/2", "a/3", "a/4"],
            NAME_COLUMN: ["meteor", "bird", "bird", UNNAMED],
            CACHED_COLUMN: [False, True, False, True],
        }
    )


def test_naming_a_cluster_names_every_track_of_it_that_has_no_name() -> None:
    group = named_group({}, members=["a/1", "a/2"], selected="a/1", neighbours=False)

    assert group.moving == ("a/1", "a/2")
    assert group.given == ""


def test_naming_a_cluster_names_the_selected_track_whatever_it_stands_under() -> None:
    """The selected track is the one being looked at while the name is given,
    so the name is about it before it is about the rest of the cluster."""
    labels = {"bird": ["a/1", "a/2"], "plane": ["a/3"]}

    group = named_group(labels, members=["a/1", "a/2", "a/3"], selected="a/3", neighbours=False)

    assert group.moving == ("a/1", "a/2", "a/3")
    assert group.given == "bird"


def test_naming_a_cluster_leaves_the_other_tracks_under_a_name_of_their_own() -> None:
    labels = {"bird": ["a/1", "a/2"], "plane": ["a/3"]}

    group = named_group(labels, members=["a/1", "a/2", "a/3"], selected="a/1", neighbours=False)

    assert group.moving == ("a/1", "a/2")
    assert group.under == 2


def test_naming_the_nearest_tracks_names_the_track_they_were_drawn_around() -> None:
    """A neighbourhood is the tracks nearest the selected one, which holds none
    of the selected track itself."""
    group = named_group({"bird": ["a/2"]}, members=["a/2", "a/3"], selected="a/1", neighbours=True)

    assert group.keys == ("a/1", "a/2", "a/3")
    assert group.moving == ("a/1", "a/2", "a/3")
    assert group.under == 1


def test_the_nearest_tracks_of_no_selected_track_are_no_group() -> None:
    group = named_group({}, members=[], selected="", neighbours=True)

    assert group.keys == ()
    assert group.moving == ()


def _gone_through() -> pd.DataFrame:
    return pd.DataFrame({"key": ["a/1", "a/2", "a/3"], VALIDATED_COLUMN: [False, True, False]})


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
