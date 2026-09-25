"""How a search names one track of the corpus, and where a cluster's name
goes on the map."""

import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from hessdalen.dashboard.map_view import (
    ANY_VALIDATION,
    BY_CACHED,
    BY_CLUSTER_NAME,
    BY_DISTANCE,
    CACHED_COLUMN,
    CLUSTER_TRACKS,
    NAME_COLUMN,
    NAMES_TITLE,
    NEIGHBOUR_TRACKS,
    PATH_DRAWING,
    PICKED_TRACKS,
    POINT_COLUMNS,
    TABLE_COLUMNS,
    TAGS_COLUMN,
    TRANSFORM_DRAWING,
    UNNAMED,
    UNVALIDATED,
    VALIDATED,
    VALIDATED_COLUMN,
    Ring,
    _scatter,
    by_validation,
    charted_signal,
    cluster_names,
    in_order,
    labelled,
    named_group,
    overlapping,
    picked_tracks,
    point_traces,
    ring_traces,
    searched,
    stretches,
    table_rows,
)
from hessdalen.dashboard.track_gallery import ADD, ONE, RANGE
from hessdalen.dashboard.track_map_chart import trace_uid

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


def test_a_trace_of_points_is_written_the_way_plotly_writes_one() -> None:
    """The plot is handed traces the page writes out itself, so they have to
    hold what a figure of plotly's own would have held."""
    tracks = _drawn()

    written = point_traces(tracks, colour="Cluster", dimmed=frozenset())[0]

    expected = json.loads(
        go.Scattergl(
            x=written["x"],
            y=written["y"],
            mode="markers",
            name=written["name"],
            uid=written["uid"],
            marker=written["marker"],
            customdata=written["customdata"],
            hoverinfo="none",
        ).to_json()
    )
    assert written == expected


def test_a_point_carries_its_numbers_to_the_digits_the_panel_shows() -> None:
    written = point_traces(_drawn(), colour="Cluster", dimmed=frozenset())[0]

    straightness = POINT_COLUMNS.index("straightness")
    peak = POINT_COLUMNS.index("peak_deviation_max")
    assert [row[straightness] for row in written["customdata"]] == [0.95, 0.5]
    assert [row[peak] for row in written["customdata"]] == [45.1, 8.2]
    assert written["x"] == [1.2346, 2.0]


def test_the_marks_over_the_tracks_stand_at_the_top_of_the_legend() -> None:
    tracks = _drawn()
    rings = [Ring(name="Cluster sample", marker={"symbol": "circle-open"}, tracks=tracks.iloc[:1])]

    traces = ring_traces(rings, names=pd.DataFrame({"x": [1.0], "y": [2.0], "name": ["bird"]}))

    assert [trace["uid"] for trace in traces] == [trace_uid("Cluster sample"), trace_uid(NAMES_TITLE)]
    assert [trace["legendrank"] for trace in traces] == [0, 1]
    assert traces[0]["hoverinfo"] == "skip"


def test_a_trace_uid_holds_only_what_a_css_class_name_may() -> None:
    """Plotly looks a trace up by a class name built from its uid when the
    trace leaves the figure, so a uid with a colon in it aborts the redraw."""
    uids = [trace["uid"] for trace in point_traces(_drawn(), colour="Cluster", dimmed=frozenset())]

    assert uids
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", uid) for uid in uids)
    assert trace_uid("cluster", "bird or insect") == "cluster-bird_or_insect"


def test_a_map_of_no_marks_holds_the_points_alone() -> None:
    figure = _scatter(_drawn(), colour="Cluster", rings=[], dimmed=frozenset(), names=pd.DataFrame())

    assert [trace["type"] for trace in figure["data"]] == ["scattergl"]
    assert figure["layout"]["legend"]["title"]["text"] == "Cluster"
    assert json.dumps(figure)


def _drawn() -> pd.DataFrame:
    """Two tracks of one cluster, as the map draws them."""
    return pd.DataFrame(
        {
            "key": ["a/1", "a/2"],
            "x": np.array([1.23456789, 2.0], dtype="float32"),
            "y": np.array([3.0, 4.0], dtype="float32"),
            "cluster": ["4", "4"],
            TAGS_COLUMN: ["bird", "bird"],
            NAME_COLUMN: ["bird", "bird"],
            "side": ["smooth", "smooth"],
            "clip": ["Cam1_2025-06-03__12-40-00", "Cam1_2025-06-03__12-40-00"],
            "track_id": np.array([7484, 12], dtype="int32"),
            "frames": np.array([117, 40], dtype="int32"),
            "straightness": np.array([0.9512, 0.4999], dtype="float32"),
            "peak_deviation_max": np.array([45.14, 8.24], dtype="float32"),
            "label": ["noInsect", "noInsect"],
        }
    )


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


def test_a_track_stands_under_its_cluster_name_until_it_is_tagged() -> None:
    carried = labelled(_mapped(), clusters={"bird": ["a/1", "a/2"]}, own={})

    assert carried[TAGS_COLUMN].tolist() == ["bird", "bird", UNNAMED, UNNAMED]


def test_a_tag_given_to_one_track_stands_over_its_cluster_name() -> None:
    """A cluster holds what its descriptors group, and one track of it can be
    something else."""
    carried = labelled(_mapped(), clusters={"bird": ["a/1", "a/2"]}, own={"plane": ["a/2"]})

    assert carried[TAGS_COLUMN].tolist() == ["bird", "plane", UNNAMED, UNNAMED]
    assert carried[NAME_COLUMN].tolist() == ["bird", "bird", UNNAMED, UNNAMED]


def test_a_track_under_several_tags_carries_them_all() -> None:
    carried = labelled(_mapped(), clusters={}, own={"plane": ["a/2"], "bird": ["a/2"], "far": ["a/2"]})

    assert carried[TAGS_COLUMN].tolist() == [UNNAMED, "bird, far, plane", UNNAMED, UNNAMED]


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
    group = named_group({}, members=["a/1", "a/2"], selected="a/1", given_to=CLUSTER_TRACKS)

    assert group.moving == ("a/1", "a/2")
    assert group.given == ""


def test_naming_a_cluster_names_the_selected_track_whatever_it_stands_under() -> None:
    """The selected track is the one being looked at while the name is given,
    so the name is about it before it is about the rest of the cluster."""
    labels = {"bird": ["a/1", "a/2"], "plane": ["a/3"]}

    group = named_group(labels, members=["a/1", "a/2", "a/3"], selected="a/3", given_to=CLUSTER_TRACKS)

    assert group.moving == ("a/1", "a/2", "a/3")
    assert group.given == "bird"


def test_naming_a_cluster_leaves_the_other_tracks_under_a_name_of_their_own() -> None:
    labels = {"bird": ["a/1", "a/2"], "plane": ["a/3"]}

    group = named_group(labels, members=["a/1", "a/2", "a/3"], selected="a/1", given_to=CLUSTER_TRACKS)

    assert group.moving == ("a/1", "a/2")
    assert group.under == 2


def test_naming_the_nearest_tracks_names_the_track_they_were_drawn_around() -> None:
    """A neighbourhood is the tracks nearest the selected one, which holds none
    of the selected track itself."""
    group = named_group({"bird": ["a/2"]}, members=["a/2", "a/3"], selected="a/1", given_to=NEIGHBOUR_TRACKS)

    assert group.keys == ("a/1", "a/2", "a/3")
    assert group.moving == ("a/1", "a/2", "a/3")
    assert group.under == 1


def test_naming_the_picked_tracks_names_those_tracks_and_no_others() -> None:
    """The picked tracks were picked by hand, so every one of them takes the
    name and the selected track is left out where it was not picked."""
    labels = {"bird": ["a/2"], "plane": ["a/3"]}

    group = named_group(labels, members=["a/2", "a/3"], selected="a/1", given_to=PICKED_TRACKS)

    assert group.keys == ("a/2", "a/3")
    assert group.moving == ("a/2", "a/3")


def test_picking_no_panel_yet_leaves_no_group_to_name() -> None:
    group = named_group({}, members=[], selected="a/1", given_to=PICKED_TRACKS)

    assert group.moving == ()


def test_a_click_on_a_panel_picks_that_one_track() -> None:
    assert picked_tracks(("a/2", "a/3"), key="a/1", reach=ONE, order=GALLERY) == ("a/1",)


def test_ctrl_and_a_click_take_a_panel_into_the_picked_tracks() -> None:
    assert picked_tracks(("a/2",), key="a/4", reach=ADD, order=GALLERY) == ("a/2", "a/4")


def test_ctrl_and_a_click_on_a_picked_panel_take_it_out_again() -> None:
    assert picked_tracks(("a/2", "a/4"), key="a/2", reach=ADD, order=GALLERY) == ("a/4",)


def test_shift_and_a_click_pick_the_run_up_to_the_panel() -> None:
    assert picked_tracks(("a/2",), key="a/4", reach=RANGE, order=GALLERY) == ("a/2", "a/3", "a/4")


def test_a_run_picked_backwards_keeps_the_track_picked_first_in_front() -> None:
    """The first of the picked tracks is the one whose video plays, so a run
    picked back towards the start of a gallery keeps it there."""
    assert picked_tracks(("a/4",), key="a/2", reach=RANGE, order=GALLERY) == ("a/4", "a/3", "a/2")


def test_shift_and_a_click_with_nothing_picked_pick_the_one_panel() -> None:
    assert picked_tracks((), key="a/3", reach=RANGE, order=GALLERY) == ("a/3",)


def test_a_run_reaches_no_further_than_the_gallery_it_is_picked_in() -> None:
    """A run is picked out of one gallery, and the track picked first can be
    one another gallery drew."""
    assert picked_tracks(("b/9",), key="a/3", reach=RANGE, order=GALLERY) == ("a/3",)


GALLERY = ["a/1", "a/2", "a/3", "a/4", "a/5"]
"""Five panels as a gallery stands them."""


def test_the_nearest_tracks_of_no_selected_track_are_no_group() -> None:
    group = named_group({}, members=[], selected="", given_to=NEIGHBOUR_TRACKS)

    assert group.keys == ()
    assert group.moving == ()


def _gone_through() -> pd.DataFrame:
    return pd.DataFrame({"key": ["a/1", "a/2", "a/3"], VALIDATED_COLUMN: [False, True, False]})


def test_a_row_of_the_table_holds_the_track_under_the_names_the_page_gives_it() -> None:
    listed = table_rows(_listed())

    assert listed.columns.tolist() == list(TABLE_COLUMNS.values())
    assert listed["Recording"].tolist() == [INSECT]
    assert listed["Tags"].tolist() == ["bird"]
    assert listed["Video cached"].tolist() == [True]


def _listed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": [f"2025-06-03/{INSECT}/7484"],
            "track_id": [7484],
            "clip": [INSECT],
            TAGS_COLUMN: ["bird"],
            NAME_COLUMN: ["bird"],
            "cluster": ["4"],
            "label": ["noInsect"],
            "side": ["smooth"],
            "frames": [117],
            "straightness": [0.95],
            "peak_deviation_max": [45.1],
            CACHED_COLUMN: [True],
            VALIDATED_COLUMN: [False],
        }
    )


def test_the_tracks_running_while_this_one_ran_come_back_longest_first() -> None:
    ranges = stretches(_frames(), clip=INSECT)

    found = overlapping(_of_one_recording(), ranges=ranges, track=_of_one_recording().iloc[0])

    assert found["track_id"].tolist() == [2, 3]
    assert found["overlap"].tolist() == [51, 11]


def test_a_track_of_another_recording_is_not_running_alongside() -> None:
    """Frame numbers count from the start of each recording, so two of them
    holding the same frames share nothing."""
    ranges = stretches(_frames(), clip=INSECT)

    found = overlapping(_of_one_recording(), ranges=ranges, track=_of_one_recording().iloc[0])

    assert ROD not in found["clip"].tolist()


def test_a_track_that_had_ended_is_left_out() -> None:
    ranges = stretches(_frames(), clip=INSECT)

    found = overlapping(_of_one_recording(), ranges=ranges, track=_of_one_recording().iloc[3])

    assert found["track_id"].tolist() == [3]


def test_the_rhythm_chart_follows_the_chosen_signal() -> None:
    assert charted_signal("Wobble") == "wobble"
    assert charted_signal("Presence") == "presence"


def test_the_rhythm_chart_is_drawn_while_the_galleries_draw_paths() -> None:
    """The chart stands under the light curve whatever the galleries hold, so
    a rhythm can be read against the blob without giving up the paths."""
    assert charted_signal(PATH_DRAWING) == "brightness"
    assert charted_signal(TRANSFORM_DRAWING) == "brightness"


def _of_one_recording() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": [f"a/{INSECT}/1", f"a/{INSECT}/2", f"a/{INSECT}/3", f"a/{INSECT}/4"],
            "clip": [INSECT, INSECT, INSECT, INSECT],
            "track_id": [1, 2, 3, 4],
        }
    )


def _frames() -> pd.DataFrame:
    """Four tracks of one recording and one of another: two running together
    from the start, a third overlapping the end of both, a fourth after the
    first."""
    stretched = {1: range(100, 151), 2: range(100, 151), 3: range(140, 191), 4: range(180, 231)}
    rows = [
        {"key": f"a/{INSECT}/{track}", "clip": INSECT, "frame_number": frame}
        for track, frames in stretched.items()
        for frame in frames
    ]
    rows += [{"key": f"b/{ROD}/1", "clip": ROD, "frame_number": frame} for frame in range(100, 151)]
    return pd.DataFrame(rows)


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
