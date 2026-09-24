"""The names given to clusters, kept against the tracks that were under
them."""

import json

import pytest

from hessdalen.dashboard.cluster_labels import (
    canonical_label,
    label_of,
    near_label,
    read_labels,
    tagged,
    tags_of,
    write_labels,
    write_tags,
)

KNOWN = ["bird", "insect", "meteor", "street light", "car"]

BIRDS = ["a/Cam1_2025-06-03__12-40-00/1", "a/Cam1_2025-06-03__12-40-00/2"]
INSECTS = ["b/Cam2_2025-01-09__23-20-00/7", "b/Cam2_2025-01-09__23-20-00/9"]


def test_a_name_is_kept_against_every_track_it_was_given_to(tmp_path) -> None:
    path = tmp_path / "cluster-labels.json"

    written = write_labels(path, read_labels(path), name="bird", keys=BIRDS)

    assert written == {"bird": sorted(BIRDS)}
    assert json.loads(path.read_text()) == {"bird": sorted(BIRDS)}


def test_names_given_to_other_clusters_stand(tmp_path) -> None:
    path = tmp_path / "cluster-labels.json"
    write_labels(path, read_labels(path), name="bird", keys=BIRDS)

    written = write_labels(path, read_labels(path), name="insect", keys=INSECTS)

    assert sorted(written) == ["bird", "insect"]
    assert read_labels(path)["bird"] == sorted(BIRDS)


def test_a_cluster_named_again_moves_to_the_new_name(tmp_path) -> None:
    """A cluster stands under one name, so naming it again is a correction and
    not a second name."""
    path = tmp_path / "cluster-labels.json"
    write_labels(path, read_labels(path), name="bird", keys=BIRDS)

    written = write_labels(path, read_labels(path), name="insect", keys=BIRDS)

    assert written == {"insect": sorted(BIRDS)}


def test_an_empty_name_takes_a_cluster_out_of_the_one_it_is_under(tmp_path) -> None:
    path = tmp_path / "cluster-labels.json"
    write_labels(path, read_labels(path), name="bird", keys=BIRDS)

    written = write_labels(path, read_labels(path), name="", keys=BIRDS)

    assert written == {}
    assert read_labels(path) == {}


def test_a_cluster_is_under_the_name_its_tracks_carry(tmp_path) -> None:
    path = tmp_path / "cluster-labels.json"
    labels = write_labels(path, read_labels(path), name="bird", keys=BIRDS)

    assert label_of(labels, keys=BIRDS) == "bird"
    assert label_of(labels, keys=INSECTS) == ""


def test_a_cluster_keeps_its_name_when_one_track_is_put_under_another(tmp_path) -> None:
    """A track reassigned on its own leaves the cluster under the name the rest
    of its tracks hold."""
    path = tmp_path / "cluster-labels.json"
    labels = write_labels(path, read_labels(path), name="bird", keys=[*BIRDS, *INSECTS])
    labels = write_labels(path, labels, name="plane", keys=INSECTS[:1])

    assert label_of(labels, keys=[*BIRDS, *INSECTS]) == "bird"
    assert label_of(labels, keys=INSECTS[:1]) == "plane"


def test_no_file_yet_means_no_name_given(tmp_path) -> None:
    assert read_labels(tmp_path / "cluster-labels.json") == {}


@pytest.mark.parametrize(
    ("written", "kept"),
    [
        ("Bird", "bird"),
        ("BIRDS", "bird"),
        ("streetLight", "street light"),
        ("StreetLights", "street light"),
        ("street-lights", "street light"),
        ("  street   light  ", "street light"),
        ("insects", "insect"),
        ("flashes", "flash"),
        ("bodies", "body"),
        ("boxes", "box"),
        ("mosquitoes", "mosquito"),
        ("gas", "gas"),
        ("aircraft", "aircraft"),
        ("species", "species"),
        ("!", ""),
    ],
)
def test_a_name_is_kept_in_one_form(written: str, kept: str) -> None:
    assert canonical_label(written) == kept


def test_a_name_is_brought_to_that_form_before_it_is_written(tmp_path) -> None:
    path = tmp_path / "cluster-labels.json"

    written = write_labels(path, read_labels(path), name="Street Lights", keys=BIRDS)

    assert written == {"street light": sorted(BIRDS)}


def test_two_names_that_come_to_one_hold_their_tracks_together(tmp_path) -> None:
    """A file written before names were kept in one form carries both, and
    reading it puts the tracks of both under the one name."""
    path = tmp_path / "cluster-labels.json"
    path.write_text(json.dumps({"bird": BIRDS[:1], "Birds": BIRDS[1:]}))

    assert read_labels(path) == {"bird": sorted(BIRDS)}


@pytest.mark.parametrize(
    ("typed", "meant"),
    [
        ("brid", "bird"),
        ("insct", "insect"),
        ("meteror", "meteor"),
        ("street ligth", "street light"),
    ],
)
def test_a_mistyped_name_finds_the_name_it_was_meant_to_be(typed: str, meant: str) -> None:
    assert near_label(typed, known=KNOWN) == meant


@pytest.mark.parametrize("name", ["bird", "insect", "street light"])
def test_a_name_already_in_use_is_near_nothing(name: str) -> None:
    """The name is the one in use, so there is nothing to put to the person."""
    assert near_label(name, known=KNOWN) == ""


@pytest.mark.parametrize("name", ["plane", "cloud", "rain or snow", ""])
def test_a_name_of_its_own_is_near_nothing(name: str) -> None:
    assert near_label(name, known=KNOWN) == ""


def test_a_short_name_is_held_to_one_letter() -> None:
    """At two letters apart the short words of the vocabulary reach each other,
    so a short name is only questioned at one."""
    assert near_label("cat", known=KNOWN) == "car"
    assert near_label("bat", known=KNOWN) == ""


def test_the_nearest_of_the_names_in_use_is_the_one_put_forward() -> None:
    assert near_label("birt", known=["insect", "bind", "bird"]) == "bird"


def test_a_track_stands_under_every_tag_given_to_it(tmp_path) -> None:
    path = tmp_path / "track-labels.json"

    written = write_tags(path, read_labels(path), key=BIRDS[0], names=["bird", "Far Away"])

    assert written == {"bird": [BIRDS[0]], "far away": [BIRDS[0]]}
    assert tags_of(read_labels(path), key=BIRDS[0]) == ["bird", "far away"]


def test_tagging_a_track_again_leaves_the_tags_it_was_not_given(tmp_path) -> None:
    """The tags handed in are what the track holds afterwards, so one dropped
    from the box is dropped from the track."""
    path = tmp_path / "track-labels.json"
    write_tags(path, read_labels(path), key=BIRDS[0], names=["bird", "far away"])

    written = write_tags(path, read_labels(path), key=BIRDS[0], names=["bird"])

    assert written == {"bird": [BIRDS[0]]}


def test_tagging_one_track_leaves_the_others_where_they_are(tmp_path) -> None:
    path = tmp_path / "track-labels.json"
    write_tags(path, read_labels(path), key=BIRDS[0], names=["bird"])

    written = write_tags(path, read_labels(path), key=BIRDS[1], names=["insect"])

    assert written == {"bird": [BIRDS[0]], "insect": [BIRDS[1]]}


def test_two_tags_give_the_tracks_that_are_both_things() -> None:
    labels = {"bird": BIRDS, "far away": [BIRDS[1], INSECTS[0]]}

    assert tagged(labels, names=["bird", "far away"]) == {BIRDS[1]}
    assert tagged(labels, names=["bird"]) == set(BIRDS)
    assert tagged(labels, names=[]) == set()
