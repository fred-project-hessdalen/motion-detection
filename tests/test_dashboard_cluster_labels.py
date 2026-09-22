"""The names given to clusters, kept against the tracks that were under
them."""

import json

import pytest

from hessdalen.dashboard.cluster_labels import canonical_label, label_of, read_labels, write_labels

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
