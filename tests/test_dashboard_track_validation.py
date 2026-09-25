"""How the tracks a person has confirmed are kept on disk."""

from pathlib import Path

from hessdalen.dashboard.track_validation import read_validated, write_validated


def test_no_track_is_confirmed_before_any_was(tmp_path: Path) -> None:
    assert read_validated(tmp_path / "validated-tracks.json") == frozenset()


def test_a_confirmed_track_is_there_again_when_the_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / "held" / "validated-tracks.json"

    write_validated(path, key="a/1", confirmed=True)

    assert read_validated(path) == frozenset({"a/1"})


def test_confirming_a_track_leaves_the_tracks_already_confirmed(tmp_path: Path) -> None:
    path = tmp_path / "validated-tracks.json"
    write_validated(path, key="a/1", confirmed=True)

    written = write_validated(path, key="a/2", confirmed=True)

    assert written == frozenset({"a/1", "a/2"})
    assert read_validated(path) == written


def test_a_track_taken_back_is_no_longer_confirmed(tmp_path: Path) -> None:
    path = tmp_path / "validated-tracks.json"
    write_validated(path, key="a/1", confirmed=True)
    write_validated(path, key="a/2", confirmed=True)

    written = write_validated(path, key="a/1", confirmed=False)

    assert written == frozenset({"a/2"})
    assert read_validated(path) == written


def test_taking_back_a_track_that_was_never_confirmed_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "validated-tracks.json"
    write_validated(path, key="a/2", confirmed=True)

    assert write_validated(path, key="a/1", confirmed=False) == frozenset({"a/2"})
