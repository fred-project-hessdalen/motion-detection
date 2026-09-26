"""The ledger of labelling decisions and what follows from it."""

import json
from pathlib import Path

from hessdalen.labelling.ledger import (
    JUDGED,
    PREDICTED,
    PROPAGATED,
    SEEN,
    Line,
    append,
    contested,
    derive,
    read,
)


def _line(*, key: str, basis: str, name: str, previous: str = "", round: int = 1) -> Line:
    return Line(round=round, key=key, basis=basis, name=name, previous=previous, confidence="sure", note="")


def test_a_line_written_is_read_back_with_the_moment_it_was_written(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"

    written = append(path, _line(key="a/1", basis=SEEN, name="bird"))

    assert written.time
    assert read(path) == [written]


def test_a_half_written_last_line_is_passed_over(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = append(path, _line(key="a/1", basis=SEEN, name="bird"))
    with path.open("a") as ledger:
        ledger.write(json.dumps({"round": 1, "key": "a/2"})[:-3])

    assert read(path) == [first]


def test_a_seen_track_carries_its_latest_verdict() -> None:
    state = derive(
        [
            _line(key="a/1", basis=SEEN, name="bird"),
            _line(key="a/1", basis=JUDGED, name="insect", previous="bird", round=3),
            _line(key="a/2", basis=PROPAGATED, name="bird"),
        ]
    )

    assert set(state.seen) == {"a/1"}
    assert state.seen["a/1"].name == "insect"
    assert state.seen["a/1"].round == 3
    assert state.rounds == 3


def test_a_track_seen_without_a_name_is_still_seen() -> None:
    state = derive([_line(key="a/1", basis=SEEN, name="")])

    assert state.seen["a/1"].name == ""


def test_the_tracks_named_by_a_model_are_the_ones_its_lines_last_named() -> None:
    from hessdalen.labelling.ledger import named_by_model

    named = named_by_model(
        [
            _line(key="a/1", basis=SEEN, name="bird"),
            _line(key="a/2", basis=PROPAGATED, name="bird"),
            _line(key="a/3", basis=SEEN, name=""),
            _line(key="a/4", basis=PREDICTED, name="bird"),
            _line(key="a/5", basis=SEEN, name="bird"),
            _line(key="a/5", basis=JUDGED, name="", previous="bird"),
        ]
    )

    assert named == frozenset({"a/1", "a/2"})


def test_a_first_naming_is_no_flip_and_a_change_of_name_is() -> None:
    state = derive(
        [
            _line(key="a/1", basis=PROPAGATED, name="bird"),
            _line(key="a/1", basis=PROPAGATED, name="insect", previous="bird"),
            _line(key="a/1", basis=SEEN, name="bird", previous="insect"),
            _line(key="a/2", basis=PREDICTED, name="insect", previous="bird"),
        ]
    )

    assert state.flips == {"a/1": 2}
    assert contested(state, limit=2) == frozenset({"a/1"})
    assert contested(state, limit=3) == frozenset()
