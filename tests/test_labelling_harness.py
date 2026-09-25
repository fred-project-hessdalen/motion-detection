"""The loop that names the corpus with a reader in it.

These need the analysis and agent groups, and are skipped where either
is not installed. The reader is one of the tests' own, and a sheet is
written as its rows alone, since no recording is drawn.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("anthropic")

from hessdalen.dashboard.cluster_labels import read_labels, write_labels  # noqa: E402
from hessdalen.dashboard.track_validation import write_validated  # noqa: E402
from hessdalen.labelling.harness import Budget, Harness, Thresholds  # noqa: E402
from hessdalen.labelling.readers import Proposal, RowContext  # noqa: E402
from hessdalen.labelling.service import NONE_OF_THESE, Labelling, Reading, Settings  # noqa: E402
from hessdalen.labelling.sheets import Layout, Sheet, sheet_path  # noqa: E402
from hessdalen.labelling.votes import Rule  # noqa: E402
from labelling_corpus import BIRDS, INSECTS, corpus  # noqa: E402

SETTINGS = Settings(signature="map", flip_limit=2)
RULE = Rule(votes=3, agreement=2, cap=1.5)
THRESHOLDS = Thresholds(calibration=0.8, verification=0.8)
TRUTH = {**{key: "bird" for key in BIRDS}, **{key: "insect" for key in INSECTS}}


class TruthfulReader:
    """A reader that knows what every track is."""

    def __init__(self, truth: dict[str, str]) -> None:
        self.truth = truth
        self.sheets_read = 0

    def read(self, sheet: Path, *, rows: Sequence[RowContext], vocabulary: dict[str, Any]) -> list[Reading]:
        self.sheets_read += 1
        return [
            Reading(key=row.key, name=self.truth.get(row.key, NONE_OF_THESE), confidence="sure", note="")
            for row in rows
        ]

    def judge(self, view: Path, *, key: str, history: Sequence[str], vocabulary: dict[str, Any]) -> Reading:
        return Reading(key=key, name=self.truth.get(key, "review"), confidence="sure", note="")

    def propose(self, sheets: Sequence[Path], *, vocabulary: dict[str, Any]) -> Proposal:
        return Proposal(name="moth", definition="a moth at the lens", examples=[])


def _harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reader: TruthfulReader, sheets: int) -> Harness:
    monkeypatch.setattr(Labelling, "sheet", _sheet_of_rows)
    monkeypatch.setattr(Labelling, "isolation", lambda self, key: _sheet_of_rows(self, [key], layout=None))
    labelling = Labelling(corpus(tmp_path), settings=SETTINGS)
    return Harness(
        labelling,
        reader,
        rule=RULE,
        budget=Budget(sheets=sheets, fetches=0),
        thresholds=THRESHOLDS,
        log=lambda line: None,
    )


def _sheet_of_rows(self: Labelling, keys: Sequence[str], *, layout: Layout | None) -> Sheet:
    """A sheet as its rows alone, written beside where the picture would
    be."""
    path = sheet_path(self.sheets_dir, keys=keys, layout=Layout(frames=0, radii=0, side=0), kind="rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps({"keys": list(keys), "layout": {}}))
    return Sheet(path=path, keys=tuple(keys))


def _validated(harness: Harness, keys: list[str], *, name: str) -> None:
    """Tracks a person named and confirmed, the ground truth calibration
    reads against."""
    write_labels(harness.labelling.places.labels, name=name, keys=keys)
    for key in keys:
        write_validated(harness.labelling.places.validated, key=key, confirmed=True)


def test_the_loop_names_every_track_on_disk_and_leaves_the_far_ones(tmp_path: Path, monkeypatch) -> None:
    reader = TruthfulReader(TRUTH)
    harness = _harness(tmp_path, monkeypatch, reader=reader, sheets=20)
    _validated(harness, BIRDS[:6], name="bird")

    report = harness.run()

    labels = read_labels(harness.labelling.places.labels)
    assert labels == {"bird": sorted(BIRDS), "insect": sorted(INSECTS)}
    assert report.stopped == ""
    assert report.sheets == reader.sheets_read
    assert report.status is not None
    assert report.status.far == 0
    assert report.status.review == 0


def test_the_loop_stops_when_the_reader_disagrees_with_the_validated_tracks(tmp_path: Path, monkeypatch) -> None:
    reader = TruthfulReader(TRUTH)
    harness = _harness(tmp_path, monkeypatch, reader=reader, sheets=20)
    _validated(harness, BIRDS[:6], name="insect")

    report = harness.run()

    assert report.calibration == 0.0
    assert report.stopped == "calibration under the threshold"
    assert reader.sheets_read == 1
    assert read_labels(harness.labelling.places.labels) == {"insect": sorted(BIRDS[:6])}


def test_a_reading_agrees_with_a_validated_track_through_its_tags_too(tmp_path: Path, monkeypatch) -> None:
    reader = TruthfulReader(TRUTH)
    harness = _harness(tmp_path, monkeypatch, reader=reader, sheets=1)
    _validated(harness, BIRDS[:6], name="clutter")
    for key in BIRDS[:6]:
        harness.labelling.tag(key, tags=["bird", "far away"], round=0)

    assert harness.calibrate() == 1.0


def test_fewer_validated_tracks_than_a_sheet_is_no_calibration(tmp_path: Path, monkeypatch) -> None:
    reader = TruthfulReader(TRUTH)
    harness = _harness(tmp_path, monkeypatch, reader=reader, sheets=20)
    _validated(harness, BIRDS[:2], name="bird")

    report = harness.run()

    assert report.stopped == "calibration under the threshold"
    assert reader.sheets_read == 0


def test_the_sheet_budget_ends_the_loop(tmp_path: Path, monkeypatch) -> None:
    reader = TruthfulReader(TRUTH)
    harness = _harness(tmp_path, monkeypatch, reader=reader, sheets=2)
    _validated(harness, BIRDS[:6], name="bird")

    report = harness.run()

    assert report.sheets == 2


def test_the_cap_moves_to_where_verified_readings_stop_agreeing(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(tmp_path, monkeypatch, reader=TruthfulReader(TRUTH), sheets=0)
    harness.verified = [(0.1, True), (0.2, True), (0.3, True), (0.4, True), (0.5, False), (0.6, False), (0.7, False)]

    assert harness.cap() == pytest.approx(0.5)


def test_the_cap_stands_while_fewer_than_a_sheet_have_been_verified(tmp_path: Path, monkeypatch) -> None:
    harness = _harness(tmp_path, monkeypatch, reader=TruthfulReader(TRUTH), sheets=0)
    harness.verified = [(0.1, False), (0.2, False)]

    assert harness.cap() == RULE.cap
