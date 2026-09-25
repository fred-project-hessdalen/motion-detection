"""Reading the corpus recording by recording, under a reader of the tests'
own.

These need the analysis and agent groups, and are skipped where either
is not installed.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("anthropic")

from hessdalen.dashboard.cluster_labels import read_labels  # noqa: E402
from hessdalen.labelling.ledger import PROPAGATED  # noqa: E402
from hessdalen.labelling.reading import Reading  # noqa: E402
from hessdalen.labelling.readers import Proposal, RowContext  # noqa: E402
from hessdalen.labelling.service import Labelling, Reading as Verdict, Settings  # noqa: E402
from hessdalen.labelling.sheets import Layout, Sheet, sheet_path  # noqa: E402
from labelling_corpus import BIRDS, INSECTS, LONERS, corpus  # noqa: E402

SETTINGS = Settings(signature="map", flip_limit=2)
TRUTH = {**{key: "bird" for key in BIRDS}, **{key: "insect" for key in INSECTS}, **{key: "meteor" for key in LONERS}}


class TruthfulReader:
    def __init__(self) -> None:
        self.sheets_read = 0

    def read(self, sheet: Path, *, rows: Sequence[RowContext], first: int, vocabulary: dict[str, Any]) -> list[Verdict]:
        self.sheets_read += 1
        return [Verdict(key=row.key, name=TRUTH[row.key], confidence="sure", note="") for row in rows]

    def judge(self, view: Path, *, key: str, history: Sequence[str], vocabulary: dict[str, Any]) -> Verdict:
        return Verdict(key=key, name=TRUTH[key], confidence="sure", note="")

    def propose(self, sheets: Sequence[Path], *, vocabulary: dict[str, Any]) -> Proposal:
        return Proposal(name="moth", definition="", examples=[])


def _reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, sheets: int) -> Reading:
    monkeypatch.setattr(Labelling, "sheet", _sheet_of_rows)
    labelling = Labelling(corpus(tmp_path), settings=SETTINGS)
    monkeypatch.setattr(labelling, "fetch", lambda recording: _fetched(labelling, recording))
    reading = Reading(labelling, TruthfulReader(), sheets=sheets, workers=1, log=lambda line: None)
    monkeypatch.setattr(
        reading, "build", lambda batches: [_sheet_of_rows(labelling, keys, layout=None, first=1) for keys in batches]
    )
    return reading


def _fetched(labelling: Labelling, recording: str) -> Path:
    """A fetch of the tests' own, which puts an empty file where the page's
    fetch would put the recording."""
    labelling.places.fetched.mkdir(parents=True, exist_ok=True)
    landed = labelling.places.fetched / recording
    landed.touch()
    labelling.tracks.loc[labelling.tracks["recording"] == recording, "cached"] = True
    return landed


def _sheet_of_rows(self: Labelling, keys: Sequence[str], *, layout: Layout | None, first: int) -> Sheet:
    path = sheet_path(self.sheets_dir, keys=keys, captions=[], layout=Layout(frames=0, radii=0, side=0), kind="rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps({"keys": list(keys), "layout": {}}))
    return Sheet(path=path, keys=tuple(keys))


def test_the_recordings_off_disk_come_with_the_most_smooth_tracks_first(tmp_path: Path) -> None:
    labelling = Labelling(corpus(tmp_path), settings=SETTINGS)

    (candidate,) = labelling.smooth_recordings()

    assert candidate.recording == "rec9.mkv"
    assert candidate.uncertain == 2


def test_a_fetched_recording_is_read_and_let_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reading = _reading(tmp_path, monkeypatch, sheets=10)

    report = reading.run()

    labels = read_labels(reading.labelling.places.labels)
    assert labels == {"meteor": sorted(LONERS)}
    assert report.recordings == ["rec9.mkv"]
    assert report.tracks == 2
    assert not (reading.labelling.places.fetched / "rec9.mkv").exists()
    assert not bool(reading.labelling.tracks.loc[reading.labelling.tracks["key"] == LONERS[0], "cached"].iloc[0])


def test_the_sheet_budget_stops_the_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reading = _reading(tmp_path, monkeypatch, sheets=0)

    report = reading.run()

    assert report.stopped == "sheet budget spent"
    assert report.recordings == []


def test_a_rough_cluster_takes_the_name_its_seen_tracks_agree_on(tmp_path: Path) -> None:
    labelling = Labelling(corpus(tmp_path), settings=SETTINGS)
    labelling.tracks.loc[labelling.tracks["cluster"] == 0, "side"] = "rough"
    sheet = _sheet_of_rows(labelling, BIRDS[:6], layout=None, first=1)
    readings = [
        Verdict(key=key, name="bird" if key != BIRDS[0] else "insect", confidence="sure", note="") for key in BIRDS[:6]
    ]
    labelling.verdict(readings, sheet=sheet, round=1)

    reached = labelling.name_rough_clusters(agreement=5, of=6, round=2)

    assert reached == {"bird": 4}
    assert read_labels(labelling.places.labels)["bird"] == sorted([*BIRDS[1:6], *BIRDS[6:]])
    assert read_labels(labelling.places.labels)["insect"] == [BIRDS[0]]
    line = labelling.ledger_lines(key=BIRDS[9], last=0)[-1]
    assert line.basis == PROPAGATED
    assert line.evidence == {"cluster": 0, "seen": 6, "share": 5}


def test_a_rough_cluster_short_of_agreement_or_of_seen_tracks_is_left(tmp_path: Path) -> None:
    labelling = Labelling(corpus(tmp_path), settings=SETTINGS)
    labelling.tracks.loc[labelling.tracks["cluster"].isin([0, 1]), "side"] = "rough"
    sheet = _sheet_of_rows(labelling, [*BIRDS[:6], *INSECTS[:3]], layout=None, first=1)
    readings = [
        Verdict(key=key, name="bird" if index % 2 else "insect", confidence="sure", note="")
        for index, key in enumerate(BIRDS[:6])
    ]
    readings += [Verdict(key=key, name="insect", confidence="sure", note="") for key in INSECTS[:3]]
    labelling.verdict(readings, sheet=sheet, round=1)

    assert labelling.name_rough_clusters(agreement=5, of=6, round=2) == {}


def test_the_keys_to_read_are_the_smooth_tracks_and_a_handful_of_rough_ones(tmp_path: Path) -> None:
    labelling = Labelling(corpus(tmp_path), settings=SETTINGS)
    labelling.tracks.loc[labelling.tracks["cluster"] == 0, "side"] = "rough"

    keys = labelling.keys_to_read("rec0.mkv", rough_seen=2)

    assert keys == [key for key in BIRDS if key.split("/")[1] == "rec0"][:2]
    assert labelling.keys_to_read("rec3.mkv", rough_seen=2) == [key for key in INSECTS if key.split("/")[1] == "rec3"]
