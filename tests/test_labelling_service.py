"""What the labelling reads, builds and writes.

These need the analysis group, and are skipped where it is not
installed.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from hessdalen.dashboard.cluster_labels import read_labels, write_labels  # noqa: E402
from hessdalen.dashboard.track_validation import write_validated  # noqa: E402
from hessdalen.labelling.ledger import PROPAGATED, SEEN  # noqa: E402
from hessdalen.labelling.service import (  # noqa: E402
    NONE_OF_THESE,
    REVIEW_TAG,
    Labelling,
    Reading,
    Settings,
)
from hessdalen.labelling.sheets import SHEET, Sheet  # noqa: E402
from hessdalen.labelling.votes import DISAGREE, FAR, Rule  # noqa: E402
from labelling_corpus import BIRDS, INSECTS, LONERS, corpus  # noqa: E402

SETTINGS = Settings(signature="map", flip_limit=2)
RULE = Rule(votes=3, agreement=2, cap=1.5)


def _labelling(tmp_path: Path) -> Labelling:
    return Labelling(corpus(tmp_path), settings=SETTINGS)


def _sheet_of(keys: list[str]) -> Sheet:
    return Sheet(path=Path("sheets/one.png"), keys=tuple(keys))


def _seen(labelling: Labelling, keys: list[str], *, name: str, round: int = 1) -> None:
    readings = [Reading(key=key, name=name, confidence="sure", note="") for key in keys]
    labelling.verdict(readings, sheet=_sheet_of(keys), round=round)


def test_the_status_counts_the_corpus_and_the_names(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    write_labels(labelling.places.labels, name="bird", keys=BIRDS[:2])

    status = labelling.status(rule=RULE)

    assert status.tracks == 18
    assert status.names == {"bird": 2}
    assert status.cached == 16
    assert status.seen == 0
    assert status.far == 16


def test_clusters_come_with_the_most_unnamed_tracks_on_disk_first(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    write_labels(labelling.places.labels, name="bird", keys=BIRDS[:8])

    clusters = labelling.clusters()

    assert [summary.cluster for summary in clusters] == [1, 0, -1]
    assert clusters[1].names == {"bird": 8}
    assert clusters[1].unnamed_cached == 2
    assert clusters[1].recordings == 3


def test_a_sample_spreads_over_the_core_the_rim_and_the_recordings(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)

    sample = labelling.sample(0, count=4)

    recordings = {key.split("/")[1] for key in sample}
    assert len(sample) == 4
    assert len(recordings) == 3
    assert BIRDS[4] in sample or BIRDS[5] in sample
    assert BIRDS[0] in sample or BIRDS[9] in sample


def test_a_sample_leaves_out_seen_and_validated_tracks(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    write_validated(labelling.places.validated, key=BIRDS[0], confirmed=True)
    _seen(labelling, [BIRDS[1]], name="bird")

    sample = labelling.sample(0, count=10)

    assert BIRDS[0] not in sample
    assert BIRDS[1] not in sample
    assert len(sample) == 8


def test_a_verdict_in_the_vocabulary_names_the_track_and_leaves_a_line(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    write_labels(labelling.places.labels, name="bird", keys=BIRDS[:1])
    labelling.vocabulary()

    written = labelling.verdict(
        [Reading(key=BIRDS[1], name="Birds", confidence="sure", note="wings")],
        sheet=_sheet_of([BIRDS[1]]),
        round=1,
    )

    assert written.changed == [BIRDS[1]]
    assert read_labels(labelling.places.labels) == {"bird": sorted(BIRDS[:2])}
    line = labelling.ledger_lines(key=BIRDS[1], last=0)[-1]
    assert line.basis == SEEN
    assert line.name == "bird"
    assert line.evidence["row"] == 0


def test_a_verdict_outside_the_vocabulary_or_on_a_validated_track_is_refused(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    write_labels(labelling.places.labels, name="bird", keys=BIRDS[:1])
    write_validated(labelling.places.validated, key=BIRDS[0], confirmed=True)

    written = labelling.verdict(
        [
            Reading(key=BIRDS[0], name="bird", confidence="sure", note=""),
            Reading(key=BIRDS[1], name="dragon", confidence="sure", note=""),
        ],
        sheet=_sheet_of(BIRDS[:2]),
        round=1,
    )

    assert written.changed == []
    assert [refusal.key for refusal in written.refused] == BIRDS[:2]
    assert labelling.state.seen == {}


def test_an_unsure_verdict_or_none_of_these_is_seen_without_a_name(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    write_labels(labelling.places.labels, name="bird", keys=BIRDS[:1])

    labelling.verdict(
        [
            Reading(key=BIRDS[1], name="bird", confidence="unsure", note=""),
            Reading(key=BIRDS[2], name=NONE_OF_THESE, confidence="sure", note=""),
        ],
        sheet=_sheet_of(BIRDS[1:3]),
        round=1,
    )

    assert read_labels(labelling.places.labels) == {"bird": BIRDS[:1]}
    assert labelling.state.seen[BIRDS[1]].name == ""
    assert labelling.state.seen[BIRDS[2]].name == ""


def test_a_name_spreads_to_the_neighbours_of_seen_tracks_and_no_further(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    write_labels(labelling.places.labels, name="bird", keys=BIRDS[:1])
    write_labels(labelling.places.labels, name="insect", keys=INSECTS[:1])
    _seen(labelling, BIRDS[:3], name="bird")
    _seen(labelling, INSECTS[:2], name="insect")

    done = labelling.propagate(rule=RULE, round=2)

    labels = read_labels(labelling.places.labels)
    assert labels["bird"] == sorted(BIRDS)
    assert labels["insect"] == sorted(INSECTS)
    assert done.named == {"bird": 7, "insect": 4}
    assert done.changed == 11
    assert done.far == 2
    assert done.contested == []
    line = labelling.ledger_lines(key=BIRDS[9], last=0)[-1]
    assert line.basis == PROPAGATED
    assert set(line.evidence["voters"]) == set(BIRDS[:3])


def test_a_preview_says_what_a_propagation_would_do_and_writes_nothing(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    _seen(labelling, BIRDS[:3], name="bird")

    preview = labelling.preview(rule=RULE)

    assert preview.named == {"bird": 7}
    assert preview.changed == 7
    assert read_labels(labelling.places.labels) == {"bird": sorted(BIRDS[:3])}


def test_the_uncertain_tracks_are_the_ones_whose_neighbours_disagree_then_the_far_ones(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    _seen(labelling, [BIRDS[0]], name="bird")
    _seen(labelling, [BIRDS[1]], name="insect")
    _seen(labelling, [BIRDS[2]], name="meteor")

    uncertain = labelling.uncertain(count=20, rule=RULE)

    assert [vote.status for vote in uncertain[:7]] == [DISAGREE] * 7
    assert [vote.status for vote in uncertain[7:]] == [FAR] * 6
    assert uncertain[7].key == INSECTS[0]


def test_a_track_renamed_as_often_as_the_limit_is_contested_and_left_alone(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    _seen(labelling, BIRDS[:3], name="bird")
    labelling.propagate(rule=RULE, round=1)
    _seen(labelling, BIRDS[3:6], name="insect", round=2)
    labelling.propagate(rule=Rule(votes=3, agreement=2, cap=10.0), round=2)
    _seen(labelling, BIRDS[6:8], name="bird", round=3)

    done = labelling.propagate(rule=Rule(votes=2, agreement=2, cap=10.0), round=3)

    assert BIRDS[9] in done.contested
    assert BIRDS[9] in labelling.contested()
    before = read_labels(labelling.places.labels)
    labelling.propagate(rule=Rule(votes=2, agreement=2, cap=10.0), round=4)
    assert read_labels(labelling.places.labels) == before


def test_verify_candidates_are_propagated_tracks_on_disk_of_the_round(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    _seen(labelling, BIRDS[:3], name="bird")
    labelling.propagate(rule=RULE, round=1)

    candidates = labelling.verify_candidates(count=3, round=1)

    assert len(candidates) == 3
    assert set(candidates) <= set(BIRDS[3:])
    assert BIRDS[9] in candidates


def test_the_vocabulary_is_fixed_at_first_use_and_grown_only_by_definition(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    (labelling.places.labelling / "vocabulary.json").unlink()
    write_labels(labelling.places.labels, name="bird", keys=BIRDS[:1])
    assert sorted(labelling.vocabulary()) == ["bird"]
    write_labels(labelling.places.labels, name="insect", keys=INSECTS[:1])

    refused = labelling.define_name("birds", definition="", examples=[], round=1)
    near = labelling.define_name("brid", definition="", examples=[], round=1)
    added = labelling.define_name("Sheep", definition="a sheep on the hill", examples=BIRDS[:1], round=1)

    assert refused.refused[0].reason == "The name is in use already."
    assert near.refused[0].reason == "The name is close to 'bird'."
    assert added.changed == ["sheep"]
    assert sorted(labelling.vocabulary()) == ["bird", "sheep"]


def test_a_snapshot_puts_the_label_files_back_as_they_were(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    write_labels(labelling.places.labels, name="bird", keys=BIRDS[:1])
    labelling.snapshot("before")
    write_labels(labelling.places.labels, name="insect", keys=BIRDS[:1])

    labelling.restore("before")

    assert read_labels(labelling.places.labels) == {"bird": BIRDS[:1]}


def test_a_sheet_of_a_track_whose_recording_is_absent_is_refused_by_name(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)

    with pytest.raises(FileNotFoundError, match="rec9.mkv"):
        labelling.sheet(LONERS[:1], layout=SHEET)


def test_a_tag_is_written_and_counted(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)

    held = labelling.tag(BIRDS[0], tags=[REVIEW_TAG], round=1)

    assert held == [REVIEW_TAG]
    assert labelling.status(rule=RULE).review == 1
    assert labelling.track(BIRDS[0])["tags"] == [REVIEW_TAG]


def test_the_fetch_list_names_the_recording_covering_most_uncertain_tracks(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    _seen(labelling, BIRDS[:3], name="bird")

    (candidate,) = labelling.fetch_list(count=5, rule=RULE)

    assert candidate.recording == "rec9.mkv"
    assert candidate.uncertain == 2


def test_the_neighbours_of_a_track_come_nearest_first_with_what_is_known_of_them(tmp_path: Path) -> None:
    labelling = _labelling(tmp_path)
    _seen(labelling, [BIRDS[1]], name="bird")

    near = labelling.neighbours(BIRDS[0], count=2, seen_only=False)
    seen = labelling.neighbours(BIRDS[0], count=2, seen_only=True)

    assert [held.key for held in near] == [BIRDS[1], BIRDS[2]]
    assert near[0].seen and near[0].name == "bird"
    assert [held.key for held in seen] == [BIRDS[1]]


def test_the_ledger_is_read_back_when_the_labelling_opens_again(tmp_path: Path) -> None:
    places = corpus(tmp_path)
    first = Labelling(places, settings=SETTINGS)
    _seen(first, BIRDS[:2], name="bird")

    again = Labelling(places, settings=SETTINGS)

    assert set(again.state.seen) == set(BIRDS[:2])
    assert json.loads(again.ledger_path.read_text().splitlines()[0])["basis"] == SEEN
