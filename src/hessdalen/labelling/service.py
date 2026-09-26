"""What the labelling reads, builds and writes, for the server and the
harness alike.

The map and the paths are read once. The label files are read afresh
on every call, because the page writes them too. The ledger is read
once and appended to, and what the labelling keeps between calls is
derived from its lines. Every write to a label file goes through the
page's own write functions, under the lock they hold, and leaves a
line in the ledger.

Nothing here decides what a track is. A verdict comes in from a model
and is written as given, a vote is cast by the rule it is handed, and
what to look at next is chosen by the order the harness asks for.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-not-found]

from hessdalen.dashboard.cluster_labels import (
    canonical_label,
    locked,
    near_label,
    read_labels,
    tags_of,
    write_labels,
    write_tags,
)
from hessdalen.dashboard.places import Places
from hessdalen.dashboard.track_clip import StoredTrack
from hessdalen.dashboard.track_validation import read_validated
from hessdalen.dashboard.video_cache import FetchProgress, archive_video, fetch_video, ledger_entries, make_room
from hessdalen.labelling import ledger
from hessdalen.labelling.ledger import DEFINED, JUDGED, LEDGER_NAME, PREDICTED, PROPAGATED, SEEN, TAGGED, Line
from hessdalen.labelling.sheets import ISOLATION, Layout, Sheet, SheetTrack, build_isolation, build_sheet
from hessdalen.labelling.signatures import SIGNATURES
from hessdalen.labelling.votes import DISAGREE, FAR, NAMED, Rule, Vote, Voters, cast

UNNAMED = ""
REVIEW_TAG = "review"
"""The tag that puts a track in front of the person."""

UNREADABLE_TAG = "cannot tell"
"""The tag a person puts on a validated track whose sheet does not show
what it is. Such a track is no ground truth for a reader of sheets,
whatever the person saw in the video."""

FROM_VIDEO_TAG = "from video"
"""The tag a person puts on a validated track they could name only from
its video. The name is ground truth, and the sheet does not show it,
so a reader of sheets is not scored on it."""

SHEET_BLIND_TAGS = (UNREADABLE_TAG, FROM_VIDEO_TAG)
"""The tags that keep a validated track out of sheet calibration."""

SMOOTH = "smooth"
ROUGH = "rough"
UNASSIGNED_CLUSTER = -1

NONE_OF_THESE = "none of these"
"""What a model answers when no name of the vocabulary fits a row. The
row is recorded as seen without a name and does not vote."""

CONFIDENCES = ("sure", "likely", "unsure")
VOTING_CONFIDENCES = frozenset({"sure", "likely"})
"""Verdicts that name a track. An unsure verdict is recorded as seen
without a name, so it neither names the track nor votes."""

VOCABULARY_NAME = "vocabulary.json"
SHEETS_DIR = "sheets"
SNAPSHOTS_DIR = "snapshots"
SNAPSHOT_FILES = ("labels", "track_labels", "validated")


@dataclass(frozen=True, slots=True)
class Reading:
    """What a model made of one row of a sheet."""

    key: str
    name: str
    confidence: str
    note: str


@dataclass(frozen=True, slots=True)
class Refusal:
    key: str
    reason: str


@dataclass(frozen=True, slots=True)
class Written:
    """What a write changed and what it refused."""

    changed: list[str]
    refused: list[Refusal]


@dataclass(frozen=True, slots=True)
class Propagation:
    """What one propagation did."""

    named: dict[str, int]
    """How many tracks now stand under each name by the vote, changed or
    not."""

    changed: int
    disagree: int
    far: int
    contested: list[str]
    """The tracks whose name changed as often as the limit for the first
    time in this propagation."""


@dataclass(frozen=True, slots=True)
class Neighbour:
    key: str
    distance: float
    name: str
    seen: bool


@dataclass(frozen=True, slots=True)
class ClusterSummary:
    cluster: int
    size: int
    cached: int
    unnamed_cached: int
    seen: int
    recordings: int
    names: dict[str, int]


@dataclass(frozen=True, slots=True)
class FetchCandidate:
    recording: str
    uncertain: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Status:
    tracks: int
    names: dict[str, int]
    seen: int
    cached: int
    cached_unseen: int
    contested: int
    review: int
    disagree: int
    far: int
    rounds: int


@dataclass(frozen=True, slots=True)
class Settings:
    signature: str
    flip_limit: int


def _quiet(progress: FetchProgress) -> None:
    return None


class Labelling:
    def __init__(self, places: Places, *, settings: Settings) -> None:
        self.places = places
        self.settings = settings
        self.table = pq.read_table(places.map)
        self.tracks = self._frame()
        self.vectors = SIGNATURES[settings.signature](self.table)
        self.place_of = {key: place for place, key in enumerate(self.tracks["key"])}
        self.everyone = NearestNeighbors().fit(self.vectors)
        self.lines = ledger.read(self.ledger_path)
        self.state = ledger.derive(self.lines)
        self._paths: pd.DataFrame | None = None

    @property
    def ledger_path(self) -> Path:
        return self.places.labelling / LEDGER_NAME

    @property
    def sheets_dir(self) -> Path:
        return self.places.labelling / SHEETS_DIR

    # Reads.

    def status(self, *, rule: Rule) -> Status:
        names = self.names_by_key()
        cached = self.tracks[self.tracks["cached"]]
        votes = self._votes(self._open(cached), rule=rule)
        return Status(
            tracks=len(self.tracks),
            names=dict(Counter(names.values())),
            seen=len(self.state.seen),
            cached=len(cached),
            cached_unseen=int((~cached["key"].isin(list(self.state.seen))).sum()),
            contested=len(self.contested()),
            review=len(self.tagged(REVIEW_TAG)),
            disagree=sum(vote.status == DISAGREE for vote in votes),
            far=sum(vote.status == FAR for vote in votes),
            rounds=self.state.rounds,
        )

    def clusters(self) -> list[ClusterSummary]:
        """Every cluster, the one with most unnamed tracks on disk first."""
        names = self.names_by_key()
        summaries = []
        for cluster, members in self.tracks.groupby("cluster"):
            held = Counter(names.get(key, UNNAMED) for key in members["key"])
            cached = members[members["cached"]]
            summaries.append(
                ClusterSummary(
                    cluster=int(cluster),
                    size=len(members),
                    cached=len(cached),
                    unnamed_cached=sum(names.get(key, UNNAMED) == UNNAMED for key in cached["key"]),
                    seen=int(members["key"].isin(list(self.state.seen)).sum()),
                    recordings=int(members["recording"].nunique()),
                    names={name: count for name, count in held.items() if name},
                )
            )
        return sorted(summaries, key=lambda summary: (-summary.unnamed_cached, -summary.size, summary.cluster))

    def track(self, key: str) -> dict[str, Any]:
        """Everything the map, the label files and the ledger hold of one
        track."""
        row = self.tracks.iloc[self.place_of[key]]
        verdict = self.state.seen.get(key)
        return {
            **{column: _plain(row[column]) for column in self.tracks.columns},
            "name": self.names_by_key().get(key, UNNAMED),
            "tags": tags_of(read_labels(self.places.track_labels), key=key),
            "validated": key in self.validated(),
            "seen": None if verdict is None else {"name": verdict.name, "confidence": verdict.confidence},
            "flips": self.state.flips.get(key, 0),
        }

    def neighbours(self, key: str, *, count: int, seen_only: bool) -> list[Neighbour]:
        """The tracks nearest this one in signature space, nearest first."""
        names = self.names_by_key()
        query = self.vectors[[self.place_of[key]]]
        if seen_only:
            voters = self._voters()
            if not voters.keys:
                return []
            found = NearestNeighbors(n_neighbors=min(count, len(voters.keys))).fit(voters.vectors)
            distances, places = found.kneighbors(query)
            keys = [voters.keys[place] for place in places[0]]
        else:
            distances, places = self.everyone.kneighbors(query, n_neighbors=min(count + 1, len(self.tracks)))
            keys = [str(self.tracks["key"].iloc[place]) for place in places[0]]
        return [
            Neighbour(key=near, distance=float(apart), name=names.get(near, UNNAMED), seen=near in self.state.seen)
            for near, apart in zip(keys, distances[0])
            if near != key
        ][:count]

    def sample(self, cluster: int, *, count: int) -> list[str]:
        """Tracks of the cluster to sheet, spread over its core, its rim and
        distinct recordings.

        Only tracks with a recording on disk that no model has seen and
        no person has validated. The core and the rim are taken in
        turns, and a recording already taken is passed over while other
        recordings are left.
        """
        members = self._open(self.tracks[(self.tracks["cluster"] == cluster) & self.tracks["cached"]])
        if members.empty:
            return []

        middle = members[["x", "y"]].median()
        apart = np.hypot(members["x"] - middle["x"], members["y"] - middle["y"])
        ordered = members.assign(apart=apart).sort_values("apart", kind="stable")
        return _spread(ordered, count=count)

    def uncertain(self, *, count: int, rule: Rule) -> list[Vote]:
        """The tracks on disk the vote cannot name, the weakest first: the
        ones whose seen neighbours disagree by how few agree, then the ones
        with no seen neighbour within the cap by how near the nearest is."""
        votes = self._votes(self._open(self.tracks[self.tracks["cached"]]), rule=rule)
        disagree = sorted((vote for vote in votes if vote.status == DISAGREE), key=lambda vote: vote.share)
        far = sorted((vote for vote in votes if vote.status == FAR), key=lambda vote: vote.distance)
        return [*disagree, *far][:count]

    def verify_candidates(self, *, count: int, round: int) -> list[str]:
        """Tracks on disk that a propagation of this round named and no model
        has seen, half of them the furthest from their voters and half the
        ones fewest voters agreed on."""
        latest: dict[str, Line] = {}
        for line in self.lines:
            if line.basis == PROPAGATED and line.round == round:
                latest[line.key] = line
        cached = set(self.tracks.loc[self.tracks["cached"], "key"])
        held = [line for key, line in latest.items() if key in cached and key not in self.state.seen]

        by_distance = sorted(held, key=lambda line: -float(line.evidence.get("distance", 0.0)))
        by_share = sorted(held, key=lambda line: int(line.evidence.get("share", 0)))
        chosen: list[str] = []
        for first, second in zip(by_distance, by_share):
            for line in (first, second):
                if line.key not in chosen and len(chosen) < count:
                    chosen.append(line.key)
        return chosen

    def calibration_keys(self) -> list[str]:
        """Every validated track on disk that holds a name, to read blind
        and compare.

        The validated tracks are the only ground truth. A name the
        earlier pass gave a whole cluster says nothing about the track
        it sits on, so no other track is read for calibration. A
        validated track tagged as unreadable on its sheet, or as named
        from its video alone, is left out.
        """
        cached = self.tracks[self.tracks["cached"]]
        blind = set().union(*(self.tagged(tag) for tag in SHEET_BLIND_TAGS))
        return [key for key in cached["key"] if key in self.validated() and key not in blind and self.truth(key)]

    def truth(self, key: str) -> list[str]:
        """What a validated track is: the name it holds and the tags on it,
        any of which a reading may agree with."""
        held = [self.names_by_key().get(key, UNNAMED), *tags_of(read_labels(self.places.track_labels), key=key)]
        return [name for name in dict.fromkeys(held) if name and name not in SHEET_BLIND_TAGS]

    def calibration_recordings(self) -> list[FetchCandidate]:
        """Recordings not on disk that hold validated tracks, the one holding
        most first."""
        validated = self.validated()
        held = self.tracks[~self.tracks["cached"] & self.tracks["key"].isin(list(validated))]
        entries = ledger_entries(self.places.ledgers)
        return [
            FetchCandidate(
                recording=str(recording),
                uncertain=int(count),
                size_bytes=int(entries.get(str(recording), {}).get("size_bytes", 0)),
            )
            for recording, count in held.groupby("recording").size().sort_values(ascending=False).items()
        ]

    def sheet(self, keys: Sequence[str], *, layout: Layout, first: int) -> Sheet:
        """The sheet of these tracks, one row each in the order given.

        A track whose recording is not on disk is refused, with the
        recording named.
        """
        tracks = [self._sheet_track(key, number=number) for number, key in enumerate(keys, start=first)]
        return build_sheet(self.sheets_dir, tracks=tracks, layout=layout)

    def isolation(self, key: str) -> Sheet:
        return build_isolation(self.sheets_dir, track=self._sheet_track(key, number=1), layout=ISOLATION)

    def preview(self, *, rule: Rule) -> Propagation:
        """What a propagation under this rule would do, with nothing
        written."""
        votes = self._votes(self._open(self.tracks), rule=rule)
        names = self.names_by_key()
        changed = sum(vote.status == NAMED and names.get(vote.key, UNNAMED) != vote.name for vote in votes)
        return Propagation(
            named=dict(Counter(vote.name for vote in votes if vote.status == NAMED)),
            changed=changed,
            disagree=sum(vote.status == DISAGREE for vote in votes),
            far=sum(vote.status == FAR for vote in votes),
            contested=[],
        )

    def ledger_lines(self, *, key: str, last: int) -> list[Line]:
        """The lines about one track, or the last few lines of all when no
        key is given."""
        if key:
            return [line for line in self.lines if line.key == key]
        return self.lines[-last:] if last else []

    def fetch_list(self, *, count: int, rule: Rule) -> list[FetchCandidate]:
        """Recordings not on disk, the one covering the most tracks the vote
        cannot name first."""
        votes = self._votes(self._open(self.tracks[~self.tracks["cached"]]), rule=rule)
        recording_of = dict(zip(self.tracks["key"], self.tracks["recording"]))
        counted = Counter(recording_of[vote.key] for vote in votes if vote.status != NAMED)
        entries = ledger_entries(self.places.ledgers)
        return [
            FetchCandidate(
                recording=recording,
                uncertain=uncertain,
                size_bytes=int(entries.get(recording, {}).get("size_bytes", 0)),
            )
            for recording, uncertain in counted.most_common(count)
        ]

    def vocabulary(self) -> dict[str, dict[str, Any]]:
        """Every name a verdict may use, with its definition and examples.

        The vocabulary is fixed at the first call to the names in use
        then, and a name is added only through define_name.
        """
        path = self.places.labelling / VOCABULARY_NAME
        if path.is_file():
            return dict(json.loads(path.read_text()))

        held = {name: {"definition": "", "examples": []} for name in sorted(read_labels(self.places.labels))}
        _write_json(path, held)
        return held

    def contested(self) -> frozenset[str]:
        return ledger.contested(self.state, limit=self.settings.flip_limit)

    def validated(self) -> frozenset[str]:
        return read_validated(self.places.validated)

    def names_by_key(self) -> dict[str, str]:
        return {key: name for name, keys in read_labels(self.places.labels).items() for key in keys}

    def tagged(self, tag: str) -> set[str]:
        return set(read_labels(self.places.track_labels).get(canonical_label(tag), []))

    # Writes.

    def predict(self, readings: Sequence[Reading], *, sheet: Sheet, round: int) -> None:
        """Record what is expected of each row before the sheet is read."""
        names = self.names_by_key()
        for reading in readings:
            self._record(
                Line(
                    round=round,
                    key=reading.key,
                    basis=PREDICTED,
                    name=reading.name,
                    previous=names.get(reading.key, UNNAMED),
                    confidence=reading.confidence,
                    note=reading.note,
                    evidence={"sheet": str(sheet.path), "row": sheet.keys.index(reading.key)},
                )
            )

    def verdict(self, readings: Sequence[Reading], *, sheet: Sheet, round: int) -> Written:
        """Write what a model made of each row of the sheet.

        A validated track is refused, and so is a name outside the
        vocabulary. A verdict of "none of these" or an unsure one is
        recorded as seen without a name and changes no label.
        """
        return self._write_readings(readings, basis=SEEN, evidence={"sheet": str(sheet.path)}, round=round, sheet=sheet)

    def judge(self, reading: Reading, *, view: Sheet, round: int) -> Written:
        """Write what a model made of a contested track's isolation view."""
        return self._write_readings([reading], basis=JUDGED, evidence={"view": str(view.path)}, round=round, sheet=view)

    def propagate(self, *, rule: Rule, round: int) -> Propagation:
        """Cast the vote over every unseen track and write the names that
        won.

        A track a person validated, a model has seen or that is
        contested keeps its name whatever the vote. A name that is what
        the track already holds is left as it is and not recorded.
        """
        before = self.contested()
        votes = self._votes(self._open(self.tracks), rule=rule)
        names = self.names_by_key()
        won = [vote for vote in votes if vote.status == NAMED and names.get(vote.key, UNNAMED) != vote.name]
        for name, group in _grouped(won).items():
            write_labels(self.places.labels, name=name, keys=[vote.key for vote in group])
            for vote in group:
                self._record(
                    Line(
                        round=round,
                        key=vote.key,
                        basis=PROPAGATED,
                        name=name,
                        previous=names.get(vote.key, UNNAMED),
                        confidence="",
                        note="",
                        evidence={"voters": list(vote.voters), "share": vote.share, "distance": vote.distance},
                    )
                )
        return Propagation(
            named=dict(Counter(vote.name for vote in votes if vote.status == NAMED)),
            changed=len(won),
            disagree=sum(vote.status == DISAGREE for vote in votes),
            far=sum(vote.status == FAR for vote in votes),
            contested=sorted(self.contested() - before),
        )

    def tag(self, key: str, *, tags: Sequence[str], round: int) -> list[str]:
        """Put the track under these tags and no others."""
        written = write_tags(self.places.track_labels, key=key, names=tags)
        held = tags_of(written, key=key)
        self._record(Line(round=round, key=key, basis=TAGGED, name=" ".join(held), previous="", confidence="", note=""))
        return held

    def define_name(self, name: str, *, definition: str, examples: Sequence[str], round: int) -> Written:
        """Add a name to the vocabulary, unless it is close enough to one in
        use to be that name mistyped."""
        given = canonical_label(name)
        held = self.vocabulary()
        near = near_label(given, known=sorted(held))
        if not given or given in held or near:
            reason = "is empty" if not given else "is in use already" if given in held else f"is close to '{near}'"
            return Written(changed=[], refused=[Refusal(key=given, reason=f"The name {reason}.")])

        held[given] = {"definition": definition, "examples": list(examples)}
        _write_json(self.places.labelling / VOCABULARY_NAME, held)
        self._record(Line(round=round, key="", basis=DEFINED, name=given, previous="", confidence="", note=definition))
        return Written(changed=[given], refused=[])

    def fetch(self, recording: str) -> Path:
        """Fetch the recording from the archive into the page's own folder,
        making room within its disk floor, and mark its tracks as on disk."""
        entry = ledger_entries(self.places.ledgers).get(recording)
        if entry is None:
            raise FileNotFoundError(f"No sift ledger names {recording}.")
        video = archive_video(entry)
        make_room(self.places.fetched, video=video)
        landed = fetch_video(self.places.fetched, video=video, on_progress=_quiet)
        self.tracks.loc[self.tracks["recording"] == recording, "cached"] = True
        return landed

    def release(self, recording: str) -> None:
        """Let a fetched recording go once its sheets are built, and mark its
        tracks as off disk. A recording the sift kept is left alone."""
        fetched = self.places.fetched / recording
        if fetched.is_file():
            fetched.unlink()
            self.tracks.loc[self.tracks["recording"] == recording, "cached"] = False

    def smooth_recordings(self) -> list[FetchCandidate]:
        """Recordings not on disk, the one holding the most smooth tracks no
        model has seen first, with sizes."""
        held = self._open(self.tracks[(self.tracks["side"] == SMOOTH) & ~self.tracks["cached"]])
        entries = ledger_entries(self.places.ledgers)
        return [
            FetchCandidate(
                recording=str(recording),
                uncertain=int(count),
                size_bytes=int(entries.get(str(recording), {}).get("size_bytes", 0)),
            )
            for recording, count in held.groupby("recording").size().sort_values(ascending=False).items()
        ]

    def keys_to_read(self, recording: str, *, rough_seen: int) -> list[str]:
        """The tracks of one recording to put on sheets: every smooth track no
        model has seen, and the rough tracks of clusters that have fewer than
        rough_seen seen tracks, as many as bring each cluster to that count.

        A rough cluster is named from a handful of its tracks, because
        the rough side is clutter of one kind per cluster, so its tracks
        beyond that handful are not read.
        """
        held = self._open(self.tracks[self.tracks["recording"] == recording])
        smooth = held[held["side"] == SMOOTH]["key"].tolist()
        seen_by_cluster = Counter(
            int(self.tracks.iloc[self.place_of[key]]["cluster"]) for key in self.state.seen if key in self.place_of
        )
        rough: list[str] = []
        for cluster, members in held[(held["side"] == ROUGH) & (held["cluster"] != UNASSIGNED_CLUSTER)].groupby(
            "cluster"
        ):
            short = rough_seen - seen_by_cluster.get(int(cluster), 0)
            rough.extend(members["key"].tolist()[: max(0, short)])
        return [*smooth, *rough]

    def name_rough_clusters(self, *, agreement: int, of: int, round: int) -> dict[str, int]:
        """Give every rough cluster the name most of its seen tracks were
        given, once at least `of` of them are seen and `agreement` of the
        last `of` agree, and say how many tracks each name reached.

        The rough side is clutter of one kind per cluster, which is what
        the clustering was tuned to, so a cluster read alike a handful
        of times is named as a whole. A track a person validated, a
        model has seen or that is contested keeps its name.
        """
        names = self.names_by_key()
        reached: dict[str, int] = {}
        rough = self.tracks[(self.tracks["side"] == ROUGH) & (self.tracks["cluster"] != UNASSIGNED_CLUSTER)]
        for cluster, members in rough.groupby("cluster"):
            verdicts = [self.state.seen[key].name for key in members["key"] if key in self.state.seen]
            verdicts = [name for name in verdicts if name]
            if len(verdicts) < of:
                continue
            name, share = Counter(verdicts).most_common(1)[0]
            if share < agreement * len(verdicts) / of:
                continue
            keys = [key for key in self._open(members)["key"] if names.get(key, UNNAMED) != name]
            if not keys:
                continue
            write_labels(self.places.labels, name=name, keys=keys)
            for key in keys:
                self._record(
                    Line(
                        round=round,
                        key=key,
                        basis=PROPAGATED,
                        name=name,
                        previous=names.get(key, UNNAMED),
                        confidence="",
                        note="",
                        evidence={"cluster": int(cluster), "seen": len(verdicts), "share": share},
                    )
                )
            reached[name] = reached.get(name, 0) + len(keys)
        return reached

    def snapshot(self, name: str) -> Path:
        """Copy the label files under the name, and say where."""
        target = self.places.labelling / SNAPSHOTS_DIR / name
        target.mkdir(parents=True, exist_ok=True)
        for attribute in SNAPSHOT_FILES:
            source: Path = getattr(self.places, attribute)
            if source.is_file():
                shutil.copy2(source, target / source.name)
        return target

    def restore(self, name: str) -> Path:
        """Put the label files back as the snapshot holds them."""
        source = self.places.labelling / SNAPSHOTS_DIR / name
        if not source.is_dir():
            raise FileNotFoundError(f"No snapshot is kept under {name}.")
        for attribute in SNAPSHOT_FILES:
            target: Path = getattr(self.places, attribute)
            held = source / target.name
            if held.is_file():
                with locked(target):
                    shutil.copy2(held, target)
        return source

    # Helpers.

    def _write_readings(
        self, readings: Sequence[Reading], *, basis: str, evidence: dict[str, Any], round: int, sheet: Sheet
    ) -> Written:
        names = self.names_by_key()
        validated = self.validated()
        vocabulary = self.vocabulary()
        changed: list[str] = []
        refused: list[Refusal] = []
        naming: list[Reading] = []
        for reading in readings:
            name = canonical_label(reading.name) if reading.name != NONE_OF_THESE else UNNAMED
            if reading.key in validated:
                refused.append(Refusal(key=reading.key, reason="A person validated this track."))
            elif name and name not in vocabulary:
                refused.append(Refusal(key=reading.key, reason=f"'{name}' is not in the vocabulary."))
            elif reading.confidence not in CONFIDENCES:
                refused.append(Refusal(key=reading.key, reason=f"'{reading.confidence}' is not a confidence."))
            else:
                naming.append(Reading(key=reading.key, name=name, confidence=reading.confidence, note=reading.note))

        for reading in naming:
            named = reading.name if reading.confidence in VOTING_CONFIDENCES else UNNAMED
            if named:
                write_labels(self.places.labels, name=named, keys=[reading.key])
                changed.append(reading.key)
            self._record(
                Line(
                    round=round,
                    key=reading.key,
                    basis=basis,
                    name=named,
                    previous=names.get(reading.key, UNNAMED),
                    confidence=reading.confidence,
                    note=reading.note,
                    evidence={**evidence, "row": sheet.keys.index(reading.key)},
                )
            )
        return Written(changed=changed, refused=refused)

    def _record(self, line: Line) -> None:
        self.lines.append(ledger.append(self.ledger_path, line))
        self.state = ledger.derive(self.lines)

    def _frame(self) -> pd.DataFrame:
        frame = self.table.to_pandas()
        frame["key"] = frame["event"] + "/" + frame["clip"] + "/" + frame["track_id"].astype(str)
        frame["cached"] = frame["recording"].isin(self._on_disk())
        return frame

    def _on_disk(self) -> frozenset[str]:
        folders = (self.places.videos, self.places.fetched)
        return frozenset(path.name for folder in folders if folder.is_dir() for path in folder.iterdir())

    def _open(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """The tracks of these a vote or a sheet may still touch: not seen,
        not validated, not contested."""
        closed = set(self.state.seen) | self.validated() | self.contested()
        return tracks[~tracks["key"].isin(closed)]

    def _voters(self) -> Voters:
        """Every seen track that was given a name, where it stands and the name
        it was given."""
        keys = [key for key, verdict in self.state.seen.items() if verdict.name]
        places = [self.place_of[key] for key in keys]
        return Voters(keys=keys, vectors=self.vectors[places], names=[self.state.seen[key].name for key in keys])

    def _votes(self, tracks: pd.DataFrame, *, rule: Rule) -> list[Vote]:
        keys = tracks["key"].tolist()
        return cast(self._voters(), keys=keys, vectors=self.vectors[[self.place_of[key] for key in keys]], rule=rule)

    def _sheet_track(self, key: str, *, number: int) -> SheetTrack:
        """One track as a sheet draws it, captioned by its row number.

        The caption names the row and never the track, because a
        recording's name carries the folder it was filed under, which
        says what the recording was kept for, and a reader shown
        "bird" reads a bird.
        """
        row = self.tracks.iloc[self.place_of[key]]
        recording = str(row["recording"])
        video = next(
            (
                folder / recording
                for folder in (self.places.videos, self.places.fetched)
                if (folder / recording).is_file()
            ),
            None,
        )
        if video is None:
            raise FileNotFoundError(f"The recording {recording} of {key} is not on disk.")

        points = self.paths().loc[[key]].sort_values("frame_number")
        return SheetTrack(
            key=key,
            caption=f"track {number} - {row['frames']} frames - {row['camera']}",
            stored=StoredTrack(
                track_id=int(row["track_id"]),
                frame_numbers=points["frame_number"].to_numpy(),
                x=points["x"].to_numpy(),
                y=points["y"].to_numpy(),
                frame_height=int(points["frame_height"].iloc[0]),
            ),
            video=video,
            brightness=points["brightness"].to_numpy(),
            pixel_count=points["pixel_count"].to_numpy(),
        )

    def paths(self) -> pd.DataFrame:
        """Every track frame by frame, indexed by key, read the first time a
        sheet asks for it."""
        if self._paths is None:
            frame = pq.read_table(self.places.paths).to_pandas()
            frame["key"] = frame["event"] + "/" + frame["clip"] + "/" + frame["track_id"].astype(str)
            self._paths = frame.set_index("key").sort_index()
        return self._paths


def _spread(ordered: pd.DataFrame, *, count: int) -> list[str]:
    """Keys taken from the two ends of the order in turns, each recording
    once while another is left, and the rest filled in the same turns."""
    keys = ordered["key"].tolist()
    recordings = ordered["recording"].tolist()
    turns = [place for pair in zip(range(len(keys)), range(len(keys) - 1, -1, -1)) for place in pair]
    turns = list(dict.fromkeys(turns))

    chosen: list[str] = []
    taken: set[str] = set()
    for place in turns:
        if len(chosen) < count and recordings[place] not in taken:
            chosen.append(keys[place])
            taken.add(recordings[place])
    for place in turns:
        if len(chosen) < count and keys[place] not in chosen:
            chosen.append(keys[place])
    return chosen


def _grouped(votes: Sequence[Vote]) -> dict[str, list[Vote]]:
    grouped: dict[str, list[Vote]] = {}
    for vote in votes:
        grouped.setdefault(vote.name, []).append(vote)
    return grouped


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, held: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(held, indent=2, sort_keys=True) + "\n")
