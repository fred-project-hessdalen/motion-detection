"""Judge the ways a track can be turned into a picture against each other.

Each candidate in analysis/track_images is asked two questions. The
first needs no labels: a track long enough is cut in two, each half is
drawn on its own, and every half is asked to find its partner among all
the others. A picture holding something about the object holds the
halves together. The run is repeated within each recording, where every
half on offer shares the scene, the camera and the night, and that
second number is the one that says anything about the object rather
than about the sky behind it.

The second reads the names in cluster-labels.json, which were given to
whole clusters and never checked track by track. They are worth only
the order they put the candidates in, because every candidate is read
against the same wrong labels.

Run with the analysis group:
uv run --group core --group analysis python scripts/dev/compare_track_images.py
"""

from __future__ import annotations

import argparse
import collections
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hessdalen.analysis.spectra import BRIGHTNESS, PRESENCE, SIZE, WOBBLE, TrackSignals, track_signals
from hessdalen.analysis.track_images import CANDIDATES, SIDE

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = REPO_ROOT / "data" / "out" / "analysis"

COARSE = 16
"""How wide a picture is cut down to before pictures are compared.

Fine detail in one of these is the detector's noise, and comparing at
full size measures that as much as anything else.
"""

SPLIT_FRAMES = 96
"""Frames a track needs before it is cut in two."""

CUT_LOW, CUT_HIGH = 0.33, 0.67
"""Where along a track the cut may fall."""

MIN_PIECE = 24
"""Frames the shorter piece of a cut track keeps."""

NEARBY = 10
"""Neighbours a track's name is read from, and the rank a partner counts
as found within."""

SHEET_TRACKS = 8
"""Tracks drawn per name on a contact sheet."""

QUERIES = 14
"""Tracks whose neighbours are drawn out for a person to judge."""

MIN_LABELLED = 50
"""Tracks a name needs before its recall is worth printing."""

BLOCK = 512
"""Tracks compared against all the others at once."""

Columns = dict[str, np.ndarray]
Drawing = Callable[[TrackSignals], np.ndarray]
Held = TypeVar("Held")


@dataclass(frozen=True, slots=True)
class Track:
    """One track's place, its name if it has one, and its rows."""

    key: str
    clip: str
    name: str
    first: int
    count: int


def main(args: argparse.Namespace) -> None:
    tracks, columns = _read(args.paths, names=_names(args.labels))
    named = [t for t in tracks if t.name]
    print(f"{len(tracks)} tracks, {len(named)} of them under a name")

    halves = [t for t in tracks if t.count >= SPLIT_FRAMES]
    if args.limit:
        halves, named = _sample(halves, args.limit), _sample(named, args.limit)
    print(f"{len(halves)} tracks long enough to cut in two")
    print(f"chance of a partner coming first among {len(halves)}: {100 / max(len(halves), 1):.3f}%")

    drawings: dict[str, Drawing] = {**CANDIDATES, **REFERENCES}
    for candidate in args.candidates or [*sorted(CANDIDATES), *sorted(REFERENCES)]:
        draw = drawings[candidate]
        print(f"\n== {candidate}", flush=True)
        _report_halves(draw=draw, tracks=halves, columns=columns, each_on_its_own=args.each_on_its_own)
        _report_names(draw=draw, tracks=named, columns=columns, each_on_its_own=args.each_on_its_own)
        stem = candidate.replace(" ", "-")
        if candidate in CANDIDATES:
            _write_sheet(args.sheets / f"{stem}.png", draw=draw, tracks=named, columns=columns)
        _write_neighbours(
            args.sheets / f"neighbours-{stem}.png",
            draw=draw,
            tracks=named,
            columns=columns,
            each_on_its_own=args.each_on_its_own,
        )


def _report_halves(*, draw: Drawing, tracks: list[Track], columns: Columns, each_on_its_own: bool) -> None:
    """How often a half of a track finds the other half of the same track.

    Every reading is printed beside the reading chance would give on the
    same pool, because the pool sets what counts as doing anything at
    all and the pools within one recording are small.
    """
    first, second = _ready(
        [draw(_signals(t, columns, half=0)) for t in tracks],
        [draw(_signals(t, columns, half=1)) for t in tracks],
        each_on_its_own=each_on_its_own,
    )

    print(f"   pooled     {_reading(_partner_ranks(first, second))}")
    print(f"      chance  {_reading(_partner_ranks(first, second[_shuffled(len(tracks))]))}")

    by_clip = collections.defaultdict(list)
    for at, track in enumerate(tracks):
        by_clip[track.clip].append(at)
    pools = [rows for rows in by_clip.values() if len(rows) >= NEARBY]

    within = [rank for rows in pools for rank in _partner_ranks(first[rows], second[rows])]
    mixed = [rank for rows in pools for rank in _partner_ranks(first[rows], second[rows][_shuffled(len(rows))])]
    held = sum(len(rows) for rows in pools)
    print(f"   in a clip  {_reading(np.array(within))}   over {len(pools)} clips, {held / max(len(pools), 1):.0f} each")
    print(f"      chance  {_reading(np.array(mixed))}")


def _shuffled(count: int) -> np.ndarray:
    """An order that pairs every half with somebody else's.

    A reading taken against it has to come out at chance, or the
    retrieval is answering something about the pool rather than about
    the tracks in it.
    """
    return np.roll(np.arange(count), max(count // 3, 1))


def _partner_ranks(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Where each track's other half stood among all the halves on offer.

    Counted rather than sorted, and a block of tracks at a time, because
    the corpus offers tens of thousands of halves and the whole table of
    one against another does not fit.

    A half that ties with others takes the middle of the run it ties in.
    Counting a tie as a win would hand a reference holding one number a
    perfect score, since every track it cannot tell apart would come
    first.
    """
    ranks = np.empty(len(first), dtype=np.int64)
    for start in range(0, len(first), BLOCK):
        stop = min(start + BLOCK, len(first))
        nearness = _nearness(first[start:stop], second)
        partner = nearness[np.arange(stop - start), np.arange(start, stop)]
        nearer = (nearness > partner[:, None]).sum(axis=1)
        level = (nearness == partner[:, None]).sum(axis=1) - 1
        ranks[start:stop] = nearer + level // 2 + 1
    return ranks


def _nearness(block: np.ndarray, others: np.ndarray) -> np.ndarray:
    """How near every row of the block is to every other row, as the negative
    of the distance between them squared.

    Distance rather than angle, because a reference is a short row of
    plain numbers whose lengths carry meaning, and on the pictures,
    whose rows are all the same length, the two put things in the same
    order anyway.
    """
    return -(np.square(block).sum(axis=1)[:, None] + np.square(others).sum(axis=1)[None, :] - 2.0 * block @ others.T)


def _ready(*groups: list[np.ndarray], each_on_its_own: bool) -> list[np.ndarray]:
    """Every drawing as a row ready to compare, all groups scaled together.

    Two ways of scaling, because they ask different questions of a
    picture. Set about its own middle, a picture keeps its pattern and
    gives up how much of anything it holds, which is what is usually
    wanted of a picture and is wrong here if how far a track strays is
    the thing that separates it. Set against the spread of each cell
    over the whole corpus, a picture keeps how much and lets one loud
    cell speak over the pattern.

    A row of plain numbers on scales that have nothing to do with each
    other can only take the second, since its own middle means nothing.

    The halves of a track are scaled together either way, since two
    halves measured against different scales cannot be put beside each
    other.
    """
    sizes = [len(group) for group in groups]
    stacked = [one for group in groups for one in group]
    flat = stacked[0].ndim == 1

    values = np.array(stacked, dtype=np.float64) if flat else np.array([_coarse(one) for one in stacked])
    if flat or not each_on_its_own:
        spread = values.std(axis=0)
        values = (values - values.mean(axis=0)) / np.maximum(spread, 0.01 * max(float(spread.mean()), 1e-12))
    else:
        values = values - values.mean(axis=1, keepdims=True)
    if not flat:
        values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)

    edges = np.cumsum([0, *sizes])
    return [values[edges[at] : edges[at + 1]] for at in range(len(groups))]


def _coarse(picture: np.ndarray) -> np.ndarray:
    """One picture cut down and flattened.

    Fine detail in one of these is the detector's noise, and comparing
    at full size measures that as much as anything else.
    """
    small = cv2.resize(np.asarray(picture, dtype=np.float32), (COARSE, COARSE), interpolation=cv2.INTER_AREA)
    return np.asarray(small.reshape(-1), dtype=np.float64)


def _reading(ranks: np.ndarray) -> str:
    if ranks.size == 0:
        return "nothing to read"
    return (
        f"first {100 * float((ranks == 1).mean()):5.1f}%   "
        f"in ten {100 * float((ranks <= NEARBY).mean()):5.1f}%   "
        f"median rank {int(np.median(ranks))}"
    )


def _report_names(*, draw: Drawing, tracks: list[Track], columns: Columns, each_on_its_own: bool) -> None:
    """What share of each name's tracks its nearest neighbours agree on.

    This asks what kind of thing a track is, where the retrieval asks
    which track it is. The two come apart: a recording holding forty
    tracks of one kind gives every one of them thirty-nine matches as
    good as its own other half, so a reading that has caught the kind
    perfectly still answers the other question at chance.

    Printed beside the same reading over the names dealt out at random,
    which is what no knowledge of a track looks like.
    """
    drawn = _ready([draw(_signals(t, columns, half=None)) for t in tracks], each_on_its_own=each_on_its_own)[0]
    given = np.array([t.name for t in tracks])
    neighbours = _neighbours(drawn)

    print(f"   names      {_recalls(given, guessed=_voted(given, neighbours=neighbours))}")
    shuffled = given[_shuffled(len(given))]
    print(f"      chance  {_recalls(shuffled, guessed=_voted(shuffled, neighbours=neighbours))}")


def _neighbours(drawn: np.ndarray) -> np.ndarray:
    """The nearest few tracks to each track, itself left out."""
    found = np.empty((len(drawn), NEARBY), dtype=np.int64)
    for start in range(0, len(drawn), BLOCK):
        stop = min(start + BLOCK, len(drawn))
        nearness = _nearness(drawn[start:stop], drawn)
        nearness[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        for row in range(stop - start):
            found[start + row] = np.argpartition(-nearness[row], NEARBY)[:NEARBY]
    return found


def _voted(given: np.ndarray, *, neighbours: np.ndarray) -> np.ndarray:
    return np.array([collections.Counter(given[row]).most_common(1)[0][0] for row in neighbours])


def _recalls(given: np.ndarray, *, guessed: np.ndarray) -> str:
    recalls = []
    parts = []
    for name, count in collections.Counter(given).most_common():
        if count < MIN_LABELLED:
            continue
        held = given == name
        recall = float((guessed[held] == name).mean())
        recalls.append(recall)
        parts.append(f"{name} {100 * recall:.0f}%")
    return f"mean recall {100 * float(np.mean(recalls)):5.1f}%   " + "  ".join(parts)


def hand_made_numbers(signals: TrackSignals) -> np.ndarray:
    """A handful of aggregates of the kind the descriptors already hold.

    Here so that every picture has something of its own kind to be read
    against. The 27 descriptors themselves cannot stand here, because
    three of the columns they are built from are not in the paths table
    and so cannot be had for half a track.
    """
    wobble = signals.values[WOBBLE]
    course = np.column_stack((signals.centre_x, signals.centre_y))
    steps = np.linalg.norm(np.diff(course, axis=0), axis=1) if len(course) > 1 else np.zeros(1)
    travelled = float(steps.sum())
    return np.array(
        [
            np.log1p(float(np.std(wobble)) * 1e3),
            _slow_share(wobble),
            float(np.std(signals.values[BRIGHTNESS])),
            float(np.std(signals.values[SIZE])),
            float(np.mean(signals.values[PRESENCE])),
            np.log1p(float(np.mean(steps))),
            float(np.linalg.norm(course[-1] - course[0])) / max(travelled, 1e-9),
            float(np.std(steps)) / max(float(np.mean(steps)), 1e-9),
        ]
    )


def _slow_share(signal: np.ndarray) -> float:
    """What share of a signal's variation sits at the slowest rates."""
    if signal.size < 4:
        return 0.0
    steps = np.arange(signal.size, dtype=np.float64)
    level = signal - np.polynomial.Polynomial.fit(steps, signal, 1)(steps)
    power = np.square(np.abs(np.fft.rfft(level)))[1:]
    rates = np.fft.rfftfreq(signal.size)[1:]
    return float(power[rates < 0.06].sum() / power.sum()) if power.sum() > 0 else 0.0


def how_long(signals: TrackSignals) -> np.ndarray:
    """The track's length alone, which every picture throws away."""
    return np.array([np.log1p(float(signals.frame_number.size))])


REFERENCES: dict[str, Drawing] = {"hand made numbers": hand_made_numbers, "length alone": how_long}
"""What the pictures are read against, beside chance."""


def _signals(track: Track, columns: Columns, *, half: int | None) -> TrackSignals:
    """One track's signals, or those of one piece of it.

    The cut falls at a different place on every track, so that the two
    pieces of one track are of different lengths. Cutting every track
    down the middle would leave both its pieces exactly as long as each
    other, and a piece could then find its partner by length alone,
    which says nothing about what either piece holds.
    """
    at = slice(track.first, track.first + track.count)
    rows = {name: values[at] for name, values in columns.items()}
    if half is not None:
        cut = _cut(track)
        piece = slice(0, cut) if half == 0 else slice(cut, track.count)
        rows = {name: values[piece] for name, values in rows.items()}
    return track_signals(
        frame_number=rows["frame_number"],
        centre_x=rows["centre_x"],
        centre_y=rows["centre_y"],
        brightness=rows["brightness"],
        pixel_count=rows["pixel_count"],
        reach=float(max(rows["frame_width"][0], rows["frame_height"][0])),
    )


def _write_neighbours(
    path: Path, *, draw: Drawing, tracks: list[Track], columns: Columns, each_on_its_own: bool
) -> None:
    """A few tracks and the tracks nearest them, every one drawn as its own
    path.

    The numbers rest on names nobody checked. This rests on nothing: a
    row holds a track and what this way of drawing thinks it resembles,
    both shown as the thing a person recognises, so the question of
    whether the resemblance is real can be put to a person.
    """
    drawn = _ready([draw(_signals(t, columns, half=None)) for t in tracks], each_on_its_own=each_on_its_own)[0]
    neighbours = _neighbours(drawn)
    asked = _sample(list(range(len(tracks))), QUERIES)

    rows = []
    for query in asked:
        held = [query, *neighbours[query][: NEARBY - 2]]
        strip = np.hstack([np.pad(_path_picture(tracks[at], columns), ((1, 1), (1, 1), (0, 0))) for at in held])
        strip[:, : SIDE + 2, 2] = np.maximum(strip[:, : SIDE + 2, 2], 90)
        label = np.zeros((16, strip.shape[1], 3), np.uint8)
        for column, at in enumerate(held):
            corner = (column * (SIDE + 2) + 3, 12)
            cv2.putText(label, tracks[at].name[:11], corner, cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)
        rows += [label, strip]

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))


def _path_picture(track: Track, columns: Columns) -> np.ndarray:
    """One track's path as it went, fitted to a square and not turned."""
    signals = _signals(track, columns, half=None)
    points = np.column_stack((signals.centre_x, signals.centre_y))
    points = points - (points.min(axis=0) + points.max(axis=0)) / 2.0
    points = points / max(float(np.abs(points).max()), 1e-9) * (SIDE / 2 - 3)

    canvas = np.zeros((SIDE, SIDE, 3), np.uint8)
    drawn = np.rint(points + SIDE / 2).astype(np.int32)
    for at in range(len(drawn) - 1):
        shade = int(60 + 195 * at / max(len(drawn) - 2, 1))
        cv2.line(canvas, tuple(drawn[at]), tuple(drawn[at + 1]), (shade, shade, 60), 1, cv2.LINE_AA)
    return canvas


def _cut(track: Track) -> int:
    """Where a track is cut in two, between a third and two thirds along.

    Settled by the track's own name, so that the two pieces of one track
    are always cut at the same place however often this is run, and two
    different tracks are cut at different places.
    """
    share = CUT_LOW + (CUT_HIGH - CUT_LOW) * (abs(hash(track.key)) % 1000) / 1000.0
    return max(MIN_PIECE, min(int(track.count * share), track.count - MIN_PIECE))


def _write_sheet(path: Path, *, draw: Drawing, tracks: list[Track], columns: Columns) -> None:
    """One row per name, a few of its tracks each, so the numbers can be read
    against what the pictures look like."""
    by_name = collections.defaultdict(list)
    for track in tracks:
        if track.name:
            by_name[track.name].append(track)

    rows = []
    for name, held in sorted(by_name.items(), key=lambda item: -len(item[1])):
        drawn = [_shown(draw(_signals(t, columns, half=None))) for t in held[:SHEET_TRACKS]]
        while len(drawn) < SHEET_TRACKS:
            drawn.append(np.zeros((SIDE, SIDE, 3), np.uint8))
        strip = np.hstack([np.pad(one, ((1, 1), (1, 1), (0, 0))) for one in drawn])
        label = np.zeros((18, strip.shape[1], 3), np.uint8)
        cv2.putText(label, name, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        rows += [label, strip]

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))


def _shown(picture: np.ndarray) -> np.ndarray:
    """One picture as something to look at."""
    if picture.ndim == 3:
        return np.ascontiguousarray((np.clip(picture, 0.0, 1.0) * 255).astype(np.uint8))
    shades = (np.clip(picture, 0.0, 1.0) * 255).astype(np.uint8)
    return np.asarray(cv2.applyColorMap(shades, cv2.COLORMAP_VIRIDIS))


def _read(path: Path, *, names: dict[str, str]) -> tuple[list[Track], Columns]:
    table = pq.read_table(path)
    keys = np.array(
        pc.binary_join_element_wise(
            table["event"].cast("string"), table["clip"].cast("string"), pc.cast(table["track_id"], "string"), "/"
        )
    )
    frames = np.array(table["frame_number"])
    order = np.lexsort((frames, keys))

    columns = {
        name: np.array(table[name])[order]
        for name in ("frame_number", "centre_x", "centre_y", "brightness", "pixel_count", "frame_width", "frame_height")
    }
    keys, clips = keys[order], np.array(table["clip"])[order]
    unique, first, count = np.unique(keys, return_index=True, return_counts=True)
    tracks = [
        Track(
            key=str(unique[at]),
            clip=str(clips[first[at]]),
            name=names.get(str(unique[at]), ""),
            first=int(first[at]),
            count=int(count[at]),
        )
        for at in range(len(unique))
    ]
    return tracks, columns


def _sample(held: list[Held], limit: int) -> list[Held]:
    """A spread of what it is given, since neighbouring tracks share a
    recording and a quick look at one recording says nothing about the
    corpus."""
    if len(held) <= limit:
        return held
    at = np.random.default_rng(0).choice(len(held), size=limit, replace=False)
    return [held[int(one)] for one in sorted(at)]


def _names(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    given = json.loads(path.read_text())
    return {key: name for name, keys in given.items() for key in keys}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", type=Path, default=ANALYSIS / "track-paths.parquet")
    parser.add_argument("--labels", type=Path, default=ANALYSIS / "cluster-labels.json")
    parser.add_argument("--sheets", type=Path, default=ANALYSIS / "signatures")
    parser.add_argument("--candidates", nargs="*", choices=[*sorted(CANDIDATES), *sorted(REFERENCES)])
    parser.add_argument("--limit", type=int, default=0, help="read at most this many tracks, for a quick look")
    parser.add_argument(
        "--each-on-its-own",
        action="store_true",
        help="set every picture about its own middle, keeping its pattern and giving up how much it holds",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse())
