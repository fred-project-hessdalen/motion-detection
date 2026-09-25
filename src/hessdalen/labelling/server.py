"""The Model Context Protocol (MCP) server an agent labels the corpus
through.

Every tool is a thin wrapper over the labelling service: the reads
hand back what the service holds, a sheet comes back as an image, and
the writes go through the service so that every one of them leaves a
ledger line. The server holds no opinion of its own.

Start it with

    uv run --group agent --group core --group dashboard --group analysis \\
        python -m hessdalen.labelling.server [--root <repo>] [--signature descriptors]

and register that command in the agent's MCP settings.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from hessdalen.dashboard.places import PLACES, Places
from hessdalen.labelling.service import Labelling, Reading, Settings
from hessdalen.labelling.sheets import SHEET, read_sheet
from hessdalen.labelling.signatures import DEFAULT_SIGNATURE, SIGNATURES
from hessdalen.labelling.votes import Rule

FLIP_LIMIT = 2
"""Changes of name from one name to another at which a track is
contested and left to a judge."""

VOTES = 5
AGREEMENT = 4
CAP = 1.0
"""The vote rule a tool runs under when its caller names none."""

INSTRUCTIONS = """\
The corpus is a map of tracks the movement detector found in sky
recordings. Names are kept per track. A cluster of the map holds
tracks of more than one kind, so a name is given to a track a sheet
showed, and spreads from there to the unseen tracks nearest it in a
signature space by a vote (propagate). Read a sheet (sheet) of tracks
sampled from a cluster (sample) or of the tracks the vote cannot name
(uncertain), then record one verdict per row (verdict) with a name from
the vocabulary (vocabulary), "none of these", or an unsure confidence.
A track a person validated is never written. A track whose name
changed twice is contested and settled from its isolation view
(track_sheet, judge) or tagged "review" for the person (tag).
"""

server = MCPServer("track-labelling", instructions=INSTRUCTIONS)

_labelling: Labelling | None = None


def labelling() -> Labelling:
    if _labelling is None:
        raise RuntimeError("The server was started without a corpus.")
    return _labelling


class Row(BaseModel):
    """What was made of one row of a sheet."""

    key: str
    name: str
    confidence: str
    note: str = ""


def _rule(votes: int, agreement: int, cap: float) -> Rule:
    return Rule(votes=votes, agreement=agreement, cap=cap)


@server.tool()
def status(votes: int = VOTES, agreement: int = AGREEMENT, cap: float = CAP) -> dict[str, Any]:
    """Counts over the corpus: tracks per name, seen, on disk, contested,
    tagged for review, and how many tracks on disk the vote under this
    rule leaves uncertain."""
    return asdict(labelling().status(rule=_rule(votes, agreement, cap)))


@server.tool()
def clusters(count: int = 50) -> list[dict[str, Any]]:
    """Every cluster of the map, the one with most unnamed tracks on disk
    first: size, tracks on disk, unnamed tracks on disk, seen tracks,
    recordings, and the names its tracks hold."""
    return [asdict(summary) for summary in labelling().clusters()[:count]]


@server.tool()
def track(keys: list[str]) -> list[dict[str, Any]]:
    """Everything known of each track: its descriptors, cluster, name, tags,
    whether a person validated it, what a model last made of it, and how
    often its name has changed."""
    return [labelling().track(key) for key in keys]


@server.tool()
def neighbours(key: str, count: int = 10, seen_only: bool = False) -> list[dict[str, Any]]:
    """The tracks nearest this one in signature space, nearest first, with
    their distance, name and whether a model has seen them."""
    return [asdict(near) for near in labelling().neighbours(key, count=count, seen_only=seen_only)]


@server.tool()
def sample(cluster: int, count: int = 6) -> list[str]:
    """Tracks of the cluster to put on a sheet, spread over its core, its
    rim and distinct recordings, none seen or validated, all on disk."""
    return labelling().sample(cluster, count=count)


@server.tool()
def uncertain(count: int = 6, votes: int = VOTES, agreement: int = AGREEMENT, cap: float = CAP) -> list[dict[str, Any]]:
    """The tracks on disk the vote cannot name, the weakest first: the ones
    whose seen neighbours disagree, then the ones with no seen neighbour
    within the cap."""
    return [asdict(vote) for vote in labelling().uncertain(count=count, rule=_rule(votes, agreement, cap))]


@server.tool()
def verify_candidates(round: int, count: int = 6) -> list[str]:
    """Tracks on disk a propagation of the round named and no model has
    seen, the furthest from their voters and the weakest votes."""
    return labelling().verify_candidates(count=count, round=round)


@server.tool()
def calibration() -> list[str]:
    """Every validated track on disk, the ground truth to read blind and
    compare with what it holds. Nothing else counts as ground truth."""
    return labelling().calibration_keys()


@server.tool()
def calibration_recordings() -> list[dict[str, Any]]:
    """Recordings not on disk that hold validated tracks, the one holding
    most first, to fetch before a calibration."""
    return [asdict(held) for held in labelling().calibration_recordings()]


@server.tool()
def sheet(keys: list[str], first: int = 1) -> list[str | Image]:
    """The sheet of these tracks, one row each in the order given, at most
    eight, the rows captioned with track numbers counting from first.
    Each row is the caption, the drawn path from dark blue to yellow, and
    six stretched close-up crops across the track. The first text names
    the sheet's path, which a verdict refers to."""
    try:
        built = labelling().sheet(keys, layout=SHEET, first=first)
    except FileNotFoundError as absent:
        raise ToolError(str(absent)) from absent
    return [f"sheet {built.path} rows {list(built.keys)}", Image(path=built.path)]


@server.tool()
def track_sheet(key: str) -> list[str | Image]:
    """The isolation view of one track: twelve close-up crops beside the
    path, the whole frame at the same steps with the detection boxed, and
    the blob's brightness and size over the track. The first text names
    the view's path, which a judgement refers to."""
    try:
        built = labelling().isolation(key)
    except FileNotFoundError as absent:
        raise ToolError(str(absent)) from absent
    return [f"view {built.path} rows {list(built.keys)}", Image(path=built.path)]


@server.tool()
def propagation_preview(votes: int = VOTES, agreement: int = AGREEMENT, cap: float = CAP) -> dict[str, Any]:
    """What a propagation under this rule would do, with nothing written:
    tracks per name, how many would change, and how many stay uncertain."""
    return asdict(labelling().preview(rule=_rule(votes, agreement, cap)))


@server.tool()
def ledger(key: str = "", last: int = 20) -> list[dict[str, Any]]:
    """The ledger lines about one track, or the last few lines of all when
    no key is given."""
    return [asdict(line) for line in labelling().ledger_lines(key=key, last=last)]


@server.tool()
def fetch_list(
    count: int = 10, votes: int = VOTES, agreement: int = AGREEMENT, cap: float = CAP
) -> list[dict[str, Any]]:
    """Recordings not on disk, the one covering most tracks the vote cannot
    name first, with its size in bytes."""
    return [asdict(held) for held in labelling().fetch_list(count=count, rule=_rule(votes, agreement, cap))]


@server.tool()
def vocabulary() -> dict[str, dict[str, Any]]:
    """Every name a verdict may use, with its definition and example
    tracks."""
    return labelling().vocabulary()


@server.tool()
def predict(rows: list[Row], sheet: str, round: int) -> int:
    """Record what is expected of each row of the sheet before it is read,
    and say how many lines were written."""
    readings = [_reading(row) for row in rows]
    labelling().predict(readings, sheet=read_sheet(Path(sheet)), round=round)
    return len(readings)


@server.tool()
def verdict(rows: list[Row], sheet: str, round: int) -> dict[str, Any]:
    """Write what was made of each row of the sheet: a name from the
    vocabulary, "none of these", and a confidence of sure, likely or
    unsure. A sure or likely name is written to the track. Unsure and
    "none of these" record the track as seen without a name. A validated
    track and a name outside the vocabulary are refused."""
    written = labelling().verdict([_reading(row) for row in rows], sheet=read_sheet(Path(sheet)), round=round)
    return asdict(written)


@server.tool()
def judge(row: Row, view: str, round: int) -> dict[str, Any]:
    """Settle a contested track from its isolation view, under the same
    terms as a verdict."""
    return asdict(labelling().judge(_reading(row), view=read_sheet(Path(view)), round=round))


@server.tool()
def propagate(round: int, votes: int = VOTES, agreement: int = AGREEMENT, cap: float = CAP) -> dict[str, Any]:
    """Cast the vote over every unseen track and write the names that won,
    leaving validated, seen and contested tracks as they are. Says how
    many tracks stand under each name by the vote, how many changed, how
    many stay uncertain, and which tracks became contested."""
    return asdict(labelling().propagate(rule=_rule(votes, agreement, cap), round=round))


@server.tool()
def tag(key: str, tags: list[str], round: int) -> list[str]:
    """Put the track under these tags and no others. The tag "review" puts
    the track in front of the person."""
    return labelling().tag(key, tags=tags, round=round)


@server.tool()
def define_name(name: str, definition: str, examples: list[str], round: int) -> dict[str, Any]:
    """Add a name to the vocabulary with a one-line definition and example
    tracks. A name in use or close to one in use is refused."""
    return asdict(labelling().define_name(name, definition=definition, examples=examples, round=round))


@server.tool()
def fetch(recording: str) -> str:
    """Fetch a recording from the archive so its tracks can be sheeted.
    Takes about a minute per recording and makes room within the disk
    floor the page keeps. Says where the recording landed."""
    try:
        return str(labelling().fetch(recording))
    except FileNotFoundError as unknown:
        raise ToolError(str(unknown)) from unknown


@server.tool()
def snapshot(name: str) -> str:
    """Copy the label files under this name, and say where."""
    return str(labelling().snapshot(name))


@server.tool()
def restore(name: str) -> str:
    """Put the label files back as the snapshot of this name holds them."""
    try:
        return str(labelling().restore(name))
    except FileNotFoundError as unknown:
        raise ToolError(str(unknown)) from unknown


def _reading(row: Row) -> Reading:
    return Reading(key=row.key, name=row.name, confidence=row.confidence, note=row.note)


def start(places: Places, *, settings: Settings) -> Labelling:
    """Open the corpus the tools answer about."""
    global _labelling
    _labelling = Labelling(places, settings=settings)
    return _labelling


def main() -> None:
    parser = argparse.ArgumentParser(description="The MCP server an agent labels the corpus through.")
    parser.add_argument("--root", type=Path, default=PLACES.root, help="the repository the corpus sits under")
    parser.add_argument("--signature", choices=sorted(SIGNATURES), default=DEFAULT_SIGNATURE)
    parser.add_argument("--flip-limit", type=int, default=FLIP_LIMIT)
    parsed = parser.parse_args()
    start(Places(root=parsed.root), settings=Settings(signature=parsed.signature, flip_limit=parsed.flip_limit))
    server.run("stdio")


if __name__ == "__main__":
    main()
