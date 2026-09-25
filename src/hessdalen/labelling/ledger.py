"""The ledger of the labelling, one line per decision, and what follows
from it.

The label files hold only the names as they stand. The ledger holds
how each name got there: what was seen on which sheet, what was
predicted before a sheet was read, which tracks voted a name onto
another, and what a judge decided of a contested track. It is
appended to and never rewritten, so a run that stops continues from
what the ledger says, and a run that went wrong is read back line by
line.

Everything the labelling keeps in memory between calls is derived
from the lines: which tracks have been seen and what was made of them,
how often each track's name has changed, and which round the run is
in.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEEN = "seen"
"""A model read the track's row on a sheet and said what it is."""

JUDGED = "judged"
"""A model read the track's isolation view and settled its name."""

PROPAGATED = "propagated"
"""The track's nearest seen tracks voted the name onto it."""

PREDICTED = "predicted"
"""What the vote said of a track before its sheet was read."""

TAGGED = "tagged"
"""A tag was put on the track, such as the one that asks for review."""

DEFINED = "defined"
"""A name was added to the vocabulary."""

NAMING = frozenset({SEEN, JUDGED, PROPAGATED})
"""The bases on which a line puts a name on a track."""

OWN_VERDICT = frozenset({SEEN, JUDGED})
"""The bases on which a model looked at the track itself."""


@dataclass(frozen=True, slots=True)
class Line:
    """One decision about one track."""

    round: int
    key: str
    basis: str
    name: str
    previous: str
    confidence: str
    note: str
    evidence: dict[str, Any] = field(default_factory=dict)
    time: str = ""


def append(path: Path, line: Line) -> Line:
    """Write the line at the end of the ledger, stamped with the moment, and
    hand it back as written."""
    stamped = Line(**{**asdict(line), "time": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as ledger:
        ledger.write(json.dumps(asdict(stamped), sort_keys=True) + "\n")
    return stamped


def read(path: Path) -> list[Line]:
    """Every line of the ledger, in the order written.

    A line only half written, which a run cut short can leave last, is
    passed over.
    """
    if not path.is_file():
        return []

    lines = []
    for text in path.read_text().splitlines():
        try:
            lines.append(Line(**json.loads(text)))
        except (json.JSONDecodeError, TypeError):
            continue
    return lines


@dataclass(frozen=True, slots=True)
class Verdict:
    """What a model last made of a track it looked at."""

    name: str
    confidence: str
    round: int
    evidence: dict[str, Any]


@dataclass(frozen=True, slots=True)
class State:
    """What the ledger says as it stands."""

    seen: dict[str, Verdict]
    """Every track a model looked at, with its latest verdict, whether or
    not the verdict named it."""

    flips: dict[str, int]
    """How often each track's name was changed from one name to another."""

    rounds: int
    """The highest round any line was written in."""


def derive(lines: list[Line]) -> State:
    """The state every line of the ledger leads to.

    A change from no name to a name is the first naming and no flip.
    A change from one name to another is a flip whatever its basis, so
    a track the vote keeps moving and a track a model keeps reading
    differently are both counted.
    """
    seen: dict[str, Verdict] = {}
    flips: dict[str, int] = {}
    rounds = 0
    for line in lines:
        rounds = max(rounds, line.round)
        if line.basis in OWN_VERDICT:
            seen[line.key] = Verdict(
                name=line.name, confidence=line.confidence, round=line.round, evidence=line.evidence
            )
        if line.basis in NAMING and line.previous and line.name and line.name != line.previous:
            flips[line.key] = flips.get(line.key, 0) + 1
    return State(seen=seen, flips=flips, rounds=rounds)


def contested(state: State, *, limit: int) -> frozenset[str]:
    """The tracks whose name has changed as often as the limit or more."""
    return frozenset(key for key, count in state.flips.items() if count >= limit)
