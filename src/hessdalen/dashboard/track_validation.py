"""The tracks a person has gone through and confirmed.

A track is validated once someone has watched it and stands by the name
it is under. The map can be held to the tracks that carry it or to the
ones still to go through, and a gallery marks a validated track with a
tick.

The tracks are kept apart from the names, because confirming a track
says that it was looked at rather than what it holds.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_validated(path: Path) -> frozenset[str]:
    """Every track confirmed so far."""
    if not path.is_file():
        return frozenset()

    return frozenset(str(key) for key in json.loads(path.read_text()))


def write_validated(path: Path, validated: frozenset[str], *, key: str, confirmed: bool) -> frozenset[str]:
    """Put this track among the confirmed or take it out, and write them all
    out again."""
    written = validated | {key} if confirmed else validated - {key}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(written), indent=2) + "\n")
    return written
