"""The names given to clusters of the track map, and the tracks under each.

A cluster is what the corpus itself groups, and a name is what a person
makes of that group once it has been drawn out. The names are kept as
the tracks they were given to rather than as the clusters, because a
later run of the analysis step numbers its clusters afresh while a track
keeps its name.

The file is what a training set is built from, so it holds the tracks
and nothing about the map they were picked on.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_labels(path: Path) -> dict[str, list[str]]:
    """Every name given so far, with the tracks under it."""
    if not path.is_file():
        return {}
    held = json.loads(path.read_text())
    return {str(name): [str(key) for key in keys] for name, keys in held.items()}


def label_of(labels: dict[str, list[str]], *, keys: list[str]) -> str:
    """The name these tracks are under, or nothing when they are under none.

    A name is given to every track of a cluster at once, so the name of
    any one of them is the name of the cluster.
    """
    wanted = set(keys)
    for name, held in labels.items():
        if wanted.intersection(held):
            return name
    return ""


def write_labels(path: Path, labels: dict[str, list[str]], *, name: str, keys: list[str]) -> dict[str, list[str]]:
    """Put these tracks under this name and write every name out again.

    The tracks are taken out of the name they were under first, so a
    cluster named again moves rather than standing under both names.
    Under an empty name they are only taken out, which is how a name is
    withdrawn.
    """
    wanted = set(keys)
    written = {held: [key for key in under if key not in wanted] for held, under in labels.items()}
    if name:
        written[name] = sorted(set(written.get(name, [])) | wanted)

    written = {held: under for held, under in written.items() if under}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(written, indent=2, sort_keys=True) + "\n")
    return written
