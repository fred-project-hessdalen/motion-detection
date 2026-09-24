"""The names given to clusters of the track map, and the tracks under each.

A cluster is what the corpus itself groups, and a name is what a person
makes of that group once it has been drawn out. The names are kept as
the tracks they were given to rather than as the clusters, because a
later run of the analysis step numbers its clusters afresh while a track
keeps its name.

The same file shape holds the tags of single tracks, where a track
stands under every name that is true of it at once rather than under
one.

A name is kept in one form, in small letters with single spaces between
its words and every word in the singular, so that one thing does not
stand under two spellings of itself. A name that the form leaves a
letter or two from one already in use is most likely that name mistyped,
which the page puts to the person before either is written.

The file is what a training set is built from, so it holds the tracks
and nothing about the map they were picked on.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

WORD_BREAK = re.compile(r"[^0-9A-Za-z]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
"""Where one word of a name ends and the next begins.

A name is written however the person writing it likes, so the break is
anything that is not a letter or a digit, and the step from a small
letter into a capital that a run-together name is written with.
"""

PLURAL_ENDINGS = (
    ("ss", "ss"),
    ("us", "us"),
    ("is", "is"),
    ("ies", "y"),
    ("sses", "ss"),
    ("shes", "sh"),
    ("ches", "ch"),
    ("xes", "x"),
    ("oes", "o"),
    ("s", ""),
)
"""What a plural ending is replaced by, the endings that stand for themselves
first.

These are the regular endings of English and reach no further. A word
whose plural is made another way keeps whichever form it was written in.
"""

KEPT_ENDINGS = frozenset({"aircraft", "bus", "canvas", "gas", "lens", "series", "species"})
"""Words that end as a plural does and are already singular."""

NEAR_EDITS = 2
SHORT_NAME = 4
NEAR_EDITS_SHORT = 1
"""How far apart two names may be before they count as one mistyped.

A short name is held to one letter, because at two letters apart most
short words of the vocabulary reach each other.
"""


def canonical_label(name: str) -> str:
    """The name in the one form it is kept in: small letters, single spaces
    between the words, and every word in the singular.

    One thing is named the same way however it was typed, so that
    "Street Lights", "streetLights" and "street-light" all come to the
    same name and hold their tracks together.
    """
    words = (word for word in WORD_BREAK.split(name) if word)
    return " ".join(_singular(word.lower()) for word in words)


def _singular(word: str) -> str:
    if word in KEPT_ENDINGS or not word.endswith("s"):
        return word

    for plural, singular in PLURAL_ENDINGS:
        if word.endswith(plural):
            return word[: len(word) - len(plural)] + singular
    return word


def read_labels(path: Path) -> dict[str, list[str]]:
    """Every name given so far, with the tracks under it.

    A name written before it was brought to the form names are kept in
    is read in that form, so a file holding both "birds" and "bird"
    hands back the one name with the tracks of both.
    """
    if not path.is_file():
        return {}

    held = json.loads(path.read_text())
    gathered: dict[str, list[str]] = {}
    for name, keys in held.items():
        canonical = canonical_label(str(name))
        if canonical:
            gathered[canonical] = gathered.get(canonical, []) + [str(key) for key in keys]
    return {name: sorted(set(keys)) for name, keys in gathered.items()}


def near_label(name: str, *, known: Sequence[str]) -> str:
    """The name already in use that this one is most likely a mistyping of, and
    nothing when it stands on its own.

    A name is kept as it is typed, so "brid" and "bird" hold their
    tracks apart while meaning the same thing. Two names lying that
    close together are almost always one name typed twice, which is put
    to the person before either is written.
    """
    if not name or name in known:
        return ""

    allowed = NEAR_EDITS_SHORT if len(name) <= SHORT_NAME else NEAR_EDITS
    nearest, closest = "", allowed + 1
    for held in known:
        distance = _edits(name, held)
        if distance < closest:
            nearest, closest = held, distance
    return nearest


def _edits(name: str, other: str) -> int:
    """How many letters have to be put in, taken out, replaced, or swapped with
    the letter beside them to turn one name into the other.

    A swap counts as one rather than as two, because two letters typed
    the wrong way round is the commonest way a name is mistyped.
    """
    rows = [[0] * (len(other) + 1) for _ in range(len(name) + 1)]
    for down in range(len(name) + 1):
        rows[down][0] = down
    for across in range(len(other) + 1):
        rows[0][across] = across

    for down, letter in enumerate(name, start=1):
        for across, held in enumerate(other, start=1):
            step = min(
                rows[down - 1][across] + 1,
                rows[down][across - 1] + 1,
                rows[down - 1][across - 1] + int(letter != held),
            )
            if down > 1 and across > 1 and letter == other[across - 2] and name[down - 2] == held:
                step = min(step, rows[down - 2][across - 2] + 1)
            rows[down][across] = step
    return rows[-1][-1]


def label_of(labels: dict[str, list[str]], *, keys: list[str]) -> str:
    """The name most of these tracks are under, and nothing while none of them
    is under one.

    A name is given to every track of a cluster at once, so the name of
    a cluster is the name its tracks hold. A track of it put under
    another name on its own leaves the cluster under the name the rest
    of them still hold. Two names holding as many of the tracks as each
    other give the first of the two by their spelling.
    """
    wanted = set(keys)
    counts = {name: len(wanted.intersection(under)) for name, under in labels.items()}
    most = max(counts.values(), default=0)
    if not most:
        return ""

    return min(name for name, count in counts.items() if count == most)


def tags_of(labels: dict[str, list[str]], *, key: str) -> list[str]:
    """Every name this one track stands under, by their spelling."""
    return sorted(name for name, under in labels.items() if key in under)


def tagged(labels: dict[str, list[str]], *, names: Sequence[str]) -> set[str]:
    """The tracks standing under every one of these names.

    Each name narrows what the ones before it left, so two names hand
    back the tracks that are both things rather than either.
    """
    wanted = [canonical_label(name) for name in names]
    if not wanted:
        return set()
    return set.intersection(*(set(labels.get(name, [])) for name in wanted))


def write_tags(path: Path, labels: dict[str, list[str]], *, key: str, names: Sequence[str]) -> dict[str, list[str]]:
    """Put this track under exactly these names and write every name out
    again.

    The names are brought to the form names are kept in first. A name
    the track stood under and is no longer given lets it go, so the
    names handed in are what the track holds afterwards. Every other
    track stays where it is, which is how a tag differs from a cluster's
    name.
    """
    given = {canonical_label(name) for name in names} - {""}
    written = {held: [under for under in keys if under != key] for held, keys in labels.items()}
    for name in given:
        written[name] = sorted(set(written.get(name, [])) | {key})

    return _kept(path, written)


def write_labels(path: Path, labels: dict[str, list[str]], *, name: str, keys: list[str]) -> dict[str, list[str]]:
    """Put these tracks under this name and write every name out again.

    The name is brought to the form names are kept in first. The tracks
    are taken out of the name they were under, so a cluster named again
    moves rather than standing under both names. Under an empty name
    they are only taken out, which is how a name is withdrawn.
    """
    given = canonical_label(name)
    wanted = set(keys)
    written = {held: [key for key in under if key not in wanted] for held, under in labels.items()}
    if given:
        written[given] = sorted(set(written.get(given, [])) | wanted)

    return _kept(path, written)


def _kept(path: Path, written: dict[str, list[str]]) -> dict[str, list[str]]:
    """Write out every name that still holds a track, and hand back what the
    file now says."""
    written = {held: under for held, under in written.items() if under}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(written, indent=2, sort_keys=True) + "\n")
    return written
