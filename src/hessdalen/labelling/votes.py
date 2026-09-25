"""The vote by which a name spreads from seen tracks to the tracks around
them.

A cluster of the track map holds tracks of more than one kind, so a
name cannot be given to a cluster. It is given to a track a model has
looked at, and spreads from there to the unseen tracks nearest it in a
signature space. An unseen track takes a name only when enough of its
nearest seen tracks agree on it and the nearest of them lies within a
cap, so that a name never reaches across an empty stretch of the space
to a track nothing seen is like.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-not-found]

NAMED = "named"
"""Enough of the nearest seen tracks agree, and the nearest is within the
cap."""

DISAGREE = "disagree"
"""Seen tracks lie within the cap and too few of them agree."""

FAR = "far"
"""No seen track lies within the cap."""


@dataclass(frozen=True, slots=True)
class Rule:
    """How many seen tracks vote, how many of them have to agree, and how far
    the nearest may be."""

    votes: int
    agreement: int
    cap: float


@dataclass(frozen=True, slots=True)
class Vote:
    """What the nearest seen tracks say of one unseen track."""

    key: str
    name: str
    """The name that won, and nothing where none did."""

    share: int
    """How many of the voters stood under the winning name."""

    distance: float
    """How far the nearest voter is."""

    voters: tuple[str, ...]
    status: str


@dataclass(frozen=True, slots=True)
class Voters:
    """The seen tracks a vote is taken among: where each stands in signature
    space and the name it holds."""

    keys: Sequence[str]
    vectors: np.ndarray
    names: Sequence[str]


def cast(voters: Voters, *, keys: Sequence[str], vectors: np.ndarray, rule: Rule) -> list[Vote]:
    """The vote of every track given, one Vote each in the order given.

    A tie between names goes to the one first by its spelling. With no
    voter at all every track is far.
    """
    if not len(voters.keys) or not len(keys):
        return [Vote(key=key, name="", share=0, distance=np.inf, voters=(), status=FAR) for key in keys]

    count = min(rule.votes, len(voters.keys))
    nearest = NearestNeighbors(n_neighbors=count).fit(voters.vectors)
    distances, places = nearest.kneighbors(vectors)

    votes = []
    for key, row, apart in zip(keys, places, distances):
        chosen = tuple(voters.keys[place] for place in row)
        counted = Counter(voters.names[place] for place in row)
        name, share = min(counted.items(), key=lambda held: (-held[1], held[0]))
        if float(apart[0]) > rule.cap:
            status = FAR
        elif share < rule.agreement:
            status = DISAGREE
        else:
            status = NAMED
        votes.append(
            Vote(
                key=key,
                name=name if status == NAMED else "",
                share=share,
                distance=float(apart[0]),
                voters=chosen,
                status=status,
            )
        )
    return votes
