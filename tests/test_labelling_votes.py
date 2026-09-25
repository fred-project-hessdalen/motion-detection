"""The vote by which a name spreads from seen tracks to their neighbours.

These need the analysis group, and are skipped where it is not
installed.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from hessdalen.labelling.votes import DISAGREE, FAR, NAMED, Rule, Voters, cast  # noqa: E402

RULE = Rule(votes=3, agreement=2, cap=1.0)


def _voters() -> Voters:
    return Voters(
        keys=["b/1", "b/2", "b/3", "i/1", "i/2"],
        vectors=np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0]]),
        names=["bird", "bird", "bird", "insect", "insect"],
    )


def test_a_track_among_agreeing_seen_tracks_takes_their_name() -> None:
    (vote,) = cast(_voters(), keys=["x/1"], vectors=np.array([[0.05, 0.05]]), rule=RULE)

    assert vote.status == NAMED
    assert vote.name == "bird"
    assert vote.share == 3
    assert set(vote.voters) == {"b/1", "b/2", "b/3"}


def test_a_track_whose_seen_neighbours_disagree_takes_no_name() -> None:
    voters = Voters(
        keys=["b/1", "i/1", "m/1"],
        vectors=np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]]),
        names=["bird", "insect", "meteor"],
    )

    (vote,) = cast(voters, keys=["x/1"], vectors=np.array([[0.1, 0.1]]), rule=RULE)

    assert vote.status == DISAGREE
    assert vote.name == ""
    assert vote.share == 1


def test_a_track_with_no_seen_track_within_the_cap_takes_no_name() -> None:
    (vote,) = cast(_voters(), keys=["x/1"], vectors=np.array([[2.5, 2.5]]), rule=RULE)

    assert vote.status == FAR
    assert vote.name == ""
    assert vote.distance == pytest.approx(np.hypot(2.5, 2.4))


def test_with_nothing_seen_every_track_is_far() -> None:
    voters = Voters(keys=[], vectors=np.zeros((0, 2)), names=[])

    votes = cast(voters, keys=["x/1", "x/2"], vectors=np.zeros((2, 2)), rule=RULE)

    assert [vote.status for vote in votes] == [FAR, FAR]


def test_fewer_seen_tracks_than_votes_still_vote() -> None:
    voters = Voters(keys=["b/1", "b/2"], vectors=np.array([[0.0, 0.0], [0.1, 0.0]]), names=["bird", "bird"])

    (vote,) = cast(voters, keys=["x/1"], vectors=np.array([[0.0, 0.05]]), rule=RULE)

    assert vote.status == NAMED
    assert vote.share == 2


def test_a_tie_goes_to_the_name_first_by_its_spelling() -> None:
    voters = Voters(keys=["i/1", "b/1"], vectors=np.array([[0.0, 0.0], [0.1, 0.0]]), names=["insect", "bird"])

    (vote,) = cast(voters, keys=["x/1"], vectors=np.array([[0.05, 0.0]]), rule=Rule(votes=2, agreement=1, cap=1.0))

    assert vote.name == "bird"
