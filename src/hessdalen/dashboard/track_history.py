"""The tracks selected so far, and the step back and forward through them.

A track is picked, then one of its neighbours, and then the one it came
from is wanted again. The page keeps the order the tracks were selected
in and stands on one of them, so that a step back reaches the track
before it and a step forward returns.

Selecting a track from anywhere but a step drops whatever stood ahead of
where the history stands, which is how the back button of a browser
behaves.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class History:
    """The tracks selected so far, in the order they were selected, and where
    among them the page stands.

    A history holding nothing stands at -1.
    """

    keys: tuple[str, ...] = ()
    place: int = -1

    @property
    def standing(self) -> str | None:
        """The track the page stands on, or nothing while none is selected."""
        return self.keys[self.place] if 0 <= self.place < len(self.keys) else None

    @property
    def behind(self) -> bool:
        """Whether a track was selected before the one stood on."""
        return self.place > 0

    @property
    def ahead(self) -> bool:
        """Whether a track was selected after the one stood on and stepped back
        from."""
        return self.place < len(self.keys) - 1


def visited(history: History, *, key: str) -> History:
    """The history standing on this track, with it among the tracks selected.

    Selecting the track already stood on leaves the history as it is.
    Selecting any other drops the tracks ahead of where it stands, so
    that a selection made after a step back carries on from there.
    """
    if key == history.standing:
        return history

    kept = (*history.keys[: history.place + 1], key)
    return History(keys=kept, place=len(kept) - 1)


def stepped(history: History, *, offset: int) -> History:
    """The history moved this many tracks back or forward, held at its ends."""
    if not history.keys:
        return history

    return History(keys=history.keys, place=min(max(history.place + offset, 0), len(history.keys) - 1))
