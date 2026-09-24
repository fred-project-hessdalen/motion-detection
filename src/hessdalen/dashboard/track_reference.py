"""A reference to a track that holds outside this repository.

A track is named here by its clip and its number, which mean nothing to
anyone without the corpus. A reference says instead where the recording
is in the archive and when in it the track ran, so that someone with the
link alone can open the video and watch the same seconds.

Seconds come from frame numbers, so they need the recording's frame
rate. The rate is read off the file where the file is on disk. Where it
is not, the reference stands on NOMINAL_RATE and says so, because the
cameras run at about that and a second either way still lands a viewer
on the event.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass

NOMINAL_RATE = 25.0
"""The frame rate a reference falls back on.

Measured over the kept recordings: 15 of 20 run at exactly this, the
rest within a twentieth of a frame per second of it.
"""

COLUMNS = ("recording", "track", "tags", "begin_s", "end_s", "begin", "end", "rate", "url")


@dataclass(frozen=True, slots=True)
class Reference:
    """Where a track's recording is and when in it the track ran."""

    recording: str
    track_id: int
    url: str
    begin_s: float
    end_s: float
    tags: tuple[str, ...]
    measured_rate: bool
    """Whether the seconds came from the recording's own frame rate."""


def reference(
    *,
    recording: str,
    track_id: int,
    url: str,
    first_frame: int,
    last_frame: int,
    frames_per_second: float,
    tags: Sequence[str] = (),
) -> Reference:
    """The reference to one track, from the frames it ran over."""
    rate = frames_per_second if frames_per_second > 0 else NOMINAL_RATE
    return Reference(
        recording=recording,
        track_id=track_id,
        url=url,
        begin_s=first_frame / rate,
        end_s=last_frame / rate,
        tags=tuple(tags),
        measured_rate=frames_per_second > 0,
    )


def reference_line(held: Reference) -> str:
    """The reference on one line, to be read by a person and pasted anywhere."""
    place = held.url or held.recording
    return f"{place} {clock(held.begin_s)}-{clock(held.end_s)}"


def clock(seconds: float) -> str:
    """The second of a recording as a person reads a player's clock, to a
    tenth."""
    hours, rest = divmod(max(seconds, 0.0), 3600.0)
    minutes, remaining = divmod(rest, 60.0)
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{remaining:04.1f}"
    return f"{int(minutes)}:{remaining:04.1f}"


def references_csv(references: Sequence[Reference]) -> str:
    """Every reference as a row, for a spreadsheet or another program."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(COLUMNS)
    for held in references:
        writer.writerow(
            (
                held.recording,
                held.track_id,
                " ".join(held.tags),
                f"{held.begin_s:.1f}",
                f"{held.end_s:.1f}",
                clock(held.begin_s),
                clock(held.end_s),
                "measured" if held.measured_rate else f"{NOMINAL_RATE:.0f} fps assumed",
                held.url,
            )
        )
    return out.getvalue()
