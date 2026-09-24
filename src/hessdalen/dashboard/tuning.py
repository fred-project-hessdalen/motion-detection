"""Clips cut out of a recording so the detector can be tuned on them.

A track on the map says where something moved, and what is worth tuning
is usually what it missed beside that track. The recordings page runs
the detector under settings a person moves, so a stretch of the
recording is cut out and put where that page lists it.

The cut copies the streams over without re-encoding, so what the
detector reads is what the archive holds, and it starts at the keyframe
at or before the stretch, which is as close as a copy can cut.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

METADATA_NAME = "metadata.csv"
METADATA_COLUMNS = ("file", "movement", "label", "begin_s", "end_s")
COLLECTION = "clips"
"""The folder of a collection the catalog reads, under the tuning root."""

MARGIN_SECONDS = 5.0
"""Seconds kept either side of the stretch a track ran over.

Enough for the background model to settle before the track opens, and
to show what else moves around it.
"""


def clip_name(video: Path, *, begin_s: float, end_s: float) -> str:
    """What a cut of this stretch is called, the way the catalog reads a
    clip's name."""
    return f"{video.stem}_clip_{begin_s:.3f}_{end_s:.3f}{video.suffix}"


def cut_command(video: Path, *, begin_s: float, end_s: float, output: Path) -> list[str]:
    """The ffmpeg call that copies this stretch out of the recording."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{begin_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(video),
        "-c",
        "copy",
        "-map",
        "0",
        str(output),
    ]


def cut_for_tuning(video: Path, *, begin_s: float, end_s: float, root: Path) -> Path:
    """Cut this stretch out of the recording and return where it landed.

    A stretch already cut is left as it is, so asking twice costs
    nothing.
    """
    directory = root / COLLECTION
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / clip_name(video, begin_s=begin_s, end_s=end_s)
    if output.is_file():
        return output

    partial = output.with_name(f"{output.stem}.part{output.suffix}")
    subprocess.run(cut_command(video, begin_s=begin_s, end_s=end_s, output=partial), check=True)
    partial.replace(output)
    return output


def note_label(root: Path, *, name: str, label: str, begin_s: float, end_s: float) -> None:
    """Record what the clip was cut for, in the file the catalog reads labels
    from.

    The seconds are the stretch in the clip's own time. A clip cut again
    keeps one row.
    """
    path = root / METADATA_NAME
    rows = {row["file"]: row for row in _held(path)}
    rows[name] = {
        "file": name,
        "movement": "1",
        "label": label,
        "begin_s": f"{begin_s:.1f}",
        "end_s": f"{end_s:.1f}",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[file] for file in sorted(rows))


def _held(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
