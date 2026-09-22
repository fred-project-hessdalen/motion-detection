"""Videos fetched from the archive for tracks whose video was not kept.

The sift keeps a track file for every video it detects on and deletes
most of the videos. A track of such a video can still be looked at by
fetching its video once, into a folder of the dashboard's own, so the
kept videos the sift tracks as its data stay as they are.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hessdalen.io.drive import ArchiveVideo, fetch_through

REMOTE = "hessdalen:"
"""The rclone remote the archive is reached through, signed in with the
project's own client."""

MIN_FREE_BYTES = 12 * 1024**3
"""Space a fetch has to leave on the disk.

The archive sift stops fetching below 8 GB free and skips every video it
reaches until it is run again, and its own queue holds up to about 12 GB
at a time, so a fetch here must not take the disk under that.
"""

FETCH_RATE = 31e6 / 26.0
"""Bytes a second a fetch moves, from a 31 MB recording that took 26 seconds
through the remote with four streams."""

WATCH_SECONDS = 0.5
"""How often a fetch's file is looked at to see how much has arrived."""


@dataclass(frozen=True, slots=True)
class FetchProgress:
    """How much of a video has arrived.

    Nothing has arrived while rclone is still reaching the archive,
    which takes some seconds before the first byte lands.
    """

    bytes_done: int
    bytes_total: int

    @property
    def fraction(self) -> float:
        return min(1.0, self.bytes_done / max(1, self.bytes_total))


def archive_video(entry: Mapping[str, Any]) -> ArchiveVideo:
    """The archive video a sift ledger line describes."""
    return ArchiveVideo(
        path=str(entry["archive_path"]),
        name=str(entry["name"]),
        file_id=str(entry["file_id"]),
        size_bytes=int(entry["size_bytes"]),
    )


def cached_video(directory: Path, *, video: ArchiveVideo) -> Path | None:
    """The fetched copy of the video, or None when no whole copy is there.

    A copy of any other size is a fetch that went wrong, and is treated
    as absent.
    """
    path = directory / video.name
    return path if path.is_file() and path.stat().st_size == video.size_bytes else None


def room_to_fetch(video: ArchiveVideo, *, free_bytes: int) -> bool:
    """Whether fetching the video leaves the space the archive sift needs."""
    return free_bytes - video.size_bytes >= MIN_FREE_BYTES


def free_bytes(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(directory).free


def fetch_video(directory: Path, *, video: ArchiveVideo, on_progress: Callable[[FetchProgress], None]) -> Path:
    """Fetch the video into the directory and return where it landed.

    The fetch writes a part file of its own and moves it into place once
    it is whole. A fetch that fails or is cut short leaves its part file
    behind, which is removed before the failure is passed on, so nothing
    half written is ever found and played.
    """
    target = directory / video.name
    part = target.with_name(f"{target.name}.part")
    on_progress(FetchProgress(bytes_done=0, bytes_total=video.size_bytes))
    try:
        _fetch_watched(video, part=part, on_progress=on_progress)
    except (OSError, subprocess.CalledProcessError):
        part.unlink(missing_ok=True)
        raise
    part.replace(target)
    on_progress(FetchProgress(bytes_done=video.size_bytes, bytes_total=video.size_bytes))
    return target


def _fetch_watched(video: ArchiveVideo, *, part: Path, on_progress: Callable[[FetchProgress], None]) -> None:
    """Fetch into the part file while another thread reports how much of it has
    arrived.

    rclone reports nothing to its caller, and it writes into the file it
    was given, whose size grows with what has arrived. The space it
    reserves on disk is the whole video from the start, so that says
    nothing.
    """
    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch, args=(part,), kwargs={"total": video.size_bytes, "on_progress": on_progress, "stop": stop}
    )
    watcher.start()
    try:
        fetch_through(REMOTE, video, part)
    finally:
        stop.set()
        watcher.join()


def _watch(part: Path, *, total: int, on_progress: Callable[[FetchProgress], None], stop: threading.Event) -> None:
    while not stop.wait(WATCH_SECONDS):
        if part.is_file():
            on_progress(FetchProgress(bytes_done=min(part.stat().st_size, total), bytes_total=total))


def fetch_seconds(video: ArchiveVideo) -> float:
    """How long the fetch is expected to take at the rate measured through the
    remote."""
    return video.size_bytes / FETCH_RATE
