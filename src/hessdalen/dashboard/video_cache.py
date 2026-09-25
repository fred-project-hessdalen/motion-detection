"""Videos fetched from the archive for tracks whose video was not kept.

The sift keeps a track file for every video it detects on and deletes
most of the videos. A track of such a video can still be looked at by
fetching its video once, into a folder of the dashboard's own, so the
kept videos the sift tracks as its data stay as they are.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hessdalen.io.drive import STREAMS, ArchiveVideo

REMOTE = "hessdalen:"
"""The rclone remote the archive is reached through, signed in with the
project's own client."""

MIN_FREE_BYTES = 2 * 1024**3
"""Space a fetch has to leave on the disk.

A fetch pulls a whole recording down and the clips drawn from it are
written beside it, so the page holds a recording's worth twice over
before it has anything to show. This leaves room for that and for
whatever else the machine is doing, and it is the page's own floor,
reached from what the page itself does.
"""

FETCH_RATE = 31e6 / 26.0
"""Bytes a second a fetch moves, from a 31 MB recording that took 26 seconds
through the remote with four streams."""

WATCH_SECONDS = 0.5
"""How often a fetch's file is looked at to see how much has arrived."""

PART_SUFFIX = ".part"
"""What a fetch under way writes into, until it is whole and takes the
recording's own name."""


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
    """Whether fetching the video leaves the disk the space the page keeps
    free."""
    return free_bytes - video.size_bytes >= MIN_FREE_BYTES


def free_bytes(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(directory).free


@dataclass(frozen=True, slots=True)
class Fetched:
    """A recording the page has fetched: where it sits, how much of the disk
    it holds, and when the page last drew from it."""

    path: Path
    size_bytes: int
    used: float


def make_room(directory: Path, *, video: ArchiveVideo) -> int:
    """Let go of fetched recordings until this one has room, and say what is
    free once they are gone.

    A fetched recording is a copy of one in the archive and can be
    fetched again, so the page keeps the ones it has lately drawn from
    and lets the rest go rather than filling the disk. The recording
    about to be fetched is left alone, and so is a fetch under way.
    """
    free = free_bytes(directory)
    wanted = MIN_FREE_BYTES + video.size_bytes - free
    if wanted <= 0:
        return free

    for path in dropped_for(fetched_videos(directory, keeping=video.name), wanted=wanted):
        path.unlink(missing_ok=True)
    return free_bytes(directory)


def evictable_bytes(directory: Path, *, video: ArchiveVideo) -> int:
    """How much of the disk the page can give back by letting go of what it
    has fetched, with this recording kept.

    A fetch is offered on what the disk would hold once the page has let
    go of what it can, so that browsing a track never costs another
    recording its place. The room itself is made when the fetch runs.
    """
    return sum(held.size_bytes for held in fetched_videos(directory, keeping=video.name))


def dropped_for(fetched: Sequence[Fetched], *, wanted: int) -> tuple[Path, ...]:
    """The fetched recordings to let go so that this many bytes come free, the
    one drawn from longest ago first.

    Nothing is let go once the room is there, and what is kept is what
    the page has drawn from most recently, which is where the next track
    of a recording finds its own recording still on disk.
    """
    freed = 0
    dropping = []
    for held in sorted(fetched, key=lambda held: held.used):
        if freed >= wanted:
            break
        dropping.append(held.path)
        freed += held.size_bytes
    return tuple(dropping)


def fetched_videos(directory: Path, *, keeping: str) -> list[Fetched]:
    """Every whole recording the page has fetched, apart from the one named.

    A part file belongs to a fetch under way and is left where it is.
    """
    held = []
    for path in sorted(directory.iterdir()) if directory.is_dir() else []:
        if path.is_file() and path.name != keeping and path.suffix != PART_SUFFIX:
            found = path.stat()
            held.append(Fetched(path=path, size_bytes=found.st_size, used=found.st_mtime))
    return held


def used_now(directory: Path, *, recording: str) -> None:
    """Mark that the page has just drawn from this fetched recording, so that
    the ones let go are the ones nothing has wanted for longest.

    A recording the sift kept is not in this folder and is left alone,
    because the page never lets go of the sift's own data.
    """
    video = directory / recording
    if video.is_file():
        video.touch()


def fetch_video(directory: Path, *, video: ArchiveVideo, on_progress: Callable[[FetchProgress], None]) -> Path:
    """Fetch the video into the directory and return where it landed.

    The fetch writes a part file of its own and moves it into place once
    it is whole. A fetch that fails or is cut short leaves its part file
    behind, which is removed before the failure is passed on, so nothing
    half written is ever found and played.

    rclone gives the copy the modification time the recording has in the
    archive, which is months old, so the fetch marks it as used to keep
    what has just arrived from looking like the least wanted file there.
    """
    target = directory / video.name
    part = target.with_name(f"{target.name}{PART_SUFFIX}")
    on_progress(FetchProgress(bytes_done=0, bytes_total=video.size_bytes))
    try:
        _copy_watched(video, part=part, on_progress=on_progress)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(target)
    target.touch()
    on_progress(FetchProgress(bytes_done=video.size_bytes, bytes_total=video.size_bytes))
    return target


def _copy_watched(video: ArchiveVideo, *, part: Path, on_progress: Callable[[FetchProgress], None]) -> None:
    """Copy the video into the part file, reporting how much of it has arrived
    until rclone exits.

    rclone reports nothing to its caller, and it writes into the file it
    was given, whose size grows with what has arrived. The space it
    reserves on disk is the whole video from the start, so that says
    nothing. A report that raises stops rclone before the error is
    passed on.
    """
    part.parent.mkdir(parents=True, exist_ok=True)
    command = copy_command(video, part)
    rclone = subprocess.Popen(command)
    try:
        while not _exited(rclone):
            if part.is_file():
                arrived = min(part.stat().st_size, video.size_bytes)
                on_progress(FetchProgress(bytes_done=arrived, bytes_total=video.size_bytes))
    finally:
        if rclone.poll() is None:
            rclone.terminate()
        rclone.wait()

    if rclone.returncode != 0:
        raise subprocess.CalledProcessError(rclone.returncode, command)
    if part.stat().st_size != video.size_bytes:
        raise OSError(f"{video.name} arrived as {part.stat().st_size} bytes against the {video.size_bytes} listed")


def copy_command(video: ArchiveVideo, target: Path) -> list[str]:
    """The rclone call that copies the video to target, the one the archive
    sift makes.

    The remote is rooted at the archive, so the first segment of a
    listing path names that root and is dropped.
    """
    source = REMOTE + video.path.split("/", 1)[1]
    return ["rclone", "copyto", "--multi-thread-streams", str(STREAMS), source, str(target)]


def _exited(process: subprocess.Popen[bytes]) -> bool:
    """Whether the process exits within one look at its file."""
    try:
        process.wait(timeout=WATCH_SECONDS)
    except subprocess.TimeoutExpired:
        return False
    return True


def fetch_seconds(video: ArchiveVideo) -> float:
    """How long the fetch is expected to take at the rate measured through the
    remote."""
    return video.size_bytes / FETCH_RATE
