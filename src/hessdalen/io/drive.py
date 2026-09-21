"""The archive's recordings, listed in a file and fetched one at a time.

The archive is a shared cloud folder whose listing is kept outside this
repository, as a CSV of one row per video. A row carries where the video
sits in the archive, the id the store knows it by, and its byte count,
which is the only size available: the store answers a HEAD for some
files with an empty body and no length.
"""

from __future__ import annotations

import csv
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
VIEW_URL = "https://drive.google.com/file/d/{file_id}/view"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"

ATTEMPTS = 3
"""Tries a fetch gets before the caller sees the failure."""

CHUNK = 1 << 20


@dataclass(frozen=True, slots=True)
class ArchiveVideo:
    """One video of the archive, as the listing describes it."""

    path: str
    """Where the video sits in the archive, as a slash-separated path."""

    name: str
    file_id: str
    size_bytes: int

    @property
    def url(self) -> str:
        """Where a person opens this video in a browser."""
        return VIEW_URL.format(file_id=self.file_id)

    @property
    def event(self) -> str:
        """The folder holding the video."""
        return self.path.rsplit("/", 2)[-2]

    @property
    def category(self) -> str:
        """The folder holding the event folder, which the archive files
        events under by what they show."""
        return self.path.rsplit("/", 3)[-3]

    @property
    def stem(self) -> str:
        """The archive path without the file's suffix."""
        return self.path.rsplit(".", 1)[0]


def read_inventory(path: Path) -> list[ArchiveVideo]:
    """Every video the listing names, in the order it names them."""
    with path.open(newline="") as handle:
        return [
            ArchiveVideo(
                path=row["path"],
                name=row["name"],
                file_id=row["id"],
                size_bytes=int(row["bytes"]),
            )
            for row in csv.DictReader(handle)
        ]


def matching(videos: list[ArchiveVideo], patterns: list[str]) -> list[ArchiveVideo]:
    """The videos whose archive path holds one of the patterns."""
    lowered = [pattern.lower() for pattern in patterns]
    return [video for video in videos if any(pattern in video.path.lower() for pattern in lowered)]


def cut_out(videos: list[ArchiveVideo]) -> list[ArchiveVideo]:
    """The videos that are cuts of another video in the listing.

    An event folder in the archive sits beside the whole recording it
    was cut from, under the same name. The cuts are a twentieth of the
    size each and carry the same scene, so a pass that reads the cuts
    has no use for the recording as well.
    """
    folders = {video.path.rsplit("/", 1)[0] for video in videos}
    return [video for video in videos if video.stem not in folders]


def fetch(video: ArchiveVideo, target: Path) -> Path:
    """Write the video to target and return where it landed.

    The bytes go to a neighbouring part file and are moved onto the
    target once the whole video is there, so an interrupted fetch never
    leaves something a later run mistakes for a finished file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    written = _download(video.file_id, partial)
    if written != video.size_bytes:
        partial.unlink(missing_ok=True)
        raise OSError(f"{video.name} arrived as {written} bytes against the {video.size_bytes} the listing gives")

    partial.replace(target)
    return target


def _download(file_id: str, partial: Path) -> int:
    """The byte count written, after as many tries as ATTEMPTS allows."""
    request = urllib.request.Request(DOWNLOAD_URL.format(file_id=file_id), headers={"User-Agent": USER_AGENT})

    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=CHUNK)
            return partial.stat().st_size
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == ATTEMPTS:
                raise
    raise AssertionError("unreachable")
