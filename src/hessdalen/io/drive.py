"""The archive's recordings, listed in a file and fetched one at a time.

The archive is a shared cloud folder whose listing is kept outside this
repository, as a CSV of one row per video. A row carries where the video
sits in the archive, the id the store knows it by, and its byte count,
which is the only size available: the store answers a HEAD for some
files with an empty body and no length.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
VIEW_URL = "https://drive.google.com/file/d/{file_id}/view"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"

ATTEMPTS = 3
"""Tries a fetch gets before the caller sees the failure."""

STREAMS = 4
"""Connections one fetch through an account opens.

One stream carried a 31 MB recording in 58 seconds and four carried the
next one in 26, so the store hands out a single stream slowly.
"""

CHUNK = 1 << 20

TRAINING_BRANCH = "trainingData"

RECORDING_START = re.compile(r"cam\d+[_-]\d{4}-?\d{2}-?\d{2}[_-]+\d{2}[-_]?\d{2}[-_]?\d{2}", re.IGNORECASE)
"""The camera and start time a recording's name opens with, which every
cut of it repeats."""

UNLABELLED_WORDS = frozenset({"utc", "p", "crop", "diff", "video", "videos", "sdr"})
"""Words in the archive's names that describe the file or the folder
layout and say nothing about what was filmed."""


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

    @property
    def label(self) -> str:
        """What the archive calls the thing this video shows.

        The training branch files every video under a class folder, and
        that folder is the label. Elsewhere the label is the name of the
        nearest folder above the video that says more than a camera, a
        date or a time, or else the video's own name.
        """
        parts = self.path.split("/")
        if parts[1] == TRAINING_BRANCH:
            return parts[2]

        for name in [*reversed(parts[1:-1]), self.name.rsplit(".", 1)[0]]:
            words = label_words(name)
            if words:
                return words
        return ""

    @property
    def recording(self) -> str:
        """The recording this video was cut from, as its folder and the
        camera and start time its name opens with.

        A folder can hold several recordings, and a name that opens some
        other way stands for a recording of its own.
        """
        folder = self.path.rsplit("/", 1)[0]
        start = RECORDING_START.match(self.name)
        return f"{folder}/{start.group(0) if start else self.name}"


def label_words(name: str) -> str:
    """The words of a name that say what was filmed, joined by
    underscores."""
    return "_".join(word for word in re.split(r"[_\-\s]+", name) if word and not unlabelled(word))


def unlabelled(word: str) -> bool:
    return (
        word.isdigit() or word.lower() in UNLABELLED_WORDS or re.fullmatch(r"cam\d+", word, re.IGNORECASE) is not None
    )


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


def distinct(videos: list[ArchiveVideo]) -> list[ArchiveVideo]:
    """The first video listed under each path.

    The store lets two files share a path, and a fetch by path can only
    ask for one of them.
    """
    first: dict[str, ArchiveVideo] = {}
    for video in videos:
        first.setdefault(video.path, video)
    return list(first.values())


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
        refusal = _refusal(partial)
        partial.unlink(missing_ok=True)
        raise OSError(
            refusal or f"{video.name} arrived as {written} bytes against the {video.size_bytes} the listing gives"
        )

    partial.replace(target)
    return target


def fetch_through(remote: str, video: ArchiveVideo, target: Path) -> Path:
    """Write the video to target, pulling it through an rclone remote.

    The remote is rooted at the archive, so the first segment of a
    listing path names that root and is dropped. Going through an
    account rather than the public endpoint is what keeps a long run
    alive: the public endpoint stops serving a file once enough of it
    has been handed out, for as long as a day.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    source = remote + video.path.split("/", 1)[1]

    subprocess.run(["rclone", "copyto", "--multi-thread-streams", str(STREAMS), source, str(target)], check=True)
    if target.stat().st_size != video.size_bytes:
        raise OSError(f"{video.name} arrived as {target.stat().st_size} bytes against the {video.size_bytes} listed")
    return target


def _refusal(partial: Path) -> str:
    """What the store said instead of sending the video, if it said
    anything.

    A refusal arrives as an HTML page under a 200, so the only sign that
    it is not a video is the body itself.
    """
    head = partial.read_bytes()[:4096] if partial.exists() else b""
    if not head.lstrip().lower().startswith(b"<!doctype html"):
        return ""

    title = re.search(rb"<title>(.*?)</title>", head, re.S)
    return title.group(1).decode("utf-8", errors="replace").strip() if title else "the store answered with a web page"


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
