"""The recordings kept for development, with the labels recorded for them."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = frozenset({".mkv", ".mp4"})
COLLECTIONS = ("videos", "clips")
CLIP_NAME = re.compile(r"^(?P<recording>.+)_clip_(?P<begin>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)$")


@dataclass(frozen=True, slots=True)
class Label:
    name: str
    begin_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class DevelopmentVideo:
    path: Path
    collection: str
    labels: tuple[Label, ...]

    @property
    def name(self) -> str:
        return self.path.name


def development_videos(examples_dir: Path) -> list[DevelopmentVideo]:
    """Every recording under the example collections, each carrying the labels
    that apply to it."""
    labels = _labels_by_file(examples_dir / "metadata.csv")
    return [video for collection in COLLECTIONS for video in _videos_in(examples_dir / collection, labels)]


def _videos_in(directory: Path, labels: dict[str, tuple[Label, ...]]) -> list[DevelopmentVideo]:
    if not directory.is_dir():
        return []
    return [
        DevelopmentVideo(path=path, collection=directory.name, labels=_labels_for(path, labels))
        for path in sorted(directory.iterdir())
        if path.suffix in VIDEO_SUFFIXES
    ]


def _labels_for(path: Path, labels: dict[str, tuple[Label, ...]]) -> tuple[Label, ...]:
    own_labels = labels.get(path.name)
    if own_labels is not None:
        return own_labels

    clip = CLIP_NAME.match(path.stem)
    if clip is None:
        return ()

    recording = f"{clip['recording']}{path.suffix}"
    return _clip_labels(labels.get(recording, ()), begin_s=float(clip["begin"]), end_s=float(clip["end"]))


def _clip_labels(recording_labels: tuple[Label, ...], *, begin_s: float, end_s: float) -> tuple[Label, ...]:
    """The labels of a recording expressed in the time of a clip cut out of
    it."""
    return tuple(
        Label(
            name=label.name,
            begin_s=max(label.begin_s, begin_s) - begin_s,
            end_s=min(label.end_s, end_s) - begin_s,
        )
        for label in recording_labels
        if label.end_s > begin_s and label.begin_s < end_s
    )


def _labels_by_file(metadata_path: Path) -> dict[str, tuple[Label, ...]]:
    if not metadata_path.is_file():
        return {}

    with metadata_path.open(newline="") as metadata:
        rows = [row for row in csv.DictReader(metadata) if row["movement"] == "1"]

    by_file: dict[str, list[Label]] = {}
    for row in rows:
        label = Label(name=row["label"], begin_s=float(row["begin_s"]), end_s=float(row["end_s"]))
        by_file.setdefault(row["file"], []).append(label)
    return {file_name: tuple(file_labels) for file_name, file_labels in by_file.items()}
