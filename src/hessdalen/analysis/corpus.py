"""Every track of a sifted corpus in one table, tagged by where it came from.

The sift writes one track file per clip under a class folder and an
event folder, and a clip holding no track writes an empty file rather
than none. A row here is one track with those three names beside its
description, so any question about the corpus is asked of one table and
the aggregation is left to whoever asks.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from hessdalen.analysis.descriptors import ARROW_TYPES, TrackDescriptor, describe_file

PLACE_COLUMNS = ("label", "event", "clip")
"""The class folder, the recording folder inside it, and the clip's stem.

The label is the folder a recording was filed under and says nothing
about which of its clips holds the event, so it is a hint about the
whole event folder rather than a label on any one clip.
"""

CORPUS_SCHEMA = pa.schema(
    [(name, pa.string()) for name in PLACE_COLUMNS]
    + [(field.name, ARROW_TYPES[str(field.type)]) for field in fields(TrackDescriptor)]
)


@dataclass(frozen=True, slots=True)
class ClipPath:
    """Where one clip's track file sits, and what it was filed under."""

    label: str
    event: str
    clip: str
    path: Path


@dataclass(frozen=True, slots=True)
class Corpus:
    """The tracks of a sifted corpus, and every clip that was looked at.

    A clip that holds no track contributes no row, so the clips are kept
    beside the table to say what the corpus was measured over.
    """

    tracks: pa.Table
    clips: list[ClipPath]


def read_corpus(root: Path) -> Corpus:
    """Describe every track under a corpus root, tagged by class and event."""
    clips = corpus_clips(root)
    rows = [row for clip in tqdm(clips, desc="Describing tracks", unit="clip") for row in _rows_for_clip(clip)]
    return Corpus(tracks=pa.Table.from_pylist(rows, schema=CORPUS_SCHEMA), clips=clips)


def corpus_clips(root: Path) -> list[ClipPath]:
    """Every clip the sift has written a track file for, in corpus order."""
    return [
        ClipPath(label=path.parent.parent.name, event=path.parent.name, clip=path.stem, path=path)
        for path in sorted(root.glob("*/*/*.parquet"))
    ]


PATH_COLUMNS = ("track_id", "frame_number", "x", "y", "centre_x", "centre_y", "pixel_count", "brightness")
"""What a track did frame by frame, as its track file holds it."""


def gather_paths(clips: list[ClipPath]) -> pa.Table:
    """Every track's frames from every clip, in one table.

    Each row carries its clip's three names and the size of the frame
    its positions are pixels of, so a track can be drawn from this table
    alone.
    """
    return pa.concat_tables(list(path_tables(clips)))


def path_tables(clips: list[ClipPath]) -> Iterator[pa.Table]:
    """Each clip's frames as a table of its own, in corpus order.

    The whole corpus runs to millions of frames, which is tens of
    gigabytes held as one table, so a caller that writes as it reads
    takes them a clip at a time.
    """
    for clip in tqdm(clips, desc="Gathering paths", unit="clip"):
        table = _path_rows(clip)
        if table.num_rows > 0:
            yield table


def write_corpus(path: Path, corpus: Corpus) -> int:
    """Write the gathered tracks out, and return how many rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(corpus.tracks, path)
    return corpus.tracks.num_rows


def _path_rows(clip: ClipPath) -> pa.Table:
    held = pq.read_table(clip.path, columns=list(PATH_COLUMNS))
    metadata = held.schema.metadata or {}
    count = held.num_rows
    return (
        held.append_column("label", pa.array([clip.label] * count, type=pa.string()))
        .append_column("event", pa.array([clip.event] * count, type=pa.string()))
        .append_column("clip", pa.array([clip.clip] * count, type=pa.string()))
        .append_column("frame_width", pa.array([int(metadata[b"hessdalen_frame_width"])] * count, type=pa.int32()))
        .append_column("frame_height", pa.array([int(metadata[b"hessdalen_frame_height"])] * count, type=pa.int32()))
        .replace_schema_metadata(None)
    )


def _rows_for_clip(clip: ClipPath) -> list[dict[str, object]]:
    place = {"label": clip.label, "event": clip.event, "clip": clip.clip}
    return [{**place, **asdict(descriptor)} for descriptor in describe_file(clip.path)]
