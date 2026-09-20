"""The tracks a run found, written for later analysis to read back.

One file holds one recording. The rows carry what the detection stage
measured frame by frame, and the settings that produced them ride along
in the file's metadata, because a model trained on a corpus of these
learns the detector as much as it learns the sky.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from hessdalen.domain.models import MovementEvent
from hessdalen.processing.movement import MovementSettings

SCHEMA = pa.schema(
    [
        ("recording", pa.string()),
        ("track_id", pa.int32()),
        ("frame_number", pa.int32()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("pixel_count", pa.int32()),
        ("peak_deviation", pa.float32()),
        ("brightness", pa.float64()),
        ("major_axis", pa.float32()),
        ("minor_axis", pa.float32()),
    ]
)
"""One row per frame of every track the run reported.

The recording is a column of its own so that files gathered into one
dataset keep saying which recording each row came from.
"""


def write_tracks(
    path: Path,
    *,
    recording: str,
    frame_height: int,
    frame_width: int,
    settings: MovementSettings,
    events: Iterable[MovementEvent],
) -> int:
    """Write every track the events report, and return how many rows."""
    table = pa.Table.from_pylist(rows_from_events(events, recording=recording), schema=SCHEMA)
    described = table.replace_schema_metadata(
        provenance(frame_height=frame_height, frame_width=frame_width, settings=settings)
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(described, path)
    return described.num_rows


def rows_from_events(events: Iterable[MovementEvent], *, recording: str) -> list[dict[str, Any]]:
    """The frames of every track, in track and then frame order.

    A track is confirmed some frames after it starts and the detector
    replays the frames it held back, so the events of one track do not
    arrive in frame order.
    """
    found = [
        {
            "recording": recording,
            "track_id": event.track_id,
            "frame_number": event.frame_number,
            "x": event.centroid[0],
            "y": event.centroid[1],
            "pixel_count": event.blob.pixel_count,
            "peak_deviation": event.blob.peak_deviation,
            "brightness": event.blob.brightness,
            "major_axis": event.blob.major_axis,
            "minor_axis": event.blob.minor_axis,
        }
        for event in events
        if event.track_id is not None and event.centroid is not None and event.blob is not None
    ]
    return sorted(found, key=lambda row: (row["track_id"], row["frame_number"]))


def provenance(*, frame_height: int, frame_width: int, settings: MovementSettings) -> dict[bytes, bytes]:
    """Everything the detector was told, including the settings no control
    offers, so two runs that differ anywhere can be told apart.

    The frame is measured after resizing, because the rows give positions
    in its pixels and analysis reads them as ratios of its larger side.
    """
    return {
        b"hessdalen_frame_height": str(frame_height).encode(),
        b"hessdalen_frame_width": str(frame_width).encode(),
        b"hessdalen_settings": json.dumps(asdict(settings), sort_keys=True).encode(),
    }
