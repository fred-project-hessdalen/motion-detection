"""What a written track file holds, and what it says about the run behind
it."""

import json

import pyarrow.parquet as pq

from hessdalen.config import config
from hessdalen.domain.models import DetectedMovement, MovementEvent
from hessdalen.io.tracks import write_tracks
from synthetic import blob_measurement

RECORDING = "Cam1_2025-02-20__11-40-00_018.mkv"
FRAME_HEIGHT, FRAME_WIDTH = 1080, 1920


def test_a_frame_without_movement_writes_no_row(tmp_path) -> None:
    path = tmp_path / "tracks.parquet"

    rows = _write(path, [MovementEvent(frame_number=frame, track_id=None, centroid=None) for frame in range(4)])

    assert rows == 0
    assert pq.read_table(path).column_names == [
        "recording",
        "track_id",
        "frame_number",
        "x",
        "y",
        "pixel_count",
        "peak_deviation",
        "brightness",
        "major_axis",
        "minor_axis",
    ]


def test_replayed_frames_are_written_in_track_and_frame_order(tmp_path) -> None:
    """A track is confirmed some frames after it starts, so its held-back
    frames arrive after a second track has already reported."""
    path = tmp_path / "tracks.parquet"

    _write(path, [_movement(7, 2), _movement(4, 1), _movement(5, 1), _movement(8, 2), _movement(6, 1)])

    table = pq.read_table(path)
    assert table.column("track_id").to_pylist() == [1, 1, 1, 2, 2]
    assert table.column("frame_number").to_pylist() == [4, 5, 6, 7, 8]


def test_a_row_carries_the_blob_the_frame_measured(tmp_path) -> None:
    path = tmp_path / "tracks.parquet"
    blob = blob_measurement()

    _write(path, [_movement(4, 1)])

    table = pq.read_table(path)
    assert table.column("recording").to_pylist() == [RECORDING]
    assert table.column("pixel_count").to_pylist() == [blob.pixel_count]
    assert table.column("brightness").to_pylist() == [blob.brightness]
    assert table.column("major_axis").to_pylist() == [blob.major_axis]


def test_the_file_says_what_the_detector_was_told(tmp_path) -> None:
    path = tmp_path / "tracks.parquet"

    _write(path, [_movement(4, 1)])

    metadata = pq.read_table(path).schema.metadata
    assert metadata[b"hessdalen_frame_height"] == str(FRAME_HEIGHT).encode()
    assert metadata[b"hessdalen_frame_width"] == str(FRAME_WIDTH).encode()
    written = json.loads(metadata[b"hessdalen_settings"])
    assert written["detection"]["detection_sigma"] == config().settings.detection.detection_sigma
    assert written["background"]["noise_floor"] == config().settings.background.noise_floor


def _movement(frame_number: int, track_id: int) -> DetectedMovement:
    return DetectedMovement(
        frame_number=frame_number,
        track_id=track_id,
        centroid=(float(frame_number), 10.0),
        blob=blob_measurement(),
    )


def _write(path, events) -> int:
    return write_tracks(
        path,
        recording=RECORDING,
        frame_height=FRAME_HEIGHT,
        frame_width=FRAME_WIDTH,
        settings=config().settings,
        events=events,
    )
