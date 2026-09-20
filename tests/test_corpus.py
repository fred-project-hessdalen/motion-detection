"""What gathering a sifted corpus of track files produces."""

import pyarrow.parquet as pq
import pytest

from hessdalen.analysis.corpus import PLACE_COLUMNS, read_corpus, write_corpus
from hessdalen.config import config
from hessdalen.domain.models import BlobMeasurement, DetectedMovement, MovementEvent
from hessdalen.io.tracks import write_tracks

FRAME_HEIGHT, FRAME_WIDTH = 1080, 1920
EVENT_FRAMES = 20


def test_a_track_carries_the_folders_it_was_filed_under(tmp_path) -> None:
    _write_clip(tmp_path, label="meteors", event="Cam2_2024-12-15__04-40-00", clip="clip_000", tracks=1)

    corpus = read_corpus(tmp_path)

    assert corpus.tracks.num_rows == 1
    assert [corpus.tracks.column(name).to_pylist()[0] for name in PLACE_COLUMNS] == [
        "meteors",
        "Cam2_2024-12-15__04-40-00",
        "clip_000",
    ]


def test_a_clip_holding_no_track_is_still_a_clip(tmp_path) -> None:
    """The sift looks at every clip of an event folder, and most of them hold
    nothing, which is what makes them the negatives of that folder."""
    _write_clip(tmp_path, label="planes", event="Cam1_2025-09-02__23-20-00", clip="clip_000", tracks=1)
    _write_clip(tmp_path, label="planes", event="Cam1_2025-09-02__23-20-00", clip="clip_001", tracks=0)

    corpus = read_corpus(tmp_path)

    assert len(corpus.clips) == 2
    assert corpus.tracks.num_rows == 1
    assert [clip.clip for clip in corpus.clips] == ["clip_000", "clip_001"]


def test_every_class_and_event_folder_is_found(tmp_path) -> None:
    _write_clip(tmp_path, label="birds", event="Cam1_2025-02-20__11-40-00", clip="clip_000", tracks=2)
    _write_clip(tmp_path, label="rain_snow", event="Cam2_2024-11-13__01-00-00", clip="clip_000", tracks=1)

    corpus = read_corpus(tmp_path)

    assert sorted({clip.label for clip in corpus.clips}) == ["birds", "rain_snow"]
    assert corpus.tracks.column("label").to_pylist() == ["birds", "birds", "rain_snow"]


def test_the_gathered_table_reads_back_whole(tmp_path) -> None:
    _write_clip(tmp_path, label="cosmics", event="Cam5_2026-03-01__00-00-00", clip="clip_000", tracks=1)
    corpus = read_corpus(tmp_path)

    rows = write_corpus(tmp_path / "gathered" / "corpus.parquet", corpus)

    written = pq.read_table(tmp_path / "gathered" / "corpus.parquet")
    assert rows == 1
    assert written.column_names[:3] == list(PLACE_COLUMNS)
    assert written.column("straightness").to_pylist()[0] == pytest.approx(1.0)


def _write_clip(root, *, label: str, event: str, clip: str, tracks: int) -> None:
    events: list[MovementEvent] = []
    for track_id in range(1, tracks + 1):
        events.extend(_straight_track(track_id))
    write_tracks(
        root / label / event / f"{clip}.parquet",
        recording=f"{event}_{clip}.mkv",
        frame_height=FRAME_HEIGHT,
        frame_width=FRAME_WIDTH,
        settings=config().settings,
        events=events,
    )


def _straight_track(track_id: int) -> list[DetectedMovement]:
    return [
        DetectedMovement(
            frame_number=frame,
            track_id=track_id,
            centroid=(100.0 + frame * 10.0, 500.0),
            blob=_blob_at(100.0 + frame * 10.0),
        )
        for frame in range(EVENT_FRAMES)
    ]


def _blob_at(x: float) -> BlobMeasurement:
    return BlobMeasurement(
        pixel_count=20,
        peak_deviation=18.0,
        brightness=900.0,
        centre_x=x,
        centre_y=500.0,
        major_axis=6.0,
        minor_axis=3.0,
    )
