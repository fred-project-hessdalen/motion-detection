"""What the sifting run keeps, and what it writes down about the rest."""

import argparse
import importlib.util
import threading
from pathlib import Path

import pytest

from hessdalen.config import config
from hessdalen.domain.models import BlobMeasurement, DetectedMovement
from hessdalen.io.tracks import write_tracks

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "dev" / "sift_drive.py"
EMPTY_BLOB = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
FRAME_HEIGHT, FRAME_WIDTH = 1080, 1920


@pytest.fixture(scope="module")
def sift():
    spec = importlib.util.spec_from_file_location("sift_drive", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_travelling_track_outscores_flicker(sift, tmp_path) -> None:
    crossing = _written(sift, tmp_path / "crossing.parquet", _crossing())
    flicker = _written(sift, tmp_path / "flicker.parquet", _flicker())

    assert crossing.value > flicker.value
    assert crossing.first_frame == 0
    assert crossing.last_frame == 50


def test_a_recording_with_no_track_scores_nothing(sift, tmp_path) -> None:
    path = tmp_path / "quiet.parquet"
    write_tracks(
        path,
        recording="quiet.mkv",
        frame_height=FRAME_HEIGHT,
        frame_width=FRAME_WIDTH,
        settings=config().settings,
        events=[],
    )

    score = sift.score_tracks(path, frame_shape=(FRAME_HEIGHT, FRAME_WIDTH), settings=config().settings)

    assert score.value == 0.0
    assert score.track_count == 0


def test_the_best_cut_of_a_recording_is_kept_with_its_close_siblings(sift) -> None:
    morning = "Cam1_2025-02-20__11-40-00"
    findings = [
        _finding(sift, "birds", morning, f"{morning}_000.mkv", score=10.0),
        _finding(sift, "birds", morning, f"{morning}_001.mkv", score=6.0),
        _finding(sift, "birds", morning, f"{morning}_002.mkv", score=1.0),
        _finding(sift, "meteors", "night", "Cam2_2024-12-15__04-40-00_000.mkv", score=4.0),
    ]

    kept = sift.winners(findings, margin=0.5)

    assert [finding.name for finding in kept] == [
        f"{morning}_000.mkv",
        f"{morning}_001.mkv",
        "Cam2_2024-12-15__04-40-00_000.mkv",
    ]


def test_two_recordings_in_one_folder_are_judged_apart(sift) -> None:
    findings = [
        _finding(sift, "2025-06", "2025-06-19", "Cam2_2025-06-19__11-40-00_rod.mkv", score=10.0),
        _finding(sift, "2025-06", "2025-06-19", "Cam1_2025-06-19__20-00-00_DownwardsLight.mkv", score=1.0),
    ]

    kept = sift.winners(findings, margin=0.5)

    assert len(kept) == 2


def test_an_event_folder_where_nothing_moved_keeps_nothing(sift) -> None:
    findings = [_finding(sift, "birds", "morning", "a.mkv", score=0.0)]

    assert sift.winners(findings, margin=0.5) == []


def test_the_ledger_reads_back_what_was_appended(sift, tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    finding = _finding(sift, "birds", "morning", "a.mkv", score=3.5)

    sift.append(ledger, finding)
    sift.append(ledger, _finding(sift, "birds", "morning", "b.mkv", score=1.0))

    assert sift.read_ledger(ledger) == [finding, _finding(sift, "birds", "morning", "b.mkv", score=1.0)]


def test_a_recording_the_ledger_holds_is_not_fetched_again(sift, tmp_path) -> None:
    inventory = tmp_path / "inventory.csv"
    inventory.write_text(
        "path,name,id,mime,modified,bytes\n"
        "cameras/trainingData/birds/morning/a.mkv,a.mkv,one,video/x-matroska,02/20/25,10\n"
        "cameras/trainingData/birds/morning/b.mkv,b.mkv,two,video/x-matroska,02/20/25,10\n"
    )
    ledger = tmp_path / "ledger.jsonl"
    sift.append(ledger, _finding(sift, "birds", "morning", "a.mkv", score=3.5, file_id="one"))

    waiting = sift.pending(argparse.Namespace(inventory=inventory, ledger=ledger, select=["trainingData"], limit=None))

    assert [video.name for video in waiting] == ["b.mkv"]


def test_a_clip_carries_its_recordings_label_in_its_own_time(sift) -> None:
    recording = sift.LabelRow(file="a_018.mkv", label="birds", begin_s=32.9, end_s=55.0)

    rows = sift.clip_labels(["a_018_clip_27.900_60.029.mkv"], {"a_018.mkv": recording})

    assert rows == [sift.LabelRow(file="a_018_clip_27.900_60.029.mkv", label="birds", begin_s=5.0, end_s=27.1)]


def test_a_label_running_past_the_clip_stops_at_its_end(sift) -> None:
    recording = sift.LabelRow(file="a_018.mkv", label="birds", begin_s=1.0, end_s=60.0)

    rows = sift.clip_labels(["a_018_clip_0.000_30.000.mkv"], {"a_018.mkv": recording})

    assert (rows[0].begin_s, rows[0].end_s) == (1.0, 30.0)


def test_a_rescore_reads_the_score_back_off_the_track_file(sift, tmp_path) -> None:
    tracks = tmp_path / "crossing.parquet"
    expected = _written(sift, tracks, _crossing())
    ledger = tmp_path / "ledger.jsonl"
    stale = _finding(sift, "birds", "morning", "crossing.mkv", score=0.0)
    sift.append(ledger, sift.replace(stale, tracks_path=str(tracks), track_count=0))

    sift.rescore(ledger)

    [finding] = sift.read_ledger(ledger)
    assert finding.score == expected.value
    assert finding.track_count == 1
    assert (finding.first_frame, finding.last_frame) == (0, 50)


def test_a_scan_only_run_keeps_nothing(sift, tmp_path, monkeypatch) -> None:
    inventory = tmp_path / "inventory.csv"
    inventory.write_text(
        "path,name,id,mime,modified,bytes\n"
        "cameras/trainingData/birds/morning/a.mkv,a.mkv,one,video/x-matroska,02/20/25,10\n"
    )
    ledger = tmp_path / "ledger.jsonl"
    sift.append(ledger, _finding(sift, "birds", "morning", "a.mkv", score=3.5, file_id="one"))
    monkeypatch.setattr(sift, "collect", lambda args: pytest.fail("a scan-only run collected"))

    sift.main(
        argparse.Namespace(
            rescore=False, scan_only=True, ledger=ledger, inventory=inventory, select=["trainingData"], limit=None
        )
    )

    assert len(sift.read_ledger(ledger)) == 1


def test_a_recording_the_decoder_cannot_read_is_skipped(sift, tmp_path, capsys) -> None:
    staged = tmp_path / "edit.mov"
    staged.write_bytes(b"not a video")
    video = sift.ArchiveVideo(path="cameras/2025/edit.mov", name="edit.mov", file_id="one", size_bytes=11)
    ledger = tmp_path / "ledger.jsonl"
    room = threading.Semaphore(0)

    sift.take(
        sift.Arrival(video, staged, None),
        args=argparse.Namespace(ledger=ledger, target_height=FRAME_HEIGHT),
        frozen=sift.blob_hash(sift.CONFIG_PATH),
        room=room,
    )

    assert "edit.mov not read" in capsys.readouterr().out
    assert not staged.exists()
    assert not ledger.exists()
    assert room.acquire(blocking=False)


def test_the_blob_hash_is_the_one_git_would_give(sift, tmp_path) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")

    assert sift.blob_hash(path) == EMPTY_BLOB


def _written(sift, path: Path, events: list[DetectedMovement]):
    write_tracks(
        path,
        recording=path.name,
        frame_height=FRAME_HEIGHT,
        frame_width=FRAME_WIDTH,
        settings=config().settings,
        events=events,
    )
    return sift.score_tracks(path, frame_shape=(FRAME_HEIGHT, FRAME_WIDTH), settings=config().settings)


def _crossing() -> list[DetectedMovement]:
    """A bright object crossing half the frame over two seconds."""
    return [_movement(frame, track_id=1, x=200.0 + 10.0 * frame, peak=40.0) for frame in range(51)]


def _flicker() -> list[DetectedMovement]:
    """Three short tracks that stay where they started."""
    return [
        _movement(frame, track_id=track, x=600.0 + frame, peak=11.0)
        for track in (1, 2, 3)
        for frame in range(track * 10, track * 10 + 3)
    ]


def _movement(frame: int, *, track_id: int, x: float, peak: float) -> DetectedMovement:
    blob = BlobMeasurement(
        pixel_count=9,
        peak_deviation=peak,
        brightness=900.0,
        centre_x=x,
        centre_y=500.0,
        major_axis=3.0,
        minor_axis=3.0,
    )
    return DetectedMovement(frame_number=frame, track_id=track_id, centroid=(x, 500.0), blob=blob)


def _finding(sift, category: str, event: str, name: str, *, score: float, file_id: str = "one"):
    return sift.Finding(
        file_id=file_id,
        url=f"https://example.invalid/{file_id}",
        archive_path=f"cameras/trainingData/{category}/{event}/{name}",
        name=name,
        event=event,
        category=category,
        size_bytes=10,
        frame_count=1500,
        detection_seconds=5.0,
        target_height=FRAME_HEIGHT,
        config_blob=EMPTY_BLOB,
        tracks_path=f"data/corpus/tracks/{category}/{event}/{Path(name).stem}.parquet",
        track_count=3,
        score=score,
        first_frame=0,
        last_frame=50,
    )
