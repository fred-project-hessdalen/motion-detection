"""The listing of the archive, and what a fetch out of it leaves behind."""

import io
import urllib.error
from pathlib import Path

import pytest

from hessdalen.io import drive
from hessdalen.io.drive import ArchiveVideo

HEADER = "path,name,id,mime,modified,bytes\n"
CLIP = "cameras/trainingData/birds/Cam1_2025-02-20__11-40-00/Cam1_2025-02-20__11-40-00_018.mkv"
RECORDING = "cameras/trainingData/birds/Cam1_2025-02-20__11-40-00.mkv"


def test_the_listing_reads_as_one_video_per_row(tmp_path) -> None:
    path = tmp_path / "inventory.csv"
    path.write_text(HEADER + f"{CLIP},Cam1_2025-02-20__11-40-00_018.mkv,abc123,video/x-matroska,02/20/25,31146896\n")

    videos = drive.read_inventory(path)

    assert len(videos) == 1
    assert videos[0].file_id == "abc123"
    assert videos[0].size_bytes == 31146896
    assert videos[0].event == "Cam1_2025-02-20__11-40-00"
    assert videos[0].url.endswith("abc123/view")


def test_a_pattern_keeps_the_videos_whose_path_holds_it() -> None:
    videos = [_video(CLIP), _video("cameras/trainingData/planes/Cam1_2025-02-11__05-40-00/plane_000.mkv")]

    kept = drive.matching(videos, ["TRAININGDATA/BIRDS"])

    assert [video.path for video in kept] == [CLIP]


def test_a_recording_beside_its_own_cuts_is_dropped() -> None:
    videos = [_video(RECORDING), _video(CLIP)]

    kept = drive.cut_out(videos)

    assert [video.path for video in kept] == [CLIP]


def test_a_recording_with_no_cuts_beside_it_is_kept() -> None:
    videos = [_video("cameras/trainingData/meteors/Cam3_2025-11-18__01-16.mp4")]

    assert drive.cut_out(videos) == videos


def test_a_fetch_leaves_the_video_and_no_part_file(tmp_path, monkeypatch) -> None:
    video = _video(CLIP, size_bytes=4)
    monkeypatch.setattr(drive.urllib.request, "urlopen", _answering(b"abcd"))
    target = tmp_path / "clip.mkv"

    drive.fetch(video, target)

    assert target.read_bytes() == b"abcd"
    assert list(tmp_path.iterdir()) == [target]


def test_a_short_arrival_raises_and_leaves_nothing(tmp_path, monkeypatch) -> None:
    video = _video(CLIP, size_bytes=8)
    monkeypatch.setattr(drive.urllib.request, "urlopen", _answering(b"abcd"))
    target = tmp_path / "clip.mkv"

    with pytest.raises(OSError, match="4 bytes against the 8"):
        drive.fetch(video, target)

    assert list(tmp_path.iterdir()) == []


def test_a_refused_fetch_raises_with_what_the_store_said(tmp_path, monkeypatch) -> None:
    page = b"<!DOCTYPE html><html><head><title>Quota exceeded</title></head><body>later</body></html>"
    monkeypatch.setattr(drive.urllib.request, "urlopen", _answering(page))

    with pytest.raises(OSError, match="Quota exceeded"):
        drive.fetch(_video(CLIP), tmp_path / "clip.mkv")

    assert list(tmp_path.iterdir()) == []


def test_a_remote_fetch_asks_for_the_path_under_the_archive_root(tmp_path, monkeypatch) -> None:
    video = _video(CLIP, size_bytes=4)
    asked = []

    def run(command, check):
        asked.append(command)
        Path(command[-1]).write_bytes(b"abcd")
        return None

    monkeypatch.setattr(drive.subprocess, "run", run)
    target = tmp_path / "clip.mkv"

    drive.fetch_through("hessdalen:", video, target)

    assert asked[0][:2] == ["rclone", "copyto"]
    assert asked[0][-2:] == ["hessdalen:" + CLIP.split("/", 1)[1], str(target)]
    assert target.read_bytes() == b"abcd"


def test_a_fetch_that_fails_once_is_tried_again(tmp_path, monkeypatch) -> None:
    video = _video(CLIP, size_bytes=4)
    answers = [urllib.error.URLError("reset"), b"abcd"]

    def urlopen(request, timeout=None):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return _Response(answer)

    monkeypatch.setattr(drive.urllib.request, "urlopen", urlopen)
    target = tmp_path / "clip.mkv"

    drive.fetch(video, target)

    assert target.read_bytes() == b"abcd"
    assert answers == []


def _video(path: str, size_bytes: int = 31146896) -> ArchiveVideo:
    return ArchiveVideo(path=path, name=path.rsplit("/", 1)[-1], file_id="abc123", size_bytes=size_bytes)


def _answering(payload: bytes):
    def urlopen(request, timeout=None):
        return _Response(payload)

    return urlopen


class _Response(io.BytesIO):
    """What urlopen hands back, as far as a download reads it."""

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_) -> None:
        self.close()
