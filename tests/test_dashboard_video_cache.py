"""When the dashboard fetches a video the sift did not keep, and what it keeps
of the fetch."""

import subprocess
import sys
import time

import pytest

from hessdalen.dashboard import video_cache
from hessdalen.dashboard.video_cache import (
    MIN_FREE_BYTES,
    FetchProgress,
    archive_video,
    cached_video,
    copy_command,
    fetch_video,
    room_to_fetch,
)

ENTRY = {
    "archive_path": "cameras/2025/2025-06/2025-06-01/Cam1_2025-06-01__15-20-00_avalanche.mkv",
    "name": "Cam1_2025-06-01__15-20-00_avalanche.mkv",
    "file_id": "an-archive-id",
    "size_bytes": 1000,
    "track_count": 89,
}


def test_a_ledger_line_names_the_video_it_describes() -> None:
    video = archive_video(ENTRY)

    assert video.name == ENTRY["name"]
    assert video.path == ENTRY["archive_path"]
    assert video.size_bytes == 1000


def test_a_copy_of_the_listed_size_is_the_video(tmp_path) -> None:
    (tmp_path / ENTRY["name"]).write_bytes(b"x" * 1000)

    assert cached_video(tmp_path, video=archive_video(ENTRY)) == tmp_path / ENTRY["name"]


def test_a_copy_of_any_other_size_is_treated_as_absent(tmp_path) -> None:
    """A fetch that went wrong can leave a short file where the video belongs,
    and playing it would show a broken recording."""
    (tmp_path / ENTRY["name"]).write_bytes(b"x" * 999)

    assert cached_video(tmp_path, video=archive_video(ENTRY)) is None


def test_a_fetch_must_leave_the_space_the_sift_needs() -> None:
    video = archive_video(ENTRY)

    assert room_to_fetch(video, free_bytes=MIN_FREE_BYTES + 1000)
    assert not room_to_fetch(video, free_bytes=MIN_FREE_BYTES + 999)


def test_the_fetch_asks_rclone_for_the_video_below_the_archive_root(tmp_path) -> None:
    command = copy_command(archive_video(ENTRY), tmp_path / "part")

    assert command[:2] == ["rclone", "copyto"]
    assert command[-2:] == [
        "hessdalen:2025/2025-06/2025-06-01/Cam1_2025-06-01__15-20-00_avalanche.mkv",
        str(tmp_path / "part"),
    ]


def test_a_whole_fetch_is_moved_into_place(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(video_cache, "copy_command", _writes(chunks=1, chunk=1000, pause=0.0, status=0))

    landed = fetch_video(tmp_path, video=archive_video(ENTRY), on_progress=_ignored)

    assert landed == tmp_path / ENTRY["name"]
    assert landed.stat().st_size == 1000
    assert sorted(path.name for path in tmp_path.iterdir()) == [ENTRY["name"]]


def test_how_much_has_arrived_is_reported_while_a_fetch_writes(tmp_path, monkeypatch) -> None:
    """Rclone says nothing while it works, so the file it writes into is
    watched as it grows."""
    monkeypatch.setattr(video_cache, "WATCH_SECONDS", 0.01)
    monkeypatch.setattr(video_cache, "copy_command", _writes(chunks=10, chunk=100, pause=0.03, status=0))
    reports: list[FetchProgress] = []

    fetch_video(tmp_path, video=archive_video(ENTRY), on_progress=reports.append)

    arrived = [report.bytes_done for report in reports]
    assert arrived[0] == 0
    assert arrived[-1] == 1000
    assert any(0 < part < 1000 for part in arrived)
    assert arrived == sorted(arrived)
    assert {report.bytes_total for report in reports} == {1000}


def test_a_report_that_raises_stops_the_fetch_and_leaves_nothing_behind(tmp_path, monkeypatch) -> None:
    """A fetch for a track nobody looks at any more should not hold up the
    track looked at now."""
    monkeypatch.setattr(video_cache, "WATCH_SECONDS", 0.01)
    monkeypatch.setattr(video_cache, "copy_command", _writes(chunks=1000, chunk=1, pause=0.01, status=0))

    def stop_once_arriving(progress: FetchProgress) -> None:
        if progress.bytes_done > 0:
            raise _Stopped

    started = time.perf_counter()
    with pytest.raises(_Stopped):
        fetch_video(tmp_path, video=archive_video(ENTRY), on_progress=stop_once_arriving)

    assert time.perf_counter() - started < 5.0
    assert list(tmp_path.iterdir()) == []


def test_a_fetch_that_arrives_short_leaves_nothing_behind(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(video_cache, "copy_command", _writes(chunks=1, chunk=999, pause=0.0, status=0))

    with pytest.raises(OSError, match="999 bytes"):
        fetch_video(tmp_path, video=archive_video(ENTRY), on_progress=_ignored)

    assert list(tmp_path.iterdir()) == []


def test_a_failed_fetch_leaves_nothing_behind(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(video_cache, "copy_command", _writes(chunks=1, chunk=10, pause=0.0, status=1))

    with pytest.raises(subprocess.CalledProcessError):
        fetch_video(tmp_path, video=archive_video(ENTRY), on_progress=_ignored)

    assert list(tmp_path.iterdir()) == []


class _Stopped(Exception):
    pass


def _ignored(progress: FetchProgress) -> None:
    return None


def _writes(*, chunks: int, chunk: int, pause: float, status: int):
    """A copy command that writes the target in chunks, the way rclone grows
    the file it was given, and exits with the status."""
    script = (
        "import sys, time\n"
        "with open(sys.argv[1], 'wb') as written:\n"
        f"    for _ in range({chunks}):\n"
        f"        written.write(b'x' * {chunk})\n"
        "        written.flush()\n"
        f"        time.sleep({pause})\n"
        f"sys.exit({status})\n"
    )

    def copy_command(video, target):
        return [sys.executable, "-c", script, str(target)]

    return copy_command
