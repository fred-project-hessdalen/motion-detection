"""When the dashboard fetches a video the sift did not keep, and what it keeps
of the fetch."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hessdalen.dashboard import video_cache
from hessdalen.dashboard.video_cache import (
    MIN_FREE_BYTES,
    FetchProgress,
    archive_video,
    cached_video,
    copy_command,
    dropped_for,
    evictable_bytes,
    fetch_video,
    fetched_videos,
    make_room,
    room_to_fetch,
    used_now,
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


def test_a_fetch_must_leave_the_space_the_page_keeps_free() -> None:
    video = archive_video(ENTRY)

    assert room_to_fetch(video, free_bytes=MIN_FREE_BYTES + 1000)
    assert not room_to_fetch(video, free_bytes=MIN_FREE_BYTES + 999)


def test_the_recording_looked_at_longest_ago_is_let_go_first(tmp_path) -> None:
    _fetched(tmp_path, name="old.mkv", size=500, used=1.0)
    _fetched(tmp_path, name="middling.mkv", size=500, used=2.0)
    _fetched(tmp_path, name="newest.mkv", size=500, used=3.0)

    dropping = dropped_for(fetched_videos(tmp_path, keeping="wanted.mkv"), wanted=600)

    assert dropping == (tmp_path / "old.mkv", tmp_path / "middling.mkv")


def test_the_recording_about_to_be_fetched_and_a_fetch_under_way_are_left_alone(tmp_path) -> None:
    """The one being fetched is what the room is being made for, and a part
    file belongs to a fetch that is still writing it."""
    _fetched(tmp_path, name="wanted.mkv", size=500, used=1.0)
    _fetched(tmp_path, name="arriving.mkv.part", size=500, used=1.0)
    _fetched(tmp_path, name="old.mkv", size=500, used=2.0)

    held = fetched_videos(tmp_path, keeping="wanted.mkv")

    assert [found.path.name for found in held] == ["old.mkv"]


def test_drawing_from_a_recording_puts_it_behind_the_others(tmp_path) -> None:
    _fetched(tmp_path, name="old.mkv", size=500, used=1.0)
    _fetched(tmp_path, name="newest.mkv", size=500, used=2.0)

    used_now(tmp_path, recording="old.mkv")

    dropping = dropped_for(fetched_videos(tmp_path, keeping="wanted.mkv"), wanted=100)
    assert dropping == (tmp_path / "newest.mkv",)


def test_what_can_be_given_back_leaves_out_the_recording_being_fetched(tmp_path) -> None:
    """The fetch is offered on this figure, so that looking at a track of a
    recording that is not on disk never costs another recording its place."""
    _fetched(tmp_path, name="old.mkv", size=500, used=1.0)
    _fetched(tmp_path, name=str(ENTRY["name"]), size=700, used=2.0)

    assert evictable_bytes(tmp_path, video=archive_video(ENTRY)) == 500


def test_room_is_made_for_the_fetch_and_no_more(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(video_cache, "free_bytes", _disk_holding(tmp_path, spare=1500))
    _fetched(tmp_path, name="old.mkv", size=500, used=1.0)
    _fetched(tmp_path, name="middling.mkv", size=500, used=2.0)
    _fetched(tmp_path, name="newest.mkv", size=500, used=3.0)

    free = make_room(tmp_path, video=archive_video(ENTRY))

    assert free == MIN_FREE_BYTES + 1000
    assert [path.name for path in sorted(tmp_path.iterdir())] == ["newest.mkv"]


def test_nothing_is_let_go_while_the_disk_has_the_room(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(video_cache, "free_bytes", _disk_holding(tmp_path, spare=3000))
    _fetched(tmp_path, name="old.mkv", size=500, used=1.0)
    _fetched(tmp_path, name="newest.mkv", size=500, used=2.0)

    free = make_room(tmp_path, video=archive_video(ENTRY))

    assert free == MIN_FREE_BYTES + 2000
    assert [path.name for path in sorted(tmp_path.iterdir())] == ["newest.mkv", "old.mkv"]


def test_what_has_just_been_fetched_is_not_the_first_thing_let_go(tmp_path, monkeypatch) -> None:
    """rclone gives the copy the modification time the recording has in the
    archive, which is older than anything the page has drawn from."""
    monkeypatch.setattr(video_cache, "copy_command", _writes_dated(chunk=1000, dated=1.0))
    _fetched(tmp_path, name="drawn-from-earlier.mkv", size=500, used=2.0)

    fetch_video(tmp_path, video=archive_video(ENTRY), on_progress=_ignored)

    dropping = dropped_for(fetched_videos(tmp_path, keeping="wanted.mkv"), wanted=100)
    assert dropping == (tmp_path / "drawn-from-earlier.mkv",)


def test_a_recording_the_sift_kept_is_not_marked(tmp_path) -> None:
    """Those videos are the sift's own data and live outside the page's
    folder, where the page has nothing to say about them."""
    used_now(tmp_path, recording="kept-by-the-sift.mkv")

    assert list(tmp_path.iterdir()) == []


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


def _fetched(directory: Path, *, name: str, size: int, used: float) -> Path:
    """A recording the page fetched, of that size, last drawn from then."""
    path = directory / name
    path.write_bytes(b"x" * size)
    os.utime(path, (used, used))
    return path


def _writes_dated(*, chunk: int, dated: float):
    """A copy command that gives what it wrote the modification time the
    archive holds, the way rclone does."""
    script = (
        f"import os, sys\nopen(sys.argv[1], 'wb').write(b'x' * {chunk})\nos.utime(sys.argv[1], ({dated}, {dated}))\n"
    )

    def copy_command(video, target):
        return [sys.executable, "-c", script, str(target)]

    return copy_command


def _disk_holding(directory: Path, *, spare: int):
    """A disk with that many bytes over the page's floor, once the fetched
    recordings have taken their share of it."""

    def free_bytes(_: Path) -> int:
        return MIN_FREE_BYTES + spare - sum(path.stat().st_size for path in directory.iterdir())

    return free_bytes


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
