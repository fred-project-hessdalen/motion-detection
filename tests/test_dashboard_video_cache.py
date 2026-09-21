"""When the dashboard fetches a video the sift did not keep, and what it keeps
of the fetch."""

import subprocess

import pytest

from hessdalen.dashboard import video_cache
from hessdalen.dashboard.video_cache import MIN_FREE_BYTES, archive_video, cached_video, fetch_video, room_to_fetch

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


def test_a_whole_fetch_is_moved_into_place(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(video_cache, "fetch_through", _arrives_with(1000))

    landed = fetch_video(tmp_path, video=archive_video(ENTRY))

    assert landed == tmp_path / ENTRY["name"]
    assert landed.stat().st_size == 1000
    assert sorted(path.name for path in tmp_path.iterdir()) == [ENTRY["name"]]


def test_a_failed_fetch_leaves_nothing_behind(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(video_cache, "fetch_through", _fails)

    with pytest.raises(subprocess.CalledProcessError):
        fetch_video(tmp_path, video=archive_video(ENTRY))

    assert list(tmp_path.iterdir()) == []


def _arrives_with(size: int):
    def fetch_through(remote, video, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * size)
        return target

    return fetch_through


def _fails(remote, video, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * 10)
    raise subprocess.CalledProcessError(1, ["rclone", "copyto"])
