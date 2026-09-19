"""Test the listing of development recordings and the labels they carry."""

from pathlib import Path

import pytest

from hessdalen.dashboard.catalog import development_videos

METADATA = """file,movement,label,begin_s,end_s
Cam2_night.mkv,1,meteor,44,46
Cam1_day.mkv,1,bird,0,3
Cam4_quiet.mkv,0,,0,0
"""

RECORDINGS = ("Cam1_day.mkv", "Cam2_night.mkv", "Cam4_quiet.mkv")
CLIPS = (
    "Cam2_night_clip_42.000_48.000.mkv",
    "Cam2_night_clip_10.000_20.000.mkv",
    "Cam5_unknown_clip_0.000_5.000.mkv",
)


@pytest.fixture
def examples_dir(tmp_path: Path) -> Path:
    (tmp_path / "metadata.csv").write_text(METADATA)
    for collection, names in (("videos", RECORDINGS), ("clips", CLIPS)):
        (tmp_path / collection).mkdir()
        for name in names:
            (tmp_path / collection / name).touch()
    (tmp_path / "videos" / "notes.txt").touch()
    return tmp_path


def test_recordings_come_before_clips_and_exclude_other_files(examples_dir: Path):
    videos = development_videos(examples_dir)

    assert [(video.collection, video.name) for video in videos] == [
        ("videos", "Cam1_day.mkv"),
        ("videos", "Cam2_night.mkv"),
        ("videos", "Cam4_quiet.mkv"),
        ("clips", "Cam2_night_clip_10.000_20.000.mkv"),
        ("clips", "Cam2_night_clip_42.000_48.000.mkv"),
        ("clips", "Cam5_unknown_clip_0.000_5.000.mkv"),
    ]


def test_a_recording_carries_the_labels_written_for_it(examples_dir: Path):
    videos = {video.name: video for video in development_videos(examples_dir)}

    labels = videos["Cam2_night.mkv"].labels
    assert [(label.name, label.begin_s, label.end_s) for label in labels] == [("meteor", 44.0, 46.0)]
    assert videos["Cam4_quiet.mkv"].labels == ()


def test_a_clip_carries_the_label_of_its_recording_in_its_own_time(examples_dir: Path):
    videos = {video.name: video for video in development_videos(examples_dir)}

    labels = videos["Cam2_night_clip_42.000_48.000.mkv"].labels
    assert [(label.name, label.begin_s, label.end_s) for label in labels] == [("meteor", 2.0, 4.0)]


def test_a_clip_outside_the_labelled_window_carries_no_label(examples_dir: Path):
    videos = {video.name: video for video in development_videos(examples_dir)}

    assert videos["Cam2_night_clip_10.000_20.000.mkv"].labels == ()
    assert videos["Cam5_unknown_clip_0.000_5.000.mkv"].labels == ()


def test_an_absent_examples_directory_yields_no_recordings(tmp_path: Path):
    assert development_videos(tmp_path / "missing") == []
