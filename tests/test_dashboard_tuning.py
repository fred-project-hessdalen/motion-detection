"""Cutting a stretch of a recording out for the recordings page to tune on."""

import csv
from pathlib import Path

from hessdalen.dashboard.catalog import CLIP_NAME
from hessdalen.dashboard.tuning import METADATA_NAME, clip_name, cut_command, cut_for_tuning, note_label

RECORDING = Path("/videos/Cam5_2026-07-25__11-40-00_UTC.mkv")


def test_a_cut_is_named_the_way_the_catalog_reads_a_clip() -> None:
    name = clip_name(RECORDING, begin_s=365.32, end_s=379.0)

    clip = CLIP_NAME.match(Path(name).stem)
    assert clip is not None
    assert clip["recording"] == RECORDING.stem
    assert (float(clip["begin"]), float(clip["end"])) == (365.32, 379.0)


def test_the_cut_copies_the_streams_over_between_the_two_seconds() -> None:
    command = cut_command(RECORDING, begin_s=365.32, end_s=379.0, output=Path("/out/clip.mkv"))

    assert command[:2] == ["ffmpeg", "-hide_banner"]
    assert command[command.index("-ss") + 1] == "365.320"
    assert command[command.index("-to") + 1] == "379.000"
    assert command[command.index("-c") + 1] == "copy"
    assert command[-1] == "/out/clip.mkv"


def test_a_stretch_already_cut_is_not_cut_again(tmp_path, monkeypatch) -> None:
    root = tmp_path / "tuning"
    standing = root / "clips" / clip_name(RECORDING, begin_s=1.0, end_s=2.0)
    standing.parent.mkdir(parents=True)
    standing.write_bytes(b"cut")
    monkeypatch.setattr("hessdalen.dashboard.tuning.subprocess.run", _refuse)

    assert cut_for_tuning(RECORDING, begin_s=1.0, end_s=2.0, root=root) == standing


def test_a_clip_carries_what_it_was_cut_for_in_the_clips_own_time(tmp_path) -> None:
    note_label(tmp_path, name="a_clip_360.000_380.000.mkv", label="bird, far away", begin_s=5.3, end_s=9.0)

    [row] = _rows(tmp_path / METADATA_NAME)
    assert row == {
        "file": "a_clip_360.000_380.000.mkv",
        "movement": "1",
        "label": "bird, far away",
        "begin_s": "5.3",
        "end_s": "9.0",
    }


def test_a_clip_cut_again_keeps_one_row(tmp_path) -> None:
    note_label(tmp_path, name="a.mkv", label="bird", begin_s=1.0, end_s=2.0)
    note_label(tmp_path, name="b.mkv", label="moth", begin_s=1.0, end_s=2.0)
    note_label(tmp_path, name="a.mkv", label="plane", begin_s=3.0, end_s=4.0)

    rows = _rows(tmp_path / METADATA_NAME)
    assert [row["file"] for row in rows] == ["a.mkv", "b.mkv"]
    assert rows[0]["label"] == "plane"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _refuse(*args, **kwargs):
    raise AssertionError("a stretch already cut was cut again")
