"""The contact sheets a model names a track from."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hessdalen.dashboard.track_clip import StoredTrack
from hessdalen.labelling.sheets import (
    CAPTION_HEIGHT,
    ISOLATION,
    MAX_ROWS,
    SHEET,
    Layout,
    SheetTrack,
    build_isolation,
    build_sheet,
    sheet_path,
)

FRAMES = 40
WIDTH, HEIGHT = 64, 48
BACKGROUND = 20
OBJECT = 60
"""A dot only a little brighter than the sky, which the stretch has to
pull apart from it."""

SMALL = Layout(frames=4, radii=1, side=40)


def test_a_sheet_holds_a_row_per_track_with_the_path_beside_the_crops(tmp_path: Path) -> None:
    tracks = [_sheet_track(tmp_path, key="a/1"), _sheet_track(tmp_path, key="a/2")]

    sheet = build_sheet(tmp_path / "sheets", tracks=tracks, layout=SMALL)

    picture = cv2.imread(str(sheet.path))
    assert sheet.keys == ("a/1", "a/2")
    assert picture.shape == (2 * (SMALL.side + CAPTION_HEIGHT), (SMALL.frames + 1) * SMALL.side, 3)


def test_a_crop_is_stretched_until_the_object_stands_out(tmp_path: Path) -> None:
    sheet = build_sheet(tmp_path / "sheets", tracks=[_sheet_track(tmp_path, key="a/1")], layout=SMALL)

    picture = cv2.imread(str(sheet.path))
    crops = picture[CAPTION_HEIGHT:, SMALL.side :]
    assert crops.max() == 255
    assert crops.min() == 0


def test_the_rows_are_written_beside_the_sheet(tmp_path: Path) -> None:
    sheet = build_sheet(tmp_path / "sheets", tracks=[_sheet_track(tmp_path, key="a/1")], layout=SMALL)

    beside = json.loads(sheet.path.with_suffix(".json").read_text())
    assert beside["keys"] == ["a/1"]
    assert beside["layout"] == {"frames": 4, "radii": 1, "side": 40}


def test_the_same_rows_under_the_same_layout_are_built_once(tmp_path: Path) -> None:
    track = _sheet_track(tmp_path, key="a/1")
    first = build_sheet(tmp_path / "sheets", tracks=[track], layout=SMALL)
    written = first.path.stat().st_mtime_ns

    again = build_sheet(tmp_path / "sheets", tracks=[track], layout=SMALL)

    assert again.path == first.path
    assert again.path.stat().st_mtime_ns == written


def test_another_layout_kind_or_caption_is_another_sheet(tmp_path: Path) -> None:
    keys = ["a/1"]
    held = sheet_path(tmp_path, keys=keys, captions=["track 1"], layout=SHEET, kind="sheet")

    assert held != sheet_path(tmp_path, keys=keys, captions=["track 1"], layout=ISOLATION, kind="sheet")
    assert held != sheet_path(tmp_path, keys=keys, captions=["track 1"], layout=SHEET, kind="isolation")
    assert held != sheet_path(tmp_path, keys=keys, captions=["track 2"], layout=SHEET, kind="sheet")


def test_a_sheet_holds_no_more_rows_than_can_be_read(tmp_path: Path) -> None:
    tracks = [_sheet_track(tmp_path, key=f"a/{index}") for index in range(MAX_ROWS + 1)]

    with pytest.raises(ValueError):
        build_sheet(tmp_path / "sheets", tracks=tracks, layout=SMALL)


def test_an_isolation_view_adds_the_whole_frame_and_the_signals(tmp_path: Path) -> None:
    layout = Layout(frames=6, radii=1, side=40)

    view = build_isolation(tmp_path / "sheets", track=_sheet_track(tmp_path, key="a/1"), layout=layout)

    picture = cv2.imread(str(view.path))
    assert view.keys == ("a/1",)
    assert picture.shape == (3 * (layout.side + CAPTION_HEIGHT), (layout.frames + 1) * layout.side, 3)


def _sheet_track(tmp_path: Path, *, key: str) -> SheetTrack:
    video = tmp_path / "recording.avi"
    if not video.is_file():
        _write_video(video)
    frames = np.arange(FRAMES, dtype=np.int64)
    return SheetTrack(
        key=key,
        caption=f"track {key}",
        stored=StoredTrack(
            track_id=1,
            frame_numbers=frames,
            x=np.array([_position(frame)[0] for frame in frames], dtype=np.float32),
            y=np.array([_position(frame)[1] for frame in frames], dtype=np.float32),
            frame_height=HEIGHT,
        ),
        video=video,
        brightness=np.full(FRAMES, 300.0),
        pixel_count=np.full(FRAMES, 4),
    )


def _write_video(path: Path) -> None:
    """A dark recording with one dot crossing it, a little brighter than the
    sky."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (WIDTH, HEIGHT))
    for frame in range(FRAMES):
        picture = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
        x, y = _position(frame)
        picture[y - 1 : y + 2, x - 1 : x + 2] = OBJECT
        writer.write(picture)
    writer.release()


def _position(frame: int) -> tuple[int, int]:
    return 8 + frame, HEIGHT // 2
