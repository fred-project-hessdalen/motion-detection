"""Contact sheets of tracks, the pictures a model names a track from.

A model reads images. The close-up video the page builds is too wide
and too faint to judge an object by, and a sheet of tight crops is
readable. A sheet holds one row per track: a caption, the track's path
drawn on its own, and crops of the recording taken at even steps
across the track. Each crop is cut one box half-width around the
detection, enlarged by nearest neighbour so its pixels stay pixels,
and stretched over the grey levels it holds, which is what turns a
dark speck on bright sky into something with a shape.

An isolation view is a sheet of one track with more steps, the whole
frame beside the close-up at each step, and the track's brightness and
size drawn under them, for a track the sheets could not settle.

A sheet is kept under a name made from what it draws, so the same rows
under the same layout are built once and a ledger line that names the
sheet keeps naming the same picture. The rows are written beside the
picture, so a sheet can be read back without the ledger.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from hessdalen.dashboard.panels import box_size
from hessdalen.dashboard.track_clip import (
    ClipProgress,
    StoredTrack,
    Stretch,
    close_up_centres,
    cut_close_up,
    stretch_source,
)
from hessdalen.io.video import TIMESTAMP_MASK_COORDS, VideoStream

MAX_ROWS = 8
"""Rows a sheet may hold, so that a row stays large enough to read."""

CAPTION_HEIGHT = 22
CAPTION_COLOUR = (200, 200, 200)
CAPTION_SCALE = 0.42
SMALLEST_CROP = 8
"""Pixels a crop is cut to at least, whatever the box half-width."""

PANEL_MARGIN = 10
"""Pixels a drawn path keeps from the edge of its panel."""

PATH_COLOURS = ((90, 30, 20), (40, 200, 220))
"""What the colour of a drawn path runs from and to, in blue green red.

The first frame of a track is dark blue and its last is yellow, as the
galleries of the page draw them.
"""

PATH_LINE = (70, 70, 70)
MARK_COLOUR = (0, 220, 255)
"""The box drawn round the detection on a whole-frame thumbnail."""

WHOLE_CAPTION = "the whole frame at the same steps, the detection boxed"

BRIGHTNESS_COLOUR = (80, 200, 255)
SIZE_COLOUR = (255, 160, 80)
SIGNAL_MARGIN = 6


@dataclass(frozen=True, slots=True)
class Layout:
    """How a sheet is drawn: steps across each track, how far a crop reaches
    from the detection in box half-widths, and the side of a panel."""

    frames: int
    radii: int
    side: int


SHEET = Layout(frames=6, radii=1, side=150)
"""What the sheets of a labelling pass are drawn under."""

ISOLATION = Layout(frames=12, radii=1, side=150)
"""What an isolation view is drawn under."""


@dataclass(frozen=True, slots=True)
class SheetTrack:
    """One track a sheet draws: what its row is captioned, where the track
    ran, the recording on disk it ran in, and how bright and how large
    its blob was on each frame it was matched."""

    key: str
    caption: str
    stored: StoredTrack
    video: Path
    brightness: np.ndarray
    pixel_count: np.ndarray


@dataclass(frozen=True, slots=True)
class Sheet:
    """A built sheet: where it is, and the tracks its rows show in order."""

    path: Path
    keys: tuple[str, ...]


def read_sheet(path: Path) -> Sheet:
    """A sheet built earlier, from the rows written beside it."""
    beside = json.loads(path.with_suffix(".json").read_text())
    return Sheet(path=path, keys=tuple(str(key) for key in beside["keys"]))


def sheet_path(directory: Path, *, keys: Sequence[str], layout: Layout, kind: str) -> Path:
    """Where the sheet of these rows under this layout is kept."""
    described = json.dumps({"kind": kind, "keys": list(keys), "layout": asdict(layout)}, sort_keys=True)
    return directory / f"{hashlib.sha1(described.encode()).hexdigest()[:12]}.png"


def build_sheet(directory: Path, *, tracks: Sequence[SheetTrack], layout: Layout) -> Sheet:
    """The sheet of these tracks, one row each, built unless it is there."""
    if not tracks or len(tracks) > MAX_ROWS:
        raise ValueError(f"A sheet holds 1 to {MAX_ROWS} rows, and {len(tracks)} were asked for.")

    keys = tuple(track.key for track in tracks)
    path = sheet_path(directory, keys=keys, layout=layout, kind="sheet")
    if path.is_file():
        return Sheet(path=path, keys=keys)

    rows = []
    for track in tracks:
        taken = _taken(track, layout=layout)
        panels = [_path_panel(track.stored, side=layout.side), *taken.close]
        rows.append(_row(track.caption, panels=panels, side=layout.side))
    return _written(path, rows=rows, keys=keys, layout=layout)


def build_isolation(directory: Path, *, track: SheetTrack, layout: Layout) -> Sheet:
    """The isolation view of one track, built unless it is there.

    Three rows: the path beside the close-up crops, the whole frame at
    the same steps with the detection boxed, and the brightness and the
    size of the blob over the track's frames.
    """
    keys = (track.key,)
    path = sheet_path(directory, keys=keys, layout=layout, kind="isolation")
    if path.is_file():
        return Sheet(path=path, keys=keys)

    taken = _taken(track, layout=layout)
    rows = [
        _row(track.caption, panels=[_path_panel(track.stored, side=layout.side), *taken.close], side=layout.side),
        _row(WHOLE_CAPTION, panels=[_blank(layout.side), *taken.whole], side=layout.side),
        _signals_row(track, width=(layout.frames + 1) * layout.side, side=layout.side),
    ]
    return _written(path, rows=rows, keys=keys, layout=layout)


@dataclass(frozen=True, slots=True)
class _Taken:
    """The crops taken across one track, close up and of the whole frame."""

    close: list[np.ndarray]
    whole: list[np.ndarray]


def _taken(track: SheetTrack, *, layout: Layout) -> _Taken:
    """Frames taken evenly across the track, each cut to the crop the close-up
    shows and stretched over the levels it holds, and the whole frame beside
    it.

    An object is a few pixels of a picture the camera records at 1920 by
    1080, so the crop is drawn pixel by pixel and its levels are pulled
    apart until the object stands out from the sky behind it.
    """
    stored = track.stored
    stretch = Stretch(begin_frame=stored.first_frame, end_frame=stored.last_frame)
    wanted = set(np.linspace(0, stretch.drawn_frames - 1, num=layout.frames).round().astype(int).tolist())

    stream = VideoStream(
        stretch_source(track.video, stretch=stretch, on_progress=_quiet),
        mask_coords=TIMESTAMP_MASK_COORDS,
        target_height=stored.frame_height,
    )
    height, width = stream.frame_shape
    size = box_size(height, width)
    cut = max(SMALLEST_CROP, 2 * layout.radii * size)
    crop = np.empty((cut, cut, 3), dtype=np.uint8)
    centres = close_up_centres(stored, stretch=stretch, size=size)

    taken = _Taken(close=[], whole=[])
    for offset, colour in enumerate(stream.stream_frames()):
        if offset >= stretch.drawn_frames:
            break
        if offset not in wanted:
            continue
        cut_close_up(crop, frame=colour.frame, centre=centres[offset])
        taken.close.append(cv2.resize(_pulled_apart(crop), (layout.side, layout.side), interpolation=cv2.INTER_NEAREST))
        taken.whole.append(_thumbnail(colour.frame, centre=centres[offset], size=size, side=layout.side))
    return taken


def _quiet(progress: ClipProgress) -> None:
    return None


def _pulled_apart(crop: np.ndarray) -> np.ndarray:
    """The crop over the whole range of levels, so a few levels of contrast
    become visible ones."""
    darkest, brightest = int(crop.min()), int(crop.max())
    if brightest - darkest < 2:
        return crop.copy()
    lifted = (crop.astype(np.float32) - darkest) * (255.0 / (brightest - darkest))
    return np.clip(lifted, 0, 255).astype(np.uint8)


def _thumbnail(frame: np.ndarray, *, centre: np.ndarray, size: int, side: int) -> np.ndarray:
    """The whole frame fitted into a panel, with a box round the detection so
    that where the object sits in the picture can be read."""
    marked = frame.copy()
    corner = (int(centre[0]) - size, int(centre[1]) - size)
    cv2.rectangle(marked, corner, (corner[0] + 2 * size, corner[1] + 2 * size), MARK_COLOUR, 2)

    height, width = marked.shape[:2]
    scale = side / max(height, width)
    fitted = cv2.resize(marked, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    panel = _blank(side)
    top = (side - fitted.shape[0]) // 2
    panel[top : top + fitted.shape[0], : fitted.shape[1]] = fitted
    return panel


def _path_panel(track: StoredTrack, *, side: int) -> np.ndarray:
    """The track's own path, fitted to a panel of the same size as a crop."""
    panel = _blank(side)
    points = np.column_stack([track.x, track.y]).astype(np.float64)
    span = max(1.0, float((points.max(axis=0) - points.min(axis=0)).max()))
    reach = side - 2 * PANEL_MARGIN
    placed = ((points - points.min(axis=0)) * (reach / span) + PANEL_MARGIN).round().astype(np.int32)

    for index in range(len(placed) - 1):
        cv2.line(panel, tuple(placed[index]), tuple(placed[index + 1]), PATH_LINE, 1)
    for index, point in enumerate(placed):
        share = index / max(1, len(placed) - 1)
        colour = tuple(int(first + share * (last - first)) for first, last in zip(*PATH_COLOURS))
        cv2.circle(panel, tuple(point), 2, colour, -1)
    return panel


def _signals_row(track: SheetTrack, *, width: int, side: int) -> np.ndarray:
    """The blob's brightness and size over the frames of the track, each drawn
    over its own range, under a caption naming the two colours."""
    row = np.zeros((side + CAPTION_HEIGHT, width, 3), dtype=np.uint8)
    _caption(row, "brightness (blue) and size (orange) over the track's frames")
    panel = row[CAPTION_HEIGHT:]
    frames = track.stored.frame_numbers
    for values, colour in ((track.brightness, BRIGHTNESS_COLOUR), (track.pixel_count, SIZE_COLOUR)):
        cv2.polylines(panel, [_curve(frames, values, width=width, height=side)], False, colour, 1)
    return row


def _curve(frames: np.ndarray, values: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """A signal as points across the panel, its frames along and its values
    up, each over the range it holds."""
    along = np.asarray(frames, dtype=np.float64)
    up = np.log1p(np.asarray(values, dtype=np.float64))
    across = (along - along.min()) / max(1.0, float(along.max() - along.min()))
    lifted = (up - up.min()) / max(1e-9, float(up.max() - up.min()))
    xs = SIGNAL_MARGIN + across * (width - 2 * SIGNAL_MARGIN)
    ys = height - SIGNAL_MARGIN - lifted * (height - 2 * SIGNAL_MARGIN)
    return np.column_stack([xs, ys]).round().astype(np.int32)


def _row(caption: str, *, panels: Sequence[np.ndarray], side: int) -> np.ndarray:
    """The panels side by side under the caption."""
    row = np.zeros((side + CAPTION_HEIGHT, len(panels) * side, 3), dtype=np.uint8)
    _caption(row, caption)
    for column, panel in enumerate(panels):
        row[CAPTION_HEIGHT:, column * side : (column + 1) * side] = panel
    return row


def _caption(row: np.ndarray, text: str) -> None:
    cv2.putText(row, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, CAPTION_SCALE, CAPTION_COLOUR, 1)


def _blank(side: int) -> np.ndarray:
    return np.zeros((side, side, 3), dtype=np.uint8)


def _written(path: Path, *, rows: Sequence[np.ndarray], keys: tuple[str, ...], layout: Layout) -> Sheet:
    """The rows stacked into one picture and written out, with the keys of
    the rows beside it."""
    width = max(row.shape[1] for row in rows)
    sheet = np.zeros((sum(row.shape[0] for row in rows), width, 3), dtype=np.uint8)
    top = 0
    for row in rows:
        sheet[top : top + row.shape[0], : row.shape[1]] = row
        top += row.shape[0]

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    path.with_suffix(".json").write_text(json.dumps({"keys": list(keys), "layout": asdict(layout)}, indent=2) + "\n")
    return Sheet(path=path, keys=keys)
