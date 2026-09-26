"""Every track as a small picture of its own shape, to find tracks that
look alike.

The map places tracks by their descriptors, and neighbours there share
speed, size and brightness more than they share shape. The picture
used here is the path similarity of hessdalen.analysis.track_images,
how far the blob stood from where it stood at every pair of moments,
which caught what kind of track a track is best of the pictures tried.
It is shrunk to a small square, so that a track's shape is a few
hundred numbers and the nearest shapes to one track are found by
plain distance over the whole corpus in a moment.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa

from hessdalen.analysis.spectra import track_signals
from hessdalen.analysis.track_images import path_similarity

SHAPE_SIDE = 16
"""Side of the shrunk picture. Sixteen keeps a hook, a loop and a
straight passage apart, and holds a track in 256 numbers."""

SHAPE_COLUMN = "shape"
KEY_COLUMN = "key"


def shape_vector(*, frame_number: np.ndarray, centre_x: np.ndarray, centre_y: np.ndarray, reach: float) -> np.ndarray:
    """One track's shape as a vector, from its centres frame by frame."""
    signals = track_signals(
        frame_number=frame_number,
        centre_x=centre_x,
        centre_y=centre_y,
        brightness=np.ones(frame_number.size),
        pixel_count=np.ones(frame_number.size, dtype=np.int64),
        reach=reach,
    )
    picture = path_similarity(signals).astype(np.float32)
    return np.asarray(cv2.resize(picture, (SHAPE_SIDE, SHAPE_SIDE), interpolation=cv2.INTER_AREA)).ravel()


def shape_table(paths: pd.DataFrame) -> pa.Table:
    """The shape of every track in the paths, keyed as the map keys them.

    The paths hold every track frame by frame with its event, clip and
    track id, which together are the key, and the frame's size, which
    the wobble is measured against.
    """
    keys = paths["event"] + "/" + paths["clip"] + "/" + paths["track_id"].astype(str)
    held = paths.assign(**{KEY_COLUMN: keys}).sort_values([KEY_COLUMN, "frame_number"])
    names: list[str] = []
    shapes: list[np.ndarray] = []
    for key, rows in held.groupby(KEY_COLUMN, sort=False):
        names.append(str(key))
        shapes.append(
            shape_vector(
                frame_number=rows["frame_number"].to_numpy(),
                centre_x=rows["centre_x"].to_numpy(dtype=np.float64),
                centre_y=rows["centre_y"].to_numpy(dtype=np.float64),
                reach=float(max(int(rows["frame_width"].iloc[0]), int(rows["frame_height"].iloc[0]))),
            )
        )
    stacked = np.vstack(shapes) if shapes else np.zeros((0, SHAPE_SIDE * SHAPE_SIDE), dtype=np.float32)
    column = pa.FixedSizeListArray.from_arrays(pa.array(stacked.ravel(), type=pa.float32()), SHAPE_SIDE * SHAPE_SIDE)
    return pa.table({KEY_COLUMN: pa.array(names, type=pa.string()), SHAPE_COLUMN: column})


def shape_matrix(table: pa.Table) -> tuple[list[str], np.ndarray]:
    """The keys and the shapes of a shape table, one row each."""
    keys = [str(key) for key in table.column(KEY_COLUMN).to_pylist()]
    flat = np.asarray(table.column(SHAPE_COLUMN).combine_chunks().flatten().to_numpy(), dtype=np.float32)
    return keys, flat.reshape(len(keys), SHAPE_SIDE * SHAPE_SIDE)


def nearest_shapes(matrix: np.ndarray, *, place: int, among: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """The rows among the given ones whose shape lies nearest the shape at
    place, nearest first, with their distances, the row itself left out."""
    others = among[among != place]
    if others.size == 0:
        return others, np.zeros(0)
    distances = np.linalg.norm(matrix[others] - matrix[place], axis=1)
    order = np.argsort(distances, kind="stable")[:count]
    return others[order], distances[order]
