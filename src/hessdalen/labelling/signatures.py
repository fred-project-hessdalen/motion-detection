"""The signature spaces a track's neighbours are found in.

A name spreads through neighbours, so what counts as near decides what
the labelling does. The first space is the descriptors the clustering
reads, as normal scores among the tracks of each camera. A track
carried into a path image told kinds apart better in the comparison
that measured both, so a second space plugs in here under a name of
its own, and the labelling is told which by that name.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyarrow as pa

from hessdalen.analysis.track_map import FEATURES, standing_within_camera

Signature = Callable[[pa.Table], np.ndarray]
"""A row per track of the table, in the order of the table."""


def descriptors(tracks: pa.Table) -> np.ndarray:
    """The descriptors the clustering reads, scored within each camera."""
    return standing_within_camera(tracks, names=FEATURES)


def layout(tracks: pa.Table) -> np.ndarray:
    """Where the tracks lie on the map, which is the space the page's Nearest
    tracks gallery reads.

    The map is laid out from the descriptors under one seed, so what
    is near here moves a little from one map to the next.
    """
    return np.column_stack([np.asarray(tracks.column(axis).to_numpy(), dtype=np.float64) for axis in ("x", "y")])


SIGNATURES: dict[str, Signature] = {"descriptors": descriptors, "map": layout}
"""Every space by the name the labelling is told."""

DEFAULT_SIGNATURE = "descriptors"
