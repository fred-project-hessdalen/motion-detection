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


SIGNATURES: dict[str, Signature] = {"descriptors": descriptors}
"""Every space by the name the labelling is told."""

DEFAULT_SIGNATURE = "descriptors"
