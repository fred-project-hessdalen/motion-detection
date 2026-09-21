"""Every track of the corpus placed on a map and sorted into clusters.

The folder labels play no part in either. They describe a whole
recording, and most tracks of a recording are its background activity,
so they are read afterwards to name what landed where and to check that
the grouping means something.

Each descriptor becomes a normal score among the tracks of its own
camera before anything else happens. Without that, the site is the first
thing any clustering separates on, because two cameras differ in sky,
lens and framing more than a bird differs from a branch.
"""

from __future__ import annotations

import re

import numpy as np
import pyarrow as pa
from sklearn.cluster import HDBSCAN  # type: ignore[import-not-found]
from sklearn.decomposition import PCA  # type: ignore[import-not-found]
from sklearn.manifold import TSNE  # type: ignore[import-not-found]
from sklearn.preprocessing import QuantileTransformer  # type: ignore[import-not-found]

from hessdalen.analysis.corpus import PLACE_COLUMNS

NOT_DESCRIPTORS = frozenset({*PLACE_COLUMNS, "recording", "track_id"})

CAMERA = re.compile(r"^(Cam\d+)")

MIN_CAMERA_TRACKS = 30
"""Tracks a camera needs before its tracks are scored among themselves.

A camera with fewer is scored against the whole corpus, because a rank
among a handful of tracks says almost nothing.
"""

COMPONENTS = 6
"""Principal components the clustering runs in.

The 27 descriptors lean on each other heavily, and density clustering in
all of them finds one blob. Six components carry about four fifths of
the variance of the example corpus.
"""

MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 5
PERPLEXITY = 30.0
LAYOUT_SEED = 0

UNASSIGNED = -1
"""The cluster of a track the clustering left out of every cluster."""


def map_corpus(tracks: pa.Table) -> pa.Table:
    """The corpus with a place on the map and a cluster added to every
    track."""
    scores = standing_within_camera(tracks)
    components = PCA(n_components=min(COMPONENTS, *scores.shape)).fit_transform(scores)
    clusters = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, copy=True).fit_predict(components)
    layout = TSNE(
        n_components=2,
        perplexity=min(PERPLEXITY, float(tracks.num_rows - 1) / 3.0),
        init="pca",
        random_state=LAYOUT_SEED,
    ).fit_transform(components)

    return (
        tracks.append_column("camera", pa.array([camera_of(clip) for clip in tracks.column("clip").to_pylist()]))
        .append_column("x", pa.array(layout[:, 0], type=pa.float32()))
        .append_column("y", pa.array(layout[:, 1], type=pa.float32()))
        .append_column("cluster", pa.array(clusters, type=pa.int32()))
    )


def standing_within_camera(tracks: pa.Table) -> np.ndarray:
    """Every descriptor as a normal score among the tracks of its own camera,
    one column per name descriptor_names gives, in that order."""
    names = descriptor_names(tracks)
    values = np.column_stack([np.asarray(tracks.column(name).to_numpy(), dtype=np.float64) for name in names])
    cameras = np.array([camera_of(clip) for clip in tracks.column("clip").to_pylist()])

    scores = _normal_scores(values, among=values)
    for camera, count in zip(*np.unique(cameras, return_counts=True)):
        if count >= MIN_CAMERA_TRACKS:
            held = cameras == camera
            scores[held] = _normal_scores(values[held], among=values[held])
    return scores


def descriptor_names(tracks: pa.Table) -> list[str]:
    """The columns of a corpus table that describe a track, in table order."""
    return [name for name in tracks.column_names if name not in NOT_DESCRIPTORS]


def camera_of(clip: str) -> str:
    """The camera a clip was recorded on, read off the front of its name."""
    match = CAMERA.match(clip)
    return match.group(1) if match else "unknown"


def _normal_scores(values: np.ndarray, *, among: np.ndarray) -> np.ndarray:
    """Where each value stands among the others, as a standard normal score."""
    transformer = QuantileTransformer(n_quantiles=min(1000, among.shape[0]), output_distribution="normal")
    return np.asarray(transformer.fit(among).transform(values))
