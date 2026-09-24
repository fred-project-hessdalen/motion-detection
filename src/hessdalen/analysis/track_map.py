"""Every track of the corpus placed on a map and sorted into clusters.

The folder labels play no part in either. They describe a whole
recording, and most tracks of a recording are its background activity,
so they are read afterwards to name what landed where and to check that
the grouping means something.

The tracks are first split by how evenly they move from step to step.
That split sits in a trough between two peaks of the corpus, and drawn
out, the tracks either side of it are clean paths on one side and
clutter on the other. Each side is then clustered on its own. Clustered
together, the clutter outnumbers the clean paths two to one and draws
their neighbourhoods, so a clean track that shares the clutter's small
blobs and low contrast is filed with it.

Each descriptor becomes a normal score among the tracks of its own
camera before anything else happens. Without that, the site is the first
thing any clustering separates on, because two cameras differ in sky,
lens and framing more than a bird differs from a branch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from sklearn.cluster import HDBSCAN  # type: ignore[import-not-found]
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-not-found]
from sklearn.preprocessing import QuantileTransformer  # type: ignore[import-not-found]
from tqdm import tqdm
from umap import UMAP  # type: ignore[import-not-found]

FEATURES = (
    "frames",
    "straightness",
    "line_residual",
    "velocity_residual",
    "acceleration_residual",
    "turn_mean",
    "speed_mean",
    "speed_deviation",
    "area_mean",
    "area_variation",
    "elongation_mean",
    "brightness_variation",
    "peak_deviation_max",
    "flicker",
    "roughness",
    "jump",
)
"""The descriptors the clustering reads.

Where a track sits in its frame and how bright its scene is are left
out, because both say more about the site than about what moved.
"""

CAMERA = re.compile(r"^(Cam\d+)")

MIN_CAMERA_TRACKS = 30
"""Tracks a camera needs before its tracks are scored among themselves.

A camera with fewer is scored against the whole corpus, because a rank
among a handful of tracks says almost nothing.
"""

SMOOTH_BELOW = 0.6
"""Roughness under which a track counts as a clean path, from the trough
between the two peaks of roughness over the corpus."""

NEIGHBOURS = 15
EMBEDDED = 5
LAYOUT_SPREAD = 0.1

CONSENSUS_SEEDS = (0, 1, 2, 3, 4)
"""UMAP seeds a side is clustered under before the runs are agreed.

A single embedding depends on its seed. Over the example corpus, the
clusters of the smooth side agreed between seeds 0, 1 and 2 only by an
adjusted Rand index of 0.36 to 0.68, though each run looked coherent
drawn out. The clusters that come back under many seeds are the ones
that belong to the tracks.
"""

TOGETHER = 0.5
"""Share of runs in which two tracks shared a cluster, at which the
agreement counts them as close."""

NEAREST = 0.01
"""Least distance between two tracks the agreement hands on.

Tracks that shared a cluster in every run would otherwise be nothing
apart, and the density clustering reads a distance of nothing as an
infinite density, which leaves it no way to weigh one cluster against
another.
"""


@dataclass(frozen=True, slots=True)
class Side:
    """One side of the roughness split, and how finely it is clustered."""

    name: str
    min_cluster_size: int


SMOOTH = Side(name="smooth", min_cluster_size=15)
ROUGH = Side(name="rough", min_cluster_size=40)
"""The rough side is clustered coarsely. Split finely, its clusters look
alike when drawn, which is what clutter of one kind does."""

MIN_SAMPLES = 5

UNASSIGNED = -1
"""The cluster of a track the clustering left out of every cluster."""

SPREAD_NEIGHBOURS = 5
"""Clustered tracks a track the clustering skipped takes its cluster from."""

PER_CLIP = 50
"""The sample size the roughness split was measured against.

A busy recording can hold thousands of tracks of one kind, and at that
weight it sets the density every other track is clustered against. On
the archive's full-day recordings, 45 of 373 clips held nine tenths of
all tracks, the largest 4316, and the trough in roughness that splits
clean paths from clutter filled in. Holding each clip to 50 tracks
brought the trough back.
"""


def sample_per_clip(tracks: pa.Table, *, per_clip: int | None) -> pa.Table:
    """At most per_clip tracks from each clip, and every track when per_clip is
    None."""
    return tracks.filter(pa.array(per_clip_mask(tracks, per_clip=per_clip)))


def per_clip_mask(tracks: pa.Table, *, per_clip: int | None) -> np.ndarray:
    """Which tracks stand when each clip is held to per_clip of them, spread
    evenly over its tracks in the order the tracker opened them, which runs
    with time.

    Every track stands when per_clip is None.
    """
    if per_clip is None:
        return np.ones(tracks.num_rows, dtype=bool)

    clips = np.array(tracks.column("clip").to_pylist())
    events = np.array(tracks.column("event").to_pylist())
    track_ids = np.asarray(tracks.column("track_id").to_numpy())
    kept = np.zeros(tracks.num_rows, dtype=bool)
    for place in set(zip(events.tolist(), clips.tolist())):
        held = np.flatnonzero((events == place[0]) & (clips == place[1]))
        ordered = held[np.argsort(track_ids[held])]
        chosen = np.unique(np.linspace(0, ordered.size - 1, min(per_clip, ordered.size)).astype(int))
        kept[ordered[chosen]] = True
    return kept


def map_corpus(tracks: pa.Table, *, seeds: tuple[int, ...], per_clip: int | None) -> pa.Table:
    """The corpus with a side, a cluster and a place on the map added to
    every track.

    Each side is clustered under every seed and the runs are agreed.
    Clusters are numbered across both sides, the smooth side's first,
    and the map is laid out under the first seed.

    The clustering is decided on at most per_clip tracks of each clip,
    for the reason PER_CLIP gives, and the tracks it was not decided on
    take the cluster their neighbours hold. Every track is laid out and
    handed back either way.
    """
    fitted = per_clip_mask(tracks, per_clip=per_clip)
    smooth = np.asarray(tracks.column("roughness").to_numpy()) < SMOOTH_BELOW
    clusters = np.full(tracks.num_rows, UNASSIGNED, dtype=np.int32)
    taken = 0
    for side, held in ((SMOOTH, smooth), (ROUGH, ~smooth)):
        if not held.any():
            continue

        scores = standing_within_camera(tracks.filter(pa.array(held)), names=FEATURES)
        on_side = fitted[held]
        found = _spread(scores, fitted=on_side, clusters=_agreed_clusters(scores[on_side], side=side, seeds=seeds))
        clusters[held] = np.where(found >= 0, found + taken, UNASSIGNED)
        taken += int(found.max()) + 1 if (found >= 0).any() else 0

    layout = _layout(standing_within_camera(tracks, names=FEATURES), seed=seeds[0])
    return (
        tracks.append_column("camera", pa.array([camera_of(clip) for clip in tracks.column("clip").to_pylist()]))
        .append_column("side", pa.array([SMOOTH.name if held else ROUGH.name for held in smooth]))
        .append_column("x", pa.array(layout[:, 0], type=pa.float32()))
        .append_column("y", pa.array(layout[:, 1], type=pa.float32()))
        .append_column("cluster", pa.array(clusters, type=pa.int32()))
    )


def _spread(scores: np.ndarray, *, fitted: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """The cluster of every track of one side, from the clusters the tracks
    nearest it were given.

    A track the clustering was not decided on takes what most of its
    SPREAD_NEIGHBOURS nearest clustered tracks hold, in the same
    descriptor space the clustering read. A tie goes to the lower
    cluster, and a side with nothing clustered leaves every track out.
    """
    found = np.full(scores.shape[0], UNASSIGNED, dtype=np.int32)
    found[fitted] = clusters

    known = np.flatnonzero(found >= 0)
    waiting = np.flatnonzero(~fitted)
    if known.size == 0 or waiting.size == 0:
        return found

    nearest = NearestNeighbors(n_neighbors=min(SPREAD_NEIGHBOURS, known.size)).fit(scores[known])
    votes = found[known][nearest.kneighbors(scores[waiting], return_distance=False)]
    found[waiting] = [np.bincount(row).argmax() for row in votes]
    return found


def _agreed_clusters(scores: np.ndarray, *, side: Side, seeds: tuple[int, ...]) -> np.ndarray:
    """The clusters of the tracks one side was fitted on that its runs under
    every seed agree on.

    Two tracks are as far apart as the share of runs that did not put them
    in one cluster, and those distances are clustered once more. A track
    most runs left out of every cluster stays out. Kept in, it would be
    the same distance from every other track, and a set of tracks all the
    same distance apart reads to the clustering as a cluster of its own.
    """
    if scores.shape[0] <= max(NEIGHBOURS, side.min_cluster_size):
        return np.full(scores.shape[0], UNASSIGNED, dtype=np.int32)

    runs = np.array(
        [
            _one_run(scores, side=side, seed=seed)
            for seed in tqdm(seeds, desc=f"Clustering {side.name} tracks", unit="run")
        ]
    )
    held = np.mean(runs >= 0, axis=0) >= TOGETHER
    agreed = np.full(scores.shape[0], UNASSIGNED, dtype=np.int32)
    if held.sum() <= side.min_cluster_size:
        return agreed

    kept = runs[:, held]
    together = np.mean([(run[:, None] == run[None, :]) & (run[:, None] >= 0) for run in kept], axis=0)
    apart = np.maximum(np.where(together >= TOGETHER, 1.0 - together, 1.0), NEAREST)
    np.fill_diagonal(apart, 0.0)
    agreed[held] = HDBSCAN(
        min_cluster_size=side.min_cluster_size, min_samples=MIN_SAMPLES, metric="precomputed", copy=True
    ).fit_predict(apart)
    return agreed


def _one_run(scores: np.ndarray, *, side: Side, seed: int) -> np.ndarray:
    embedded = UMAP(
        n_neighbors=NEIGHBOURS, min_dist=0.0, n_components=EMBEDDED, random_state=seed, n_jobs=1
    ).fit_transform(scores)
    return np.asarray(
        HDBSCAN(min_cluster_size=side.min_cluster_size, min_samples=MIN_SAMPLES, copy=True).fit_predict(embedded)
    )


def _layout(scores: np.ndarray, *, seed: int) -> np.ndarray:
    """Two dimensions for the map, over every track, so neighbours on the map
    are neighbours in the descriptors."""
    return np.asarray(
        UMAP(
            n_neighbors=min(NEIGHBOURS, scores.shape[0] - 1),
            min_dist=LAYOUT_SPREAD,
            n_components=2,
            random_state=seed,
            n_jobs=1,
        ).fit_transform(scores)
    )


def standing_within_camera(tracks: pa.Table, *, names: tuple[str, ...]) -> np.ndarray:
    """The named descriptors as normal scores among the tracks of each one's
    own camera, a column per name in the order given."""
    values = np.column_stack([np.asarray(tracks.column(name).to_numpy(), dtype=np.float64) for name in names])
    cameras = np.array([camera_of(clip) for clip in tracks.column("clip").to_pylist()])

    scores = _normal_scores(values, among=values)
    for camera, count in zip(*np.unique(cameras, return_counts=True)):
        if count >= MIN_CAMERA_TRACKS:
            held = cameras == camera
            scores[held] = _normal_scores(values[held], among=values[held])
    return scores


def camera_of(clip: str) -> str:
    """The camera a clip was recorded on, read off the front of its name."""
    match = CAMERA.match(clip)
    return match.group(1) if match else "unknown"


def _normal_scores(values: np.ndarray, *, among: np.ndarray) -> np.ndarray:
    """Where each value stands among the others, as a standard normal score."""
    transformer = QuantileTransformer(n_quantiles=min(1000, among.shape[0]), output_distribution="normal")
    return np.asarray(transformer.fit(among).transform(values))
