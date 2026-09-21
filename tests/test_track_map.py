"""What placing the corpus on a map and clustering it produces.

These need the analysis group, and are skipped where it is not
installed.
"""

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("umap")

from sklearn.metrics import adjusted_rand_score  # noqa: E402

from hessdalen.analysis.corpus import CORPUS_SCHEMA  # noqa: E402
from hessdalen.analysis.track_map import (  # noqa: E402
    FEATURES,
    SMOOTH_BELOW,
    UNASSIGNED,
    camera_of,
    map_corpus,
    standing_within_camera,
)

GROUP_SIZE = 50
ONE_SEED = (0,)
"""One run of each side, which the agreement then holds as it is, to keep
these tests quick."""


def test_a_clean_path_and_clutter_never_share_a_cluster() -> None:
    """The two sides of the roughness split are clustered apart, so no
    cluster can hold both."""
    mapped = map_corpus(_table([_streak(index) for index in range(GROUP_SIZE)] + _wanderers()), seeds=ONE_SEED)

    clusters = np.array(mapped.column("cluster").to_pylist())
    sides = np.array(mapped.column("side").to_pylist())
    for cluster in set(clusters[clusters != UNASSIGNED].tolist()):
        assert len(set(sides[clusters == cluster].tolist())) == 1


def test_two_kinds_of_clean_path_fall_into_two_clusters() -> None:
    """Long slow straight paths and short fast hooked ones differ on every
    descriptor that matters, so no two of them should share a cluster."""
    slow = [_streak(index) for index in range(GROUP_SIZE)]
    fast = [_hook(index) for index in range(GROUP_SIZE)]

    clusters = np.array(map_corpus(_table(slow + fast), seeds=ONE_SEED).column("cluster").to_pylist())

    slow_found, fast_found = clusters[:GROUP_SIZE], clusters[GROUP_SIZE:]
    assert set(slow_found[slow_found != UNASSIGNED]).isdisjoint(set(fast_found[fast_found != UNASSIGNED]))
    assert np.mean(slow_found != UNASSIGNED) > 0.5
    assert np.mean(fast_found != UNASSIGNED) > 0.5


def test_runs_that_agree_leave_the_clusters_as_one_run_finds_them() -> None:
    """The agreement between runs reclusters their shared memberships, so
    runs under one seed repeated give back that seed's clusters."""
    tracks = _table([_streak(index) for index in range(GROUP_SIZE)] + [_hook(index) for index in range(GROUP_SIZE)])

    once = np.array(map_corpus(tracks, seeds=ONE_SEED).column("cluster").to_pylist())
    twice = np.array(map_corpus(tracks, seeds=(0, 0)).column("cluster").to_pylist())

    assert adjusted_rand_score(once, twice) == pytest.approx(1.0)


def test_every_track_is_placed_on_the_map() -> None:
    tracks = _table([_streak(index) for index in range(GROUP_SIZE)] + _wanderers())

    mapped = map_corpus(tracks, seeds=ONE_SEED)

    assert mapped.num_rows == tracks.num_rows
    assert np.isfinite(np.array(mapped.column("x").to_pylist())).all()
    assert mapped.column("camera").to_pylist()[0] == "Cam1"


def test_a_camera_whose_tracks_all_stand_out_more_is_scored_against_itself() -> None:
    """A camera whose every track stands out twice as far from its sky is one
    site among several, and scoring within the camera takes that away."""
    first = [_streak(index, clip=f"Cam1_2025-01-01__00-00-00_{index:03d}") for index in range(GROUP_SIZE)]
    second = [
        {**_streak(index, clip=f"Cam2_2025-01-01__00-00-00_{index:03d}"), "peak_deviation_max": 2.0 * (110.0 + index)}
        for index in range(GROUP_SIZE)
    ]

    scores = standing_within_camera(_table(first + second), names=FEATURES)

    column = FEATURES.index("peak_deviation_max")
    assert scores[:GROUP_SIZE, column] == pytest.approx(scores[GROUP_SIZE:, column])


def test_the_camera_is_read_off_the_front_of_the_clip() -> None:
    assert camera_of("Cam2_2024-12-15__04-40-00_000") == "Cam2"
    assert camera_of("Cam1-20250820-125851-1755687531339-7_005") == "Cam1"
    assert camera_of("2025-12-25__14_20_00-UTC_crop_black") == "unknown"


def _table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=CORPUS_SCHEMA)


def _wanderers() -> list[dict[str, object]]:
    return [_wanderer(index) for index in range(GROUP_SIZE)]


def _streak(index: int, *, clip: str | None = None) -> dict[str, object]:
    wobble = 0.01 * (index % 7)
    return _row(
        index,
        clip=clip or f"Cam1_2025-01-01__00-00-00_{index:03d}",
        frames=120 + index % 5,
        straightness=0.97 - wobble,
        line_residual=0.0002 + wobble / 100.0,
        turn_mean=0.1 + wobble,
        speed_mean=0.002 + wobble / 100.0,
        peak_deviation_max=110.0 + index,
        roughness=0.1 + wobble,
        jump=1.5 + wobble,
    )


def _hook(index: int) -> dict[str, object]:
    wobble = 0.01 * (index % 7)
    return _row(
        index,
        clip=f"Cam1_2025-01-01__00-00-00_{200 + index:03d}",
        frames=14 + index % 5,
        straightness=0.5 + wobble,
        line_residual=0.02 + wobble / 100.0,
        turn_mean=0.9 + wobble,
        speed_mean=0.012 + wobble / 100.0,
        peak_deviation_max=30.0 + index / 10.0,
        roughness=0.35 + wobble,
        jump=3.0 + wobble,
    )


def _wanderer(index: int) -> dict[str, object]:
    wobble = 0.01 * (index % 7)
    return _row(
        index,
        clip=f"Cam1_2025-01-01__00-00-00_{100 + index:03d}",
        frames=40 + index % 5,
        straightness=0.2 + wobble,
        line_residual=0.01 + wobble / 100.0,
        turn_mean=1.8 + wobble,
        peak_deviation_max=20.0 + index / 10.0,
        roughness=SMOOTH_BELOW + 0.8 + wobble,
        jump=12.0 + wobble,
    )


def _row(index: int, *, clip: str, **descriptors: float) -> dict[str, object]:
    row: dict[str, object] = {name: 0.0 for name in CORPUS_SCHEMA.names}
    row.update(
        label="birds",
        event="Cam1_2025-01-01__00-00-00",
        clip=clip,
        recording=f"{clip}.mkv",
        track_id=index + 1,
        missed_frames=0,
    )
    row.update(descriptors)
    return row
