"""What placing the corpus on a map and clustering it produces.

These need the analysis group, and are skipped where it is not
installed.
"""

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("sklearn")

from hessdalen.analysis.corpus import CORPUS_SCHEMA  # noqa: E402
from hessdalen.analysis.track_map import (  # noqa: E402
    UNASSIGNED,
    camera_of,
    descriptor_names,
    map_corpus,
    standing_within_camera,
)

GROUP_SIZE = 40


def test_two_kinds_of_track_fall_into_two_clusters() -> None:
    """Straight bright streaks and crooked dim wanderers differ on every
    descriptor that matters, so no two of them should share a cluster."""
    tracks = _table([_streak(index) for index in range(GROUP_SIZE)] + [_wanderer(index) for index in range(GROUP_SIZE)])

    clusters = np.array(map_corpus(tracks).column("cluster").to_pylist())

    streaks, wanderers = clusters[:GROUP_SIZE], clusters[GROUP_SIZE:]
    assert set(streaks[streaks != UNASSIGNED]).isdisjoint(set(wanderers[wanderers != UNASSIGNED]))
    assert np.mean(streaks != UNASSIGNED) > 0.5
    assert np.mean(wanderers != UNASSIGNED) > 0.5


def test_every_track_is_placed_on_the_map() -> None:
    tracks = _table([_streak(index) for index in range(GROUP_SIZE)] + [_wanderer(index) for index in range(GROUP_SIZE)])

    mapped = map_corpus(tracks)

    assert mapped.num_rows == tracks.num_rows
    assert np.isfinite(np.array(mapped.column("x").to_pylist())).all()
    assert mapped.column("camera").to_pylist()[0] == "Cam1"


def test_a_camera_that_reads_brighter_throughout_is_scored_against_itself() -> None:
    """A camera whose every track is twice as bright is one site among several,
    and scoring within the camera takes that away."""
    dim = [_streak(index, clip=f"Cam1_2025-01-01__00-00-00_{index:03d}") for index in range(GROUP_SIZE)]
    bright = [
        {**_streak(index, clip=f"Cam2_2025-01-01__00-00-00_{index:03d}"), "brightness_mean": 2.0 * (1000.0 + index)}
        for index in range(GROUP_SIZE)
    ]

    tracks = _table(dim + bright)

    scores = standing_within_camera(tracks)

    column = descriptor_names(tracks).index("brightness_mean")
    assert scores[:GROUP_SIZE, column] == pytest.approx(scores[GROUP_SIZE:, column])


def test_the_camera_is_read_off_the_front_of_the_clip() -> None:
    assert camera_of("Cam2_2024-12-15__04-40-00_000") == "Cam2"
    assert camera_of("Cam1-20250820-125851-1755687531339-7_005") == "Cam1"
    assert camera_of("2025-12-25__14_20_00-UTC_crop_black") == "unknown"


def _table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=CORPUS_SCHEMA)


def _streak(index: int, *, clip: str | None = None) -> dict[str, object]:
    wobble = 0.01 * (index % 7)
    return _row(
        index,
        clip=clip or f"Cam1_2025-01-01__00-00-00_{index:03d}",
        straightness=0.97 - wobble,
        line_residual=0.0002 + wobble / 100.0,
        turn_mean=0.1 + wobble,
        peak_deviation_max=110.0 + index,
        brightness_mean=1000.0 + index,
    )


def _wanderer(index: int) -> dict[str, object]:
    wobble = 0.01 * (index % 7)
    return _row(
        index,
        clip=f"Cam1_2025-01-01__00-00-00_{100 + index:03d}",
        straightness=0.2 + wobble,
        line_residual=0.01 + wobble / 100.0,
        turn_mean=1.8 + wobble,
        peak_deviation_max=20.0 + index / 10.0,
        brightness_mean=300.0 + index,
    )


def _row(index: int, *, clip: str, **descriptors: float) -> dict[str, object]:
    row: dict[str, object] = {name: 0.0 for name in CORPUS_SCHEMA.names}
    row.update(
        label="birds",
        event="Cam1_2025-01-01__00-00-00",
        clip=clip,
        recording=f"{clip}.mkv",
        track_id=index + 1,
        frames=40,
        missed_frames=0,
    )
    row.update(descriptors)
    return row
