"""Every track as a small picture of its shape, and the tracks nearest one
in that picture."""

import numpy as np
import pandas as pd

from hessdalen.analysis.track_shapes import SHAPE_SIDE, nearest_shapes, shape_matrix, shape_table, shape_vector


def _straight(frames: int, *, scale: float) -> dict[str, np.ndarray]:
    steps = np.arange(frames)
    return {"frame_number": steps, "centre_x": 100.0 + steps * scale, "centre_y": np.full(frames, 300.0)}


def _circle(frames: int) -> dict[str, np.ndarray]:
    steps = np.arange(frames)
    angle = steps / frames * 2 * np.pi
    return {"frame_number": steps, "centre_x": 500 + 80 * np.cos(angle), "centre_y": 500 + 80 * np.sin(angle)}


def test_a_shape_is_a_fixed_number_of_values_whatever_the_length() -> None:
    short = shape_vector(**_straight(12, scale=3.0), reach=1920.0)
    long = shape_vector(**_straight(300, scale=3.0), reach=1920.0)

    assert short.shape == long.shape == (SHAPE_SIDE * SHAPE_SIDE,)


def test_a_straight_track_looks_like_a_straight_track_at_any_scale_and_unlike_a_circle() -> None:
    small = shape_vector(**_straight(50, scale=1.0), reach=1920.0)
    large = shape_vector(**_straight(50, scale=9.0), reach=1920.0)
    round_ = shape_vector(**_circle(50), reach=1920.0)

    assert np.linalg.norm(small - large) < 0.1 * np.linalg.norm(small - round_)


def test_the_shape_table_holds_one_row_per_track_under_the_map_key() -> None:
    frames = []
    for clip, track_id, points in (
        ("c1", 1, _straight(20, scale=2.0)),
        ("c1", 2, _circle(20)),
        ("c2", 1, _straight(30, scale=1.0)),
    ):
        frames.append(
            pd.DataFrame(
                {**points, "event": "e", "clip": clip, "track_id": track_id, "frame_width": 1920, "frame_height": 1080}
            )
        )
    paths = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=0)

    table = shape_table(paths)
    keys, matrix = shape_matrix(table)

    assert sorted(keys) == ["e/c1/1", "e/c1/2", "e/c2/1"]
    assert matrix.shape == (3, SHAPE_SIDE * SHAPE_SIDE)
    straight_a, straight_b, circle = (keys.index(key) for key in ("e/c1/1", "e/c2/1", "e/c1/2"))
    found, distances = nearest_shapes(matrix, place=straight_a, among=np.arange(3), count=2)
    assert found.tolist() == [straight_b, circle]
    assert distances[0] < distances[1]


def test_the_nearest_shapes_are_looked_for_among_the_given_rows_only() -> None:
    matrix = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [5.0, 0.0]])

    found, _ = nearest_shapes(matrix, place=0, among=np.array([0, 2, 3]), count=5)

    assert found.tolist() == [2, 3]
