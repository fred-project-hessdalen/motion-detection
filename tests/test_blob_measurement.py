"""What a blob's shape and brightness come out as, given the pixels it
holds."""

import numpy as np
import pytest

from hessdalen.processing.detection import Detection, FOREGROUND, blob_axes, blobs_around_peaks

WIDTH, HEIGHT = 48, 32
STREAK_LEVEL = 100
NEIGHBOUR_LEVEL = 200


def test_a_line_is_stretched_where_a_square_is_round() -> None:
    stretched = blob_axes(np.ones((1, 9), dtype=bool))
    round_blob = blob_axes(np.ones((3, 3), dtype=bool))

    assert stretched.minor == pytest.approx(0.0, abs=1e-9)
    assert stretched.major > round_blob.major
    assert round_blob.major == pytest.approx(round_blob.minor)


def test_a_blob_is_summed_over_its_own_pixels_alone() -> None:
    """A diagonal streak's bounding box holds a neighbour that was measured
    first, and the streak must not carry that neighbour's grey levels."""
    gray = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    foreground = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    gray[10:12, 20:22] = NEIGHBOUR_LEVEL
    foreground[10:12, 20:22] = FOREGROUND
    for step in range(11):
        gray[11 + step, 11 + step] = STREAK_LEVEL
        foreground[11 + step, 11 + step] = FOREGROUND

    neighbour, streak = _detections(gray, foreground, seeds=[(10, 20), (11, 11)])

    assert neighbour.blob.brightness == 4.0 * NEIGHBOUR_LEVEL
    assert streak.blob.brightness == 11.0 * STREAK_LEVEL
    assert streak.blob.pixel_count == 11


def test_a_blob_carries_the_grey_levels_under_it() -> None:
    gray = np.full((HEIGHT, WIDTH), 30, dtype=np.uint8)
    foreground = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    gray[5:8, 5:8] = STREAK_LEVEL
    foreground[5:8, 5:8] = FOREGROUND

    (square,) = _detections(gray, foreground, seeds=[(5, 5)])

    assert square.blob.brightness == 9.0 * STREAK_LEVEL
    assert square.blob.major_axis == pytest.approx(square.blob.minor_axis)


def _detections(gray: np.ndarray, foreground: np.ndarray, *, seeds: list[tuple[int, int]]) -> list[Detection]:
    rows = np.array([row for row, _column in seeds])
    columns = np.array([column for _row, column in seeds])
    return blobs_around_peaks(
        gray=gray,
        foreground=foreground,
        rows=rows,
        columns=columns,
        deviations=np.full(rows.size, 20.0),
        min_pixels=1,
    )
