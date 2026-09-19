"""Pytest tests for movement detection using pre-generated test videos.

Run with: uv run --group test pytest tests/test_grid.py -v
"""

from pathlib import Path

import pytest

from hessdalen.io.video import VideoStream, FileFrameSource
from hessdalen.processing.movement import MovementDetector, MovementSettings, TrackingSettings


TEST_DATA_DIR = Path("tests/data/test_grid")
CIRCLE_SIZES = [1, 5, 10]
NOISE_LEVELS = [0, 25, 50]


@pytest.fixture(scope="module")
def test_data_dir():
    if not TEST_DATA_DIR.exists():
        pytest.skip(
            f"Test data directory not found: {TEST_DATA_DIR}. "
            "Run scripts/dev/generate_test_data.py to generate test videos."
        )
    return TEST_DATA_DIR


def count_detections(video_path: Path) -> int:
    stream = VideoStream(FileFrameSource(video_path))
    detector = MovementDetector(
        stream=stream,
        settings=MovementSettings(tracking=TrackingSettings(min_consecutive_frames=3)),
    )

    detected_count = 0
    for event in detector.detect():
        if event.centroid is not None:
            detected_count += 1

    return detected_count


@pytest.mark.parametrize("circle_size", CIRCLE_SIZES)
@pytest.mark.parametrize("noise_level", NOISE_LEVELS)
def test_movement_detection_grid(test_data_dir, circle_size, noise_level):
    video_path = test_data_dir / f"test_r{circle_size}_n{noise_level}.mp4"

    if not video_path.exists():
        pytest.fail(
            f"Test video not found: {video_path}. Run scripts/dev/generate_test_data.py to generate test videos."
        )

    detected_count = count_detections(video_path)

    assert detected_count >= 10, (
        f"Expected at least 10 detections for circle_size={circle_size}, "
        f"noise_level={noise_level}, but got {detected_count}"
    )
