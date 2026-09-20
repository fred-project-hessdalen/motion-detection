"""The detector reports the same movement wherever the pixels are measured.

The GPU stage is skipped when no card answers, so the host stage is the
one every machine checks.
"""

import numpy as np
import pytest

from hessdalen.io.video import VideoStream
from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.detection import CpuDetectionStage, DetectionSettings
from hessdalen.processing.devices import Device, cuda_available, detection_stage
from hessdalen.processing.movement import MovementDetector, MovementSettings
from synthetic import MockFrameSource, Scene, draw_object, scene_frames

WIDTH, HEIGHT = 320, 240
WARMUP_FRAMES = 40
OBJECT_FRAMES = 30
STEP = 4
FIRST_X = 60
OBJECT_Y = 90
CONTRAST_IN_NOISE = 12

SCENE = Scene(name="moonlit night", level=60, noise=8.0)


def test_host_stage_is_chosen_when_asked_for() -> None:
    stage = detection_stage(
        device="cpu",
        background=BackgroundSettings(),
        detection=DetectionSettings(),
        timestamp_mask=None,
    )

    assert isinstance(stage, CpuDetectionStage)


def test_a_still_scene_yields_no_detections() -> None:
    stage = _host_stage()

    detections = [stage.detections(frame) for frame in _still_frames()]

    assert detections[-1] == []


def test_an_object_is_reported_where_it_was_drawn() -> None:
    stage = _host_stage()
    frames = _still_frames()
    for frame in frames:
        stage.detections(frame)

    x, y = 160, 120
    moved = frames[-1].copy()
    moved[y - 2 : y + 3, x - 2 : x + 3] = 255
    detections = stage.detections(moved)

    assert len(detections) == 1
    assert detections[0].centroid == pytest.approx((x, y), abs=2.0)


def test_the_deviation_image_covers_the_frame() -> None:
    stage = _host_stage()
    frames = _still_frames()
    for frame in frames:
        stage.detections(frame)

    image = stage.deviation_image()

    assert image.shape == frames[0].shape
    assert image.dtype == np.uint8


@pytest.mark.skipif(not cuda_available(), reason="no CUDA device on this machine")
def test_both_stages_track_the_same_object() -> None:
    frames = _frames_with_moving_object()

    on_host = _tracks(frames, device="cpu")
    on_card = _tracks(frames, device="cuda")

    assert len(on_host) == 1
    assert on_card == on_host


def _host_stage() -> CpuDetectionStage:
    return CpuDetectionStage(
        background=BackgroundSettings(),
        detection=DetectionSettings(),
        timestamp_mask=None,
    )


def _still_frames() -> list[np.ndarray]:
    colour = scene_frames(SCENE, count=WARMUP_FRAMES, width=WIDTH, height=HEIGHT, seed=11)
    return [frame[:, :, 0] for frame in colour]


def _frames_with_moving_object() -> list[np.ndarray]:
    frames = scene_frames(SCENE, count=WARMUP_FRAMES + OBJECT_FRAMES, width=WIDTH, height=HEIGHT, seed=5)
    contrast = CONTRAST_IN_NOISE * SCENE.noise
    for index in range(OBJECT_FRAMES):
        draw_object(
            frames[WARMUP_FRAMES + index],
            x=FIRST_X + index * STEP,
            y=OBJECT_Y,
            contrast=contrast,
        )
    return frames


def _tracks(frames: list[np.ndarray], *, device: Device) -> dict[int, list[tuple[float, float]]]:
    stream = VideoStream(MockFrameSource(frames), target_height=HEIGHT)
    detector = MovementDetector(stream=stream, settings=MovementSettings(device=device))

    tracks: dict[int, list[tuple[float, float]]] = {}
    for event in detector.detect():
        if event.track_id is not None and event.centroid is not None:
            tracks.setdefault(event.track_id, []).append(event.centroid)
    return tracks
