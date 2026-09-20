"""One set of settings has to carry every sky the cameras see.

The scenes below span a factor of thirty in brightness and ten in noise,
which is the range the example recordings cover between a dark night and
a snowy afternoon. Each is run through the settings the config holds, so
a set of settings saved from the dashboard is measured against every
scene here.
"""

import pytest

from hessdalen.config import config
from hessdalen.io.video import VideoStream
from hessdalen.processing.movement import MovementDetector
from synthetic import MockFrameSource, Scene, draw_object, scene_frames

WIDTH, HEIGHT = 320, 240
WARMUP_FRAMES = 40
OBJECT_FRAMES = 30
STEP = 4
FIRST_X = 60
OBJECT_Y = 90
CONTRAST_IN_NOISE = 12

SCENES = [
    Scene(name="dark night", level=5, noise=2.0),
    Scene(name="moonlit night", level=60, noise=8.0),
    Scene(name="overcast day", level=180, noise=20.0),
]
SCENE_IDS = [scene.name for scene in SCENES]


def detect_tracks(frames: list) -> dict[int, list[tuple[float, float]]]:
    stream = VideoStream(MockFrameSource(frames), target_height=HEIGHT)
    detector = MovementDetector(stream=stream, settings=config().settings)

    tracks: dict[int, list[tuple[float, float]]] = {}
    for event in detector.detect():
        if event.track_id is not None and event.centroid is not None:
            tracks.setdefault(event.track_id, []).append(event.centroid)
    return tracks


def frames_with_moving_object(scene: Scene, seed: int) -> list:
    frames = scene_frames(scene, count=WARMUP_FRAMES + OBJECT_FRAMES, width=WIDTH, height=HEIGHT, seed=seed)

    contrast = CONTRAST_IN_NOISE * scene.noise
    if scene.level > 128:
        contrast = -contrast

    for step, frame in enumerate(frames[WARMUP_FRAMES:]):
        draw_object(frame, x=FIRST_X + step * STEP, y=OBJECT_Y, contrast=contrast)
    return frames


@pytest.mark.parametrize("scene", SCENES, ids=SCENE_IDS)
def test_moving_object_is_tracked_in_every_scene(scene: Scene):
    tracks = detect_tracks(frames_with_moving_object(scene, seed=1))

    assert len(tracks) == 1, f"Expected one track in {scene.name}, got {len(tracks)}: {tracks.keys()}"

    centroids = next(iter(tracks.values()))
    travelled = max(x for x, _y in centroids) - min(x for x, _y in centroids)
    assert travelled >= 0.6 * STEP * OBJECT_FRAMES, (
        f"Track in {scene.name} covers {travelled:.0f} px of the object's {STEP * OBJECT_FRAMES} px path"
    )


@pytest.mark.parametrize("scene", SCENES, ids=SCENE_IDS)
def test_noise_alone_is_not_tracked(scene: Scene):
    frames = scene_frames(scene, count=WARMUP_FRAMES + OBJECT_FRAMES, width=WIDTH, height=HEIGHT, seed=2)

    tracks = detect_tracks(frames)

    assert tracks == {}, f"Noise alone produced {len(tracks)} tracks in {scene.name}"
