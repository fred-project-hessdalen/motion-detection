"""How the background queue builds track clips, and what it says about them
meanwhile."""

import threading
from pathlib import Path

from hessdalen.dashboard.clip_queue import ClipQueue
from hessdalen.dashboard.track_clip import ClipProgress, Stretch

WAIT = 5.0
STRETCH = Stretch(begin_frame=100, end_frame=149)


def test_a_requested_clip_is_built_and_reported_ready() -> None:
    queue = ClipQueue()
    done = threading.Event()

    queue.request("a", _job(Path("a.mp4"), finished=done))

    assert done.wait(WAIT)
    _settle(queue, "a")
    assert queue.state("a").stage == "ready"
    assert queue.state("a").clip == Path("a.mp4")


def test_how_far_a_build_has_got_is_reported_while_it_runs() -> None:
    queue = ClipQueue()
    reported, release = threading.Event(), threading.Event()

    def job(report) -> Path:
        report(ClipProgress(frames_done=60, stretch=STRETCH))
        reported.set()
        release.wait(WAIT)
        return Path("a.mp4")

    queue.request("a", job)

    assert reported.wait(WAIT)
    progress = queue.state("a").progress
    release.set()
    assert queue.state("a").stage in ("building", "ready")
    assert progress is not None
    assert progress.frames_done == 60


def test_asking_for_another_track_drops_a_request_still_waiting() -> None:
    """Clicking through a cluster should build the track looked at last, and
    not every track clicked on the way."""
    queue = ClipQueue()
    release = threading.Event()
    queue.request("running", _held(release))

    queue.request("waiting", _job(Path("waiting.mp4")))
    queue.request("latest", _job(Path("latest.mp4")))
    release.set()

    _settle(queue, "latest")
    assert queue.state("waiting").stage == "absent"
    assert queue.state("latest").stage == "ready"
    assert queue.state("running").stage == "ready"


def test_a_failed_clip_is_reported_and_tried_again_on_the_next_request() -> None:
    queue = ClipQueue()
    queue.request("a", _fails)
    _settle(queue, "a")
    assert queue.state("a").stage == "failed"
    assert "no frames" in queue.state("a").error

    queue.request("a", _job(Path("a.mp4")))
    _settle(queue, "a")
    assert queue.state("a").stage == "ready"


def test_a_track_nobody_asked_for_has_no_clip() -> None:
    assert ClipQueue().state("never").stage == "absent"


def _job(clip: Path, *, finished: threading.Event | None = None):
    def job(report) -> Path:
        if finished is not None:
            finished.set()
        return clip

    return job


def _held(release: threading.Event):
    def job(report) -> Path:
        release.wait(WAIT)
        return Path("running.mp4")

    return job


def _fails(report) -> Path:
    raise OSError("the recording holds no frames there")


def _settle(queue: ClipQueue, key: str) -> None:
    for _ in range(500):
        if queue.state(key).stage in ("ready", "failed", "absent"):
            return
        threading.Event().wait(0.01)
