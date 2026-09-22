"""Track clips built away from the page, one at a time.

Drawing a track on its video can take a minute when the track sits deep
in a long recording, and longer when the video has to be fetched first.
The page hands that work here and goes on showing the track's stored
path, and picks the clip up once it is there.

One clip is built at a time, because the machine is shared with the
archive sift. A request for another track drops every request still
waiting, so clicking through a cluster queues nothing but the track
looked at last, and a clip already under way is finished and kept.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from hessdalen.dashboard.track_clip import ClipProgress
from hessdalen.dashboard.video_cache import FetchProgress

Stage = Literal["absent", "queued", "building", "ready", "failed"]

Progress: TypeAlias = FetchProgress | ClipProgress
"""How far a job has got: first how much of its video has arrived, when
the video has to be fetched, then how many frames it has handled."""

Report: TypeAlias = Callable[[Progress], None]
"""What a job calls to say how far it has got."""

Job: TypeAlias = Callable[[Report], Path]
"""The work that builds one clip, handed a way to report its progress."""


@dataclass(frozen=True, slots=True)
class ClipState:
    """How far the clip of one track has got.

    Progress is None until the job first reports.
    """

    stage: Stage
    clip: Path | None
    error: str
    progress: Progress | None


ABSENT = ClipState(stage="absent", clip=None, error="", progress=None)


class ClipQueue:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="track-clip")
        self._jobs: dict[str, Future[Path]] = {}
        self._progress: dict[str, Progress] = {}
        self._lock = threading.Lock()

    def request(self, key: str, job: Job) -> None:
        """Have the clip of this track built, unless it is already waiting,
        under way or done.

        A request that failed is tried again.
        """
        with self._lock:
            for other, future in self._jobs.items():
                if other != key:
                    future.cancel()
            if _standing(self._jobs.get(key)):
                return
            self._progress.pop(key, None)
            self._jobs[key] = self._executor.submit(job, self._reporter(key))

    def state(self, key: str) -> ClipState:
        with self._lock:
            future = self._jobs.get(key)
            progress = self._progress.get(key)
        if future is None or future.cancelled():
            return ABSENT
        if not future.done():
            stage: Stage = "building" if future.running() else "queued"
            return ClipState(stage=stage, clip=None, error="", progress=progress)

        failure = future.exception()
        if failure is not None:
            return ClipState(stage="failed", clip=None, error=str(failure), progress=progress)
        return ClipState(stage="ready", clip=future.result(), error="", progress=progress)

    def _reporter(self, key: str) -> Report:
        def report(progress: Progress) -> None:
            with self._lock:
                self._progress[key] = progress

        return report


def _standing(future: Future[Path] | None) -> bool:
    """Whether a request is still waiting, under way, or has built its clip."""
    if future is None or future.cancelled():
        return False
    return not future.done() or future.exception() is None
