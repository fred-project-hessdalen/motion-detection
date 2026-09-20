"""Turning drawn frames into a video a browser can play.

A stored run and a live segment are written the same way and differ only
in how hard the encoder is asked to work at each frame.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np

PIXEL_FORMAT = "yuv420p"
FALLBACK_FPS = 25.0


@dataclass(frozen=True, slots=True)
class Encoding:
    """How hard the encoder works at each frame."""

    preset: str
    crf: int


STORED = Encoding(preset="veryfast", crf=26)
"""What a stored run is written with, and kept at for as long as its settings
stand."""

LIVE = Encoding(preset="ultrafast", crf=26)
"""What a live segment is written with, where the wait to see it is what
counts.

The quality target is the one a stored run uses. The faint single pixels
a setting is moved to find are the first thing a coarser one loses.
"""


def encode(
    frames: Iterator[np.ndarray],
    *,
    output: Path,
    width: int,
    height: int,
    frames_per_second: float,
    encoding: Encoding,
) -> int:
    """Write planar frames to output as H.264, and report how many were
    written.

    The first frame is pulled before ffmpeg starts, so a source holding
    none writes no file and leaves no process behind. A caller that
    stops early still closes the pipe and reaps the encoder.
    """
    first = next(frames, None)
    if first is None:
        return 0

    encoder = _open_encoder(
        output,
        width=width,
        height=height,
        frames_per_second=frames_per_second,
        encoding=encoding,
    )
    stdin = encoder.stdin
    if stdin is None:
        raise RuntimeError("ffmpeg was started without an input pipe.")

    written = 0
    try:
        for frame in chain((first,), frames):
            stdin.write(frame)
            written += 1
    finally:
        stdin.close()
        encoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with status {encoder.returncode}.")
    return written


def _open_encoder(
    output: Path,
    *,
    width: int,
    height: int,
    frames_per_second: float,
    encoding: Encoding,
) -> subprocess.Popen[bytes]:
    command = encoder_command(
        output,
        width=width,
        height=height,
        frames_per_second=frames_per_second,
        encoding=encoding,
    )
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def encoder_command(
    output: Path,
    *,
    width: int,
    height: int,
    frames_per_second: float,
    encoding: Encoding,
) -> list[str]:
    """The ffmpeg call that turns raw planar frames into a playable video.

    Handing the encoder colour frames makes it convert them itself,
    which costs ten times what OpenCV charges and sends twice the bytes
    down the pipe.
    """
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        PIXEL_FORMAT,
        "-s",
        f"{width}x{height}",
        "-r",
        f"{frames_per_second:.6f}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        encoding.preset,
        "-crf",
        str(encoding.crf),
        "-movflags",
        "+faststart",
        str(output),
    ]


def even(size: int) -> int:
    """The largest even size at or below this one.

    Colour is stored for every second row and column, so a frame with an
    odd side loses that side's last line.
    """
    return size - size % 2


def playback_rate(frames_per_second: float) -> float:
    """The rate a video is written at, for a container that reports none."""
    return frames_per_second if frames_per_second > 0 else FALLBACK_FPS
