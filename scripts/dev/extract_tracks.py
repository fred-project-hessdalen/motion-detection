"""Write the tracks the detector finds in every recording of a directory.

One file per recording, named after it. This is what a corpus of
trajectories is built from, so it writes no video and draws nothing.
"""

import argparse
import time
from pathlib import Path

from hessdalen.config import config
from hessdalen.domain.models import MovementEvent
from hessdalen.io.tracks import write_tracks
from hessdalen.io.video import VideoStream, masked_stream
from hessdalen.processing.movement import MovementDetector, MovementSettings

SUFFIXES = (".mkv", ".mp4")


def main(args: argparse.Namespace) -> None:
    settings = config().settings
    recordings = [path for path in sorted(args.videos.iterdir()) if path.suffix in SUFFIXES]
    if not recordings:
        raise SystemExit(f"No recordings under {args.videos}.")

    for recording in recordings:
        started = time.monotonic()
        stream = masked_stream(recording, target_height=args.target_height)
        height, width = stream.frame_shape
        rows = write_tracks(
            args.output / f"{recording.stem}.parquet",
            recording=recording.name,
            frame_height=height,
            frame_width=width,
            settings=settings,
            events=_events(stream, settings=settings),
        )
        print(f"{recording.name} {rows} rows in {time.monotonic() - started:.1f}s", flush=True)


def _events(stream: VideoStream, *, settings: MovementSettings) -> list[MovementEvent]:
    return list(MovementDetector(stream=stream, settings=settings).detect())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write detected tracks to one file per recording")
    parser.add_argument("videos", type=Path, help="Directory holding the recordings")
    parser.add_argument("output", type=Path, help="Directory to write the track files into")
    parser.add_argument("--target-height", type=int, default=config().frame_height, help="Frame height to detect at")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
