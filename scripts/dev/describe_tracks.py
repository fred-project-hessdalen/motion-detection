"""Describe every track in a directory of track files, into one file.

The descriptions are what clustering and outlier ranking read, and they
are small enough that the whole corpus fits in memory.
"""

import argparse
from pathlib import Path

from hessdalen.analysis.descriptors import TrackDescriptor, describe_file, write_descriptors


def main(args: argparse.Namespace) -> None:
    files = sorted(args.tracks.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No track files under {args.tracks}.")

    described: list[TrackDescriptor] = []
    for path in files:
        found = describe_file(path)
        described.extend(found)
        print(f"{path.name} {len(found)} tracks", flush=True)

    print(f"{write_descriptors(args.output, described)} tracks written to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe written tracks for clustering")
    parser.add_argument("tracks", type=Path, help="Directory holding the track files")
    parser.add_argument("output", type=Path, help="File to write the descriptions to")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
