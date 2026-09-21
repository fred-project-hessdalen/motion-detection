"""Place every corpus track on a map, sort it into clusters, and write the map.

The dashboard's track map reads the file this writes. The corpus can
still be growing while this runs, and the map holds whatever track
files existed at that moment.

Run with the analysis group: uv run --group analysis python scripts/dev/map_tracks.py
"""

import argparse
import collections
from pathlib import Path

import pyarrow.parquet as pq

from hessdalen.analysis.corpus import gather_paths, read_corpus
from hessdalen.analysis.track_map import CONSENSUS_SEEDS, UNASSIGNED, map_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKS = REPO_ROOT / "data" / "corpus" / "tracks"
MAP = REPO_ROOT / "data" / "out" / "analysis" / "track-map.parquet"
PATHS_NAME = "track-paths.parquet"
"""The file beside the map holding every mapped track frame by frame, which the
dashboard draws a track from without opening its track file."""


def main(args: argparse.Namespace) -> None:
    corpus = read_corpus(args.tracks)
    if corpus.tracks.num_rows == 0:
        raise SystemExit(f"No tracks under {args.tracks}.")

    mapped = map_corpus(corpus.tracks, seeds=CONSENSUS_SEEDS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(gather_paths(corpus.clips), args.output.with_name(PATHS_NAME))
    pq.write_table(mapped, args.output)

    clusters = mapped.column("cluster").to_pylist()
    labels = mapped.column("label").to_pylist()
    print(f"{mapped.num_rows} tracks from {len(corpus.clips)} clips written to {args.output}")
    for cluster in sorted(set(clusters), key=lambda held: (held == UNASSIGNED, held)):
        members = [label for label, place in zip(labels, clusters) if place == cluster]
        common = ", ".join(f"{label} {count}" for label, count in collections.Counter(members).most_common(4))
        name = "none" if cluster == UNASSIGNED else str(cluster)
        print(f"  cluster {name:>4s}  {len(members):5d} tracks  {common}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map and cluster every corpus track")
    parser.add_argument("--tracks", type=Path, default=TRACKS, help="Corpus root holding label and event folders")
    parser.add_argument("--output", type=Path, default=MAP, help="File to write the map to")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
