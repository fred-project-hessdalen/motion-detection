"""Place the corpus tracks on a map, sort them into clusters, and write the map.

The dashboard's track map reads the file this writes. The corpus can
still be growing while this runs, and the map holds whatever track
files existed at that moment. A busy recording is held to a sample of
its tracks, spread over the recording.

Run with the analysis group: uv run --group analysis python scripts/dev/map_tracks.py
"""

import argparse
import collections
from pathlib import Path

import numba
import pyarrow as pa
import pyarrow.parquet as pq

from hessdalen.analysis.corpus import ClipPath, gather_paths, read_corpus
from hessdalen.analysis.track_map import CONSENSUS_SEEDS, PER_CLIP, UNASSIGNED, map_corpus, sample_per_clip

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

    sampled = sample_per_clip(corpus.tracks)
    mapped = map_corpus(sampled, seeds=CONSENSUS_SEEDS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_mapped_paths(corpus.clips, mapped), args.output.with_name(PATHS_NAME))
    pq.write_table(mapped, args.output)

    clusters = mapped.column("cluster").to_pylist()
    labels = mapped.column("label").to_pylist()
    print(
        f"{mapped.num_rows} of {corpus.tracks.num_rows} tracks, at most {PER_CLIP} from any clip, "
        f"from {len(corpus.clips)} clips written to {args.output}"
    )
    for cluster in sorted(set(clusters), key=lambda held: (held == UNASSIGNED, held)):
        members = [label for label, place in zip(labels, clusters) if place == cluster]
        common = ", ".join(f"{label} {count}" for label, count in collections.Counter(members).most_common(4))
        name = "none" if cluster == UNASSIGNED else str(cluster)
        print(f"  cluster {name:>4s}  {len(members):5d} tracks  {common}")


def _mapped_paths(clips: list[ClipPath], mapped: pa.Table) -> pa.Table:
    """The frames of the tracks on the map, and of no other."""
    paths = gather_paths(clips)
    keys = {
        (event, clip, track_id)
        for event, clip, track_id in zip(
            mapped.column("event").to_pylist(), mapped.column("clip").to_pylist(), mapped.column("track_id").to_pylist()
        )
    }
    held = [
        (event, clip, track_id) in keys
        for event, clip, track_id in zip(
            paths.column("event").to_pylist(), paths.column("clip").to_pylist(), paths.column("track_id").to_pylist()
        )
    ]
    return paths.filter(pa.array(held))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map and cluster every corpus track")
    parser.add_argument("--tracks", type=Path, default=TRACKS, help="Corpus root holding label and event folders")
    parser.add_argument("--output", type=Path, default=MAP, help="File to write the map to")
    return parser.parse_args()


if __name__ == "__main__":
    # Numba's default threading layer tries TBB first and warns when the
    # system's TBB is older than it supports. Seeded UMAP runs on one thread,
    # so numba's own layer costs nothing.
    numba.config.THREADING_LAYER = "workqueue"  # type: ignore[attr-defined]
    main(parse_args())
