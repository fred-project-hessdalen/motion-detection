"""Place the corpus tracks on a map, sort them into clusters, and write the map.

The dashboard's track map reads the file this writes. The corpus can
still be growing while this runs, and the map holds whatever track
files existed at that moment. Every track is mapped unless --per-clip
holds a busy recording to a sample of its tracks.

Run with the analysis group: uv run --group analysis python scripts/dev/map_tracks.py
"""

import argparse
import collections
from pathlib import Path

import numba
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hessdalen.analysis.corpus import ClipPath, path_tables, read_corpus
from hessdalen.analysis.track_map import CONSENSUS_SEEDS, PER_CLIP, UNASSIGNED, map_corpus

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

    mapped = map_corpus(corpus.tracks, seeds=CONSENSUS_SEEDS, per_clip=args.per_clip or None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_paths(args.output.with_name(PATHS_NAME), clips=corpus.clips, mapped=mapped)
    pq.write_table(mapped, args.output)

    clusters = mapped.column("cluster").to_pylist()
    labels = mapped.column("label").to_pylist()
    decided = f", clustered on at most {args.per_clip} of each clip" if args.per_clip else ""
    print(f"{mapped.num_rows} tracks from {len(corpus.clips)} clips{decided}, written to {args.output}")
    for cluster in sorted(set(clusters), key=lambda held: (held == UNASSIGNED, held)):
        members = [label for label, place in zip(labels, clusters) if place == cluster]
        common = ", ".join(f"{label} {count}" for label, count in collections.Counter(members).most_common(4))
        name = "none" if cluster == UNASSIGNED else str(cluster)
        print(f"  cluster {name:>4s}  {len(members):5d} tracks  {common}")


def _write_paths(path: Path, *, clips: list[ClipPath], mapped: pa.Table) -> None:
    """Write the frames of the tracks on the map, and of no other.

    Each clip is filtered and written as it is read, because the whole
    corpus is tens of gigabytes of frames and a run that held them all
    at once was killed for it. Both sides are matched on one string a
    row at a time rather than on three columns, for the same reason.
    """
    wanted = _row_keys(mapped)
    writer = None
    try:
        for table in path_tables(clips):
            held = table.filter(pc.is_in(_row_keys(table), value_set=wanted))
            if held.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(path, held.schema)
            writer.write_table(held)
    finally:
        if writer is not None:
            writer.close()


def _row_keys(table: pa.Table) -> pa.Array:
    """One string per row naming the track it belongs to."""
    return pc.binary_join_element_wise(
        table.column("event").cast(pa.string()),
        table.column("clip").cast(pa.string()),
        pc.cast(table.column("track_id"), pa.string()),
        "/",
    ).combine_chunks()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map and cluster every corpus track")
    parser.add_argument("--tracks", type=Path, default=TRACKS, help="Corpus root holding label and event folders")
    parser.add_argument("--output", type=Path, default=MAP, help="File to write the map to")
    parser.add_argument(
        "--per-clip",
        type=int,
        default=PER_CLIP,
        help="Most tracks of one clip the clustering is decided on, spread over the clip. Every track is "
        "mapped either way, and a track the clustering skipped takes the cluster its neighbours hold. "
        "Passing 0 decides the clustering on every track, which costs a matrix of every pair.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Numba's default threading layer tries TBB first and warns when the
    # system's TBB is older than it supports. Seeded UMAP runs on one thread,
    # so numba's own layer costs nothing.
    numba.config.THREADING_LAYER = "workqueue"  # type: ignore[attr-defined]
    main(parse_args())
