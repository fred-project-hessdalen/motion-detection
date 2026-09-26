"""Write every mapped track's shape beside the track map.

The dashboard's Similar shape gallery reads the file this writes. Run
it after map_tracks.py, which writes the paths this reads.

Run with the analysis group: uv run --group analysis python scripts/dev/shape_tracks.py
"""

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from hessdalen.analysis.track_shapes import shape_table

REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS = REPO_ROOT / "data" / "out" / "analysis" / "track-paths.parquet"
SHAPES_NAME = "track-shapes.parquet"


def main(args: argparse.Namespace) -> None:
    started = time.time()
    paths = pq.read_table(args.paths).to_pandas()
    shapes = shape_table(paths)
    pq.write_table(shapes, args.paths.with_name(SHAPES_NAME))
    print(f"{shapes.num_rows} tracks shaped in {time.time() - started:.0f} s, written beside {args.paths}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=PATHS, help="the paths file the map step wrote")
    main(parser.parse_args())
