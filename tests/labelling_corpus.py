"""A small track map on disk for the labelling tests to open."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hessdalen.analysis.track_map import FEATURES
from hessdalen.dashboard.places import Places

BIRDS = [f"e/rec{index % 3}/{index}" for index in range(10)]
"""Ten tracks near the origin of the map, over three recordings."""

INSECTS = [f"e/rec{3 + index % 2}/{20 + index}" for index in range(6)]
"""Six tracks ten units away, over two recordings."""

LONERS = ["e/rec9/90", "e/rec9/91"]
"""Two tracks far from everything, on a recording not on disk."""

VOCABULARY = ("bird", "insect", "meteor")


def corpus(tmp_path: Path) -> Places:
    """The map under the root, with the recordings of the birds and insects
    present, the loners' absent, and the vocabulary fixed."""
    places = Places(root=tmp_path)
    rows = []
    for place, key in enumerate(BIRDS):
        rows.append(_row(key, x=0.1 * place, y=0.0, cluster=0))
    for place, key in enumerate(INSECTS):
        rows.append(_row(key, x=10.0 + 0.1 * place, y=0.0, cluster=1))
    for place, key in enumerate(LONERS):
        rows.append(_row(key, x=50.0 + place, y=50.0, cluster=-1))
    places.map.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), places.map)

    places.videos.mkdir(parents=True)
    for recording in {row["recording"] for row in rows if not row["clip"].startswith("rec9")}:
        (places.videos / recording).touch()

    places.labelling.mkdir(parents=True)
    vocabulary = {name: {"definition": "", "examples": []} for name in VOCABULARY}
    (places.labelling / "vocabulary.json").write_text(json.dumps(vocabulary))
    return places


def _row(key: str, *, x: float, y: float, cluster: int) -> dict[str, object]:
    event, clip, track_id = key.split("/")
    return {
        "label": "unlabelled",
        "event": event,
        "clip": clip,
        "recording": f"{clip}.mkv",
        "track_id": int(track_id),
        "frames": 30,
        "camera": "Cam1",
        "side": "smooth",
        "x": x,
        "y": y,
        "cluster": cluster,
        **{feature: 0.0 for feature in FEATURES},
    }
