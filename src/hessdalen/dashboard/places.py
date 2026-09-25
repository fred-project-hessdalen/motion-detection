"""Where the files the track map and the labelling read and write sit.

The page and the labelling server read the same map and write the same
label files, so the places are named once here and both take them from
under one root. The root is the repository, and a check instance or a
test hands in a root of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class Places:
    """The files under one root."""

    root: Path

    @property
    def map(self) -> Path:
        return self.root / "data" / "out" / "analysis" / "track-map.parquet"

    @property
    def paths(self) -> Path:
        return self.map.with_name("track-paths.parquet")

    @property
    def labels(self) -> Path:
        return self.map.with_name("cluster-labels.json")

    @property
    def track_labels(self) -> Path:
        return self.map.with_name("track-labels.json")

    @property
    def validated(self) -> Path:
        return self.map.with_name("validated-tracks.json")

    @property
    def videos(self) -> Path:
        """The recordings the sift kept."""
        return self.root / "data" / "corpus" / "videos"

    @property
    def fetched(self) -> Path:
        """The recordings the page fetched from the archive."""
        return self.root / "data" / "out" / "dashboard" / "videos"

    @property
    def tuning(self) -> Path:
        return self.root / "data" / "out" / "dashboard" / "tuning"

    @property
    def clips(self) -> Path:
        return self.root / "data" / "out" / "dashboard" / "tracks"

    @property
    def ledgers(self) -> tuple[Path, ...]:
        """What the sift wrote about every video it looked at."""
        sift = self.root / "data" / "out" / "sift"
        return (sift / "ledger.jsonl", sift / "bulk-ledger.jsonl")

    @property
    def labelling(self) -> Path:
        """What the labelling writes of its own, beside the label files."""
        return self.map.with_name("labelling")


PLACES = Places(root=REPO_ROOT)
