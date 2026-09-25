"""Reading the corpus recording by recording, with no vote.

The vote in descriptor space names bird and foliage and little else,
so names come from reading. The smooth side of the map holds every
kind that is not clutter, and every smooth track is read. The rough
side is clutter of one kind per cluster, so a rough cluster is named
from a handful of its tracks read alike.

Recordings are taken in order of how many unread smooth tracks they
hold. Each is fetched, its sheets are built on several cores at once,
the sheets are read one after another, and the recording is let go,
so the disk never holds more than a recording or two. A run stopped
at any point continues from the ledger.

Run with

    uv run --group agent --group core --group dashboard \\
        --group analysis python -m hessdalen.labelling.reading \\
        --provider openai --model gpt-6-sol --sheets 1800
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from hessdalen.dashboard.places import PLACES, Places
from hessdalen.labelling.readers import Readers, RowContext
from hessdalen.labelling.service import Labelling, Settings
from hessdalen.labelling.sheets import SHEET, Layout, Sheet, SheetTrack, build_sheet
from hessdalen.labelling.signatures import DEFAULT_SIGNATURE, SIGNATURES

ROWS = 6
ROUGH_SEEN = 6
"""Tracks of a rough cluster read before the cluster is named."""

ROUGH_AGREEMENT = 5
"""Of ROUGH_SEEN, how many have to agree for the cluster to take the
name."""


@dataclass(slots=True)
class Report:
    sheets: int = 0
    tracks: int = 0
    recordings: list[str] = field(default_factory=list)
    rough_named: dict[str, int] = field(default_factory=dict)
    stopped: str = ""


class Reading:
    def __init__(
        self,
        labelling: Labelling,
        readers: Readers,
        *,
        sheets: int,
        workers: int,
        log: Callable[[str], None],
    ) -> None:
        self.labelling = labelling
        self.readers = readers
        self.budget = sheets
        self.workers = workers
        self.log = log
        self.round = labelling.state.rounds + 1
        self.report = Report()

    def run(self) -> Report:
        self.labelling.snapshot(f"reading-{self.round:03d}")
        for candidate in self.labelling.smooth_recordings():
            if self.report.sheets >= self.budget:
                self.report.stopped = "sheet budget spent"
                break
            self.log(f"fetching {candidate.recording} for {candidate.uncertain} smooth tracks")
            self.labelling.fetch(candidate.recording)
            try:
                self.read_recording(candidate.recording)
            finally:
                self.labelling.release(candidate.recording)
            self.report.recordings.append(candidate.recording)
        self.report.rough_named = self.labelling.name_rough_clusters(
            agreement=ROUGH_AGREEMENT, of=ROUGH_SEEN, round=self.round
        )
        self.log(f"done: {self.report}")
        return self.report

    def read_recording(self, recording: str) -> None:
        """Every sheet of the recording, built together and read in turn,
        within what is left of the budget."""
        keys = self.labelling.keys_to_read(recording, rough_seen=ROUGH_SEEN)
        batches = [keys[start : start + ROWS] for start in range(0, len(keys), ROWS)]
        batches = batches[: max(0, self.budget - self.report.sheets)]
        if not batches:
            return
        sheets = self.build(batches)
        for batch, sheet in zip(batches, sheets):
            rows = [RowContext(key=key, neighbour_names=[]) for key in batch]
            readings = self.readers.read(sheet.path, rows=rows, first=1, vocabulary=self.labelling.vocabulary())
            written = self.labelling.verdict(readings, sheet=sheet, round=self.round)
            self.report.sheets += 1
            self.report.tracks += len(readings)
            self.log(f"{recording}: {_counted([reading.name for reading in readings])}, refused {len(written.refused)}")

    def build(self, batches: Sequence[Sequence[str]]) -> list[Sheet]:
        """The sheets of these batches, built on the workers at once."""
        jobs = [
            [self.labelling._sheet_track(key, number=number) for number, key in enumerate(batch, start=1)]
            for batch in batches
        ]
        directory = self.labelling.sheets_dir
        if self.workers <= 1:
            return [build_sheet(directory, tracks=tracks, layout=SHEET) for tracks in jobs]
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(_build, [(directory, tracks, SHEET) for tracks in jobs]))


def _build(job: tuple[Path, list[SheetTrack], Layout]) -> Sheet:
    directory, tracks, layout = job
    return build_sheet(directory, tracks=tracks, layout=layout)


def _counted(names: Sequence[str]) -> dict[str, int]:
    counted: dict[str, int] = {}
    for name in names:
        counted[name] = counted.get(name, 0) + 1
    return counted


def main() -> None:
    from anthropic import Anthropic
    from openai import OpenAI

    from hessdalen.labelling.readers import (
        MODEL,
        OLLAMA_URL,
        AnthropicBackend,
        Backend,
        ModelReaders,
        OllamaBackend,
        OpenAIBackend,
        ollama_post,
    )

    parser = argparse.ArgumentParser(description="Read the corpus recording by recording.")
    parser.add_argument("--root", type=Path, default=PLACES.root, help="the repository the corpus sits under")
    parser.add_argument("--signature", choices=sorted(SIGNATURES), default=DEFAULT_SIGNATURE)
    parser.add_argument("--provider", choices=("anthropic", "openai", "ollama"), default="anthropic")
    parser.add_argument("--model", default=MODEL, help="the model the provider serves")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--sheets", type=int, default=400, help="sheets to build and read at most")
    parser.add_argument("--workers", type=int, default=8, help="sheets built at once")
    parsed = parser.parse_args()

    backend: Backend
    if parsed.provider == "ollama":
        backend = OllamaBackend(model=parsed.model, post=ollama_post(parsed.ollama_url))
    elif parsed.provider == "openai":
        backend = OpenAIBackend(OpenAI(), model=parsed.model)
    else:
        backend = AnthropicBackend(Anthropic(), model=parsed.model)
    labelling = Labelling(Places(root=parsed.root), settings=Settings(signature=parsed.signature, flip_limit=2))
    Reading(
        labelling,
        ModelReaders(backend),
        sheets=parsed.sheets,
        workers=parsed.workers,
        log=lambda line: print(line, file=sys.stderr, flush=True),
    ).run()


if __name__ == "__main__":
    main()
