"""The loop that names the corpus, with the model called for what needs
eyes.

Choosing what to look at, propagating, verifying and stopping are
rules, so they live here where they run the same way every round and
leave a ledger that can be replayed. A model is called to read a
sheet, to judge a contested track, and to propose a name for tracks
no name fits.

The rounds, as docs/labelling-agent.md sets them down:

0. Calibration. Sheets of tracks whose name is settled are read
   blind and compared. Under the threshold the run stops.
1. Seed. One sheet per cluster, spread over the cluster.
2. Propagate. The vote names the unseen tracks it can.
3. Verify. Sheets of tracks the propagation named, the vote as the
   prediction, and the cap moved to where agreement falls.
4. Aim. Sheets of the tracks the vote cannot name, until none is left
   on disk or the sheet budget is spent. Rounds 2 to 4 repeat.
5. Fetch. Recordings covering the most uncertain tracks, within the
   fetch budget, and rounds 2 to 4 over them.
6. Isolate. Every contested track goes to the judge.
7. Report.

Run with

    uv run --group agent --group core --group dashboard --group analysis \\
        python -m hessdalen.labelling.harness [--sheets N] [--fetches N] ...

with ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hessdalen.dashboard.places import PLACES, Places
from hessdalen.labelling.ledger import PROPAGATED
from hessdalen.labelling.readers import REVIEW, Readers, RowContext
from hessdalen.labelling.service import NONE_OF_THESE, REVIEW_TAG, Labelling, Reading, Settings, Status
from hessdalen.labelling.sheets import SHEET, Sheet
from hessdalen.labelling.signatures import DEFAULT_SIGNATURE, SIGNATURES
from hessdalen.labelling.votes import Rule

ROWS = 6
"""Rows per sheet."""

NEIGHBOUR_NAMES = 5
"""Seen tracks whose names a row is read with."""

PROPOSAL_ROWS = 6
"""Rows read as none of these before a name is proposed."""

PROPOSALS_NAME = "proposals.jsonl"


@dataclass(frozen=True, slots=True)
class Budget:
    sheets: int
    fetches: int


@dataclass(frozen=True, slots=True)
class Thresholds:
    """The least share of readings that has to agree with what the tracks
    hold, at calibration and at verification."""

    calibration: float
    verification: float


@dataclass(slots=True)
class Report:
    calibration: float
    verification: list[float] = field(default_factory=list)
    sheets: int = 0
    fetched: list[str] = field(default_factory=list)
    judged: int = 0
    proposals: int = 0
    stopped: str = ""
    status: Status | None = None


class Harness:
    def __init__(
        self,
        labelling: Labelling,
        readers: Readers,
        *,
        rule: Rule,
        budget: Budget,
        thresholds: Thresholds,
        log: Callable[[str], None],
    ) -> None:
        self.labelling = labelling
        self.readers = readers
        self.rule = rule
        self.budget = budget
        self.thresholds = thresholds
        self.log = log
        self.round = labelling.state.rounds + 1
        self.report = Report(calibration=0.0)
        self.verified: list[tuple[float, bool]] = []
        """Every verification reading, as how far the track was from its
        voters and whether the reading agreed with the vote."""

        self.unnamed_sheets: list[Sheet] = []
        """Sheets holding rows read as none of these, for a proposal."""

    def run(self) -> Report:
        self.labelling.snapshot(f"round-{self.round:03d}")
        self.report.calibration = self.calibrate()
        if self.report.calibration < self.thresholds.calibration:
            self.report.stopped = "calibration under the threshold"
            return self.finish()

        self.seed()
        self.spread()
        for candidate in self.labelling.fetch_list(count=self.budget.fetches, rule=self.rule):
            if self.report.sheets >= self.budget.sheets:
                break
            self.log(f"fetching {candidate.recording} for {candidate.uncertain} uncertain tracks")
            self.labelling.fetch(candidate.recording)
            self.report.fetched.append(candidate.recording)
            self.spread()
        self.isolate()
        return self.finish()

    def calibrate(self) -> float:
        """How often the reader agrees with the validated tracks, read blind.

        The validated tracks are the only ground truth, so recordings
        holding validated tracks are fetched first, within the fetch
        budget, and nothing is written. Fewer than a sheet's worth of
        validated tracks on disk is no calibration, and reads as no
        agreement.
        """
        for candidate in self.labelling.calibration_recordings()[: self.budget.fetches]:
            self.log(f"calibration: fetching {candidate.recording} for {candidate.uncertain} validated tracks")
            self.labelling.fetch(candidate.recording)
            self.report.fetched.append(candidate.recording)

        keys = self.labelling.calibration_keys()
        if len(keys) < ROWS:
            self.log(f"calibration: {len(keys)} validated tracks on disk, fewer than one sheet's worth")
            return 0.0

        agreed = 0
        for batch in _batches(keys, ROWS):
            _, readings = self.read(batch)
            for reading in readings:
                truth = self.labelling.truth(reading.key)
                if reading.name in truth:
                    agreed += 1
                else:
                    self.log(f"calibration: {reading.key} holds {' or '.join(truth)}, read as {reading.name}")
        self.log(f"calibration: {agreed} of {len(keys)} readings agree")
        return agreed / len(keys)

    def seed(self) -> None:
        """One sheet per cluster, the one with most unnamed tracks on disk
        first, every row its own verdict."""
        for summary in self.labelling.clusters():
            if self.report.sheets >= self.budget.sheets:
                return
            keys = self.labelling.sample(summary.cluster, count=ROWS)
            if not keys:
                continue
            sheet, readings = self.read(keys)
            self.labelling.verdict(readings, sheet=sheet, round=self.round)
            self.log(f"seed: cluster {summary.cluster} read as {_counted(readings)}")

    def spread(self) -> None:
        """Rounds 2 to 4, until no track on disk is uncertain or the sheet
        budget is spent."""
        while self.report.sheets < self.budget.sheets:
            self.propagate()
            self.verify()
            uncertain = self.labelling.uncertain(count=ROWS, rule=self.rule)
            if not uncertain:
                return
            sheet, readings = self.read([vote.key for vote in uncertain])
            self.labelling.verdict(readings, sheet=sheet, round=self.round)
            self.log(f"aim: {len(uncertain)} uncertain tracks read as {_counted(readings)}")
            self.propose_if_due()

    def propagate(self) -> None:
        done = self.labelling.propagate(rule=self.rule, round=self.round)
        self.log(
            f"round {self.round}: propagated {done.named}, changed {done.changed}, "
            f"{done.disagree} disagree, {done.far} far, contested {done.contested}"
        )

    def verify(self) -> None:
        """Sheets of tracks the propagation of this round named, the vote as
        the prediction. The cap moves to where agreement falls under the
        threshold, and the round ends whatever the agreement."""
        keys = self.labelling.verify_candidates(count=ROWS, round=self.round)
        if keys and self.report.sheets < self.budget.sheets:
            names = self.labelling.names_by_key()
            sheet = self.labelling.sheet(keys, layout=SHEET)
            predictions = [Reading(key=key, name=names.get(key, ""), confidence="", note="") for key in keys]
            self.labelling.predict(predictions, sheet=sheet, round=self.round)
            readings = self.readers.read(sheet.path, rows=self.contexts(keys), vocabulary=self.labelling.vocabulary())
            self.report.sheets += 1

            distances = self.distances(keys)
            agreed = 0
            for reading in readings:
                if reading.name in (NONE_OF_THESE, "") or reading.confidence == "unsure":
                    continue
                agreed += reading.name == names.get(reading.key)
                self.verified.append((distances[reading.key], reading.name == names.get(reading.key)))
            rate = agreed / len(readings)
            self.report.verification.append(rate)
            self.labelling.verdict(readings, sheet=sheet, round=self.round)
            self.rule = Rule(votes=self.rule.votes, agreement=self.rule.agreement, cap=self.cap())
            self.log(f"verify: {agreed} of {len(readings)} agree with the vote, cap now {self.rule.cap:.3f}")
        self.round += 1

    def cap(self) -> float:
        """The distance up to which verified readings agree with the vote at
        the threshold, from every verification so far, and the cap as it
        stands while fewer than a sheet's worth have been read."""
        if len(self.verified) < ROWS:
            return self.rule.cap

        agreed = 0
        reached = 0.0
        for count, (distance, held) in enumerate(sorted(self.verified), start=1):
            agreed += held
            if agreed / count >= self.thresholds.verification:
                reached = distance
        return max(reached, min(distance for distance, _ in self.verified))

    def isolate(self) -> None:
        """Every contested track to the judge, and those the judge cannot
        settle to the person."""
        for key in sorted(self.labelling.contested()):
            if key in self.labelling.state.seen and self.labelling.state.seen[key].round >= self.round:
                continue
            view = self.labelling.isolation(key)
            history = [
                f"{line.basis} {line.name or 'no name'} in round {line.round}"
                for line in self.labelling.ledger_lines(key=key, last=0)
            ]
            reading = self.readers.judge(view.path, key=key, history=history, vocabulary=self.labelling.vocabulary())
            if reading.name == REVIEW:
                self.labelling.tag(key, tags=[REVIEW_TAG], round=self.round)
            else:
                self.labelling.judge(reading, view=view, round=self.round)
            self.report.judged += 1
            self.log(f"judge: {key} settled as {reading.name}")

    def propose_if_due(self) -> None:
        """A name proposed once a sheet's worth of rows were read as none of
        these, written to the proposals file and the rows tagged for
        review."""
        rows = [key for sheet in self.unnamed_sheets for key in sheet.keys]
        if len(rows) < PROPOSAL_ROWS:
            return
        sheets = self.unnamed_sheets[-3:]
        proposal = self.readers.propose([sheet.path for sheet in sheets], vocabulary=self.labelling.vocabulary())
        held = {"round": self.round, **proposal.model_dump(), "sheets": [str(sheet.path) for sheet in sheets]}
        path = self.labelling.places.labelling / PROPOSALS_NAME
        with path.open("a") as proposals:
            proposals.write(json.dumps(held) + "\n")
        for key in proposal.examples:
            if key in self.labelling.place_of:
                self.labelling.tag(key, tags=[REVIEW_TAG], round=self.round)
        self.report.proposals += 1
        self.unnamed_sheets = []
        self.log(f"proposal: {proposal.name}, {proposal.definition}")

    def read(self, keys: Sequence[str]) -> tuple[Sheet, list[Reading]]:
        """The sheet of these tracks and what the reader made of it."""
        sheet = self.labelling.sheet(keys, layout=SHEET)
        readings = self.readers.read(sheet.path, rows=self.contexts(keys), vocabulary=self.labelling.vocabulary())
        self.report.sheets += 1
        if any(reading.name == NONE_OF_THESE for reading in readings):
            unnamed = tuple(reading.key for reading in readings if reading.name == NONE_OF_THESE)
            self.unnamed_sheets.append(Sheet(path=sheet.path, keys=unnamed))
        return sheet, readings

    def contexts(self, keys: Sequence[str]) -> list[RowContext]:
        return [
            RowContext(
                key=key,
                neighbour_names=[
                    near.name
                    for near in self.labelling.neighbours(key, count=NEIGHBOUR_NAMES, seen_only=True)
                    if near.name
                ],
            )
            for key in keys
        ]

    def distances(self, keys: Sequence[str]) -> dict[str, float]:
        """How far each track was from its voters, from its latest
        propagation line."""
        held = {}
        for line in self.labelling.lines:
            if line.basis == PROPAGATED and line.key in keys:
                held[line.key] = float(line.evidence.get("distance", 0.0))
        return {key: held.get(key, 0.0) for key in keys}

    def finish(self) -> Report:
        self.report.status = self.labelling.status(rule=self.rule)
        self.log(f"done: {self.report}")
        return self.report


def _batches(keys: Sequence[str], size: int) -> list[list[str]]:
    return [list(keys[start : start + size]) for start in range(0, len(keys), size)]


def _counted(readings: Sequence[Reading]) -> dict[str, int]:
    counted: dict[str, int] = {}
    for reading in readings:
        counted[reading.name] = counted.get(reading.name, 0) + 1
    return counted


def main() -> None:
    from anthropic import Anthropic

    from hessdalen.labelling.readers import (
        MODEL,
        OLLAMA_URL,
        AnthropicBackend,
        ModelReaders,
        OllamaBackend,
        ollama_post,
    )

    parser = argparse.ArgumentParser(description="Name the corpus with a model reading sheets.")
    parser.add_argument("--root", type=Path, default=PLACES.root, help="the repository the corpus sits under")
    parser.add_argument("--signature", choices=sorted(SIGNATURES), default=DEFAULT_SIGNATURE)
    parser.add_argument("--provider", choices=("anthropic", "ollama"), default="anthropic")
    parser.add_argument("--model", default=MODEL, help="the model the provider serves")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--sheets", type=int, default=400, help="sheets to build and read at most")
    parser.add_argument("--fetches", type=int, default=0, help="recordings to fetch from the archive at most")
    parser.add_argument("--votes", type=int, default=5)
    parser.add_argument("--agreement", type=int, default=4)
    parser.add_argument("--cap", type=float, default=1.0)
    parser.add_argument("--flip-limit", type=int, default=2)
    parser.add_argument("--calibration", type=float, default=0.8, help="least share of calibration readings agreeing")
    parser.add_argument("--verification", type=float, default=0.8, help="least share of verified readings agreeing")
    parsed = parser.parse_args()

    labelling = Labelling(
        Places(root=parsed.root), settings=Settings(signature=parsed.signature, flip_limit=parsed.flip_limit)
    )
    harness = Harness(
        labelling,
        ModelReaders(
            OllamaBackend(model=parsed.model, post=ollama_post(parsed.ollama_url))
            if parsed.provider == "ollama"
            else AnthropicBackend(Anthropic(), model=parsed.model)
        ),
        rule=Rule(votes=parsed.votes, agreement=parsed.agreement, cap=parsed.cap),
        budget=Budget(sheets=parsed.sheets, fetches=parsed.fetches),
        thresholds=Thresholds(calibration=parsed.calibration, verification=parsed.verification),
        log=lambda line: print(line, file=sys.stderr, flush=True),
    )
    harness.run()


if __name__ == "__main__":
    main()
