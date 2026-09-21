"""Sift the archive: fetch a recording, detect, keep what moved.

The archive holds far more video than this project can store, and most
of it holds nothing. The run walks a selection of the listing, fetching
one recording at a time, running the detector over it, writing its track
file and deleting the video. What is left behind is a track file for
every recording and a ledger line saying what the detector found.

The cuts whose tracks scored highest among the cuts of their recording
are then fetched again into the corpus, where the dashboard and the tests
can reach them. Scoring is a placeholder for the classifier being built
beside this: it reads the written track files and nothing else, so
--rescore re-decides a finished run without fetching anything twice.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from hashlib import sha1
from pathlib import Path
from queue import Queue
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from hessdalen.config import CONFIG_PATH, config
from hessdalen.dashboard.catalog import CLIP_NAME, VIDEO_SUFFIXES
from hessdalen.dashboard.runs import probe
from hessdalen.domain.models import MovementEvent
from hessdalen.io.drive import ArchiveVideo, cut_out, fetch, fetch_through, matching, read_inventory
from hessdalen.io.tracks import write_tracks
from hessdalen.io.video import VideoStream, masked_stream
from hessdalen.processing.movement import MovementDetector, MovementSettings

METADATA_COLUMNS = ("file", "movement", "label", "begin_s", "end_s")
CLIPPER = Path(__file__).resolve().parent / "extract_example_clips.py"
TABLE_COLUMNS = ("score", "kept", "label", "category", "event", "name", "track_count", "frame_count", "url")


@dataclass(frozen=True, slots=True)
class Finding:
    """What one recording gave up, as the ledger keeps it."""

    file_id: str
    url: str
    archive_path: str
    name: str
    event: str
    category: str
    size_bytes: int
    frame_count: int
    detection_seconds: float
    target_height: int
    config_blob: str
    tracks_path: str
    track_count: int
    score: float
    first_frame: int
    last_frame: int


@dataclass(frozen=True, slots=True)
class Score:
    """The best track of a recording, as the placeholder measures it."""

    value: float
    track_count: int
    first_frame: int
    last_frame: int


@dataclass(frozen=True, slots=True)
class Arrival:
    """A fetched recording, or the reason it did not arrive."""

    video: ArchiveVideo
    path: Path | None
    failure: str | None


@dataclass(frozen=True, slots=True)
class LabelRow:
    """One labelled stretch of a video, as a metadata file lists it."""

    file: str
    label: str
    begin_s: float
    end_s: float


def main(args: argparse.Namespace) -> None:
    if args.rescore:
        rescore(args.ledger)
    else:
        scan(args, frozen=blob_hash(CONFIG_PATH))

    collect(args)


def rescore(ledger: Path) -> None:
    """Score every recording in the ledger again from its track file,
    fetching nothing."""
    settings = config().settings
    findings = [rescored(finding, settings=settings) for finding in read_ledger(ledger)]

    partial = ledger.with_suffix(ledger.suffix + ".part")
    partial.write_text("".join(json.dumps(asdict(finding)) + "\n" for finding in findings))
    partial.replace(ledger)


def rescored(finding: Finding, *, settings: MovementSettings) -> Finding:
    path = Path(finding.tracks_path)
    held = pq.read_schema(path).metadata
    frame_shape = (int(held[b"hessdalen_frame_height"]), int(held[b"hessdalen_frame_width"]))

    score = score_tracks(path, frame_shape=frame_shape, settings=settings)
    return replace(
        finding,
        score=score.value,
        track_count=score.track_count,
        first_frame=score.first_frame,
        last_frame=score.last_frame,
    )


def fetcher(args: argparse.Namespace) -> Callable[[ArchiveVideo, Path], Path]:
    """How this run pulls a recording down.

    The public endpoint serves a file until enough of it has been handed
    out and then refuses it for as long as a day, so a run over hundreds
    of recordings goes through an account instead.
    """
    if not args.remote:
        return fetch
    return partial(fetch_through, args.remote)


def scan(args: argparse.Namespace, *, frozen: str) -> None:
    """Fetch, detect and discard every recording not already in the
    ledger."""
    videos = pending(args)
    if not videos:
        return

    staging = args.staging / "clips"
    staging.mkdir(parents=True, exist_ok=True)

    room = threading.Semaphore(args.fetch_workers + 1)
    arrivals: Queue[Arrival] = Queue()
    stopping = threading.Event()
    pool = ThreadPoolExecutor(max_workers=args.fetch_workers)

    pull = fetcher(args)
    try:
        for video in videos:
            pool.submit(
                stage,
                video,
                staging,
                room=room,
                arrivals=arrivals,
                stopping=stopping,
                min_free_gb=args.min_free_gb,
                pull=pull,
            )
        for _ in videos:
            take(arrivals.get(), args=args, frozen=frozen, room=room)
    finally:
        stopping.set()
        for _ in range(args.fetch_workers + 1):
            room.release()
        pool.shutdown(cancel_futures=True)


def pending(args: argparse.Namespace) -> list[ArchiveVideo]:
    """The selected recordings the ledger has no line for yet."""
    listed = cut_out(matching(read_inventory(args.inventory), args.select))
    done = {finding.file_id for finding in read_ledger(args.ledger)}

    waiting = sorted((video for video in listed if video.file_id not in done), key=lambda video: video.path)
    return waiting[: args.limit] if args.limit else waiting


def stage(
    video: ArchiveVideo,
    staging: Path,
    *,
    room: threading.Semaphore,
    arrivals: "Queue[Arrival]",
    stopping: threading.Event,
    min_free_gb: float,
    pull: Callable[[ArchiveVideo, Path], Path],
) -> None:
    """Fetch one recording, waiting until the consumer has room for it."""
    room.acquire()
    if stopping.is_set():
        return

    free = free_gigabytes(staging)
    if free < min_free_gb:
        arrivals.put(Arrival(video, None, f"{free:.1f} GB free is under the {min_free_gb:.1f} GB floor"))
        return

    try:
        arrivals.put(Arrival(video, pull(video, staging / video.name), None))
    except (OSError, subprocess.CalledProcessError) as failure:
        arrivals.put(Arrival(video, None, str(failure)))


def take(arrival: Arrival, *, args: argparse.Namespace, frozen: str, room: threading.Semaphore) -> None:
    """Detect over an arrival, write its ledger line and drop the video.

    A recording that did not arrive is reported and left out of the
    ledger, so the next run fetches it again.
    """
    if arrival.path is None:
        print(f"{arrival.video.name} not fetched: {arrival.failure}", flush=True)
        room.release()
        return

    try:
        finding = measure(arrival.video, arrival.path, args=args, frozen=frozen)
    finally:
        arrival.path.unlink(missing_ok=True)
        room.release()

    append(args.ledger, finding)
    print(
        f"{finding.category}/{finding.event}/{finding.name} "
        f"{finding.track_count} tracks score {finding.score:.2f} in {finding.detection_seconds:.1f}s",
        flush=True,
    )


def measure(video: ArchiveVideo, path: Path, *, args: argparse.Namespace, frozen: str) -> Finding:
    """Run the detector over a fetched recording and write its tracks.

    The settings are frozen for the length of a run. A run that straddled
    a change would hold two populations of tracks under one ledger, and
    the file the dashboard writes on Save is the one the detector reads,
    so the check is against the file rather than against what git holds.
    """
    blob = blob_hash(CONFIG_PATH)
    if blob != frozen:
        raise SystemExit(f"{CONFIG_PATH} changed from {frozen} to {blob}. Stopped before {video.name}.")

    settings = config().settings
    started = time.monotonic()
    stream = masked_stream(path, target_height=args.target_height)
    height, width = stream.frame_shape

    tracks_path = args.tracks / video.label / video.event / f"{Path(video.name).stem}.parquet"
    write_tracks(
        tracks_path,
        recording=video.name,
        frame_height=height,
        frame_width=width,
        settings=settings,
        events=events(stream, settings=settings),
    )
    detection_seconds = time.monotonic() - started

    score = score_tracks(tracks_path, frame_shape=(height, width), settings=settings)
    return Finding(
        file_id=video.file_id,
        url=video.url,
        archive_path=video.path,
        name=video.name,
        event=video.event,
        category=video.category,
        size_bytes=video.size_bytes,
        frame_count=probe(path).frame_count,
        detection_seconds=detection_seconds,
        target_height=args.target_height,
        config_blob=blob,
        tracks_path=str(tracks_path),
        track_count=score.track_count,
        score=score.value,
        first_frame=score.first_frame,
        last_frame=score.last_frame,
    )


def events(stream: VideoStream, *, settings: MovementSettings) -> list[MovementEvent]:
    return list(MovementDetector(stream=stream, settings=settings).detect())


def score_tracks(path: Path, *, frame_shape: tuple[int, int], settings: MovementSettings) -> Score:
    """How much the best track of a recording looks like an object.

    A track scores on three things at once: how far it travels across
    the frame, how long it lasts, and how far its brightest pixel stands
    above the noise. Foliage and rain move a little and briefly, a
    keyframe pulse stands out but does not travel, and an object does
    all three. This is the seam the classifier replaces; nothing else in
    the run reads a track.
    """
    table = pq.read_table(path)
    if table.num_rows == 0:
        return Score(value=0.0, track_count=0, first_frame=0, last_frame=0)

    height, width = frame_shape
    diagonal = math.hypot(height, width)
    tracks = grouped(table)

    best = max(tracks.values(), key=lambda frames: track_value(frames, diagonal=diagonal, settings=settings))
    return Score(
        value=track_value(best, diagonal=diagonal, settings=settings),
        track_count=len(tracks),
        first_frame=int(best[0]["frame_number"]),
        last_frame=int(best[-1]["frame_number"]),
    )


def track_value(frames: list[dict[str, Any]], *, diagonal: float, settings: MovementSettings) -> float:
    """What one track scores: how far it went, how long it lasted, and
    how far it stood above the noise, multiplied together."""
    first, last = frames[0], frames[-1]
    span = last["frame_number"] - first["frame_number"] + 1
    travel = math.hypot(last["centre_x"] - first["centre_x"], last["centre_y"] - first["centre_y"]) / diagonal
    strength = max(frame["peak_deviation"] for frame in frames) / settings.detection.detection_sigma

    return travel * math.sqrt(span) * strength


def grouped(table: pa.Table) -> dict[int, list[dict[str, Any]]]:
    """The rows of a track file, gathered per track in frame order."""
    tracks: dict[int, list[dict[str, Any]]] = {}
    for row in table.to_pylist():
        tracks.setdefault(row["track_id"], []).append(row)

    for frames in tracks.values():
        frames.sort(key=lambda frame: frame["frame_number"])
    return tracks


def collect(args: argparse.Namespace) -> None:
    """Fetch the winning recordings into the corpus and describe them."""
    findings = read_ledger(args.ledger)
    if not findings:
        raise SystemExit(f"No ledger yet at {args.ledger}.")

    keeping = winners(findings, margin=args.keep_margin)
    videos = args.corpus / "videos"
    pull = fetcher(args)
    for finding in keeping:
        target = videos / finding.name
        if not target.exists():
            pull(as_video(finding), target)
            print(f"kept {finding.category}/{finding.event}/{finding.name} at score {finding.score:.2f}", flush=True)

    labels = [recording_label(finding, videos=videos) for finding in keeping]
    write_labels(args.corpus / "metadata.csv", labels)
    cut(args, labels)
    if args.table:
        write_table(args.table, findings, keeping=keeping)


def winners(findings: list[Finding], *, margin: float) -> list[Finding]:
    """The best-scoring cut of every recording, and the ones close behind
    it."""
    recordings: dict[str, list[Finding]] = {}
    for finding in findings:
        recordings.setdefault(as_video(finding).recording, []).append(finding)

    kept: list[Finding] = []
    for siblings in recordings.values():
        top = max(finding.score for finding in siblings)
        if top <= 0.0:
            continue
        kept.extend(finding for finding in siblings if finding.score >= margin * top)
    return sorted(kept, key=lambda finding: -finding.score)


def recording_label(finding: Finding, *, videos: Path) -> LabelRow:
    """Describe a kept recording the way the example set is described.

    The label is what the archive calls the recording and the seconds
    are where the best track ran, so a row is what the detector claims
    rather than what a person confirmed.
    """
    fps = probe(videos / finding.name).frames_per_second
    return LabelRow(
        file=finding.name,
        label=as_video(finding).label,
        begin_s=finding.first_frame / fps,
        end_s=finding.last_frame / fps,
    )


def cut(args: argparse.Namespace, labels: list[LabelRow]) -> None:
    """Cut a copy of every kept recording down to where its best track
    ran, and describe the copies in a metadata file of their own.

    The copies are the tracked product. They derive entirely from the
    full recordings and their labels, so they are rebuilt from scratch
    on every pass, which drops the copy of a recording a rescore no
    longer keeps.
    """
    clips = args.cuts / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for stale in clips.iterdir():
        if stale.suffix in VIDEO_SUFFIXES:
            stale.unlink()

    subprocess.run(
        [
            sys.executable,
            str(CLIPPER),
            "--metadata",
            str(args.corpus / "metadata.csv"),
            "--videos-dir",
            str(args.corpus / "videos"),
            "--output-dir",
            str(clips),
            "--padding-seconds",
            str(args.clip_margin),
        ],
        check=True,
    )

    names = sorted(path.name for path in clips.iterdir() if path.suffix in VIDEO_SUFFIXES)
    write_labels(args.cuts / "metadata.csv", clip_labels(names, {label.file: label for label in labels}))


def clip_labels(names: list[str], recordings: dict[str, LabelRow]) -> list[LabelRow]:
    """The label of each clip's recording, moved into the clip's own
    time."""
    rows = []
    for name in names:
        path = Path(name)
        clip = CLIP_NAME.match(path.stem)
        if clip is None:
            raise ValueError(f"{name} is not named the way a cut clip is")

        start = float(clip["begin"])
        length = float(clip["end"]) - start
        recording = recordings[f"{clip['recording']}{path.suffix}"]
        rows.append(
            LabelRow(
                file=name,
                label=recording.label,
                begin_s=max(recording.begin_s - start, 0.0),
                end_s=min(recording.end_s - start, length),
            )
        )
    return rows


def write_labels(path: Path, rows: list[LabelRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(METADATA_COLUMNS)
        for row in rows:
            writer.writerow((row.file, 1, row.label, f"{row.begin_s:.1f}", f"{row.end_s:.1f}"))


def write_table(path: Path, findings: list[Finding], *, keeping: list[Finding]) -> None:
    """The ledger as one row per recording, best score first."""
    kept = {finding.file_id for finding in keeping}
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TABLE_COLUMNS)
        for finding in sorted(findings, key=lambda finding: -finding.score):
            writer.writerow(
                (
                    f"{finding.score:.3f}",
                    int(finding.file_id in kept),
                    as_video(finding).label,
                    finding.category,
                    finding.event,
                    finding.name,
                    finding.track_count,
                    finding.frame_count,
                    finding.url,
                )
            )


def as_video(finding: Finding) -> ArchiveVideo:
    return ArchiveVideo(
        path=finding.archive_path, name=finding.name, file_id=finding.file_id, size_bytes=finding.size_bytes
    )


def read_ledger(path: Path) -> list[Finding]:
    """Every line the ledger holds, oldest first."""
    if not path.exists():
        return []
    return [Finding(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def append(path: Path, finding: Finding) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(asdict(finding)) + "\n")


def blob_hash(path: Path) -> str:
    """The hash git would give this file's contents."""
    content = path.read_bytes()
    return sha1(b"blob %d\0" % len(content) + content).hexdigest()


def free_gigabytes(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch archive recordings, detect over them and keep what moved",
        epilog="@FILE reads further arguments from FILE, one per line, so a long selection keeps its spaces.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--inventory", type=Path, required=True, help="CSV listing of the archive")
    parser.add_argument("--remote", default="", help="rclone remote rooted at the archive, such as hessdalen:")
    parser.add_argument("--select", nargs="+", default=["cameras/trainingData"], help="Archive path patterns to walk")
    parser.add_argument("--ledger", type=Path, default=Path("data/out/sift/ledger.jsonl"), help="Where findings go")
    parser.add_argument("--staging", type=Path, default=Path("data/out/sift"), help="Where a fetch lands")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"), help="Where kept recordings go")
    parser.add_argument("--tracks", type=Path, default=Path("data/corpus/tracks"), help="Where track files go")
    parser.add_argument("--cuts", type=Path, default=Path("data/cuts"), help="Where the cut copies go")
    parser.add_argument("--clip-margin", type=float, default=5.0, help="Seconds kept either side of the best track")
    parser.add_argument("--table", type=Path, help="Write the ledger as a CSV table here")
    parser.add_argument("--target-height", type=int, default=config().frame_height, help="Frame height to detect at")
    parser.add_argument("--fetch-workers", type=int, default=3, help="Recordings fetched at once")
    parser.add_argument("--keep-margin", type=float, default=0.5, help="Share of the best score a sibling has to reach")
    parser.add_argument("--min-free-gb", type=float, default=8.0, help="Free space the run refuses to go below")
    parser.add_argument("--limit", type=int, help="Stop after this many recordings")
    parser.add_argument("--rescore", action="store_true", help="Re-decide from the written track files, fetching none")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
