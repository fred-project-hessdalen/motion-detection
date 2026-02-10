#!/usr/bin/env python3

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class TimeRangeColumns:
    start_col: str
    end_col: str


def _is_start_col(name: str) -> bool:
    n = name.strip().lower()
    return (
        n in {"begin_s", "start_s", "begin", "start"}
        or n.endswith("begin_s")
        or n.endswith("start_s")
        or n.startswith("begin")
        or n.startswith("start")
    )


def _is_end_col(name: str) -> bool:
    n = name.strip().lower()
    return (
        n in {"end_s", "stop_s", "end", "stop"}
        or n.endswith("end_s")
        or n.endswith("stop_s")
        or n.startswith("end")
        or n.startswith("stop")
    )


def _pick_timerange_columns(fieldnames: list[str]) -> TimeRangeColumns:
    starts = [c for c in fieldnames if _is_start_col(c)]
    ends = [c for c in fieldnames if _is_end_col(c)]

    if not starts or not ends:
        raise ValueError(f"Could not find start/end columns in metadata. Found starts={starts} ends={ends}")

    pairs: list[TimeRangeColumns] = []
    for i in range(min(len(starts), len(ends))):
        pairs.append(TimeRangeColumns(start_col=starts[i], end_col=ends[i]))

    return pairs[1] if len(pairs) >= 2 else pairs[0]


def _coerce_float(value: str, *, field: str, row_index: int) -> float:
    try:
        return float(value)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Row {row_index}: could not parse {field}={value!r} as float") from e


def _run_ffmpeg(*, input_path: Path, output_path: Path, start_s: float, end_s: float) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _run_ffmpeg_reencode(*, input_path: Path, output_path: Path, start_s: float, end_s: float) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _get_video_duration_seconds(input_path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None

    value = proc.stdout.strip()
    try:
        duration_s = float(value)
    except Exception:  # noqa: BLE001
        return None

    if duration_s <= 0:
        return None
    return duration_s


def _has_decodable_frames(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False

        for _ in range(10):
            ret, _ = cap.read()
            if ret:
                return True
        return False
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract clips from data/examples/videos referenced by data/examples/metadata.csv. "
            "If the CSV contains multiple start/end column pairs, the *second* pair is used; "
            "otherwise the only pair is used."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/examples/metadata.csv"),
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/examples/videos"),
        help="Directory containing the videos referenced by the CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/examples/clips"),
        help="Directory to write extracted clips.",
    )
    parser.add_argument(
        "--padding-seconds",
        type=float,
        default=0.0,
        help="Padding to subtract from start and add to end (seconds).",
    )

    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH.", file=sys.stderr)
        return 2

    if not args.metadata.exists():
        print(f"Metadata CSV not found: {args.metadata}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.metadata.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Metadata CSV has no header.", file=sys.stderr)
            return 2

        fieldnames = list(reader.fieldnames)
        file_col = "file" if "file" in fieldnames else fieldnames[0]
        time_cols = _pick_timerange_columns(fieldnames)

        for idx, row in enumerate(reader, start=2):
            filename = (row.get(file_col) or "").strip()
            if not filename:
                continue

            start_raw = (row.get(time_cols.start_col) or "").strip()
            end_raw = (row.get(time_cols.end_col) or "").strip()
            if not start_raw or not end_raw:
                continue

            start_s = _coerce_float(start_raw, field=time_cols.start_col, row_index=idx)
            end_s = _coerce_float(end_raw, field=time_cols.end_col, row_index=idx)

            start_s = max(0.0, start_s - args.padding_seconds)
            end_s = end_s + args.padding_seconds

            if end_s <= start_s:
                print(
                    f"Row {idx}: skipping invalid range {start_s:.3f}-{end_s:.3f} for {filename}",
                    file=sys.stderr,
                )
                continue

            input_path = args.videos_dir / filename
            if not input_path.exists():
                print(f"Row {idx}: missing video: {input_path}", file=sys.stderr)
                continue

            duration_s = _get_video_duration_seconds(input_path)
            if duration_s is not None:
                if start_s >= duration_s:
                    print(
                        f"Row {idx}: skipping out-of-range start {start_s:.3f}s for {filename} (duration {duration_s:.3f}s)",
                        file=sys.stderr,
                    )
                    continue

                end_s = min(end_s, duration_s)
                if end_s <= start_s:
                    print(
                        f"Row {idx}: skipping invalid/clamped range {start_s:.3f}-{end_s:.3f} for {filename}",
                        file=sys.stderr,
                    )
                    continue

            stem = input_path.stem
            suffix = input_path.suffix or ".mkv"
            out_name = f"{stem}_clip_{start_s:.3f}_{end_s:.3f}{suffix}"
            output_path = args.output_dir / out_name

            try:
                _run_ffmpeg(
                    input_path=input_path,
                    output_path=output_path,
                    start_s=start_s,
                    end_s=end_s,
                )

                if output_path.exists() and not _has_decodable_frames(output_path):
                    try:
                        _run_ffmpeg_reencode(
                            input_path=input_path,
                            output_path=output_path,
                            start_s=start_s,
                            end_s=end_s,
                        )
                    except subprocess.CalledProcessError as e:
                        output_path.unlink(missing_ok=True)
                        print(
                            f"Row {idx}: stream-copy clip not decodable and re-encode failed; removed: {output_path}. Error: {e}",
                            file=sys.stderr,
                        )

                if not output_path.exists() or not _has_decodable_frames(output_path):
                    output_path.unlink(missing_ok=True)
                    print(
                        f"Row {idx}: extracted clip had no decodable frames; removed: {output_path}",
                        file=sys.stderr,
                    )
            except subprocess.CalledProcessError as e:
                print(
                    f"Row {idx}: ffmpeg failed for {input_path}: {e}",
                    file=sys.stderr,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
