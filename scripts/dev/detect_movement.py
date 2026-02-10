"""Development script for movement detection in videos."""

import argparse
import shutil
import subprocess
from pathlib import Path


from hessdalen.io.video import VideoStream, FileFrameSource
from hessdalen.processing.debug import TwoPanelVideoDebugSink
from hessdalen.processing.movement import MovementDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Detect movement in a video")
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument(
        "--target-height",
        type=int,
        default=1080,
        help="Resize frames to this height (preserves aspect ratio)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (either a directory or an .mp4 filename)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Temporal smoothing factor (lower = more smoothing)",
    )
    parser.add_argument(
        "--adaptive-change-threshold",
        type=int,
        default=10,
        help="Per-pixel difference threshold (in grayscale intensity) for adaptive smoothing",
    )
    parser.add_argument(
        "--alpha-small-change",
        type=float,
        default=None,
        help="Alpha for small changes (higher suppresses slow small motion like tree sway)",
    )
    parser.add_argument(
        "--alpha-large-change",
        type=float,
        default=None,
        help="Alpha for large changes (lower keeps fast/large motion like birds in the diff)",
    )
    parser.add_argument("--threshold", type=int, default=20, help="Pixel difference threshold")
    parser.add_argument("--min-area", type=int, default=20, help="Minimum changed pixels for detection")
    parser.add_argument("--kernel-size", type=int, default=5, help="Morphological kernel size")
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=6,
        help="Minimum consecutive frames with movement",
    )
    parser.add_argument(
        "--max-movement-distance",
        type=float,
        default=10.0,
        help="Maximum pixel distance for spatial continuity",
    )
    parser.add_argument(
        "--min-movement-distance",
        type=float,
        default=2.0,
        help="Minimum pixel distance to reject stationary brightness changes",
    )
    parser.add_argument(
        "--min-trajectory-span-ratio",
        type=float,
        default=0.02,
        help="Minimum trajectory bounding box span as ratio of max frame dimension (filters stationary noise)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
    return parser.parse_args()


def _export_gif(*, input_video: Path, output_gif: Path) -> None:
    fps = 15
    width = 640
    palette_path = output_gif.with_suffix(".palette.png")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            f"{vf},palettegen",
            str(palette_path),
        ],
        check=True,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(palette_path),
            "-filter_complex",
            f"{vf}[x];[x][1:v]paletteuse",
            str(output_gif),
        ],
        check=True,
    )

    palette_path.unlink(missing_ok=True)


def main(args: argparse.Namespace) -> None:
    # Timestamp area in relative coords
    mask_coords = (0.8, 0.8, 1, 1)

    stream = VideoStream(
        FileFrameSource(args.video),
        mask_coords=mask_coords,
        target_height=args.target_height,
    )

    # Get frame dimensions for debug video
    temp_stream = VideoStream(
        FileFrameSource(args.video),
        target_height=args.target_height,
    )
    first_frame = next(temp_stream.stream_frames())
    height, width = first_frame.frame.shape[:2]

    if args.output:
        output_is_dir = args.output.exists() and args.output.is_dir()
        output_has_suffix = bool(args.output.suffix)
        if output_is_dir or not output_has_suffix:
            output_dir = args.output
        else:
            output_dir = args.output.parent
    else:
        output_dir = args.video.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    debug_path = output_dir / f"{args.video.stem}_filtered.mp4"
    gif_path = output_dir / f"{args.video.stem}_filtered.gif"

    detector = MovementDetector(
        stream=stream,
        alpha=args.alpha,
        adaptive_temporal_filter=True,
        adaptive_change_threshold=args.adaptive_change_threshold,
        alpha_small_change=args.alpha_small_change,
        alpha_large_change=args.alpha_large_change,
        diff_threshold=args.threshold,
        min_area=args.min_area,
        kernel_size=args.kernel_size,
        min_consecutive_frames=args.min_consecutive_frames,
        max_movement_distance=args.max_movement_distance,
        min_movement_distance=args.min_movement_distance,
        min_trajectory_span_ratio=args.min_trajectory_span_ratio,
    )

    debug_sink = TwoPanelVideoDebugSink(
        debug_path,
        width=width,
        height=height,
        fps=args.fps,
        buffer_size=int(args.min_consecutive_frames),
    )

    for event in detector.detect(debug_sink=debug_sink):
        debug_sink.record_event(event)

    debug_sink.close()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH; skipping GIF export")
        return

    if not debug_path.exists():
        print("Filtered video not found; skipping GIF export")
        return

    _export_gif(
        input_video=debug_path,
        output_gif=gif_path,
    )

    debug_path.unlink(missing_ok=True)
    print(f"Generated debug GIF: {gif_path}")


if __name__ == "__main__":
    main(parse_args())
