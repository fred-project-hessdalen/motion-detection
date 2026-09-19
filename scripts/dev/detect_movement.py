"""Development script for movement detection in videos."""

import argparse
import shutil
import subprocess
from pathlib import Path


from hessdalen.io.video import VideoStream, FileFrameSource
from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.debug import TwoPanelVideoDebugSink
from hessdalen.processing.movement import (
    DetectionSettings,
    MovementDetector,
    MovementSettings,
    TrackingSettings,
)


BACKGROUND_DEFAULTS = BackgroundSettings()
DETECTION_DEFAULTS = DetectionSettings()
TRACKING_DEFAULTS = TrackingSettings()


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
        "--mean-alpha",
        type=float,
        default=BACKGROUND_DEFAULTS.mean_alpha,
        help="Rate at which the background mean follows the frame (lower = slower)",
    )
    parser.add_argument(
        "--variance-alpha",
        type=float,
        default=BACKGROUND_DEFAULTS.variance_alpha,
        help="Rate at which the per-pixel noise estimate follows the frame",
    )
    parser.add_argument(
        "--foreground-sigma",
        type=float,
        default=DETECTION_DEFAULTS.foreground_sigma,
        help="Deviation in noise sigma at which a pixel joins a blob",
    )
    parser.add_argument(
        "--detection-sigma",
        type=float,
        default=DETECTION_DEFAULTS.detection_sigma,
        help="Peak deviation in noise sigma a blob needs to be reported",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=DETECTION_DEFAULTS.min_pixels,
        help="Minimum blob size in pixels",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=TRACKING_DEFAULTS.min_consecutive_frames,
        help="Minimum consecutive frames with movement",
    )
    parser.add_argument(
        "--max-movement-ratio",
        type=float,
        default=TRACKING_DEFAULTS.max_movement_ratio,
        help="Maximum movement per frame as ratio of max frame dimension",
    )
    parser.add_argument(
        "--min-movement-ratio",
        type=float,
        default=TRACKING_DEFAULTS.min_movement_ratio,
        help="Minimum movement per frame as ratio of max frame dimension (rejects stationary brightness changes)",
    )
    parser.add_argument(
        "--min-trajectory-span-ratio",
        type=float,
        default=TRACKING_DEFAULTS.min_trajectory_span_ratio,
        help="Minimum trajectory bounding box span as ratio of max frame dimension (filters stationary noise)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate")
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> MovementSettings:
    return MovementSettings(
        background=BackgroundSettings(
            mean_alpha=args.mean_alpha,
            variance_alpha=args.variance_alpha,
        ),
        detection=DetectionSettings(
            foreground_sigma=args.foreground_sigma,
            detection_sigma=args.detection_sigma,
            min_pixels=args.min_pixels,
        ),
        tracking=TrackingSettings(
            min_consecutive_frames=args.min_consecutive_frames,
            max_movement_ratio=args.max_movement_ratio,
            min_movement_ratio=args.min_movement_ratio,
            min_trajectory_span_ratio=args.min_trajectory_span_ratio,
        ),
    )


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
    height, width = stream.frame_shape

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

    detector = MovementDetector(stream=stream, settings=settings_from_args(args))

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
