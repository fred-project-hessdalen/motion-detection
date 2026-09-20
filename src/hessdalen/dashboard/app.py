"""Streamlit dashboard for the movement detector.

It lists the development recordings, runs the detector over one or all
of them and plays a result back with the tracks drawn onto it. The
README gives the command that starts it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.typing import DataframeState

from hessdalen.dashboard.catalog import DevelopmentVideo, Label, development_videos
from hessdalen.dashboard.runs import DetectionRun, VideoProbe, load_run, probe, run_detection
from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.devices import DEVICES, Device
from hessdalen.processing.movement import (
    DetectionSettings,
    MovementSettings,
    TrackingSettings,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "data" / "out" / "dashboard"
EXAMPLES_DIR_VARIABLE = "HESSDALEN_EXAMPLES_DIR"
FRAME_HEIGHTS = (270, 540, 720, 1080, 1440, 2160)

BACKGROUND_DEFAULTS = BackgroundSettings()
DETECTION_DEFAULTS = DetectionSettings()
TRACKING_DEFAULTS = TrackingSettings()

PLAYBACK_HELP = (
    "The recording as the detector sees it, with the timestamp corner masked out. "
    "Each confirmed track carries a box on the frames it was seen in and the trail it has travelled so far. "
    "The number beside a box is the track id."
)
TRACKS_HELP = "One row per track the detector confirmed, in the order the tracks were opened."
RECORDINGS_HELP = (
    "Click a row to show that recording below. "
    "Tracks and run time are filled in once a recording has been run with the settings in the sidebar."
)
DURATION_HELP = "Taken from the container until the recording has been run, and from the decoded frames after that."
DEVICE_HELP = (
    "Where the per-pixel work runs. Auto takes the graphics card when one answers. "
    "The card rounds a float differently from the processor, so a few detections can land a pixel apart."
)


def main() -> None:
    st.set_page_config(page_title="Movement detection", layout="wide")
    st.title("Movement detection")

    videos = development_videos(_examples_dir())
    if not videos:
        st.error(f"No recordings under {_examples_dir()}. Fetch them with `dvc pull`.")
        return

    with st.sidebar:
        run_controls = st.container()
        target_height = st.select_slider(
            "Frame height",
            options=FRAME_HEIGHTS,
            value=1080,
            help="Frames are resized to this height before detection. A lower value runs faster.",
        )
        device = st.radio(
            "Device",
            options=DEVICES,
            horizontal=True,
            help=DEVICE_HELP,
        )
        settings = _settings_controls(device)

    selected = _recordings_table(videos, settings=settings, target_height=target_height)

    with run_controls:
        queued = _run_controls(videos=videos, selected=selected)
    if queued:
        _execute(queued, settings=settings, target_height=target_height)

    _playback(selected, settings=settings, target_height=target_height)


def _examples_dir() -> Path:
    override = os.environ.get(EXAMPLES_DIR_VARIABLE)
    if override:
        return Path(override)
    return REPO_ROOT / "data" / "examples"


def _run_controls(*, videos: list[DevelopmentVideo], selected: DevelopmentVideo) -> list[DevelopmentVideo]:
    """The recordings the run buttons ask for, empty when neither was
    pressed."""
    run_selected = st.button(
        "Run",
        type="primary",
        width="stretch",
        help="Run the detector over the selected recording.",
    )
    run_all = st.button(
        "Run all",
        width="stretch",
        help="Run the detector over every recording in the list.",
    )

    if run_selected:
        return [selected]
    if run_all:
        return videos
    return []


def _settings_controls(device: Device) -> MovementSettings:
    with st.expander("Detection", expanded=True):
        foreground_sigma = st.slider(
            "Foreground sigma",
            min_value=1.0,
            max_value=15.0,
            value=DETECTION_DEFAULTS.foreground_sigma,
            step=0.5,
            help="How far a pixel has to depart from its own background noise to join a blob.",
        )
        detection_sigma = st.slider(
            "Detection sigma",
            min_value=5.0,
            max_value=40.0,
            value=DETECTION_DEFAULTS.detection_sigma,
            step=0.5,
            help="How far the brightest pixel of a blob has to depart before the blob is reported.",
        )
        min_pixels = st.slider(
            "Minimum pixels",
            min_value=1,
            max_value=50,
            value=DETECTION_DEFAULTS.min_pixels,
            help="Smallest blob the detector reports.",
        )

    with st.expander("Tracking"):
        min_consecutive_frames = st.slider(
            "Minimum consecutive frames",
            min_value=1,
            max_value=20,
            value=TRACKING_DEFAULTS.min_consecutive_frames,
            help="How many frames in a row a detection has to be matched before the track is confirmed.",
        )
        max_movement_ratio = st.slider(
            "Maximum movement ratio",
            min_value=0.002,
            max_value=0.100,
            value=TRACKING_DEFAULTS.max_movement_ratio,
            step=0.002,
            format="%.3f",
            help="Furthest a track may move between frames, as a ratio of the larger frame dimension.",
        )
        min_movement_ratio = st.slider(
            "Minimum movement ratio",
            min_value=0.0000,
            max_value=0.0100,
            value=TRACKING_DEFAULTS.min_movement_ratio,
            step=0.0005,
            format="%.4f",
            help="Least a track has to move between frames, which drops stationary brightness changes.",
        )
        max_missed_frames = st.slider(
            "Maximum missed frames",
            min_value=0,
            max_value=30,
            value=TRACKING_DEFAULTS.max_missed_frames,
            help="How long a track may go unmatched and still take up a later detection.",
        )
        min_trajectory_span_ratio = st.slider(
            "Minimum trajectory span ratio",
            min_value=0.000,
            max_value=0.100,
            value=TRACKING_DEFAULTS.min_trajectory_span_ratio,
            step=0.005,
            format="%.3f",
            help="Least ground a track has to cover before it is confirmed, as a ratio of the larger frame dimension.",
        )

    with st.expander("Background"):
        mean_alpha = st.slider(
            "Mean alpha",
            min_value=0.05,
            max_value=1.00,
            value=BACKGROUND_DEFAULTS.mean_alpha,
            step=0.05,
            help="Rate at which the background mean follows the frame.",
        )
        variance_alpha = st.slider(
            "Variance alpha",
            min_value=0.005,
            max_value=0.500,
            value=BACKGROUND_DEFAULTS.variance_alpha,
            step=0.005,
            format="%.3f",
            help="Rate at which the per-pixel noise estimate follows the frame.",
        )

    return MovementSettings(
        background=BackgroundSettings(mean_alpha=mean_alpha, variance_alpha=variance_alpha),
        detection=DetectionSettings(
            foreground_sigma=foreground_sigma,
            detection_sigma=detection_sigma,
            min_pixels=min_pixels,
        ),
        tracking=TrackingSettings(
            min_consecutive_frames=min_consecutive_frames,
            max_movement_ratio=max_movement_ratio,
            min_movement_ratio=min_movement_ratio,
            max_missed_frames=max_missed_frames,
            min_trajectory_span_ratio=min_trajectory_span_ratio,
        ),
        device=device,
    )


def _execute(videos: list[DevelopmentVideo], *, settings: MovementSettings, target_height: int) -> None:
    progress = st.progress(0.0)
    for index, video in enumerate(videos):
        run_detection(
            video.path,
            settings=settings,
            target_height=target_height,
            output_dir=OUTPUT_DIR,
            on_progress=_progress_reporter(progress, index=index, total=len(videos), name=video.name),
        )
    progress.empty()
    st.rerun()


def _progress_reporter(
    progress: DeltaGenerator,
    *,
    index: int,
    total: int,
    name: str,
) -> Callable[[float], None]:
    """Report the progress of one recording as its share of the whole run.

    The bar moves to the start of that share right away, so the name of
    the recording is on screen before the first frame is decoded.
    """

    def report(fraction: float) -> None:
        progress.progress((index + fraction) / total, text=f"Running {name}")

    report(0.0)
    return report


def _recordings_table(
    videos: list[DevelopmentVideo],
    *,
    settings: MovementSettings,
    target_height: int,
) -> DevelopmentVideo:
    """Draw the list of recordings and return the one whose row is picked."""
    st.subheader("Recordings", help=RECORDINGS_HELP)
    rows = [_recording_row(video, settings=settings, target_height=target_height) for video in videos]

    state = st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        selection_default={"selection": {"rows": [0]}},
        column_config={"Duration": st.column_config.TextColumn(help=DURATION_HELP)},
        key="recordings",
    )
    picked = _picked_rows(state)
    return videos[picked[0]] if picked else videos[0]


def _picked_rows(state: DeltaGenerator | DataframeState) -> list[int]:
    if isinstance(state, dict):
        return list(state["selection"]["rows"])
    return []


def _recording_row(
    video: DevelopmentVideo,
    *,
    settings: MovementSettings,
    target_height: int,
) -> dict[str, object]:
    details = _probe_cached(video.path)
    run = load_run(video.path, settings=settings, target_height=target_height, output_dir=OUTPUT_DIR)
    frame_count = run.frame_count if run is not None else details.frame_count
    return {
        "File": video.name,
        "Collection": video.collection,
        "Duration": _duration_text(frame_count, details.frames_per_second),
        "Size": f"{details.width}x{details.height}",
        "Label": _label_text(video.labels),
        "Tracks": str(len(run.trajectories)) if run is not None else "",
        "Run time": f"{run.detection_seconds:.0f} s" if run is not None else "",
    }


def _duration_text(frame_count: int, frames_per_second: float) -> str:
    if frames_per_second <= 0:
        return f"{frame_count} frames"
    return f"{frame_count / frames_per_second:.1f} s"


@st.cache_data(show_spinner=False)
def _probe_cached(video: Path) -> VideoProbe:
    return probe(video)


def _label_text(labels: tuple[Label, ...]) -> str:
    return ", ".join(f"{label.name} {label.begin_s:.0f}-{label.end_s:.0f} s" for label in labels)


def _playback(video: DevelopmentVideo, *, settings: MovementSettings, target_height: int) -> None:
    st.subheader("Playback", help=PLAYBACK_HELP)
    run = load_run(video.path, settings=settings, target_height=target_height, output_dir=OUTPUT_DIR)
    if run is None:
        st.info("This recording has not been run with the current settings. Press Run in the sidebar.")
        return

    player, tracks = st.columns([2, 1])
    with player:
        st.video(str(run.annotated_video), start_time=_start_time(video.labels))
        st.caption(f"{len(run.trajectories)} tracks over {run.frame_count} frames.")
    with tracks:
        _tracks_table(run)


def _start_time(labels: tuple[Label, ...]) -> int:
    """Second playback opens at, a moment before the first labelled event."""
    if not labels:
        return 0
    return max(0, int(min(label.begin_s for label in labels)) - 1)


def _tracks_table(run: DetectionRun) -> None:
    st.subheader("Tracks", help=TRACKS_HELP)
    if not run.trajectories:
        st.caption("No tracks.")
        return

    rows = [
        {
            "Track": trajectory.track_id,
            "First frame": trajectory.first_frame,
            "Last frame": trajectory.last_frame,
            "Frames": len(trajectory.points),
            "Span": round(trajectory.span),
        }
        for trajectory in run.trajectories
    ]
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Span": st.column_config.NumberColumn(help="Larger side of the box the track fits into, in pixels."),
            "Frames": st.column_config.NumberColumn(help="Frames the track was matched on."),
        },
    )


if __name__ == "__main__":
    main()
