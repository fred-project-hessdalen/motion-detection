"""Streamlit dashboard for the movement detector.

It lists the development recordings, runs the detector over one or all
of them and plays a result back with the tracks drawn onto it. The
README gives the command that starts it.
"""

from __future__ import annotations

import filecmp
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.typing import DataframeState

from hessdalen.dashboard.catalog import DevelopmentVideo, Label, development_videos
from hessdalen.dashboard.encoder import playback_rate
from hessdalen.dashboard.live import BuiltSegment, Progress, Segment, build_segment, load_segment
from hessdalen.dashboard.panels import PANEL_CHOICES, Panels
from hessdalen.dashboard.runs import (
    DetectionRun,
    VideoProbe,
    load_run,
    probe,
    run_detection,
)
from hessdalen.dashboard.settings import (
    FILE_NAME,
    SETTINGS,
    Reading,
    as_file,
    defaults,
    from_file,
)
from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.devices import DEVICES, Device
from hessdalen.processing.movement import (
    DetectionSettings,
    MovementSettings,
    TrackingSettings,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "data" / "out" / "dashboard"
LIVE_DIR = OUTPUT_DIR / "live"
EXAMPLES_DIR_VARIABLE = "HESSDALEN_EXAMPLES_DIR"
FRAME_HEIGHTS = (270, 540, 720, 1080, 1440, 2160)
SEGMENT_STEP = 0.5
SEGMENT_SECONDS = 10.0
SEGMENT_MARGIN_SECONDS = 3.0
STATUS_FRAMES = 25
REACHING_TEXT = "Reaching the segment"
DRAWING_TEXT = "Drawing the segment"
# Session key holding the clip on screen, so a build can leave it there. A
# string of its own here would be drawn onto the page, the way Streamlit draws
# every loose expression in the script it runs.
PLAYING = "playing"

# Session keys for the file being read in and for what reading it did. The
# uploader is named after the number of files taken so far, so taking one
# empties the box and the same file can be read in again.
SETTINGS_FILE = "settings_file"
IMPORTS = "imports"
IMPORT_REPORT = "import_report"

PLAYBACK_HELP = (
    "The recording panel shows the frames as the detector sees them, with the timestamp corner masked out. "
    "The deviation panel shows how far each pixel stands from its own background, "
    "with the detection threshold at full white. "
    "Each confirmed track carries a box on the frames it was seen in and the trail it has travelled so far. "
    "The number beside a box is the track id."
)
TRACKS_HELP = "One row per track the detector confirmed, in the order the tracks were opened."
LIVE_HELP = (
    "Build a segment of the selected recording into a clip and loop it. "
    "Moving a setting builds the segment again under the new value, so its effect can be watched while it is being "
    "found, and the clip built last keeps playing while the next one is built. "
    "A clip already built for the settings in the sidebar is played straight away, so two values can be compared "
    "without waiting for either again. "
    "The whole recording still has to be run to be played back with its tracks."
)
SEGMENT_HELP = (
    "The stretch of the recording the clip holds, and the only stretch the detector measures. "
    "A background model needs about a hundred frames to settle, so a segment reads high over its own first frames "
    "where a run over the whole recording would not. Allow for that at the front, or start the segment earlier. "
    "Reaching a segment still costs a decode of the recording ahead of it, so one starting late takes longer to "
    "build."
)
PANELS_HELP = (
    "Which panels the run draws. Each one costs a pass over the recording and its share of the encode, "
    "so one panel takes about half as long as both."
)
RECORDINGS_HELP = (
    "Click a row to show that recording below. "
    "Tracks and run time are filled in once a recording has been run with the settings in the sidebar."
)
DURATION_HELP = "Taken from the container until the recording has been run, and from the decoded frames after that."
RESET_HELP = (
    "Put the detection, tracking and background settings back to the values the detector ships with. "
    "Frame height, panels and device are left as they are."
)
EXPORT_HELP = (
    "Write the detection, tracking and background settings to a file. "
    "Frame height, panels and device are left out, the same ones Reset leaves alone."
)
IMPORT_HELP = (
    "Read settings back from a file written by Export. "
    "A setting the file does not name is left where it stands, and one outside what its slider offers is "
    "brought to the nearest end."
)
DEVICE_HELP = (
    "Where the per-pixel work runs. Auto takes the graphics card when one answers, "
    "which finds the same tracks in about half the time."
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
        panels = st.segmented_control(
            "Panels",
            options=PANEL_CHOICES,
            default="both",
            format_func=str.capitalize,
            help=PANELS_HELP,
        )
        device = st.radio(
            "Device",
            options=DEVICES,
            horizontal=True,
            help=DEVICE_HELP,
        )
        settings = _settings_controls(device)

    view = _RunView(settings=settings, target_height=target_height, panels=panels or "both")
    selected = _recordings_table(videos, view=view)

    with run_controls:
        queued = _run_controls(videos=videos, selected=selected)
        live = st.toggle("Live", help=LIVE_HELP)
    if queued:
        _execute(queued, view=view)

    if live:
        _live(selected, view=view)
        return
    _playback(selected, view=view)


@dataclass(frozen=True, slots=True)
class _RunView:
    """What the sidebar asks a run to produce, which also addresses it on
    disk."""

    settings: MovementSettings
    target_height: int
    panels: Panels

    def stored(self, video: Path) -> DetectionRun | None:
        return load_run(
            video,
            settings=self.settings,
            target_height=self.target_height,
            panels=self.panels,
            output_dir=OUTPUT_DIR,
        )


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
    _seed_settings()
    foreground = SETTINGS["foreground_sigma"]
    detection = SETTINGS["detection_sigma"]
    pixels = SETTINGS["min_pixels"]
    with st.expander("Detection", expanded=True):
        foreground_sigma = st.slider(
            "Foreground sigma",
            min_value=foreground.lowest,
            max_value=foreground.highest,
            step=0.5,
            key="foreground_sigma",
            help="How far a pixel has to depart from its own background noise to join a blob.",
        )
        detection_sigma = st.slider(
            "Detection sigma",
            min_value=detection.lowest,
            max_value=detection.highest,
            step=0.5,
            key="detection_sigma",
            help="How far the brightest pixel of a blob has to depart before the blob is reported.",
        )
        min_pixels = st.slider(
            "Minimum pixels",
            min_value=int(pixels.lowest),
            max_value=int(pixels.highest),
            key="min_pixels",
            help="Smallest blob the detector reports.",
        )

    consecutive = SETTINGS["min_consecutive_frames"]
    furthest = SETTINGS["max_movement_ratio"]
    least = SETTINGS["min_movement_ratio"]
    missed = SETTINGS["max_missed_frames"]
    span = SETTINGS["min_trajectory_span_ratio"]
    with st.expander("Tracking"):
        min_consecutive_frames = st.slider(
            "Minimum consecutive frames",
            min_value=int(consecutive.lowest),
            max_value=int(consecutive.highest),
            key="min_consecutive_frames",
            help="How many frames in a row a detection has to be matched before the track is confirmed.",
        )
        max_movement_ratio = st.slider(
            "Maximum movement ratio",
            min_value=furthest.lowest,
            max_value=furthest.highest,
            step=0.002,
            format="%.3f",
            key="max_movement_ratio",
            help="Furthest a track may move between frames, as a ratio of the larger frame dimension.",
        )
        min_movement_ratio = st.slider(
            "Minimum movement ratio",
            min_value=least.lowest,
            max_value=least.highest,
            step=0.0005,
            format="%.4f",
            key="min_movement_ratio",
            help="Least a track has to move between frames, which drops stationary brightness changes.",
        )
        max_missed_frames = st.slider(
            "Maximum missed frames",
            min_value=int(missed.lowest),
            max_value=int(missed.highest),
            key="max_missed_frames",
            help="How long a track may go unmatched and still take up a later detection.",
        )
        min_trajectory_span_ratio = st.slider(
            "Minimum trajectory span ratio",
            min_value=span.lowest,
            max_value=span.highest,
            step=0.005,
            format="%.3f",
            key="min_trajectory_span_ratio",
            help="Least ground a track has to cover before it is confirmed, as a ratio of the larger frame dimension.",
        )

    mean = SETTINGS["mean_alpha"]
    variance = SETTINGS["variance_alpha"]
    with st.expander("Background"):
        mean_alpha = st.slider(
            "Mean alpha",
            min_value=mean.lowest,
            max_value=mean.highest,
            step=0.05,
            key="mean_alpha",
            help="Rate at which the background mean follows the frame.",
        )
        variance_alpha = st.slider(
            "Variance alpha",
            min_value=variance.lowest,
            max_value=variance.highest,
            step=0.005,
            format="%.3f",
            key="variance_alpha",
            help="Rate at which the per-pixel noise estimate follows the frame.",
        )

    _settings_file_controls()

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


def _settings_file_controls() -> None:
    """Draw the controls that put the settings back, write them out and read
    them in.

    The uploader carries the number of files taken so far in its name,
    so taking one draws an empty box next time. A box left holding the
    file it has already taken would say nothing about whether the
    sliders still stand where that file put them.
    """
    restore, export = st.columns(2)
    with restore:
        st.button("Reset", width="stretch", on_click=_restore_defaults, help=RESET_HELP)
    with export:
        st.download_button(
            "Export",
            data=as_file({key: st.session_state[key] for key in SETTINGS}),
            file_name=FILE_NAME,
            mime="application/json",
            width="stretch",
            help=EXPORT_HELP,
        )

    taken = st.session_state.get(IMPORTS, 0)
    st.file_uploader(
        "Import",
        type=["json"],
        key=f"{SETTINGS_FILE}_{taken}",
        on_change=_read_settings_file,
        help=IMPORT_HELP,
    )
    report = st.session_state.get(IMPORT_REPORT)
    if report:
        st.caption(report)


def _read_settings_file() -> None:
    """Take the settings an uploaded file holds into the sliders.

    Streamlit refuses to set a widget from the script once that widget
    has been drawn, so this runs as the uploader's own callback, the way
    Reset does.
    """
    taken = st.session_state.get(IMPORTS, 0)
    uploaded = st.session_state.get(f"{SETTINGS_FILE}_{taken}")
    if uploaded is None:
        st.session_state[IMPORT_REPORT] = ""
        return

    reading = from_file(uploaded.getvalue())
    st.session_state[IMPORT_REPORT] = _reading_text(reading)
    if reading.settings:
        st.session_state.update(reading.settings)
        st.session_state[IMPORTS] = taken + 1


def _reading_text(reading: Reading) -> str:
    if reading.problem:
        return reading.problem
    taken = _count(len(reading.settings), "setting")
    if reading.held:
        return f"Took {taken}, {reading.held} of them held to what the slider offers."
    return f"Took {taken}."


def _seed_settings() -> None:
    """Put each setting in the session before its slider is drawn.

    A slider carrying a starting value of its own as well as a key has
    Streamlit report that the two disagree, so the value is kept in the
    session alone and the sliders read it from there.
    """
    for key, setting in SETTINGS.items():
        if key not in st.session_state:
            st.session_state[key] = setting.default


def _restore_defaults() -> None:
    """Put every setting slider back to the value it opens on.

    Streamlit refuses to set a widget from the script once that widget
    has been drawn, so this runs as the button's own callback, which it
    takes before the page is drawn again.
    """
    st.session_state.update(defaults())
    st.session_state[IMPORT_REPORT] = ""


def _execute(videos: list[DevelopmentVideo], *, view: _RunView) -> None:
    progress = st.progress(0.0)
    for index, video in enumerate(videos):
        run_detection(
            video.path,
            settings=view.settings,
            target_height=view.target_height,
            panels=view.panels,
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


def _recordings_table(videos: list[DevelopmentVideo], *, view: _RunView) -> DevelopmentVideo:
    """Draw the list of recordings and return the one whose row is picked."""
    st.subheader("Recordings", help=RECORDINGS_HELP)
    rows = [_recording_row(video, view=view) for video in videos]

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


def _recording_row(video: DevelopmentVideo, *, view: _RunView) -> dict[str, object]:
    details = _probe_cached(video.path)
    run = view.stored(video.path)
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


def _live(video: DevelopmentVideo, *, view: _RunView) -> None:
    """Show the segment built for the settings in the sidebar, building it
    first when it has not been built before.

    The clip built last keeps looping in the page while the next one is
    built, because the browser plays it without the script's help.
    Reporting progress is also what lets a setting moved during a build
    stop it, since every mark on the page is a point at which Streamlit
    hands the script a waiting change.
    """
    st.subheader("Live", help=LIVE_HELP)
    details = _probe_cached(video.path)
    segment = _segment_control(video, details=details)
    stored = _stored_segment(video, view=view, segment=segment)

    picture = st.empty()
    status = st.empty()
    if stored is not None:
        _play(stored, picture=picture, status=status)
        return

    held = _playing()
    _play_again(picture, clip=held)
    progress = st.progress(0.0, text=REACHING_TEXT)
    try:
        built = build_segment(
            video.path,
            settings=view.settings,
            target_height=view.target_height,
            panels=view.panels,
            segment=segment,
            frames_per_second=details.frames_per_second,
            output_dir=LIVE_DIR,
            on_progress=_build_reporter(progress, status=status),
        )
    finally:
        progress.empty()

    if built.frame_count == 0:
        picture.empty()
        status.caption(f"{video.name} ends before the segment starts.")
        return
    _play(built, picture=picture, status=status, already_shown=_same_clip(built.video, held))


def _play(
    built: BuiltSegment,
    *,
    picture: DeltaGenerator,
    status: DeltaGenerator,
    already_shown: bool = False,
) -> None:
    """Put a clip in the player and start it, unless it is already there."""
    if not already_shown:
        _play_again(picture, clip=built.video)
    st.session_state[PLAYING] = str(built.video)
    status.caption(
        f"{_count(built.track_count, 'track')} over {built.frame_count} frames, built in {built.build_seconds:.0f} s."
    )


def _play_again(picture: DeltaGenerator, *, clip: Path | None) -> None:
    """Draw a clip into the player, drawing nothing when there is none.

    The player is filled in the same breath as it is made, so the two
    reach the browser together. A clip drawn exactly as the run before
    drew it leaves the player untouched and playback carries on through
    the build. Anything slow in between, or any change to what is asked
    of the player, takes it down and stops playback for as long as the
    next clip takes to build.
    """
    if clip is not None:
        picture.video(str(clip), loop=True, autoplay=True, muted=True, width="stretch")


def _playing() -> Path | None:
    """The clip on screen, which keeps playing while the next one builds."""
    shown = st.session_state.get(PLAYING)
    if shown is None:
        return None
    return Path(shown) if Path(shown).is_file() else None


def _same_clip(built: Path, held: Path | None) -> bool:
    """Whether a build came out as the clip the page is already holding.

    Streamlit names a video by a digest of its bytes and refuses to hand
    the same name out twice in one script run, so a build that lands on
    the clip being held over must not be drawn a second time. Settings
    that turn out to make no difference to the drawing do land there.
    """
    if held is None or not held.is_file():
        return False
    return filecmp.cmp(built, held, shallow=False)


def _stored_segment(video: DevelopmentVideo, *, view: _RunView, segment: Segment) -> BuiltSegment | None:
    return load_segment(
        video.path,
        settings=view.settings,
        target_height=view.target_height,
        panels=view.panels,
        segment=segment,
        output_dir=LIVE_DIR,
    )


def _build_reporter(progress: DeltaGenerator, *, status: DeltaGenerator) -> Callable[[Progress], None]:
    def report(point: Progress) -> None:
        progress.progress(point.fraction, text=DRAWING_TEXT)
        if point.frame_number % STATUS_FRAMES == 0:
            status.caption(f"Frame {point.frame_number} of {point.frame_count}.")

    return report


def _segment_control(video: DevelopmentVideo, *, details: VideoProbe) -> Segment:
    rate = playback_rate(details.frames_per_second)
    duration = max(round(details.frame_count / rate, 1), SEGMENT_STEP)
    begin, end = st.slider(
        "Segment",
        min_value=0.0,
        max_value=duration,
        value=_default_segment(video.labels, duration=duration),
        step=SEGMENT_STEP,
        format="%.1f s",
        help=SEGMENT_HELP,
    )
    return Segment(begin_frame=int(begin * rate), end_frame=int(end * rate))


def _default_segment(labels: tuple[Label, ...], *, duration: float) -> tuple[float, float]:
    """The stretch the live view opens on, which is the labelled event."""
    if not labels:
        return 0.0, min(SEGMENT_SECONDS, duration)

    begin = max(0.0, min(label.begin_s for label in labels) - SEGMENT_MARGIN_SECONDS)
    end = min(duration, max(label.end_s for label in labels) + SEGMENT_MARGIN_SECONDS)
    return begin, end


def _playback(video: DevelopmentVideo, *, view: _RunView) -> None:
    st.subheader("Playback", help=PLAYBACK_HELP)
    run = view.stored(video.path)
    if run is None:
        st.info("This recording has not been run with the current settings. Press Run in the sidebar.")
        return

    player, tracks = st.columns([1, 1])
    with player:
        st.video(str(run.annotated_video), start_time=_start_time(video.labels))
        st.caption(f"{_count(len(run.trajectories), 'track')} over {run.frame_count} frames.")
    with tracks:
        _tracks_table(run)


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


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
