"""The track map, where clicking a track shows it at once and then plays it
where it was found.

Every track of the corpus is a point, placed and coloured by the map the
analysis step writes. Clicking one draws its stored path straight away,
and hands drawing it on its video to a background queue, because that
has to pass every frame of the recording ahead of the track and can take
a minute. A gallery below draws a random sample of any one cluster from
the stored paths alone, which is how a cluster is judged at a glance,
and a second one draws the tracks nearest the selected one, from
whatever cluster they are in.

The page never detects anything. A track whose video the sift did not
keep has its video fetched from the archive before it is drawn.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from plotly.colors import qualitative

from hessdalen.dashboard.clip_queue import ClipQueue, Job, Report
from hessdalen.dashboard.runs import probe
from hessdalen.dashboard.track_clip import (
    ClipProgress,
    StoredTrack,
    build_track_clip,
    stretch_around,
    track_clip_path,
)
from hessdalen.dashboard.track_preview import close_up, frame_view, gallery_html, light_curve
from hessdalen.dashboard.video_cache import (
    MIN_FREE_BYTES,
    FetchProgress,
    archive_video,
    cached_video,
    fetch_seconds,
    fetch_video,
    free_bytes,
    room_to_fetch,
)
from hessdalen.io.drive import ArchiveVideo

REPO_ROOT = Path(__file__).resolve().parents[3]
MAP_PATH = REPO_ROOT / "data" / "out" / "analysis" / "track-map.parquet"
PATHS_PATH = MAP_PATH.with_name("track-paths.parquet")
VIDEOS_DIR = REPO_ROOT / "data" / "corpus" / "videos"
FETCHED_DIR = REPO_ROOT / "data" / "out" / "dashboard" / "videos"
CLIPS_DIR = REPO_ROOT / "data" / "out" / "dashboard" / "tracks"
LEDGERS = (
    REPO_ROOT / "data" / "out" / "sift" / "ledger.jsonl",
    REPO_ROOT / "data" / "out" / "sift" / "bulk-ledger.jsonl",
)

MAP_COMMAND = "uv run --group analysis python scripts/dev/map_tracks.py"
UNASSIGNED_NAME = "none"
UNASSIGNED_COLOR = "#b8b8b8"
CLUSTER_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#eeca3b", "#b279a2", "#ff9da6", "#9d755d")
COLOUR_CHOICES = ("Cluster", "Side", "Label", "Camera")
COLOUR_COLUMNS = {"Cluster": "cluster", "Side": "side", "Label": "label", "Camera": "camera"}
SELECTED_CLUSTER = "Selected track's cluster"
OTHER_COLORS = qualitative.Dark24
HOVER_COLUMNS = ["key", "label", "cluster", "side", "clip", "track_id", "frames", "straightness", "peak_deviation_max"]
HOVER_TEMPLATE = (
    "Label %{customdata[1]}<br>Cluster %{customdata[2]}<br>Side %{customdata[3]}<br>"
    "Recording %{customdata[4]}<br>Track %{customdata[5]}<br>Frames %{customdata[6]}<br>"
    "Straightness %{customdata[7]:.2f}<br>Peak deviation %{customdata[8]:.1f}<extra></extra>"
)
GALLERY_SIZE = 30
NEIGHBOUR_RADIUS = 0.5
SHUFFLE_KEY = "gallery_shuffle"
PANEL_CACHE_ENTRIES = 64
POLL_SECONDS = 2.0

AUTO_FETCH_BYTES = 100 * 1024**2
"""Largest video a click fetches without being asked.

The one-minute cuts are about 30 MB and arrive in under a minute. A
whole 20-minute recording takes several minutes, and holds up every
other clip meanwhile, so that fetch waits for a press.
"""

COLOUR_HELP = (
    "What the points are coloured by. Clusters come from the descriptors alone. The side says whether a "
    "track moves evenly from step to step, as a clean path does, or hops about as clutter does, and "
    "each side is clustered on its own. A label is the folder the recording was filed under, which names "
    "the whole recording, so most tracks under a label are that scene's background activity and not the "
    "thing the folder is named for."
)
LABELS_HELP = "Show only the tracks of recordings filed under these labels."
GALLERY_CHOICE_HELP = "The cluster the gallery below the map draws a sample of."
MAP_HELP = (
    "Every track the corpus holds, placed so that tracks with similar descriptors sit close together. "
    "Grey points are tracks the clustering left out of every cluster. Click a point to see its track. "
    "Scroll to zoom, drag to pan, and double-click to zoom back out. Click an entry in the legend to hide "
    "or show its tracks."
)
PATH_HELP = (
    "The track drawn from its stored path through the blob's centre, which is what the descriptors "
    "were computed from. Colour runs from dark blue on its first frame to yellow on its last."
)
VIDEO_HELP = (
    "The track drawn on its recording, from a second before it starts to a second after it ends. "
    "It is built in the background, one track at a time, and choosing another track stops the build "
    "under way."
)
GALLERY_HELP = (
    f"Up to {GALLERY_SIZE} tracks drawn at random from the cluster, each drawn from its stored path and "
    "fitted to its own panel. Colour runs from dark blue on a track's first frame to yellow on its last."
)
SHUFFLE_HELP = "Draw another random sample of the cluster."
NEIGHBOURS_HELP = (
    f"The {GALLERY_SIZE} tracks that lie nearest the selected track on the map, from any cluster, and no "
    "further from it than the neighbour radius. Each panel names the cluster its track is in."
)
RADIUS_HELP = (
    "How far from the selected track, in the map's own units, a track may lie to count among its nearest "
    f"tracks. At {NEIGHBOUR_RADIUS} most tracks have {GALLERY_SIZE} such neighbours."
)
UNSOURCED_TEXT = (
    "The video behind this track was not kept after detection, and no ledger says where in the archive "
    "it came from, so it cannot be fetched."
)


@dataclass(frozen=True, slots=True)
class VideoSource:
    """Where a track's recording can be had from: on disk, or from the
    archive."""

    path: Path | None
    archived: ArchiveVideo | None
    link: str


def page() -> None:
    st.title("Track map")
    if not (MAP_PATH.is_file() and PATHS_PATH.is_file()):
        st.error(f"No track map at {MAP_PATH}. Write it with `{MAP_COMMAND}`.")
        return

    tracks = _map_frame(MAP_PATH.stat().st_mtime)
    paths = _path_frame(PATHS_PATH.stat().st_mtime)
    with st.sidebar:
        colour = st.segmented_control("Colour", options=COLOUR_CHOICES, default="Cluster", help=COLOUR_HELP)
        labels = st.multiselect("Labels", options=sorted(tracks["label"].unique()), help=LABELS_HELP)
        chosen_cluster = st.selectbox(
            "Gallery",
            options=[SELECTED_CLUSTER, *_cluster_names(tracks)],
            format_func=_cluster_title,
            help=GALLERY_CHOICE_HELP,
        )
        radius = st.number_input(
            "Neighbour radius", min_value=0.05, max_value=5.0, value=NEIGHBOUR_RADIUS, step=0.05, help=RADIUS_HELP
        )

    shown = tracks[tracks["label"].isin(labels)] if labels else tracks
    map_column, track_column = st.columns([3, 2])
    with map_column:
        st.subheader("Tracks", help=MAP_HELP)
        st.caption(f"{len(shown)} of {len(tracks)} tracks")
        event = st.plotly_chart(
            _scatter(shown, colour_column=COLOUR_COLUMNS[colour or "Cluster"]),
            on_select="rerun",
            selection_mode="points",
            key="track_map",
            width="stretch",
            config={"scrollZoom": True, "displaylogo": False},
        )
        picked = _picked(shown, event)
        _gallery(shown, cluster=_gallery_cluster(str(chosen_cluster), picked=picked))
        if picked is not None:
            _neighbours(shown, track=picked, radius=float(radius))

    with track_column:
        if picked is None:
            st.caption("No track selected.")
        else:
            _selected(picked, points=_points(paths, key=str(picked["key"])))


def _scatter(tracks: pd.DataFrame, *, colour_column: str) -> go.Figure:
    """The tracks drawn by the graphics card, one trace per colour, so the
    legend names each colour and a click on it hides or shows those tracks.

    Drawing on the card keeps zooming and panning smooth over thousands
    of points. The fixed UI revision keeps the zoom when the page is
    drawn again after a click.
    """
    traces = []
    for name, colour in _colours(tracks, column=colour_column).items():
        members = tracks[tracks[colour_column] == name]
        traces.append(
            go.Scattergl(
                x=members["x"],
                y=members["y"],
                mode="markers",
                name=str(name),
                marker={"color": colour, "size": 6, "opacity": 0.7},
                selected={"marker": {"opacity": 1.0, "size": 11}},
                unselected={"marker": {"opacity": 0.3}},
                customdata=members[HOVER_COLUMNS].to_numpy(dtype=object),
                hovertemplate=HOVER_TEMPLATE,
            )
        )
    figure = go.Figure(traces)
    figure.update_layout(
        height=620,
        dragmode="pan",
        uirevision="track-map",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"title": {"text": colour_column.capitalize()}},
        xaxis={"showgrid": True, "zeroline": False},
        yaxis={"showgrid": True, "zeroline": False},
    )
    return figure


def _colours(tracks: pd.DataFrame, *, column: str) -> dict[str, str]:
    """The colour of each value of the column, with the clusters in their
    fixed colours and tracks outside every cluster in grey."""
    if column != "cluster":
        names = sorted(tracks[column].unique())
        return {name: OTHER_COLORS[index % len(OTHER_COLORS)] for index, name in enumerate(names)}

    named = [name for name in _cluster_names(tracks) if name != UNASSIGNED_NAME]
    colours = {name: CLUSTER_COLORS[index % len(CLUSTER_COLORS)] for index, name in enumerate(named)}
    if (tracks["cluster"] == UNASSIGNED_NAME).any():
        colours[UNASSIGNED_NAME] = UNASSIGNED_COLOR
    return colours


def _picked(shown: pd.DataFrame, event: Any) -> pd.Series | None:
    points = event.selection.get("points", []) if isinstance(event.selection, Mapping) else []
    if not points:
        return None
    held = shown.loc[shown["key"] == str(points[0]["customdata"][0])]
    return None if held.empty else held.iloc[0]


def _selected(track: pd.Series, *, points: pd.DataFrame) -> None:
    """The picked track drawn at once from its stored path, and its video once
    that is built."""
    st.subheader(f"Track {int(track['track_id'])} in {track['clip']}", help=PATH_HELP)
    st.caption(_facts(track))
    st.altair_chart(frame_view(points))
    st.altair_chart(close_up(points))
    st.altair_chart(light_curve(points))

    st.subheader("Video", help=VIDEO_HELP)
    _video(str(track["key"]), recording=str(track["recording"]), stored=_stored_track(points))


def _video(key: str, *, recording: str, stored: StoredTrack) -> None:
    """The clip of the track when it is built, and its progress until then."""
    source = _source(recording)
    if source is None:
        st.info(UNSOURCED_TEXT)
        return

    if source.path is not None:
        clip = _clip_path(source.path, track=stored)
        if clip.is_file():
            st.video(str(clip), loop=True, autoplay=True, muted=True)
            return
    elif not _may_fetch(source):
        return

    _queue().request(key, _clip_job(source, track=stored))
    _clip_progress(key, fetching=source.path is None)


def _source(recording: str) -> VideoSource | None:
    kept = VIDEOS_DIR / recording
    if kept.is_file():
        return VideoSource(path=kept, archived=None, link="")

    entry = _ledger_entries(_ledger_stamp()).get(recording)
    if entry is None:
        return None
    archived = archive_video(entry)
    return VideoSource(
        path=cached_video(FETCHED_DIR, video=archived), archived=archived, link=str(entry.get("url", ""))
    )


def _may_fetch(source: VideoSource) -> bool:
    """Whether the recording may be fetched now: there is room for it, and a
    large one has been asked for."""
    archived = source.archived
    if archived is None:
        return False

    megabytes = archived.size_bytes / 1e6
    free = free_bytes(FETCHED_DIR)
    if not room_to_fetch(archived, free_bytes=free):
        st.warning(
            f"Fetching this {megabytes:.0f} MB video would leave {(free - archived.size_bytes) / 1e9:.1f} GB "
            f"free, under the {MIN_FREE_BYTES / 1024**3:.0f} GB the archive sift needs to keep fetching."
        )
        if source.link:
            st.link_button("Open on Drive", source.link)
        return False

    return archived.size_bytes <= AUTO_FETCH_BYTES or st.button(
        "Fetch video",
        help=f"The sift did not keep this {megabytes:.0f} MB recording. Fetching it takes about "
        f"{fetch_seconds(archived) / 60.0:.0f} minutes, and no other clip is built meanwhile.",
    )


@st.fragment(run_every=POLL_SECONDS)
def _clip_progress(key: str, *, fetching: bool) -> None:
    """How far the clip has got, looked at again every few seconds until it is
    there, when the whole page is drawn again to play it."""
    state = _queue().state(key)
    if state.stage == "ready":
        st.rerun()
    elif state.stage == "failed":
        st.error(f"The clip could not be built: {state.error}")
    elif state.stage == "queued":
        st.caption("Stopping the previous clip")
    elif state.progress is None:
        st.caption("Fetching the video from the archive" if fetching else "Opening the recording")
    elif isinstance(state.progress, FetchProgress):
        st.progress(state.progress.fraction, text=_fetch_text(state.progress))
    else:
        st.progress(state.progress.fraction, text=_progress_text(state.progress))


def _fetch_text(progress: FetchProgress) -> str:
    """How much of the video has arrived, and how much there is."""
    total = f"{progress.bytes_total / 1e6:.1f} MB"
    if progress.bytes_done == 0:
        return f"Fetching the video: reaching the archive · {total}"
    return f"Fetching the video: {progress.bytes_done / 1e6:.1f} of {total}"


def _progress_text(progress: ClipProgress) -> str:
    """How far a build has got, in frames, and how many it handles in all."""
    stretch = progress.stretch
    in_all = f"{progress.frames_total:,} frames in all"
    if progress.drawing:
        drawn = progress.frames_done - stretch.begin_frame
        return f"Drawing the track: frame {drawn:,} of {stretch.drawn_frames:,} · {in_all}"
    return (
        f"Reaching the track: frame {progress.frames_done:,} of {stretch.begin_frame:,}, "
        f"then drawing {stretch.drawn_frames:,} · {in_all}"
    )


def _clip_job(source: VideoSource, *, track: StoredTrack) -> Job:
    """The work that turns a track's source into its clip, run away from the
    page."""

    def job(report: Report) -> Path:
        video = source.path or fetch_video(FETCHED_DIR, video=_archived(source), on_progress=report)
        clip = _clip_path(video, track=track)
        if not clip.is_file():
            details = probe(video)
            stretch = stretch_around(
                track, frames_per_second=details.frames_per_second, frame_count=details.frame_count
            )
            part = clip.with_name(f"{clip.stem}.part{clip.suffix}")
            try:
                build_track_clip(
                    video,
                    track=track,
                    stretch=stretch,
                    frames_per_second=details.frames_per_second,
                    output=part,
                    on_progress=report,
                )
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            part.replace(clip)
        return clip

    return job


def _archived(source: VideoSource) -> ArchiveVideo:
    if source.archived is None:
        raise FileNotFoundError("The recording is neither on disk nor named by a ledger.")
    return source.archived


def _clip_path(video: Path, *, track: StoredTrack) -> Path:
    details = probe(video)
    stretch = stretch_around(track, frames_per_second=details.frames_per_second, frame_count=details.frame_count)
    return track_clip_path(video, track=track, stretch=stretch, output_dir=CLIPS_DIR)


def _gallery(tracks: pd.DataFrame, *, cluster: str | None) -> None:
    """A random sample of the cluster's tracks, drawn anew on Shuffle."""
    if cluster is None:
        st.caption("Select a track, or choose a cluster for the gallery in the sidebar.")
        return

    members = tracks[tracks["cluster"] == cluster]
    sample = members.sample(n=min(GALLERY_SIZE, len(members)), random_state=st.session_state.get(SHUFFLE_KEY, 0))
    heading, shuffle = st.columns([4, 1], vertical_alignment="bottom")
    heading.subheader(_cluster_title(cluster), help=GALLERY_HELP)
    shuffle.button("Shuffle", on_click=_shuffle, help=SHUFFLE_HELP)
    st.caption(f"{len(sample)} of {len(members)} tracks")
    _panels(sample, captions=[f"{place}. {label}" for place, label in enumerate(sample["label"], start=1)])


def _shuffle() -> None:
    st.session_state[SHUFFLE_KEY] = st.session_state.get(SHUFFLE_KEY, 0) + 1


def _neighbours(tracks: pd.DataFrame, *, track: pd.Series, radius: float) -> None:
    """The tracks nearest the selected one on the map, in any cluster, each
    captioned with the cluster it is in."""
    st.subheader("Nearest tracks", help=NEIGHBOURS_HELP)
    others = tracks[tracks["key"] != track["key"]]
    distance = np.hypot(others["x"] - track["x"], others["y"] - track["y"])
    nearest = others.loc[distance[distance <= radius].nsmallest(GALLERY_SIZE).index]
    st.caption(f"{len(nearest)} tracks within {radius:.2f}")
    captions = [
        f"{place}. {_cluster_caption(cluster)} · {label}"
        for place, (cluster, label) in enumerate(zip(nearest["cluster"], nearest["label"]), start=1)
    ]
    _panels(nearest, captions=captions)


def _cluster_caption(cluster: str) -> str:
    return "no cluster" if cluster == UNASSIGNED_NAME else f"cluster {cluster}"


def _panels(chosen: pd.DataFrame, *, captions: list[str]) -> None:
    """The chosen tracks drawn from their stored paths, a panel each, in rows
    that wrap to the width of the column."""
    st.html(_gallery_markup(PATHS_PATH.stat().st_mtime, tuple(zip(chosen["key"], captions))))


@st.cache_data(show_spinner=False, max_entries=PANEL_CACHE_ENTRIES)
def _gallery_markup(stamp: float, captions: tuple[tuple[str, str], ...]) -> str:
    """The panels of these tracks under these captions, kept so that a gallery
    that comes out the same on the next click is not drawn again."""
    keyed = dict(captions)
    paths = _path_frame(stamp)
    return gallery_html(paths[paths["key"].isin(keyed)], captions=keyed)


def _gallery_cluster(chosen: str, *, picked: pd.Series | None) -> str | None:
    """The cluster the gallery draws, or None when it is to follow a selected
    track and none is selected."""
    if chosen != SELECTED_CLUSTER:
        return chosen
    return None if picked is None else str(picked["cluster"])


def _cluster_title(cluster: str) -> str:
    if cluster == SELECTED_CLUSTER:
        return SELECTED_CLUSTER
    if cluster == UNASSIGNED_NAME:
        return "Unassigned tracks"
    return f"Cluster {cluster}"


def _facts(track: pd.Series) -> str:
    parts = [
        f"Label {track['label']}",
        _cluster_caption(str(track["cluster"])),
        f"{track['side']} side",
        f"camera {track['camera']}",
        f"straightness {track['straightness']:.2f}",
        f"roughness {track['roughness']:.2f}",
        f"peak deviation {track['peak_deviation_max']:.1f}",
    ]
    return " · ".join(parts)


def _points(paths: pd.DataFrame, *, key: str) -> pd.DataFrame:
    return paths.loc[paths["key"] == key].sort_values("frame_number")


def _stored_track(points: pd.DataFrame) -> StoredTrack:
    return StoredTrack(
        track_id=int(points["track_id"].iloc[0]),
        frame_numbers=points["frame_number"].to_numpy(),
        x=points["x"].to_numpy(),
        y=points["y"].to_numpy(),
        frame_height=int(points["frame_height"].iloc[0]),
    )


def _cluster_names(tracks: pd.DataFrame) -> list[str]:
    named = sorted((name for name in tracks["cluster"].unique() if name != UNASSIGNED_NAME), key=int)
    return [*named, UNASSIGNED_NAME] if (tracks["cluster"] == UNASSIGNED_NAME).any() else named


@st.cache_resource
def _queue() -> ClipQueue:
    """The one queue every session of the dashboard hands its clips to."""
    return ClipQueue()


@st.cache_data(show_spinner=False)
def _map_frame(stamp: float) -> pd.DataFrame:
    """The map as a frame the chart reads, read again whenever the file
    changes.

    The stamp is the file's modification time. It is what the cache is
    keyed on, which is why it is passed although the body never reads
    it.
    """
    frame = pq.read_table(MAP_PATH).to_pandas()
    frame["cluster"] = [UNASSIGNED_NAME if value < 0 else str(value) for value in frame["cluster"]]
    frame["key"] = _keys(frame)
    return frame


@st.cache_data(show_spinner=False)
def _path_frame(stamp: float) -> pd.DataFrame:
    """Every mapped track frame by frame, keyed the way the map is, read again
    whenever the file changes."""
    frame = pq.read_table(PATHS_PATH).to_pandas()
    frame["key"] = _keys(frame)
    return frame


def _keys(frame: pd.DataFrame) -> pd.Series:
    return frame["event"] + "/" + frame["clip"] + "/" + frame["track_id"].astype(str)


def _ledger_stamp() -> tuple[float, ...]:
    return tuple(path.stat().st_mtime if path.is_file() else 0.0 for path in LEDGERS)


@st.cache_data(show_spinner=False)
def _ledger_entries(stamp: tuple[float, ...]) -> dict[str, dict[str, Any]]:
    """What the sift ledgers record about every video they name, by the video's
    file name.

    A ledger another run is still appending to can end on a line that is
    only half written, and such a line is passed over. The stamp holds
    the ledgers' modification times, and is what the cache is keyed on.
    """
    entries: dict[str, dict[str, Any]] = {}
    for path in LEDGERS:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries[str(entry.get("name"))] = entry
    return entries
