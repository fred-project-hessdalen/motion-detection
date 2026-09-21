"""The track map, where clicking a track shows it at once and then plays it
where it was found.

Every track of the corpus is a point, placed and coloured by the map the
analysis step writes. Clicking one draws its stored path straight away,
and hands drawing it on its video to a background queue, because that
has to pass every frame of the recording ahead of the track and can take
a minute. A gallery below draws a sample of any one cluster from the
stored paths alone, which is how a cluster is judged at a glance.

The page never detects anything. A track whose video the sift did not
keep has its video fetched from the archive before it is drawn.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from hessdalen.dashboard.clip_queue import ClipQueue
from hessdalen.dashboard.runs import probe
from hessdalen.dashboard.track_clip import StoredTrack, build_track_clip, stretch_around, track_clip_path
from hessdalen.dashboard.track_preview import close_up, frame_view, gallery, light_curve
from hessdalen.dashboard.video_cache import (
    MIN_FREE_BYTES,
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
SELECTION = "track"
GALLERY_SIZE = 30
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
    "Grey points are tracks the clustering left out of every cluster. Click a point to see its track."
)
PATH_HELP = (
    "The track drawn from its stored path through the blob's centre, which is what the descriptors "
    "were computed from. Colour runs from dark blue on its first frame to yellow on its last."
)
VIDEO_HELP = (
    "The track drawn on its recording, from a second before it starts to a second after it ends. "
    "It is built in the background, one track at a time, and choosing another track drops a build "
    "still waiting."
)
GALLERY_HELP = (
    f"Up to {GALLERY_SIZE} tracks of the cluster, spread across it, each drawn from its stored path and "
    "fitted to its own panel. Colour runs from dark blue on a track's first frame to yellow on its last."
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

    shown = tracks[tracks["label"].isin(labels)] if labels else tracks
    map_column, track_column = st.columns([3, 2])
    with map_column:
        st.subheader("Tracks", help=MAP_HELP)
        st.caption(f"{len(shown)} of {len(tracks)} tracks")
        event = st.altair_chart(
            _scatter(shown, colour_column=COLOUR_COLUMNS[colour or "Cluster"]),
            on_select="rerun",
            key="track_map",
            width="stretch",
        )
        picked = _picked(shown, event)
        _gallery(shown, paths, cluster=_gallery_cluster(str(chosen_cluster), picked=picked))

    with track_column:
        if picked is None:
            st.caption("No track selected.")
        else:
            _selected(picked, points=_points(paths, key=str(picked["key"])))


def _scatter(tracks: pd.DataFrame, *, colour_column: str) -> alt.Chart:
    selection = alt.selection_point(name=SELECTION, fields=["key"], on="click")
    return (
        alt.Chart(tracks)
        .mark_circle(size=45)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=_colour(tracks, column=colour_column),
            opacity=alt.condition(selection, alt.value(0.95), alt.value(0.4)),
            tooltip=[
                alt.Tooltip("label:N", title="Label"),
                alt.Tooltip("cluster:N", title="Cluster"),
                alt.Tooltip("side:N", title="Side"),
                alt.Tooltip("clip:N", title="Recording"),
                alt.Tooltip("track_id:Q", title="Track"),
                alt.Tooltip("frames:Q", title="Frames"),
                alt.Tooltip("straightness:Q", title="Straightness", format=".2f"),
                alt.Tooltip("peak_deviation_max:Q", title="Peak deviation", format=".1f"),
            ],
        )
        .add_params(selection)
        .properties(height=620)
    )


def _colour(tracks: pd.DataFrame, *, column: str) -> alt.Color:
    """Colour by the chosen column, with tracks outside every cluster in
    grey."""
    if column != "cluster":
        return alt.Color(f"{column}:N", title=column.capitalize(), scale=alt.Scale(scheme="category20"))

    named = [name for name in _cluster_names(tracks) if name != UNASSIGNED_NAME]
    colours = [CLUSTER_COLORS[index % len(CLUSTER_COLORS)] for index in range(len(named))]
    return alt.Color(
        "cluster:N",
        title="Cluster",
        scale=alt.Scale(domain=[*named, UNASSIGNED_NAME], range=[*colours, UNASSIGNED_COLOR]),
    )


def _picked(shown: pd.DataFrame, event: Any) -> pd.Series | None:
    selected = event.selection.get(SELECTION, []) if isinstance(event.selection, Mapping) else []
    if not selected:
        return None
    held = shown.loc[shown["key"] == str(selected[0].get("key"))]
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
        st.caption("Waiting for the clip before it to finish")
    else:
        st.caption("Fetching the video, then drawing the track on it" if fetching else "Drawing the track on its video")


def _clip_job(source: VideoSource, *, track: StoredTrack) -> Callable[[], Path]:
    """The work that turns a track's source into its clip, run away from the
    page."""

    def job() -> Path:
        video = source.path or fetch_video(FETCHED_DIR, video=_archived(source))
        clip = _clip_path(video, track=track)
        if not clip.is_file():
            details = probe(video)
            stretch = stretch_around(
                track, frames_per_second=details.frames_per_second, frame_count=details.frame_count
            )
            part = clip.with_name(f"{clip.stem}.part{clip.suffix}")
            build_track_clip(
                video, track=track, stretch=stretch, frames_per_second=details.frames_per_second, output=part
            )
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


def _gallery(tracks: pd.DataFrame, paths: pd.DataFrame, *, cluster: str | None) -> None:
    if cluster is None:
        st.caption("Select a track, or choose a cluster for the gallery in the sidebar.")
        return

    members = tracks[tracks["cluster"] == cluster].sort_values("key")
    sample = members.iloc[np.unique(np.linspace(0, len(members) - 1, min(GALLERY_SIZE, len(members))).astype(int))]
    st.subheader(_cluster_title(cluster), help=GALLERY_HELP)
    st.caption(f"{len(sample)} of {len(members)} tracks")
    if sample.empty:
        return

    places = {key: place for place, key in enumerate(sample["key"])}
    drawn = paths[paths["key"].isin(places)].copy()
    drawn["place"] = drawn["key"].map(places)
    drawn["panel"] = [f"{place + 1}. {label}" for place, label in zip(drawn["place"], drawn["label"])]
    st.altair_chart(gallery(drawn.sort_values(["place", "frame_number"])))


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
    cluster = track["cluster"]
    parts = [
        f"Label {track['label']}",
        "no cluster" if cluster == UNASSIGNED_NAME else f"cluster {cluster}",
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
