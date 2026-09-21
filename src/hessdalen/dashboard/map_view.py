"""The track map, where clicking a track plays it where it was found.

Every track of the corpus is a point, placed and coloured by the map the
analysis step writes. The page reads that file and the track files and
never detects anything, so a click costs a pass over the frames ahead of
the track and a drawing of its stretch. A track whose video the sift did
not keep has its video fetched from the archive first.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from hessdalen.dashboard.runs import VideoProbe, probe
from hessdalen.dashboard.track_clip import StoredTrack, Stretch, build_track_clip, stretch_around, track_clip_path
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
TRACKS_DIR = REPO_ROOT / "data" / "corpus" / "tracks"
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
COLOUR_CHOICES = ("Cluster", "Label", "Camera")
COLOUR_COLUMNS = {"Cluster": "cluster", "Label": "label", "Camera": "camera"}
SELECTION = "track"
FALLBACK_RATE = 25.0

COLOUR_HELP = (
    "What the points are coloured by. Clusters come from the descriptors alone. A label is the folder "
    "the recording was filed under, which names the whole recording, so most tracks under a label are "
    "that scene's background activity and not the thing the folder is named for."
)
LABELS_HELP = "Show only the tracks of recordings filed under these labels."
MAP_HELP = (
    "Every track the corpus holds, placed so that tracks with similar descriptors sit close together. "
    "Grey points are tracks the clustering left out of every cluster. Click a point to play its track."
)
AUTO_FETCH_BYTES = 100 * 1024**2
"""Largest video a click fetches without being asked.

The one-minute cuts are about 30 MB and arrive in under a minute. A
whole 20-minute recording takes several minutes, and the page can do
nothing else meanwhile, so that fetch waits for a press.
"""

UNSOURCED_TEXT = (
    "The video behind this track was not kept after detection, and no ledger says where in the archive "
    "it came from, so it cannot be fetched."
)


def page() -> None:
    st.title("Track map")
    if not MAP_PATH.is_file():
        st.error(f"No track map at {MAP_PATH}. Write it with `{MAP_COMMAND}`.")
        return

    tracks = _map_frame(MAP_PATH.stat().st_mtime)
    with st.sidebar:
        colour = st.segmented_control("Colour", options=COLOUR_CHOICES, default="Cluster", help=COLOUR_HELP)
        labels = st.multiselect("Labels", options=sorted(tracks["label"].unique()), help=LABELS_HELP)

    shown = tracks[tracks["label"].isin(labels)] if labels else tracks
    st.subheader("Tracks", help=MAP_HELP)
    st.caption(f"{len(shown)} of {len(tracks)} tracks")
    event = st.altair_chart(
        _scatter(shown, colour_column=COLOUR_COLUMNS[colour or "Cluster"]),
        on_select="rerun",
        key="track_map",
        width="stretch",
    )

    picked = _picked_key(event)
    if picked is None or picked not in set(shown["key"]):
        st.caption("No track selected.")
        return
    _play(shown.loc[shown["key"] == picked].iloc[0])


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

    named = sorted((name for name in tracks["cluster"].unique() if name != UNASSIGNED_NAME), key=int)
    colours = [CLUSTER_COLORS[index % len(CLUSTER_COLORS)] for index in range(len(named))]
    return alt.Color(
        "cluster:N",
        title="Cluster",
        scale=alt.Scale(domain=[*named, UNASSIGNED_NAME], range=[*colours, UNASSIGNED_COLOR]),
    )


def _picked_key(event: Any) -> str | None:
    selected = event.selection.get(SELECTION, []) if isinstance(event.selection, Mapping) else []
    if not selected:
        return None
    return str(selected[0].get("key"))


def _play(track: pd.Series) -> None:
    stored = _stored_track(
        TRACKS_DIR / str(track["label"]) / str(track["event"]) / f"{track['clip']}.parquet",
        track_id=int(track["track_id"]),
    )

    st.subheader(f"Track {int(track['track_id'])} in {track['clip']}")
    st.caption(_facts(track))
    video = _video(str(track["recording"]))
    if video is None:
        return

    details = _probe_cached(video)
    stretch = stretch_around(stored, frames_per_second=details.frames_per_second, frame_count=details.frame_count)
    clip = track_clip_path(video, track=stored, stretch=stretch, output_dir=CLIPS_DIR)
    if not clip.is_file():
        with st.spinner("Drawing the track"):
            _build(video, track=stored, stretch=stretch, frames_per_second=details.frames_per_second, clip=clip)

    rate = details.frames_per_second or FALLBACK_RATE
    st.caption(
        f"Frames {stored.first_frame} to {stored.last_frame}, "
        f"{stored.first_frame / rate:.1f} to {stored.last_frame / rate:.1f} s into the recording"
    )
    st.video(str(clip), loop=True, autoplay=True, muted=True)


def _video(recording: str) -> Path | None:
    """The recording on disk, fetched from the archive when the sift did not
    keep it, or None when it cannot be had now."""
    kept = VIDEOS_DIR / recording
    if kept.is_file():
        return kept

    entry = _ledger_entries(_ledger_stamp()).get(recording)
    if entry is None:
        st.info(UNSOURCED_TEXT)
        return None

    archived = archive_video(entry)
    return cached_video(FETCHED_DIR, video=archived) or _fetch(archived, link=str(entry.get("url", "")))


def _fetch(video: ArchiveVideo, *, link: str) -> Path | None:
    """Fetch the video when there is room for it and, for a large one, when
    asked to."""
    megabytes = video.size_bytes / 1e6
    free = free_bytes(FETCHED_DIR)
    if not room_to_fetch(video, free_bytes=free):
        st.warning(
            f"Fetching this {megabytes:.0f} MB video would leave {(free - video.size_bytes) / 1e9:.1f} GB free, "
            f"under the {MIN_FREE_BYTES / 1024**3:.0f} GB the archive sift needs to keep fetching."
        )
        if link:
            st.link_button("Open on Drive", link)
        return None

    minutes = fetch_seconds(video) / 60.0
    asked = video.size_bytes <= AUTO_FETCH_BYTES or st.button(
        "Fetch video",
        help=f"The sift did not keep this {megabytes:.0f} MB recording. Fetching it takes about "
        f"{minutes:.0f} minutes, and the page waits until it has arrived.",
    )
    if not asked:
        return None

    try:
        with st.spinner(f"Fetching {megabytes:.0f} MB from the archive"):
            return fetch_video(FETCHED_DIR, video=video)
    except (OSError, subprocess.CalledProcessError) as failure:
        st.error(f"The fetch failed: {failure}")
        return None


def _build(video: Path, *, track: StoredTrack, stretch: Stretch, frames_per_second: float, clip: Path) -> None:
    """Build into a file of its own and move it into place once whole.

    A click on another point stops the page mid-build, and the part left
    behind would otherwise be found next time and played as the clip.
    """
    part = clip.with_name(f"{clip.stem}.part{clip.suffix}")
    build_track_clip(video, track=track, stretch=stretch, frames_per_second=frames_per_second, output=part)
    part.replace(clip)


def _facts(track: pd.Series) -> str:
    cluster = track["cluster"]
    parts = [
        f"Label {track['label']}",
        "no cluster" if cluster == UNASSIGNED_NAME else f"cluster {cluster}",
        f"camera {track['camera']}",
        f"straightness {track['straightness']:.2f}",
        f"peak deviation {track['peak_deviation_max']:.1f}",
    ]
    return " · ".join(parts)


def _stored_track(path: Path, *, track_id: int) -> StoredTrack:
    table = pq.read_table(path)
    frame_height = int((table.schema.metadata or {})[b"hessdalen_frame_height"])
    ids = np.asarray(table.column("track_id").to_numpy())
    held = np.flatnonzero(ids == track_id)
    order = np.argsort(np.asarray(table.column("frame_number").to_numpy())[held])
    rows = held[order]
    return StoredTrack(
        track_id=track_id,
        frame_numbers=np.asarray(table.column("frame_number").to_numpy())[rows],
        x=np.asarray(table.column("x").to_numpy())[rows],
        y=np.asarray(table.column("y").to_numpy())[rows],
        frame_height=frame_height,
    )


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
    frame["key"] = frame["event"] + "/" + frame["clip"] + "/" + frame["track_id"].astype(str)
    return frame


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


@st.cache_data(show_spinner=False)
def _probe_cached(video: Path) -> VideoProbe:
    return probe(video)
