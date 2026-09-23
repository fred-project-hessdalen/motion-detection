"""The track map, where clicking a track shows it at once and then plays it
where it was found.

Every track of the corpus is a point, placed and coloured by the map the
analysis step writes. Clicking one draws its stored path straight away,
and hands drawing it on its video to a background queue, because that
has to pass every frame of the recording ahead of the track and can take
a minute. A gallery below draws a random sample of any one cluster from
the stored paths alone, which is how a cluster is judged at a glance,
and a second one draws the tracks nearest the selected one, from
whatever cluster they are in. A panel of either gallery plays its track
beside the selected one when it is clicked, and the arrow on a panel
selects its track. The steps over the selected track go back through the
tracks selected before it.

The page never detects anything. A track whose video the sift did not
keep has its video fetched from the archive before it is drawn.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from plotly.colors import hex_to_rgb, qualitative
from streamlit.delta_generator import DeltaGenerator

from hessdalen.config import config
from hessdalen.dashboard.clip_queue import ClipQueue, Job, Report
from hessdalen.dashboard.cluster_labels import canonical_label, label_of, near_label, read_labels, write_labels
from hessdalen.dashboard.panels import DEVIATION, RECORDING
from hessdalen.dashboard.runs import probe
from hessdalen.dashboard.track_clip import (
    ClipProgress,
    StoredTrack,
    TrackClips,
    build_track_clip,
    stretch_around,
    track_clip_paths,
)
from hessdalen.dashboard.track_gallery import track_gallery
from hessdalen.dashboard.track_history import History, cleared, stepped, visited
from hessdalen.dashboard.track_map_chart import MAP_HEIGHT, track_map_chart
from hessdalen.dashboard.track_preview import GalleryEntry, close_up, frame_view, gallery_html, light_curve
from hessdalen.dashboard.track_validation import read_validated, write_validated
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
LABELS_PATH = MAP_PATH.with_name("cluster-labels.json")
TRACK_LABELS_PATH = MAP_PATH.with_name("track-labels.json")
VALIDATED_PATH = MAP_PATH.with_name("validated-tracks.json")
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
NAME_COLUMN = "cluster_label"
"""Column holding the name given to the cluster each track is in."""

TRACK_NAME_COLUMN = "track_label"
"""Column holding the name a track itself stands under, which is the name of
its cluster until the track is given one of its own."""

CACHED_COLUMN = "cached"
"""Column saying whether the track's recording is on disk."""

VALIDATED_COLUMN = "validated"
"""Column saying whether someone has confirmed the track."""

UNNAMED = "unlabelled"
"""What a track whose cluster has no name yet is shown under."""

COLOUR_CHOICES = ("Cluster", "Cluster label", "Track label", "Side", "Folder", "Camera")
COLOUR_COLUMNS = {
    "Cluster": "cluster",
    "Cluster label": NAME_COLUMN,
    "Track label": TRACK_NAME_COLUMN,
    "Side": "side",
    "Folder": "label",
    "Camera": "camera",
}
VALIDATION_CHOICES = ("All", "Validated", "Unvalidated")
ANY_VALIDATION, VALIDATED, UNVALIDATED = VALIDATION_CHOICES
NAMING_CHOICES = ("Cluster", "Neighbours")
CLUSTER_TRACKS, NEIGHBOUR_TRACKS = NAMING_CHOICES
SELECTED_CLUSTER = "Selected track's cluster"
ROUGH_SIDE = "rough"
OTHER_COLORS = qualitative.Dark24
HOVER_COLUMNS = [
    "key",
    TRACK_NAME_COLUMN,
    NAME_COLUMN,
    "cluster",
    "side",
    "clip",
    "track_id",
    "frames",
    "straightness",
    "peak_deviation_max",
    "label",
]
HOVER_TEMPLATE = (
    "Track label %{customdata[1]}<br>Cluster label %{customdata[2]}<br>Cluster %{customdata[3]}<br>"
    "Side %{customdata[4]}<br>Recording %{customdata[5]}<br>Track %{customdata[6]}<br>"
    "Frames %{customdata[7]}<br>Straightness %{customdata[8]:.2f}<br>"
    "Peak deviation %{customdata[9]:.1f}<br>Folder %{customdata[10]}<extra></extra>"
)
GALLERY_SIZE = 30
NEIGHBOUR_RADIUS = 0.5
SHUFFLE_KEY = "gallery_shuffle"
MAP_KEY = "track_map"
HISTORY_KEY = "track_history"
PLAYING_KEY = "playing_track"
SAMPLE_GALLERY_KEY = "sample_gallery"
NEAREST_GALLERY_KEY = "nearest_gallery"
SEARCH_KEY = "track_search"
LABEL_KEY = "cluster_label"
SAVE_KEY = "save_cluster_label"
NAMING_KEY = "naming_choice"
TRACK_LABEL_KEY = "track_label"
SAVE_TRACK_LABEL_KEY = "save_track_label"
REASSIGN_KEY = "reassign_track"
SAVE_REASSIGN_KEY = "save_reassign"
VALIDATED_KEY = "track_validated"
SIMILAR_KEY = "similar_name"
REPLACING_KEY = "replacing_name"
PANEL_KEY = "clip_panel"
FETCH_KEY = "fetch_video"
VIDEO_PLAYER = "video"
COMPARISON_PLAYER = "comparison"
SEARCH_WORDS = ("track", "in")
"""Words a search may carry around what it names, from the heading over a
selected track."""
SAMPLE_TITLE = "Cluster sample"
NEAREST_TITLE = "Nearest tracks"
SELECTED_TITLE = "Selected track"
SIMILAR_TITLE = "Similar name"
REPLACING_TITLE = "Change of name"
NAME_PLACEHOLDER = "Choose or add a name"
NAME_ROW = (2, 1, 2, 3)
TRACK_NAME_ROW = (3, 1, 1, 2)
REASSIGN_ROW = (3, 1, 3)
"""How the row a name is given in is divided, the last of them left empty.

A name is a word or two, so its box takes a small part of the width it
stands in and the rest is left alone. The whole width would put the
button that keeps the name at the far edge of the page.
"""
NAMES_TITLE = "Cluster labels"
WHOLE_TITLE = "In the frame"
CLOSE_UP_TITLE = "Close up"
CLIP_PANELS = (RECORDING, DEVIATION)
"""The panels a track's videos are built for, in the order they are offered."""
SAMPLE_COLOR = "#e6007e"
NEAREST_COLOR = "#0091d5"
SAMPLE_MARKER = {"color": SAMPLE_COLOR, "symbol": "circle-open", "size": 12, "line": {"width": 2}}
NEAREST_MARKER = {"color": NEAREST_COLOR, "symbol": "diamond-open", "size": 12, "line": {"width": 2}}
SELECTED_MARKER = {"color": "#ffd400", "symbol": "star", "size": 18, "line": {"width": 1, "color": "#333333"}}
NAMES_FONT = {"size": 13}
"""Size the name of a cluster is written at.

The colour is left to the page's own text colour, which the plot is
handed, so a name stays readable whichever theme the page is in.
"""
GRID_COLOR = "rgba(128, 128, 128, 0.25)"
DIM_ALPHA = 0.2
PANEL_CACHE_ENTRIES = 64
POLL_SECONDS = 2.0

GIGABYTE = 1024**3
"""What a gigabyte is taken to be wherever the page writes one.

The room left on the disk is held against the room the archive sift
needs, so the two are counted the same way or the sentence that puts
them side by side says one is under the other while the numbers say it
is not.
"""

AUTO_FETCH_BYTES = 100 * 1024**2
"""Largest video a click fetches without being asked.

The one-minute cuts are about 30 MB and arrive in under a minute. A
whole 20-minute recording takes several minutes, and holds up every
other clip meanwhile, so that fetch waits for a press.
"""

COLOUR_HELP = (
    "What the points are coloured by. Clusters come from the descriptors alone, and a cluster label is "
    "the name someone gave one of them. A track label is the name one track stands under, which is its "
    "cluster's name until the track is given a name of its own. The side says whether a track moves "
    "evenly from step to step, as a clean path does, or hops about as clutter does, and each side is "
    "clustered on its own. A folder is where the recording was filed in the archive, which names the "
    "whole recording, so most tracks under a folder are that scene's background activity and not the "
    "thing the folder is named for."
)
SEARCH_HELP = (
    "Select a track by its number, by its recording, or by both, as the heading over a selected track "
    'gives them. "Track 7484 in Cam1_2025-06-03__12-40-00_noInsect" and "7484" both work, and so does a '
    "recording's name on its own. The first of the matching tracks is selected."
)
NAMES_HELP = (
    "Show only the tracks whose cluster has been given one of these names. Tracks of a cluster with no "
    f"name yet, and tracks the clustering left out of every cluster, are under {UNNAMED}."
)
FOLDERS_HELP = "Show only the tracks of recordings filed under these folders of the archive."
ROUGH_HELP = (
    "Show the tracks that hop about from step to step rather than moving evenly, which is what clutter "
    "does. Turn it off to leave the map and the galleries to the clean paths."
)
CACHED_HELP = (
    "Draw the tracks whose recording is not on disk faintly. Such a track has its recording fetched from "
    "the archive before it can be drawn on it, which takes a minute or more."
)
GALLERY_CHOICE_HELP = "The cluster the gallery below the map draws a sample of."
MAP_HELP = (
    "Every track the corpus holds, placed so that tracks with similar descriptors sit close together. "
    "Grey points are tracks the clustering left out of every cluster. Click a point to see its track. "
    "Scroll to zoom, drag to pan, and double-click to zoom back out. Click an entry in the legend to hide "
    "or show its tracks. A click away from every track takes the selection off, and so does Escape, "
    "which leaves the steps to go back through the tracks selected so far. A star marks the selected "
    "track, and rings mark the tracks the galleries below the map show. The name a cluster has been "
    "given stands over the middle of its points, and the "
    f"{NAMES_TITLE} entry in the legend takes every name off the map."
)
PATH_HELP = (
    "The track drawn from its stored path through the blob's centre, which is what the descriptors "
    "were computed from. Colour runs from dark blue on its first frame to yellow on its last."
)
CLIP_HELP = (
    "In the frame shows the whole picture with the track's path and a box drawn on it. Close up shows a "
    "crop that keeps the detection in the middle and nothing drawn over it, so the object itself can be "
    "seen. That crop is placed on an average of the detections around each frame rather than on the "
    "detection itself, because the pixel a track is matched on hops about inside its blob and would "
    "shake the picture. An object holding its course is still followed with no lag. All of them are "
    "built in the background, one track at a time."
)
VIDEO_HELP = (
    f"The selected track on its recording, from a second before it starts to a second after it ends. {CLIP_HELP}"
)
COMPARISON_HELP = (
    "The track whose gallery panel was clicked, on its recording, to hold against the selected track "
    f"above. Clicking another panel brings that track here. {CLIP_HELP} One at a time means this video "
    "waits while the one above is being built."
)
VIDEO_TITLES = {
    VIDEO_PLAYER: ("Video", VIDEO_HELP),
    COMPARISON_PLAYER: ("Comparison", COMPARISON_HELP),
}
"""What stands over each of the two players, and what its tooltip says."""

WAITING_TEXT = "Waiting for the video above to be built."
PANEL_HELP = (
    "Recording shows the picture as the camera recorded it. Deviation shows how far each pixel stands "
    "from the background the detector measures it against, with the detection threshold at full white. "
    "The deviation is measured over this stretch alone, so the background model opens on the stretch's "
    "first frame and the frames before the track starts are what it has to settle in. Both panels are "
    "built together, so switching between them plays at once."
)
MARKS_HELP = (
    "Click a panel to play its track's video, which leaves the selected track where it is, and press the "
    "arrow on a panel to jump to its track, which selects it the way a click on its point does. The "
    "panel whose video is "
    "playing carries a play mark and is framed in the colour of that mark. A camera stands over a track "
    "whose recording is on disk, which is a track that can be played without waiting for a fetch, and a "
    "green tick over a track someone has confirmed."
)
GALLERY_HELP = (
    f"Up to {GALLERY_SIZE} tracks drawn at random from the cluster, each drawn from its stored path and "
    "fitted to its own panel. Colour runs from dark blue on a track's first frame to yellow on its last. "
    f"Each panel names the folder of the archive its recording was filed in. {MARKS_HELP} The map rings "
    "these tracks in the colour their panels are framed in."
)
SHUFFLE_HELP = "Draw another random sample of the cluster."
LABEL_HELP = (
    "What this cluster holds, in a word of your own, such as insect or plane. The names given so far are "
    "offered in the list, which narrows to what is typed into it, and anything else typed there is a new "
    f"name. The name is kept against the cluster's tracks in {LABELS_PATH.name}, which is what a training "
    "set is built from. It is kept in small letters with single spaces between its words and every word "
    'in the singular, so that "Street Lights" and "streetLight" come to the one name. A name within a '
    "letter or two of one already in use is put to you before it is written. Naming the cluster again "
    "moves its tracks to the new name, and clearing the box takes them out of the one they are under. A "
    "track reassigned to another cluster on its own stays where it was put."
)
SAVE_LABEL_HELP = "Keep this name against every track the name is being given to."
NAMING_HELP = (
    "Which tracks the name is given to. Cluster gives it to every track of the cluster the gallery above "
    "draws, wherever those tracks lie on the map. Neighbours gives it to the tracks nearest the selected "
    "one, which the Nearest tracks gallery draws, whichever cluster each of them is in, and is how a name "
    "is given to a neighbourhood the clustering cut in two. Naming a cluster leaves alone the tracks put "
    "under a name of their own, and naming the nearest tracks takes all of them."
)
SIMILAR_TEXT = (
    "A name holds the tracks given it and nothing more, so two spellings of one thing keep their tracks "
    "apart. Both names are kept if that is what you meant."
)
REPLACING_TEXT = "Changing the name takes every one of those tracks out of the name they are under."
TRACK_LABEL_HELP = (
    "What this one track holds, in a word of your own. It starts as the name the track's cluster is "
    "under, and is changed where the track is not what the rest of its cluster is. The name is kept "
    f"against the track alone in {TRACK_LABELS_PATH.name}, in the same form as a cluster label, and it is "
    "what a training set reads for this track. The names given so far are offered in the list, and "
    "clearing the box takes the track back to its cluster's name."
)
SAVE_TRACK_LABEL_HELP = (
    "Keep this name against this track alone. Naming a track is someone looking at it and saying what it "
    "is, so the track is marked validated at the same time."
)
REASSIGN_HELP = (
    "The cluster this track is counted with, by the name that cluster is under. Giving it another name "
    f"puts this one track under that name in {LABELS_PATH.name}, and a name no cluster is under yet "
    "opens one holding this track. The cluster it came from keeps the name the rest of its tracks hold. "
    "The clustering itself is left as it is, because it is run afresh over the whole corpus and would "
    "undo a change made here. Track label names what the track holds, and this names the cluster it "
    "belongs with."
)
SAVE_REASSIGN_HELP = (
    "Put this track under the cluster named in the box. It is marked validated at the same time, the way "
    "a track label is."
)
VALIDATED_HELP = (
    "Mark that you have watched this track and stand by the name it is under. Saving a track label marks "
    f"it as well. The tracks marked so far are kept in {VALIDATED_PATH.name}, a gallery draws a green "
    "tick over each of them, and the map can be held to them or to the tracks still to go through."
)
VALIDATION_HELP = (
    "Show the tracks someone has confirmed, the tracks still to go through, or every track whichever it is."
)
BACK_HELP = "Go back to the track selected before this one."
FORWARD_HELP = "Go forward to the track selected after this one."
NEIGHBOURS_HELP = (
    f"The {GALLERY_SIZE} tracks that lie nearest the selected track on the map, from any cluster, and no "
    "further from it than the neighbour radius. Each panel names the cluster its track is in and the name "
    f"that cluster is under. {MARKS_HELP} The map rings these tracks in the colour their panels are "
    "framed in."
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
class Ring:
    """Tracks marked on the map over their points: the selected track, and the
    tracks a gallery shows, in the colour the gallery frames its panels
    in."""

    name: str
    marker: dict[str, Any]
    tracks: pd.DataFrame


@dataclass(frozen=True, slots=True)
class Naming:
    """What a press of Save is to write: the file the name is kept in, the
    tracks it is given to, the box it was given in, and whether keeping it
    confirms those tracks as well."""

    path: Path
    keys: tuple[str, ...]
    box: str
    confirms: bool


@dataclass(frozen=True, slots=True)
class Similar:
    """A name that was given and lies within a letter or two of one already in
    use, held until the person says which of the two they meant."""

    naming: Naming
    name: str
    near: str


@dataclass(frozen=True, slots=True)
class Replacing:
    """A name that was given to tracks already standing under another, held
    until the person says whether to move them."""

    naming: Naming
    name: str
    held: str


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

    tracks = _tracks_frame()
    paths = _path_frame(PATHS_PATH.stat().st_mtime)
    with st.sidebar:
        wanted = str(st.text_input("Search", key=SEARCH_KEY, on_change=_search, help=SEARCH_HELP))
        if wanted.strip():
            st.caption(f"{len(searched(tracks, wanted=wanted))} of {len(tracks)} tracks matched")
        colour = st.segmented_control("Colour", options=COLOUR_CHOICES, default="Cluster", help=COLOUR_HELP)
        chosen_names = st.multiselect("Cluster labels", options=sorted(tracks[NAME_COLUMN].unique()), help=NAMES_HELP)
        folders = st.multiselect("Folders", options=sorted(tracks["label"].unique()), help=FOLDERS_HELP)
        validation = st.segmented_control(
            "Validation", options=VALIDATION_CHOICES, default=ANY_VALIDATION, help=VALIDATION_HELP
        )
        rough = st.toggle("Rough tracks", value=False, help=ROUGH_HELP)
        cached = st.toggle("Video cached", value=True, help=CACHED_HELP)
        chosen_cluster = st.selectbox(
            "Gallery",
            options=[SELECTED_CLUSTER, *_cluster_names(tracks)],
            format_func=_cluster_title,
            help=GALLERY_CHOICE_HELP,
        )
        radius = st.number_input(
            "Neighbour radius", min_value=0.05, max_value=5.0, value=NEIGHBOUR_RADIUS, step=0.05, help=RADIUS_HELP
        )

    shown = tracks if rough else tracks[tracks["side"] != ROUGH_SIDE]
    shown = shown[shown[NAME_COLUMN].isin(chosen_names)] if chosen_names else shown
    shown = shown[shown["label"].isin(folders)] if folders else shown
    shown = by_validation(shown, choice=str(validation or ANY_VALIDATION))
    picked = _picked(shown, tracks=tracks, key=_history().standing)
    playing = _playing(tracks, picked=picked)
    cluster = _gallery_cluster(str(chosen_cluster), picked=picked)
    sample = shown.iloc[0:0] if cluster is None else _sample(shown, cluster=cluster)
    nearest = shown.iloc[0:0] if picked is None else _nearest(shown, track=picked, radius=float(radius))
    chosen = shown.iloc[0:0] if picked is None else shown[shown["key"] == picked["key"]]
    rings = [
        Ring(name=SAMPLE_TITLE, marker=SAMPLE_MARKER, tracks=sample),
        Ring(name=NEAREST_TITLE, marker=NEAREST_MARKER, tracks=nearest),
        Ring(name=SELECTED_TITLE, marker=SELECTED_MARKER, tracks=chosen),
    ]

    map_column, track_column = st.columns([3, 2])
    with map_column:
        st.subheader("Tracks", help=MAP_HELP)
        st.caption(f"{len(shown)} of {len(tracks)} tracks")
        dimmed = _uncached(shown) if cached else frozenset()
        figure = _scatter(
            shown,
            colour=str(colour or "Cluster"),
            rings=rings,
            dimmed=dimmed,
            names=cluster_names(shown, labels=_cluster_labels(_labels_stamp())),
        )
        track_map_chart(figure, key=MAP_KEY, on_click=_map_clicked, on_clear=_map_cleared)
        _gallery(shown, cluster=cluster, sample=sample, playing=playing)
        if cluster is not None:
            _labelling(
                tracks,
                cluster=cluster,
                nearest=nearest,
                selected="" if picked is None else str(picked["key"]),
            )
        if picked is not None:
            _neighbours(nearest, radius=float(radius), playing=playing)

    with track_column:
        _steps()
        if picked is None:
            st.caption("No track selected.")
        else:
            _selected(picked, points=_points(paths, key=str(picked["key"])))
        _videos(paths, picked=picked, playing=playing)

    similar = st.session_state.pop(SIMILAR_KEY, None)
    replacing = st.session_state.pop(REPLACING_KEY, None)
    if similar is not None:
        _similar_name(similar)
    elif replacing is not None:
        _replacing_name(replacing)


def _steps() -> None:
    """The step back and forward through the tracks selected so far."""
    history = _history()
    back, forward, _ = st.columns([1, 1, 3])
    back.button("Back", on_click=_step, args=(-1,), disabled=not history.behind, help=BACK_HELP, width="stretch")
    forward.button("Forward", on_click=_step, args=(1,), disabled=not history.ahead, help=FORWARD_HELP, width="stretch")


def _map_clicked() -> None:
    """Keep the clicked track as the selected one until another is clicked.

    A click is taken here rather than while the page is drawn, because
    the page reads the selection before the map is drawn, and the map
    rings the tracks the galleries show, which follow the selected
    track.
    """
    clicked = _reported(MAP_KEY, event="clicked")
    if clicked:
        _select(clicked)


def _map_cleared() -> None:
    """Take the selection off, leaving the tracks selected so far to step back
    through."""
    if _reported(MAP_KEY, event="cleared"):
        st.session_state[HISTORY_KEY] = cleared(_history())
        st.session_state.pop(PLAYING_KEY, None)


def _sample_clicked() -> None:
    _gallery_clicked(SAMPLE_GALLERY_KEY)


def _nearest_clicked() -> None:
    _gallery_clicked(NEAREST_GALLERY_KEY)


def _sample_jumped() -> None:
    _gallery_jumped(SAMPLE_GALLERY_KEY)


def _nearest_jumped() -> None:
    _gallery_jumped(NEAREST_GALLERY_KEY)


def _gallery_clicked(component: str) -> None:
    """Play the video of the clicked track, leaving the selected track where it
    is, so that a cluster can be gone through video by video."""
    clicked = _reported(component, event="clicked")
    if clicked:
        st.session_state[PLAYING_KEY] = clicked


def _gallery_jumped(component: str) -> None:
    """Select the track whose Jump button was pressed, which is what a click on
    its point on the map does."""
    jumped = _reported(component, event="jumped")
    if jumped:
        _select(jumped)


def _reported(component: str, *, event: str) -> str:
    """The track a component reports this event on, and nothing while it
    reports none."""
    return str(st.session_state[component].get(event) or "")


def _search() -> None:
    """Select the track a search names, before the page is drawn again.

    A search selects from here rather than while the page is drawn, so
    that text left standing in the box does not take the selection back
    from a point clicked afterwards.
    """
    found = searched(_map_frame(MAP_PATH.stat().st_mtime), wanted=str(st.session_state[SEARCH_KEY]))
    if not found.empty:
        _select(str(found.iloc[0]["key"]))


def _history() -> History:
    return st.session_state.get(HISTORY_KEY, History())


def _select(key: str) -> None:
    """Stand on this track, and keep it among the tracks the steps go back
    through.

    Its video is the one that plays from here, until a gallery panel is
    clicked.
    """
    st.session_state[HISTORY_KEY] = visited(_history(), key=key)
    st.session_state.pop(PLAYING_KEY, None)


def _step(offset: int) -> None:
    st.session_state[HISTORY_KEY] = stepped(_history(), offset=offset)


def _tracks_frame() -> pd.DataFrame:
    """Every mapped track with the names it stands under, whether its recording
    is on disk, and whether someone has confirmed it."""
    tracks = labelled(
        _map_frame(MAP_PATH.stat().st_mtime),
        clusters=_cluster_labels(_labels_stamp()),
        own=_track_labels(_track_labels_stamp()),
    )
    on_disk = _videos_on_disk(_videos_stamp())
    validated = _validated_tracks(_validated_stamp())
    return tracks.assign(
        **{
            CACHED_COLUMN: tracks["recording"].isin(on_disk),
            VALIDATED_COLUMN: tracks["key"].isin(validated),
        }
    )


def searched(tracks: pd.DataFrame, *, wanted: str) -> pd.DataFrame:
    """The tracks a search names, in the order the map holds them.

    A search carries a track's number, part of its recording's name, or
    both, as the heading over a selected track gives them. Each word of
    it that is a number is read as the track's number, and every other
    word as part of the recording's name.
    """
    words = [word for word in re.split(r"[\s,]+", wanted.strip()) if word and word.lower() not in SEARCH_WORDS]
    if not words:
        return tracks.iloc[0:0]

    found = tracks
    for word in words:
        if word.isdigit():
            found = found[found["track_id"] == int(word)]
        else:
            found = found[found["clip"].str.contains(word, case=False, regex=False)]
    return found


def _playing(tracks: pd.DataFrame, *, picked: pd.Series | None) -> pd.Series | None:
    """The track whose video plays, which is the track a gallery panel was last
    clicked on and the selected track until one was."""
    clicked = str(st.session_state.get(PLAYING_KEY) or "")
    held = tracks.loc[tracks["key"] == clicked]
    return picked if held.empty else held.iloc[0]


def _picked(shown: pd.DataFrame, *, tracks: pd.DataFrame, key: str | None) -> pd.Series | None:
    """The selected track, taken from the whole corpus when the filters leave
    out the track a search named."""
    held = shown.loc[shown["key"] == key]
    if held.empty:
        held = tracks.loc[tracks["key"] == key]
    return None if held.empty else held.iloc[0]


def by_validation(tracks: pd.DataFrame, *, choice: str) -> pd.DataFrame:
    """The tracks the validation choice leaves on the map."""
    if choice == VALIDATED:
        return tracks[tracks[VALIDATED_COLUMN]]
    if choice == UNVALIDATED:
        return tracks[~tracks[VALIDATED_COLUMN]]
    return tracks


def _sample(tracks: pd.DataFrame, *, cluster: str) -> pd.DataFrame:
    """A random sample of the cluster's tracks, drawn anew on Shuffle."""
    members = tracks[tracks["cluster"] == cluster]
    return members.sample(n=min(GALLERY_SIZE, len(members)), random_state=st.session_state.get(SHUFFLE_KEY, 0))


def _nearest(tracks: pd.DataFrame, *, track: pd.Series, radius: float) -> pd.DataFrame:
    """The tracks nearest the selected one on the map, in any cluster and
    within the radius."""
    others = tracks[tracks["key"] != track["key"]]
    distance = np.hypot(others["x"] - track["x"], others["y"] - track["y"])
    return others.loc[distance[distance <= radius].nsmallest(GALLERY_SIZE).index]


def _uncached(tracks: pd.DataFrame) -> frozenset[str]:
    """The tracks whose recording is on neither the sift's shelf nor the page's
    own, which the map draws faintly."""
    return frozenset(tracks.loc[~tracks[CACHED_COLUMN], "key"])


def labelled(tracks: pd.DataFrame, *, clusters: dict[str, list[str]], own: dict[str, list[str]]) -> pd.DataFrame:
    """The tracks with the name of the cluster each one is in beside it, and
    the name the track itself stands under.

    A track whose cluster has no name yet, and a track the clustering
    left out of every cluster, stand under one name of their own, so
    that the map can be coloured and filtered by the name without those
    tracks falling off it. A track stands under its cluster's name until
    it is given a name of its own, which is then the name it stands
    under wherever the page gives one.
    """
    cluster_name = tracks["key"].map(_by_key(clusters)).fillna(UNNAMED)
    return tracks.assign(
        **{
            NAME_COLUMN: cluster_name,
            TRACK_NAME_COLUMN: tracks["key"].map(_by_key(own)).fillna(cluster_name),
        }
    )


def _by_key(labels: dict[str, list[str]]) -> dict[str, str]:
    """The name each track is under, from the tracks kept under each name."""
    return {key: name for name, keys in labels.items() for key in keys}


def cluster_names(tracks: pd.DataFrame, *, labels: dict[str, list[str]]) -> pd.DataFrame:
    """The name each named cluster is under, and where on the map it goes.

    A name stands over the middle of its cluster's points, taken as the
    median of them so that a track lying far out does not carry the name
    away with it. A cluster holding tracks of more than one name, which
    a clustering run afresh can leave, takes the name most of them are
    under.
    """
    named = tracks[tracks["cluster"] != UNASSIGNED_NAME].copy()
    named["name"] = named["key"].map({key: name for name, keys in labels.items() for key in keys})
    named = named.dropna(subset=["name"])
    if named.empty:
        return pd.DataFrame(columns=["x", "y", "name"])

    middles = named.groupby("cluster")[["x", "y"]].median()
    middles["name"] = named.groupby("cluster")["name"].agg(_commonest)
    return middles


def _commonest(names: pd.Series) -> str:
    return str(names.mode().iat[0])


def _scatter(
    tracks: pd.DataFrame, *, colour: str, rings: list[Ring], dimmed: frozenset[str], names: pd.DataFrame
) -> go.Figure:
    """The tracks drawn by the graphics card, one trace per colour, so the
    legend names each colour and a click on it hides or shows those tracks.

    Drawing on the card keeps zooming and panning smooth over thousands
    of points. The fixed UI revision keeps the zoom when the plot is
    handed the next figure, and each trace's uid keeps it hidden or
    shown as the legend left it. The rings over the selected track and
    the tracks the galleries show take no hover or click, so a click on
    a ringed track selects the track under the ring.

    The names of the clusters are drawn by the browser, which puts them
    over the points the card draws.
    """
    column = COLOUR_COLUMNS[colour]
    traces = []
    for name, shade in _colours(tracks, column=column).items():
        members = tracks[tracks[column] == name]
        traces.append(
            go.Scattergl(
                x=members["x"],
                y=members["y"],
                mode="markers",
                name=str(name),
                uid=f"{column}:{name}",
                marker={"color": _point_colours(members, colour=shade, dimmed=dimmed), "size": 6, "opacity": 0.7},
                customdata=members[HOVER_COLUMNS].to_numpy(dtype=object),
                hovertemplate=HOVER_TEMPLATE,
            )
        )
    for ring in rings:
        if ring.tracks.empty:
            continue
        traces.append(
            go.Scattergl(
                x=ring.tracks["x"],
                y=ring.tracks["y"],
                mode="markers",
                name=ring.name,
                uid=ring.name,
                marker=ring.marker,
                hoverinfo="skip",
            )
        )
    if not names.empty:
        traces.append(
            go.Scatter(
                x=names["x"],
                y=names["y"],
                mode="text",
                name=NAMES_TITLE,
                uid=NAMES_TITLE,
                text=names["name"],
                textfont=NAMES_FONT,
                hoverinfo="skip",
            )
        )
    figure = go.Figure(traces)
    figure.update_layout(
        template="plotly_white",
        height=MAP_HEIGHT,
        dragmode="pan",
        uirevision="track-map",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"title": {"text": colour}},
        xaxis={"showgrid": True, "gridcolor": GRID_COLOR, "zeroline": False},
        yaxis={"showgrid": True, "gridcolor": GRID_COLOR, "zeroline": False},
    )
    return figure


def _point_colours(members: pd.DataFrame, *, colour: str, dimmed: frozenset[str]) -> str | list[str]:
    """The trace's colour at each point, faint where the track is one of the
    dimmed, or the one colour for the whole trace when none of them is."""
    if not dimmed:
        return colour

    red, green, blue = hex_to_rgb(colour)
    faint = f"rgba({red}, {green}, {blue}, {DIM_ALPHA})"
    return [faint if key in dimmed else colour for key in members["key"]]


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


def _selected(track: pd.Series, *, points: pd.DataFrame) -> None:
    """The picked track drawn at once from its stored path."""
    st.subheader(f"Track {int(track['track_id'])} in {track['clip']}", help=PATH_HELP)
    st.caption(_facts(track))
    _track_labelling(track)
    st.altair_chart(frame_view(points))
    st.altair_chart(close_up(points))
    st.altair_chart(light_curve(points))


def _track_labelling(track: pd.Series) -> None:
    """The name this one track stands under, the box that changes it, and
    whether someone has confirmed the track.

    The name starts as the cluster's, so a track that is what the rest
    of its cluster is needs no name of its own. The name it starts at is
    part of the box's key, because a box holds whatever it was left at
    and would otherwise go on offering a start the track has moved on
    from.
    """
    key = str(track["key"])
    given = label_of(_track_labels(_track_labels_stamp()), keys=[key])
    cluster_name = str(track[NAME_COLUMN])
    start = given or ("" if cluster_name == UNNAMED else cluster_name)
    naming = Naming(path=TRACK_LABELS_PATH, keys=(key,), box=f"{TRACK_LABEL_KEY}:{key}:{start}", confirms=True)

    chooser, save, tick, _rest = st.columns(TRACK_NAME_ROW, vertical_alignment="bottom")
    _name_box(chooser, label="Track label", given=start, naming=naming, help=TRACK_LABEL_HELP)
    save.button(
        "Save",
        key=f"{SAVE_TRACK_LABEL_KEY}:{key}",
        on_click=_save_name,
        args=(naming,),
        help=SAVE_TRACK_LABEL_HELP,
        width="stretch",
    )
    box = f"{VALIDATED_KEY}:{key}"
    _seeded(box, value=bool(track[VALIDATED_COLUMN]))
    tick.checkbox("Validated", key=box, on_change=_validate, args=(key,), help=VALIDATED_HELP)
    _reassigning(track)


def _reassigning(track: pd.Series) -> None:
    """The cluster this one track is counted with, and the box that puts it
    with another.

    The box holds a cluster label, and giving it another one puts this
    track under that name on its own. The cluster the track came from
    keeps the name the rest of its tracks hold, and a name no cluster is
    under yet opens one holding this track.
    """
    key = str(track["key"])
    given = label_of(_cluster_labels(_labels_stamp()), keys=[key])
    naming = Naming(path=LABELS_PATH, keys=(key,), box=f"{REASSIGN_KEY}:{key}", confirms=True)

    chooser, save, _rest = st.columns(REASSIGN_ROW, vertical_alignment="bottom")
    _name_box(chooser, label="Reassign", given=given, naming=naming, help=REASSIGN_HELP)
    save.button(
        "Save",
        key=f"{SAVE_REASSIGN_KEY}:{key}",
        on_click=_save_name,
        args=(naming,),
        help=SAVE_REASSIGN_HELP,
        width="stretch",
    )


def _name_box(column: DeltaGenerator, *, label: str, given: str, naming: Naming, help: str) -> None:
    """The box a name is given in: the names in use so far in a drop-down that
    narrows as it is typed in, and a name of its own where none of them fits.

    Every name of either file is offered, because a cluster and a track
    stand under names of one vocabulary.
    """
    options = sorted(set(_known_names()) | ({given} if given else set()))
    _seeded(naming.box, value=given or None)
    column.selectbox(
        label,
        options=options,
        key=naming.box,
        accept_new_options=True,
        placeholder=NAME_PLACEHOLDER,
        help=help,
    )


def _seeded(box: str, *, value: object) -> None:
    """Put the value a box opens on into the session, and leave a box that has
    been opened before as the person left it.

    The value is held in the session rather than handed to the box,
    because a box that carries both and is written to as well is one
    Streamlit warns about on every run.
    """
    if box not in st.session_state:
        st.session_state[box] = value


def _save_name(naming: Naming) -> None:
    """Keep the name given in the box, unless it is close enough to a name
    already in use to be that name mistyped, which is put to the person
    first."""
    wanted = canonical_label(str(st.session_state[naming.box] or ""))
    near = near_label(wanted, known=_known_names())
    if near:
        st.session_state[SIMILAR_KEY] = Similar(naming=naming, name=wanted, near=near)
        return

    _settle(naming, name=wanted)


def _settle(naming: Naming, *, name: str) -> None:
    """Keep this name, unless the tracks stand under another one, which is put
    to the person first.

    Which of two close names was meant is settled before this, because
    the answer to that is what would be moved.
    """
    held = label_of(read_labels(naming.path), keys=list(naming.keys))
    if held and held != name:
        st.session_state[REPLACING_KEY] = Replacing(naming=naming, name=name, held=held)
        return

    _keep_name(naming, name=name)


def _keep_name(naming: Naming, *, name: str) -> None:
    """Put the tracks this name was given to under it, and take the box to what
    the file now holds.

    The box has to be set rather than emptied, because a box left to
    itself hands back whatever was typed into it and would take the page
    away from the file again.

    Naming a track is someone looking at it and saying what it is, which
    is what confirming it says, so a track keeps its name and its
    confirmation together. Withdrawing a name says nothing, and leaves
    the track as it was.
    """
    write_labels(naming.path, read_labels(naming.path), name=name, keys=list(naming.keys))
    st.session_state[naming.box] = name or None
    if naming.confirms and name:
        _confirm(naming.keys[0], confirmed=True)


@st.dialog(SIMILAR_TITLE)
def _similar_name(similar: Similar) -> None:
    """What to do about a name that is close to one already in use."""
    st.write(f"**{similar.name}** is close to **{similar.near}**, which is already in use.")
    st.caption(SIMILAR_TEXT)

    use, keep = st.columns(2)
    if use.button(f"Use {similar.near}", type="primary", width="stretch"):
        _settle(similar.naming, name=similar.near)
        st.rerun()
    if keep.button(f"Keep {similar.name}", width="stretch"):
        _settle(similar.naming, name=similar.name)
        st.rerun()


@st.dialog(REPLACING_TITLE)
def _replacing_name(replacing: Replacing) -> None:
    """What to do about tracks that stand under a name already."""
    count = len(replacing.naming.keys)
    st.write(f"{count} track{'' if count == 1 else 's'} under **{replacing.held}**.")
    st.caption(REPLACING_TEXT)

    change, keep = st.columns(2)
    given = f"Change to {replacing.name}" if replacing.name else f"Take {replacing.held} off"
    if change.button(given, type="primary", width="stretch"):
        _keep_name(replacing.naming, name=replacing.name)
        st.rerun()
    if keep.button(f"Keep {replacing.held}", width="stretch"):
        st.session_state[replacing.naming.box] = replacing.held
        st.rerun()


def _known_names() -> list[str]:
    """Every name given so far, to a cluster or to a track."""
    return sorted(set(_cluster_labels(_labels_stamp())) | set(_track_labels(_track_labels_stamp())))


def _validate(key: str) -> None:
    _confirm(key, confirmed=bool(st.session_state[f"{VALIDATED_KEY}:{key}"]))


def _confirm(key: str, *, confirmed: bool) -> None:
    """Keep on disk whether this track has been confirmed, and hold the box on
    the page to it.

    The box holds whatever it was left at, so naming a track has to put
    the box where the file now stands or the next press would take the
    track back out.
    """
    write_validated(VALIDATED_PATH, read_validated(VALIDATED_PATH), key=key, confirmed=confirmed)
    st.session_state[f"{VALIDATED_KEY}:{key}"] = confirmed


def _videos(paths: pd.DataFrame, *, picked: pd.Series | None, playing: pd.Series | None) -> None:
    """The video of the selected track, and under it the video of a track a
    gallery panel was clicked on to hold against it.

    One clip is built at a time, so the second waits while the first is
    being built. With no track selected the clicked one takes the first
    player, because there is nothing to hold it against.
    """
    first = picked if picked is not None else playing
    if first is None:
        return

    building = _video(first, points=_points(paths, key=str(first["key"])), player=VIDEO_PLAYER, may_build=True)
    compared = _compared(playing, picked=picked)
    if compared is not None:
        _video(
            compared,
            points=_points(paths, key=str(compared["key"])),
            player=COMPARISON_PLAYER,
            may_build=not building,
        )


def _compared(playing: pd.Series | None, *, picked: pd.Series | None) -> pd.Series | None:
    """The track held against the selected one, which is the track a gallery
    panel was clicked on while another track is selected."""
    if playing is None or picked is None or str(playing["key"]) == str(picked["key"]):
        return None
    return playing


def _video(track: pd.Series, *, points: pd.DataFrame, player: str, may_build: bool) -> bool:
    """The video of one track under a line naming it, and whether a build was
    asked for."""
    title, note = VIDEO_TITLES[player]
    st.subheader(title, help=note)
    st.caption(f"Track {int(track['track_id'])} in {track['clip']}")
    return _clip(
        str(track["key"]),
        recording=str(track["recording"]),
        stored=_stored_track(points),
        player=player,
        may_build=may_build,
    )


def _clip(key: str, *, recording: str, stored: StoredTrack, player: str, may_build: bool) -> bool:
    """The clips of the track when they are built, and their progress until
    then, and whether a build was asked for.

    A build is asked for only where it may be, because one clip is built
    at a time and a second asked for would stop the first.
    """
    source = _source(recording)
    if source is None:
        st.info(UNSOURCED_TEXT)
        return False

    if source.path is not None:
        clips = _clips(source.path, track=stored)
        if clips.built:
            _play(clips, player=player)
            return False
    elif not _may_fetch(source, player=player):
        return False

    if not may_build:
        st.caption(WAITING_TEXT)
        return False

    _queue().request(key, _clip_job(source, track=stored))
    _clip_progress(key, fetching=source.path is None)
    return True


def _play(clips: TrackClips, *, player: str) -> None:
    """The whole frame and the crop that follows the detection, a tab each, of
    whichever panel is chosen.

    Both panels are built together, so the choice costs nothing to
    change.
    """
    panel = st.segmented_control(
        "Panel",
        options=CLIP_PANELS,
        default=RECORDING,
        format_func=str.capitalize,
        help=PANEL_HELP,
        key=f"{PANEL_KEY}:{player}",
    )
    pair = clips.pair(str(panel or RECORDING))

    whole, close = st.tabs([WHOLE_TITLE, CLOSE_UP_TITLE])
    whole.video(str(pair.whole), loop=True, autoplay=True, muted=True)
    close.video(str(pair.close_up), loop=True, autoplay=True, muted=True)


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


def _may_fetch(source: VideoSource, *, player: str) -> bool:
    """Whether the recording may be fetched now: there is room for it, and a
    large one has been asked for."""
    archived = source.archived
    if archived is None:
        return False

    megabytes = archived.size_bytes / 1e6
    free = free_bytes(FETCHED_DIR)
    if not room_to_fetch(archived, free_bytes=free):
        st.warning(
            f"Fetching this {megabytes:.0f} MB video would leave "
            f"{(free - archived.size_bytes) / GIGABYTE:.1f} GB free, under the "
            f"{MIN_FREE_BYTES / GIGABYTE:.0f} GB the archive sift needs to keep fetching."
        )
        if source.link:
            st.link_button("Open on Drive", source.link)
        return False

    return archived.size_bytes <= AUTO_FETCH_BYTES or st.button(
        "Fetch video",
        key=f"{FETCH_KEY}:{player}",
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
        clips = _clips(video, track=track)
        if not clips.built:
            details = probe(video)
            stretch = stretch_around(
                track, frames_per_second=details.frames_per_second, frame_count=details.frame_count
            )
            parts = clips.parts
            try:
                build_track_clip(
                    video,
                    track=track,
                    stretch=stretch,
                    frames_per_second=details.frames_per_second,
                    settings=config().settings,
                    output=parts,
                    on_progress=report,
                )
            except BaseException:
                for path in parts.paths:
                    path.unlink(missing_ok=True)
                raise
            for part, clip in zip(parts.paths, clips.paths):
                part.replace(clip)
        return clips.recording.whole

    return job


def _archived(source: VideoSource) -> ArchiveVideo:
    if source.archived is None:
        raise FileNotFoundError("The recording is neither on disk nor named by a ledger.")
    return source.archived


def _clips(video: Path, *, track: StoredTrack) -> TrackClips:
    details = probe(video)
    stretch = stretch_around(track, frames_per_second=details.frames_per_second, frame_count=details.frame_count)
    return track_clip_paths(video, track=track, stretch=stretch, settings=config().settings, output_dir=CLIPS_DIR)


def _gallery(tracks: pd.DataFrame, *, cluster: str | None, sample: pd.DataFrame, playing: pd.Series | None) -> None:
    """The random sample of the cluster, framed in the colour that rings it on
    the map."""
    if cluster is None:
        st.caption("Select a track, or choose a cluster for the gallery in the sidebar.")
        return

    heading, shuffle = st.columns([4, 1], vertical_alignment="bottom")
    heading.subheader(SAMPLE_TITLE, help=GALLERY_HELP)
    shuffle.button("Shuffle", on_click=_shuffle, help=SHUFFLE_HELP)
    members = int((tracks["cluster"] == cluster).sum())
    st.caption(f"{_cluster_title(cluster)} · {len(sample)} of {members} tracks")
    captions = [f"{place}. folder {folder}" for place, folder in enumerate(sample["label"], start=1)]
    _panels(
        sample,
        captions=captions,
        frame=SAMPLE_COLOR,
        playing=playing,
        key=SAMPLE_GALLERY_KEY,
        on_click=_sample_clicked,
        on_jump=_sample_jumped,
    )


def _shuffle() -> None:
    st.session_state[SHUFFLE_KEY] = st.session_state.get(SHUFFLE_KEY, 0) + 1


def _labelling(tracks: pd.DataFrame, *, cluster: str, nearest: pd.DataFrame, selected: str) -> None:
    """The name a group of tracks is under, and the box that gives it one.

    The group is the cluster the gallery above draws, or the tracks
    nearest the selected one, which is how a name is given to a
    neighbourhood the clustering cut in two.

    Every track of a cluster takes the name, including the tracks the
    sidebar's filters leave off the map, because the name is about what
    the cluster holds. A track put under a name of its own is left
    alone, so that naming the cluster again does not take a reassignment
    back. The nearest tracks all take it, because they were picked by
    hand and each one stands under the name of whichever cluster it came
    from.

    The box is keyed by the group it names, so that it opens on that
    group's own name. The nearest tracks are the group of the selected
    track, which is why the selected track is part of the key.
    """
    chooser, save, over, _rest = st.columns(NAME_ROW, vertical_alignment="bottom")
    given_to = str(
        over.segmented_control(
            "Apply to", options=NAMING_CHOICES, default=CLUSTER_TRACKS, key=NAMING_KEY, help=NAMING_HELP
        )
        or CLUSTER_TRACKS
    )

    labels = _cluster_labels(_labels_stamp())
    held = _by_key(labels)
    neighbours = given_to == NEIGHBOUR_TRACKS
    keys = nearest["key"].tolist() if neighbours else tracks.loc[tracks["cluster"] == cluster, "key"].tolist()
    given = label_of(labels, keys=keys)
    moving = tuple(keys) if neighbours else tuple(key for key in keys if held.get(key, "") in ("", given))
    under = sum(1 for key in keys if held.get(key) == given)
    group = f"{given_to}:{selected}" if neighbours else f"{given_to}:{cluster}"
    naming = Naming(path=LABELS_PATH, keys=moving, box=f"{LABEL_KEY}:{group}", confirms=False)

    _name_box(chooser, label="Cluster label", given=given, naming=naming, help=LABEL_HELP)
    st.caption(_naming_caption(given_to, given=given, under=under, held=len(keys)))
    save.button(
        "Save",
        key=f"{SAVE_KEY}:{cluster}",
        on_click=_save_name,
        args=(naming,),
        help=SAVE_LABEL_HELP,
        width="stretch",
    )


def _naming_caption(given_to: str, *, given: str, under: int, held: int) -> str:
    """How many of the tracks the name would be given to are under it
    already."""
    group = "nearest tracks" if given_to == NEIGHBOUR_TRACKS else "tracks"
    if not held:
        return "Select a track to name the ones nearest it."
    if not given:
        return f"None of these {held} {group} has a name yet."
    return f"{under} of {held} {group} under {given}"


def _neighbours(nearest: pd.DataFrame, *, radius: float, playing: pd.Series | None) -> None:
    """The tracks nearest the selected one, each captioned with the cluster it
    is in and framed in the colour that rings them on the map."""
    st.subheader(NEAREST_TITLE, help=NEIGHBOURS_HELP)
    st.caption(f"{len(nearest)} tracks within {radius:.2f}")
    captions = [
        f"{place}. {_cluster_caption(cluster)} · {name}"
        for place, (cluster, name) in enumerate(zip(nearest["cluster"], nearest[NAME_COLUMN]), start=1)
    ]
    _panels(
        nearest,
        captions=captions,
        frame=NEAREST_COLOR,
        playing=playing,
        key=NEAREST_GALLERY_KEY,
        on_click=_nearest_clicked,
        on_jump=_nearest_jumped,
    )


def _cluster_caption(cluster: str) -> str:
    return "no cluster" if cluster == UNASSIGNED_NAME else f"cluster {cluster}"


def _panels(
    chosen: pd.DataFrame,
    *,
    captions: list[str],
    frame: str,
    playing: pd.Series | None,
    key: str,
    on_click: Callable[[], None],
    on_jump: Callable[[], None],
) -> None:
    """The chosen tracks drawn from their stored paths, a panel each, in rows
    that wrap to the width of the column, each panel playing its track's video
    when it is clicked and selecting the track from its Jump button."""
    played = "" if playing is None else str(playing["key"])
    entries = tuple(
        GalleryEntry(
            key=str(track),
            caption=caption,
            cached=bool(cached),
            validated=bool(validated),
            playing=str(track) == played,
        )
        for track, cached, validated, caption in zip(
            chosen["key"], chosen[CACHED_COLUMN], chosen[VALIDATED_COLUMN], captions
        )
    )
    track_gallery(
        _gallery_markup(PATHS_PATH.stat().st_mtime, entries, frame=frame),
        key=key,
        on_click=on_click,
        on_jump=on_jump,
    )


@st.cache_data(show_spinner=False, max_entries=PANEL_CACHE_ENTRIES)
def _gallery_markup(stamp: float, entries: tuple[GalleryEntry, ...], *, frame: str) -> str:
    """The panels of these tracks under these captions, kept so that a gallery
    that comes out the same on the next click is not drawn again."""
    return gallery_html(_path_frame(stamp), entries=entries, frame=frame)


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
        str(track[TRACK_NAME_COLUMN]),
        _cluster_caption(str(track["cluster"])),
        f"folder {track['label']}",
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


def _videos_stamp() -> tuple[float, ...]:
    return tuple(path.stat().st_mtime if path.is_dir() else 0.0 for path in (VIDEOS_DIR, FETCHED_DIR))


@st.cache_data(show_spinner=False)
def _videos_on_disk(stamp: tuple[float, ...]) -> frozenset[str]:
    """The name of every video the sift kept or the page has fetched, read
    again whenever either folder changes.

    The stamp holds those folders' modification times, and is what the
    cache is keyed on, which is why it is passed although the body never
    reads it.
    """
    folders = (VIDEOS_DIR, FETCHED_DIR)
    return frozenset(path.name for folder in folders if folder.is_dir() for path in folder.iterdir())


def _labels_stamp() -> float:
    return LABELS_PATH.stat().st_mtime if LABELS_PATH.is_file() else 0.0


@st.cache_data(show_spinner=False)
def _cluster_labels(stamp: float) -> dict[str, list[str]]:
    """The names given to clusters so far, read again whenever the file
    changes.

    The stamp is the file's modification time, and is what the cache is
    keyed on, which is why it is passed although the body never reads
    it.
    """
    return read_labels(LABELS_PATH)


def _track_labels_stamp() -> float:
    return TRACK_LABELS_PATH.stat().st_mtime if TRACK_LABELS_PATH.is_file() else 0.0


@st.cache_data(show_spinner=False)
def _track_labels(stamp: float) -> dict[str, list[str]]:
    """The names given to single tracks so far, read again whenever the file
    changes.

    The stamp is the file's modification time, and is what the cache is
    keyed on, which is why it is passed although the body never reads
    it.
    """
    return read_labels(TRACK_LABELS_PATH)


def _validated_stamp() -> float:
    return VALIDATED_PATH.stat().st_mtime if VALIDATED_PATH.is_file() else 0.0


@st.cache_data(show_spinner=False)
def _validated_tracks(stamp: float) -> frozenset[str]:
    """The tracks confirmed so far, read again whenever the file changes.

    The stamp is the file's modification time, and is what the cache is
    keyed on, which is why it is passed although the body never reads
    it.
    """
    return read_validated(VALIDATED_PATH)


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
