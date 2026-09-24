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
from collections.abc import Callable, Sequence
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

from hessdalen.analysis.spectra import SIGNALS
from hessdalen.config import config
from hessdalen.dashboard.clip_queue import ClipQueue, Job, Report
from hessdalen.dashboard.cluster_labels import (
    canonical_label,
    label_of,
    near_label,
    read_labels,
    tagged,
    tags_of,
    write_labels,
    write_tags,
)
from hessdalen.dashboard.panels import DEVIATION, RECORDING
from hessdalen.dashboard.runs import VideoProbe, probe
from hessdalen.dashboard.track_clip import (
    ClipProgress,
    StoredTrack,
    TrackClips,
    build_track_clip,
    stretch_around,
    track_clip_paths,
)
from hessdalen.dashboard.track_gallery import ADD, RANGE, track_gallery
from hessdalen.dashboard.track_history import History, cleared, stepped, visited
from hessdalen.dashboard.track_map_chart import MAP_HEIGHT, track_map_chart
from hessdalen.dashboard.track_preview import (
    GalleryEntry,
    close_up,
    frame_view,
    gallery_html,
    light_curve,
    rhythm_chart,
    spectra_html,
)
from hessdalen.dashboard.track_reference import (
    NOMINAL_RATE,
    Reference,
    reference,
    reference_line,
    references_csv,
)
from hessdalen.dashboard.tuning import MARGIN_SECONDS, cut_for_tuning, note_label
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
TUNING_DIR = REPO_ROOT / "data" / "out" / "dashboard" / "tuning"
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

TAGS_COLUMN = "tags"
"""Column holding the tags a track itself stands under, one after another,
which is the name of its cluster until the track is given tags of its own."""

CACHED_COLUMN = "cached"
"""Column saying whether the track's recording is on disk."""

VALIDATED_COLUMN = "validated"
"""Column saying whether someone has confirmed the track."""

UNNAMED = "unlabelled"
"""What a track whose cluster has no name yet is shown under."""

COLOUR_CHOICES = ("Cluster", "Cluster label", "Tags", "Side", "Folder", "Camera")
COLOUR_COLUMNS = {
    "Cluster": "cluster",
    "Cluster label": NAME_COLUMN,
    "Tags": TAGS_COLUMN,
    "Side": "side",
    "Folder": "label",
    "Camera": "camera",
}
PATH_DRAWING = "Path"
DRAWING_CHOICES = (PATH_DRAWING, "Brightness", "Size", "Wobble", "Presence")
"""What a track is drawn as, in the galleries and beside its light curve.

Every choice but the path names one of the signals a track's rhythm can
be read from, and the signals are named here as they are shown.
"""

VALIDATION_CHOICES = ("All", "Validated", "Unvalidated")
ANY_VALIDATION, VALIDATED, UNVALIDATED = VALIDATION_CHOICES
NAMING_CHOICES = ("Cluster", "Neighbours", "Picked")
CLUSTER_TRACKS, NEIGHBOUR_TRACKS, PICKED_TRACKS = NAMING_CHOICES
GALLERY_ORDERS = ("Distance", "Video cached", "Cluster label")
BY_DISTANCE, BY_CACHED, BY_CLUSTER_NAME = GALLERY_ORDERS
SAMPLE_ORDERS = (BY_DISTANCE, BY_CACHED)
"""The orders a gallery's panels can stand in.

The tracks of one cluster stand under one name, so the sample of a
cluster is not offered that order.
"""
VIEW_CHOICES = ("Map", "Table")
MAP_VIEW, TABLE_VIEW = VIEW_CHOICES
SELECTED_CLUSTER = "Selected track's cluster"
ROUGH_SIDE = "rough"
OTHER_COLORS = qualitative.Dark24

TABLE_COLUMNS = {
    "track_id": "Track",
    "clip": "Recording",
    TAGS_COLUMN: "Tags",
    CACHED_COLUMN: "Video cached",
    VALIDATED_COLUMN: "Validated",
    NAME_COLUMN: "Cluster label",
    "cluster": "Cluster",
    "label": "Folder",
    "side": "Side",
    "frames": "Frames",
    "straightness": "Straightness",
    "peak_deviation_max": "Peak deviation",
}
"""The columns of the table, in the order they stand in it, and the name over
each.

They are the lines the panel under the map holds. What names a track
comes first, then how far through it someone is, then what it was
grouped with and what it was measured as, because the table is wider
than the page and the last of the columns are reached by scrolling.
"""

RECORDING_WIDTH = 250
TAGS_WIDTH = 150
"""How wide the two columns of the table are that hold names of their own, in
pixels.

A recording's name is 34 letters at its longest and the rest of the
columns hold a word or a number, which the table sizes for itself.
"""


@dataclass(frozen=True, slots=True)
class Detail:
    """A line of the panel under the map: the column its value is read from,
    the name it stands under, the digits a number is shown to, with none where
    the value is shown as it stands, and whether it takes two of the panel's
    columns."""

    column: str
    label: str
    digits: int | None
    wide: bool


DETAILS = (
    Detail(column=TAGS_COLUMN, label="Tags", digits=None, wide=True),
    Detail(column=NAME_COLUMN, label="Cluster label", digits=None, wide=False),
    Detail(column="cluster", label="Cluster", digits=None, wide=False),
    Detail(column="side", label="Side", digits=None, wide=False),
    Detail(column="clip", label="Recording", digits=None, wide=True),
    Detail(column="track_id", label="Track", digits=None, wide=False),
    Detail(column="frames", label="Frames", digits=None, wide=False),
    Detail(column="straightness", label="Straightness", digits=2, wide=False),
    Detail(column="peak_deviation_max", label="Peak deviation", digits=1, wide=False),
    Detail(column="label", label="Folder", digits=None, wide=False),
)
"""What the panel under the map tells of the track under the pointer.

A recording's name is as long as the rest of the line put together, so
it is given two of the panel's columns and the others one each.
"""

DETAIL_LINES = [{"label": detail.label, "digits": detail.digits, "wide": detail.wide} for detail in DETAILS]
"""The same lines as the map's component takes them, which holds no column
because a point carries its values in the order the details are given in."""

POINT_COLUMNS = ["key", *(detail.column for detail in DETAILS)]
"""What every point on the map carries: the track's key, which a click
reports, and the values the panel under the map shows."""

GALLERY_SIZE = 30
NEIGHBOUR_RADIUS = 0.5
SHUFFLE_KEY = "gallery_shuffle"
SAMPLE_ORDER_KEY = "sample_order"
NEAREST_ORDER_KEY = "nearest_order"
MAP_KEY = "track_map"
VIEW_KEY = "track_view"
TABLE_KEY = "track_table"
TABLE_ROWS_KEY = "track_table_rows"
HISTORY_KEY = "track_history"
PLAYING_KEY = "playing_track"
PICKED_KEY = "picked_tracks"
SAMPLE_GALLERY_KEY = "sample_gallery"
NEAREST_GALLERY_KEY = "nearest_gallery"
TOGETHER_GALLERY_KEY = "together_gallery"
SEARCH_KEY = "track_search"
LABEL_KEY = "cluster_label"
SAVE_KEY = "save_cluster_label"
NAMING_KEY = "naming_choice"
TAGS_KEY = "track_tags"
SAVE_TAGS_KEY = "save_track_tags"
DRAWING_KEY = "track_drawing"
TAGS_FILTER_KEY = "tags_filter"
SIMILAR_TAGS_KEY = "similar_tags"
TUNE_KEY = "tune_track"
TUNING_PICK_KEY = "tuning_pick"
"""Session key holding the clip the recordings page is to open on."""

RECORDINGS_PAGE_KEY = "recordings_page"
"""Session key holding the recordings page, which this page switches to."""
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
TOGETHER_TITLE = "At the same time"
SELECTED_TITLE = "Selected track"
SIMILAR_TITLE = "Similar name"
REPLACING_TITLE = "Change of name"
NAME_PLACEHOLDER = "Choose or add a name"
TAGS_PLACEHOLDER = "Choose or add tags"
TRACKS_ROW = (3, 2)
"""How the row over the map is divided: its heading, and whether the tracks
are drawn as points or listed."""

GALLERY_ROW = (2, 3, 1)
NEIGHBOUR_ROW = (2, 3)
"""How the row over each gallery is divided: its heading, the order its panels
stand in, and for the cluster's sample the button that draws another one."""

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
TOGETHER_COLOR = "#00a878"
SAMPLE_MARKER = {"color": SAMPLE_COLOR, "symbol": "circle-open", "size": 12, "line": {"width": 2}}
NEAREST_MARKER = {"color": NEAREST_COLOR, "symbol": "diamond-open", "size": 12, "line": {"width": 2}}
TOGETHER_MARKER = {"color": TOGETHER_COLOR, "symbol": "square-open", "size": 12, "line": {"width": 2}}
SELECTED_MARKER = {"color": "#ffd400", "symbol": "star", "size": 18, "line": {"width": 1, "color": "#333333"}}
POINT_REACH = 6
"""How near a point the pointer has to come, in pixels, for the point to be the
one it is on.

A point is drawn 6 pixels across, so this reaches a few pixels past its
edge. Plotly stands at 20 pixels, which in a crowd takes a point the
pointer is nowhere near, and the arrow then stands over a point beside
the one being pointed at.
"""

NAMES_FONT = {"size": 13}
"""Size the name of a cluster is written at.

The colour is left to the page's own text colour, which the plot is
handed, so a name stays readable whichever theme the page is in.
"""
GRID_COLOR = "rgba(128, 128, 128, 0.25)"
DIM_ALPHA = 0.2
PANEL_CACHE_ENTRIES = 64
POLL_SECONDS = 2.0

PLACE_DIGITS = 4
"""Digits a point's place on the map is written to.

The map is a few hundred pixels across and every digit of every point
crosses to the browser, so four hold a point well inside the pixel it
is drawn in at any zoom the page allows.
"""

FIGURE_CACHE_ENTRIES = 8
"""Sets of points the map keeps.

One is kept for each set of tracks lately drawn, under each colour, so
that going back and forth between two filters or two colours draws
from what is already built.
"""

GIGABYTE = 1024**3
"""What a gigabyte is taken to be wherever the page writes one.

The room left on the disk is written beside the room the page keeps
free, so the two are counted the same way or the sentence holding both
says one is under the other while the numbers say it is not.
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
DRAWING_HELP = (
    "Which signal the rhythm chart under the selected track's light curve draws, and what every "
    "gallery panel holds. A rhythm drawing reads its signal on every frame the track spans and gives "
    "how strongly it repeated, the rate rising upwards and the frames running left to right, so a "
    "track that repeats at one rate carries a bright band across it. Brightness and size are the "
    "blob's own. Wobble is how far it strayed from its straight line. Presence is whether it was "
    "found at all, which repeats at the rate the detector loses a track and picks it up again. Path "
    "returns the galleries to the tracks' paths and leaves the chart on brightness. Grey is where "
    "the track was too short to read that rate, which near either end is every slow rate."
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
    "Grey points are tracks the clustering left out of every cluster. Pointing at a track fills the "
    "lines under the map with what it is and marks it with an arrow above it, so that the points around "
    "it stay in view. Click a point to see its track. "
    "Scroll to zoom, drag to pan, and double-click to zoom back out. Click an entry in the legend to hide "
    "or show its tracks. A click away from every track takes the selection off, and so does Escape, "
    "which leaves the steps to go back through the tracks selected so far. A star marks the selected "
    "track, and rings mark the tracks the galleries below the map show, which the top of the legend "
    "names. Both stand on the map while a track is selected. The name a cluster has been "
    "given stands over the middle of its points, and the "
    f"{NAMES_TITLE} entry in the legend takes every name off the map."
)
VIEW_HELP = (
    "Whether the tracks are drawn as points or listed a row each. The table holds the tracks the map "
    "holds, and a search in the sidebar lists every track it names out of the whole corpus, so a "
    "recording's name there lists that recording's tracks whichever of them the sidebar leaves off the "
    "map. Ticking a row selects its track, the way a click on a point does."
)
TABLE_HELP = (
    "Every track the map holds, a row each, and every track a search names while there is one in the "
    "sidebar, so a recording's name there lists the tracks of that recording. Click a column heading to "
    "sort by it, and tick the box at the start of a row to select its track."
)
STRAIGHTNESS_HELP = "How straight the track's path is, from 0 for a path that doubles back to 1 for a line."
PEAK_DEVIATION_HELP = "How far the track's brightest pixel stood from the background, in standard deviations."
CACHED_TABLE_HELP = "Whether the track's recording is on disk, which is a track that plays without a fetch."
VALIDATED_TABLE_HELP = "Whether someone has watched the track and stands by the name it is under."
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
    "arrow on a panel to jump to its track, which selects it the way a click on its point does. Ctrl and "
    "a click take a panel into the picked tracks or out of them again, and Shift and a click take in the "
    "run from the first picked panel to the clicked one, which Apply to under the galleries then gives a "
    "name to. A picked panel is boxed, and the video stays with the panel picked first. The "
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
SAMPLE_ORDER_HELP = (
    "What the panels stand in order of. Distance puts the tracks nearest the selected track first, and "
    "the tracks nearest the middle of the cluster while none is selected, which are the ones most of the "
    "cluster is like. Video cached puts the tracks whose recording is on disk first, the ones that play "
    "without waiting for a fetch, and leaves them nearest first within each kind."
)
NEIGHBOUR_ORDER_HELP = (
    "What the panels stand in order of. Distance puts the track nearest the selected one first. Video "
    "cached puts the tracks whose recording is on disk first, the ones that play without waiting for a "
    "fetch. Cluster label gathers the tracks of each name together and leaves the unnamed ones last. "
    "Either of the last two leaves the tracks nearest first within each group."
)
LABEL_HELP = (
    "What this cluster holds, in a word of your own, such as insect or plane. The names given so far are "
    "offered in the list, which narrows to what is typed into it, and anything else typed there is a new "
    f"name. The name is kept against the cluster's tracks in {LABELS_PATH.name}, which is what a training "
    "set is built from. It is kept in small letters with single spaces between its words and every word "
    'in the singular, so that "Street Lights" and "streetLight" come to the one name. A name within a '
    "letter or two of one already in use is put to you before it is written. Naming the cluster again "
    "moves its tracks to the new name, and clearing the box takes them out of the one they are under. "
    "The selected track takes the name whichever group is named, and any other track reassigned to "
    "another cluster on its own stays where it was put."
)
SAVE_LABEL_HELP = "Keep this name against every track the name is being given to."
NAMING_HELP = (
    "Which tracks the name is given to. Cluster gives it to every track of the cluster the gallery above "
    "draws, wherever those tracks lie on the map. Neighbours gives it to the selected track and the "
    "tracks nearest it, which the Nearest tracks gallery draws, whichever cluster each of them is in, and "
    "is how a name is given to a neighbourhood the clustering cut in two. Picked gives it to the tracks "
    "picked out of the galleries with Ctrl and a click, and to nothing else, which is how a name is "
    "given to the tracks of a cluster that hold one thing while the rest hold another. The selected "
    "track takes the name of its cluster and of its neighbourhood, and is among the picked tracks only "
    "where it was picked. Naming a cluster leaves alone the other tracks put under a name of their own, "
    "and naming the nearest or the picked tracks takes all of them."
)
PICKING_TEXT = "Ctrl and a click on the panels of a gallery pick the tracks to name."
SIMILAR_TEXT = (
    "A name holds the tracks given it and nothing more, so two spellings of one thing keep their tracks "
    "apart. Both names are kept if that is what you meant."
)
REPLACING_TEXT = "Changing the name takes every one of those tracks out of the name they are under."
TAGS_HELP = (
    "What this one track holds, in words of your own, as many as are true of it at once. They start as "
    "the name the track's cluster is under, and are changed where the track is not what the rest of its "
    f"cluster is. They are kept against the track alone in {TRACK_LABELS_PATH.name}, in the same form as "
    "a cluster label, and are what a training set reads for this track. The tags given so far are offered "
    "in the list, a tag of your own is typed into it, and emptying the box takes the track back to its "
    "cluster's name."
)
SAVE_TAGS_HELP = (
    "Keep these tags against this track alone. Tagging a track is someone looking at it and saying what "
    "it is, so the track is marked validated at the same time."
)
TAGS_FILTER_HELP = (
    "Leave on the map only the tracks holding every tag chosen here, so two tags give the tracks that are "
    "both things rather than either."
)
SIMILAR_TAGS_TITLE = "Similar tags"
SIMILAR_TAGS_TEXT = (
    "A tag a letter or two from one already in use is most often that tag mistyped, which would hold the "
    "same tracks apart under two spellings."
)
TOGETHER_HELP = (
    "The tracks of this same recording that were running while the selected one was, the longest overlap "
    "first. Two objects in the sky at once are two tracks that nothing else on the page puts together, "
    "since they are alike in neither shape nor place. A square rings them on the map."
)
REFERENCE_HELP = (
    "Where this track is, for someone without this repository: the recording in the archive, and the "
    "seconds of it the track ran over. The seconds come from the recording's own frame rate where the "
    "video is on disk, and from a nominal rate otherwise, which the line says."
)
TUNING_HELP = (
    "Cut this stretch of the recording out and open it on the recordings page, where the settings can be "
    f"moved and the detector run again. The cut runs {MARGIN_SECONDS:.0f} seconds either side of the track, "
    "so what the detector missed beside it is in the clip as well. The clips are kept out of the way of the "
    "example set, and the same stretch is cut only once."
)
TUNING_UNSOURCED_TEXT = "The recording is not on disk, so there is nothing to cut. Fetch it below first."
EXPORT_HELP = (
    "Every tagged track as a row: its recording, its tags, the seconds it ran over, and the link to the "
    "recording in the archive."
)
REFERENCES_NAME = "track-references.csv"
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
class Group:
    """The tracks a name is being given to: the whole group, the ones the name
    is written against, the name they stand under now, and how many of them are
    under it."""

    keys: tuple[str, ...]
    moving: tuple[str, ...]
    given: str
    under: int


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
class Tagging:
    """What a press of Save on the tags is to write: the track they are given
    to, and the box they were given in."""

    key: str
    box: str


@dataclass(frozen=True, slots=True)
class SimilarTags:
    """Tags that were given and lie within a letter or two of names already in
    use, held until the person says which of the two they meant."""

    tagging: Tagging
    names: tuple[str, ...]
    swaps: tuple[tuple[str, str], ...]
    """Each tag as it was typed, beside the name in use it is close to."""


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
    with st.sidebar:
        wanted = str(st.text_input("Search", key=SEARCH_KEY, on_change=_search, help=SEARCH_HELP))
        if wanted.strip():
            st.caption(f"{len(searched(tracks, wanted=wanted))} of {len(tracks)} tracks matched")
        colour = st.segmented_control("Colour", options=COLOUR_CHOICES, default="Cluster", help=COLOUR_HELP)
        st.segmented_control(
            "Drawing", options=DRAWING_CHOICES, default=PATH_DRAWING, key=DRAWING_KEY, help=DRAWING_HELP
        )
        chosen_names = st.multiselect("Cluster labels", options=sorted(tracks[NAME_COLUMN].unique()), help=NAMES_HELP)
        chosen_tags = st.multiselect(
            "Tags", options=sorted(_track_labels(_track_labels_stamp())), key=TAGS_FILTER_KEY, help=TAGS_FILTER_HELP
        )
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
    if chosen_tags:
        shown = shown[shown["key"].isin(tagged(_track_labels(_track_labels_stamp()), names=chosen_tags))]
    shown = shown[shown["label"].isin(folders)] if folders else shown
    shown = by_validation(shown, choice=str(validation or ANY_VALIDATION))
    picked = _picked(shown, tracks=tracks, key=_history().standing)
    playing = _playing(tracks, picked=picked)
    cluster = _gallery_cluster(str(chosen_cluster), picked=picked)
    sample = shown.iloc[0:0] if cluster is None else _sample(shown, cluster=cluster, selected=picked)
    nearest = shown.iloc[0:0] if picked is None else _nearest(shown, track=picked, radius=float(radius))
    together = (
        shown.iloc[0:0]
        if picked is None
        else overlapping(shown, ranges=_stretches(PATHS_PATH.stat().st_mtime, clip=str(picked["clip"])), track=picked)
    )
    chosen = shown.iloc[0:0] if picked is None else shown[shown["key"] == picked["key"]]
    rings = [
        Ring(name=SAMPLE_TITLE, marker=SAMPLE_MARKER, tracks=sample),
        Ring(name=NEAREST_TITLE, marker=NEAREST_MARKER, tracks=nearest),
        Ring(name=TOGETHER_TITLE, marker=TOGETHER_MARKER, tracks=together),
        Ring(name=SELECTED_TITLE, marker=SELECTED_MARKER, tracks=chosen),
    ]

    map_column, track_column = st.columns([3, 2])
    with map_column:
        heading, view = st.columns(TRACKS_ROW, vertical_alignment="bottom")
        drawn_as = str(
            view.segmented_control("View", options=VIEW_CHOICES, default=MAP_VIEW, key=VIEW_KEY, help=VIEW_HELP)
            or MAP_VIEW
        )
        listed = drawn_as == TABLE_VIEW
        heading.subheader("Tracks", help=TABLE_HELP if listed else MAP_HELP)
        rows = searched(tracks, wanted=wanted) if listed and wanted.strip() else shown
        st.caption(f"{len(rows)} of {len(tracks)} tracks")
        if listed:
            _table(rows)
        else:
            dimmed = _uncached(shown) if cached else frozenset()
            figure = _scatter(
                shown,
                colour=str(colour or "Cluster"),
                rings=rings,
                dimmed=dimmed,
                names=cluster_names(shown, labels=_cluster_labels(_labels_stamp())),
            )
            track_map_chart(figure, key=MAP_KEY, details=DETAIL_LINES, on_click=_map_clicked, on_clear=_map_cleared)
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
            _together(together, track=picked, playing=playing)

    with track_column:
        _steps()
        if picked is None:
            st.caption("No track selected.")
        else:
            _selected(picked, points=_track_points(PATHS_PATH.stat().st_mtime, key=str(picked["key"])))
        _videos(picked=picked, playing=playing)

    similar = st.session_state.pop(SIMILAR_KEY, None)
    replacing = st.session_state.pop(REPLACING_KEY, None)
    similar_tags = st.session_state.pop(SIMILAR_TAGS_KEY, None)
    if similar is not None:
        _similar_name(similar)
    elif replacing is not None:
        _replacing_name(replacing)
    elif similar_tags is not None:
        _similar_tags(similar_tags)


def _steps() -> None:
    """The step back and forward through the tracks selected so far."""
    history = _history()
    back, forward, _ = st.columns([1, 1, 3])
    back.button("Back", on_click=_step, args=(-1,), disabled=not history.behind, help=BACK_HELP, width="stretch")
    forward.button("Forward", on_click=_step, args=(1,), disabled=not history.ahead, help=FORWARD_HELP, width="stretch")


def _table(tracks: pd.DataFrame) -> None:
    """The tracks as a row each, in place of the map, one of them selected by a
    click on its row.

    The keys are kept beside the table, because a click reports the row
    it landed on and the table itself holds no key.
    """
    st.session_state[TABLE_ROWS_KEY] = tracks["key"].tolist()
    st.dataframe(
        table_rows(tracks),
        hide_index=True,
        width="stretch",
        height=MAP_HEIGHT,
        on_select=_table_clicked,
        selection_mode="single-row",
        column_config={
            "Recording": st.column_config.TextColumn(width=RECORDING_WIDTH),
            "Tags": st.column_config.TextColumn(width=TAGS_WIDTH),
            "Straightness": st.column_config.NumberColumn(format="%.2f", help=STRAIGHTNESS_HELP),
            "Peak deviation": st.column_config.NumberColumn(format="%.1f", help=PEAK_DEVIATION_HELP),
            "Video cached": st.column_config.CheckboxColumn(help=CACHED_TABLE_HELP),
            "Validated": st.column_config.CheckboxColumn(help=VALIDATED_TABLE_HELP),
        },
        key=TABLE_KEY,
    )


def table_rows(tracks: pd.DataFrame) -> pd.DataFrame:
    """The tracks as the table lists them, a row each under the names the rest
    of the page gives them."""
    return tracks[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)


def _table_clicked() -> None:
    """Select the track whose row was clicked, which is what a click on its
    point on the map does."""
    picked = list(st.session_state[TABLE_KEY]["selection"]["rows"])
    keys = list(st.session_state.get(TABLE_ROWS_KEY, []))
    if picked and picked[0] < len(keys):
        _select(str(keys[picked[0]]))


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


def _together_clicked() -> None:
    _gallery_clicked(TOGETHER_GALLERY_KEY)


def _together_jumped() -> None:
    _gallery_jumped(TOGETHER_GALLERY_KEY)


def _gallery_clicked(component: str) -> None:
    """Pick the clicked track out of the gallery and play its video, leaving
    the selected track where it is, so that a cluster can be gone through video
    by video.

    A click on its own picks the one panel. Ctrl and Shift pick more of
    them, and the video stays with the first of the picked tracks, which
    is the one a plain click left.
    """
    clicked = _reported(component, event="clicked")
    if not clicked:
        return

    reach, _moment, key = clicked.split(" ", 2)
    picked = picked_tracks(
        _picked_keys(), key=key, reach=reach, order=list(st.session_state.get(_order_key(component), []))
    )
    st.session_state[PICKED_KEY] = picked
    if picked:
        st.session_state[PLAYING_KEY] = picked[0]


def picked_tracks(picked: Sequence[str], *, key: str, reach: str, order: Sequence[str]) -> tuple[str, ...]:
    """The tracks picked out of the galleries once this panel is clicked.

    A click on its own leaves the one panel picked. Ctrl takes the panel
    in beside the tracks already picked, or out again where it is one of
    them. Shift picks the run from the first picked track to the clicked
    one, as the gallery holding them stands, which is how a stretch of a
    gallery is picked at once.

    The track picked first stays first, because its video is the one
    playing. A run is picked out of one gallery, so a shift click while
    the first picked track is in another gallery picks the clicked panel
    alone.
    """
    if reach == ADD:
        return tuple(held for held in picked if held != key) if key in picked else (*picked, key)

    anchor = picked[0] if picked else ""
    if reach != RANGE or anchor not in order or key not in order:
        return (key,)

    first, last = order.index(anchor), order.index(key)
    run = order[min(first, last) : max(first, last) + 1]
    return tuple(run if first <= last else reversed(run))


def _picked_keys() -> tuple[str, ...]:
    """The tracks picked out of the galleries, in the order they were
    picked."""
    return tuple(st.session_state.get(PICKED_KEY, ()))


def _order_key(component: str) -> str:
    """Where a gallery keeps the tracks it drew, in the order it drew them,
    which is what a shift click reads the run it picks off."""
    return f"{component}:order"


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
    clicked. The tracks picked out of the galleries go with the track
    they were picked around, because the galleries are drawn afresh for
    the track stood on now.
    """
    st.session_state[HISTORY_KEY] = visited(_history(), key=key)
    st.session_state.pop(PLAYING_KEY, None)
    st.session_state.pop(PICKED_KEY, None)


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


def _sample(tracks: pd.DataFrame, *, cluster: str, selected: pd.Series | None) -> pd.DataFrame:
    """A random sample of the cluster's tracks, drawn anew on Shuffle and
    nearest first.

    The sample is measured from the selected track, and from the middle
    of the cluster while no track is selected, where the tracks nearest
    the middle are the ones most of the cluster is like.
    """
    members = tracks[tracks["cluster"] == cluster]
    drawn = members.sample(n=min(GALLERY_SIZE, len(members)), random_state=st.session_state.get(SHUFFLE_KEY, 0))
    middle = members[["x", "y"]].median()
    return _by_distance(drawn, from_track=middle if selected is None else selected)


def _nearest(tracks: pd.DataFrame, *, track: pd.Series, radius: float) -> pd.DataFrame:
    """The tracks nearest the selected one on the map, in any cluster and
    within the radius, nearest first."""
    others = tracks[tracks["key"] != track["key"]]
    distance = np.hypot(others["x"] - track["x"], others["y"] - track["y"])
    return others.loc[distance[distance <= radius].nsmallest(GALLERY_SIZE).index]


def stretches(paths: pd.DataFrame, *, clip: str) -> pd.DataFrame:
    """The first and last frame of every track of one recording, by key."""
    held = paths[paths["clip"] == clip]
    return held.groupby("key")["frame_number"].agg(first="min", last="max")


def overlapping(tracks: pd.DataFrame, *, ranges: pd.DataFrame, track: pd.Series) -> pd.DataFrame:
    """The tracks of the same recording that were running while this one was,
    the longest overlap first.

    Two objects crossing the sky at once are two tracks of one
    recording, and nothing on the map puts them together: they are alike
    in neither shape nor place. What they share is the stretch of frames
    they ran over.
    """
    key = str(track["key"])
    if key not in ranges.index:
        return tracks.iloc[0:0].assign(overlap=[])

    first, last = int(ranges.loc[key, "first"]), int(ranges.loc[key, "last"])
    others = tracks[(tracks["clip"] == track["clip"]) & (tracks["key"] != key)]
    held = others[others["key"].isin(ranges.index)]

    starts = held["key"].map(ranges["first"])
    ends = held["key"].map(ranges["last"])
    overlap = np.minimum(ends, last) - np.maximum(starts, first) + 1
    shared = held.assign(overlap=overlap)
    return shared[shared["overlap"] > 0].nlargest(GALLERY_SIZE, "overlap")


def _by_distance(tracks: pd.DataFrame, *, from_track: pd.Series) -> pd.DataFrame:
    distance = np.hypot(tracks["x"] - from_track["x"], tracks["y"] - from_track["y"])
    return tracks.loc[distance.sort_values(kind="stable").index]


def in_order(tracks: pd.DataFrame, *, order: str) -> pd.DataFrame:
    """A gallery's tracks in the order picked for it, out of the tracks as they
    come in, which is nearest first.

    Either of the other two orders gathers the tracks of one kind
    together and leaves them nearest first within that kind. The tracks
    whose recording is on disk come first, because those are the ones
    that play without waiting for a fetch, and the names come in the
    order they are spelled, which leaves the unnamed tracks last.
    """
    if order == BY_CACHED:
        return tracks.sort_values(CACHED_COLUMN, ascending=False, kind="stable")
    if order == BY_CLUSTER_NAME:
        return tracks.sort_values(NAME_COLUMN, kind="stable")
    return tracks


def _uncached(tracks: pd.DataFrame) -> frozenset[str]:
    """The tracks whose recording is on neither the sift's shelf nor the page's
    own, which the map draws faintly."""
    return frozenset(tracks.loc[~tracks[CACHED_COLUMN], "key"])


def labelled(tracks: pd.DataFrame, *, clusters: dict[str, list[str]], own: dict[str, list[str]]) -> pd.DataFrame:
    """The tracks with the name of the cluster each one is in beside it, and
    the tags the track itself stands under.

    A track whose cluster has no name yet, and a track the clustering
    left out of every cluster, stand under one name of their own, so
    that the map can be coloured and filtered by the name without those
    tracks falling off it. A track stands under its cluster's name until
    it is tagged, and under its own tags from then on, which is what the
    page shows wherever it names the track.
    """
    cluster_name = tracks["key"].map(_by_key(clusters)).fillna(UNNAMED)
    return tracks.assign(
        **{
            NAME_COLUMN: cluster_name,
            TAGS_COLUMN: tracks["key"].map(_tag_text(own)).fillna(cluster_name),
        }
    )


def _tag_text(labels: dict[str, list[str]]) -> dict[str, str]:
    """The tags of each tagged track, one after another as the page shows
    them."""
    held: dict[str, list[str]] = {}
    for name, keys in labels.items():
        for key in keys:
            held.setdefault(key, []).append(name)
    return {key: ", ".join(sorted(names)) for key, names in held.items()}


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
) -> dict[str, Any]:
    """The figure the plot is handed: the tracks as points, the marks over
    them, and the layout they are drawn in.

    The points are the same on every run of the page until the tracks
    shown, what colours them or which of them are faint changes, so they
    are kept under a mark of those things and built again only when the
    mark moves. The marks over them follow the selected track and are
    built every run, which costs nothing beside the points.
    """
    held = _point_traces(_map_mark(tracks, colour=colour, dimmed=dimmed), _tracks=tracks, _dimmed=dimmed, colour=colour)
    return {"data": [*held, *ring_traces(rings, names=names)], "layout": map_layout(colour)}


def _map_mark(tracks: pd.DataFrame, *, colour: str, dimmed: frozenset[str]) -> tuple[Any, ...]:
    """What the points of the map are built from, as one value to keep them
    under.

    The tracks are marked by how many they are and by a hash of their
    keys, so that any filter of the sidebar, and any filter added later,
    moves the mark by dropping rows. The files every point carries a
    value from are marked by when they were written, because a name
    saved on this page changes what a point holds while the tracks
    themselves stay as they are. A value taken from a file that is not
    marked here would go stale.
    """
    return (
        colour,
        bool(dimmed),
        MAP_PATH.stat().st_mtime,
        _labels_stamp(),
        _track_labels_stamp(),
        _videos_stamp(),
        len(tracks),
        int(pd.util.hash_pandas_object(tracks["key"], index=False).sum()),
    )


@st.cache_data(show_spinner=False, max_entries=FIGURE_CACHE_ENTRIES)
def _point_traces(
    mark: tuple[Any, ...], *, _tracks: pd.DataFrame, _dimmed: frozenset[str], colour: str
) -> list[dict[str, Any]]:
    """The traces of points kept under the mark of what they were built from.

    The tracks and the faint ones are passed under a leading underscore,
    which is how a cache is told to leave an argument out of the key,
    because hashing a frame of tens of thousands of rows costs more than
    the mark that stands for it.
    """
    return point_traces(_tracks, colour=colour, dimmed=_dimmed)


def point_traces(tracks: pd.DataFrame, *, colour: str, dimmed: frozenset[str]) -> list[dict[str, Any]]:
    """The tracks as points drawn by the graphics card, one trace per colour,
    so the legend names each colour and a click on it hides or shows those
    tracks.

    Drawing on the card keeps zooming and panning smooth over thousands
    of points. Each trace's uid keeps it hidden or shown as the legend
    left it.

    A trace is written as the plot reads it. Building it as a figure
    first and writing that out costs a second of its own on a map this
    size, and the values a figure writes are the ones written here.
    """
    column = COLOUR_COLUMNS[colour]
    return [
        {
            "type": "scattergl",
            "x": _places(members["x"]),
            "y": _places(members["y"]),
            "mode": "markers",
            "name": str(name),
            "uid": f"{column}:{name}",
            "marker": {"color": _point_colours(members, colour=shade, dimmed=dimmed), "size": 6, "opacity": 0.7},
            "customdata": _point_details(members),
            "hoverinfo": "none",
        }
        for name, shade, members in _by_colour(tracks, column=column)
    ]


def _by_colour(tracks: pd.DataFrame, *, column: str) -> list[tuple[str, str, pd.DataFrame]]:
    """Each colour of the map, with the tracks drawn in it."""
    return [(name, shade, tracks[tracks[column] == name]) for name, shade in _colours(tracks, column=column).items()]


def _point_details(members: pd.DataFrame) -> list[list[Any]]:
    """What each point carries for the panel under the map, a row per point.

    A number is cut to the digits the panel shows it to, because the
    whole of a stored number crosses to the browser and only those
    digits are ever drawn.
    """
    columns = [_column_values(members, column=detail.column, digits=detail.digits) for detail in DETAILS]
    return [list(row) for row in zip(members["key"].tolist(), *columns)]


def _column_values(members: pd.DataFrame, *, column: str, digits: int | None) -> list[Any]:
    held = members[column]
    return held.tolist() if digits is None else held.astype("float64").round(digits).tolist()


def _places(places: pd.Series) -> list[float]:
    """Where the points sit, to the digits the plot can draw."""
    return places.astype("float64").round(PLACE_DIGITS).tolist()


def ring_traces(rings: list[Ring], *, names: pd.DataFrame) -> list[dict[str, Any]]:
    """The marks over the selected track and the tracks the galleries show,
    and the name of each named cluster over the middle of its points.

    The marks take no hover or click, so a click on a ringed track
    selects the track under the mark. They stand at the top of the
    legend, which is what says what each mark on the map means, and the
    colours the map is drawn in run to as many entries as there are
    clusters. The names are drawn by the browser, which puts them over
    the points the card draws.
    """
    traces: list[dict[str, Any]] = [
        {
            "type": "scattergl",
            "x": _places(ring.tracks["x"]),
            "y": _places(ring.tracks["y"]),
            "mode": "markers",
            "name": ring.name,
            "uid": ring.name,
            "legendrank": place,
            "marker": ring.marker,
            "hoverinfo": "skip",
        }
        for place, ring in enumerate(rings)
        if not ring.tracks.empty
    ]
    if not names.empty:
        traces.append(
            {
                "type": "scatter",
                "x": _places(names["x"]),
                "y": _places(names["y"]),
                "mode": "text",
                "name": NAMES_TITLE,
                "uid": NAMES_TITLE,
                "legendrank": len(rings),
                "text": names["name"].tolist(),
                "textfont": NAMES_FONT,
                "hoverinfo": "skip",
            }
        )
    return traces


@st.cache_data(show_spinner=False, max_entries=len(COLOUR_CHOICES))
def map_layout(colour: str) -> dict[str, Any]:
    """The layout the map is drawn in, which only the legend's title changes.

    The fixed UI revision keeps the zoom when the plot is handed the
    next figure. A point is taken as pointed at within a few pixels of
    it, so the panel under the map and the arrow over the point both
    follow the point the pointer is on rather than one lying near it.

    It is written out through a figure, because the theme it names
    stands for a page of settings that plotly expands.
    """
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        height=MAP_HEIGHT,
        dragmode="pan",
        hovermode="closest",
        hoverdistance=POINT_REACH,
        uirevision="track-map",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"title": {"text": colour}},
        xaxis={"showgrid": True, "gridcolor": GRID_COLOR, "zeroline": False},
        yaxis={"showgrid": True, "gridcolor": GRID_COLOR, "zeroline": False},
    )
    return dict(json.loads(figure.to_json())["layout"])


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
    _track_tagging(track)
    _reference(track, points=points)
    st.altair_chart(frame_view(points))
    st.altair_chart(close_up(points))
    st.altair_chart(light_curve(points))
    st.altair_chart(rhythm_chart(points, signal=charted_signal(_drawing())))


def _track_tagging(track: pd.Series) -> None:
    """The tags this one track stands under, the box that changes them, and
    whether someone has confirmed the track.

    The tags start as the cluster's name, so a track that is what the
    rest of its cluster is needs no tag of its own. What they start at
    is part of the box's key, because a box holds whatever it was left
    at and would otherwise go on offering a start the track has moved on
    from.
    """
    key = str(track["key"])
    given = tags_of(_track_labels(_track_labels_stamp()), key=key)
    cluster_name = str(track[NAME_COLUMN])
    start = given or ([] if cluster_name == UNNAMED else [cluster_name])
    box = f"{TAGS_KEY}:{key}:{','.join(start)}"

    chooser, save, tick, _rest = st.columns(TRACK_NAME_ROW, vertical_alignment="bottom")
    chooser.multiselect(
        "Tags",
        options=_known_names(),
        default=start,
        key=box,
        accept_new_options=True,
        placeholder=TAGS_PLACEHOLDER,
        help=TAGS_HELP,
    )
    save.button(
        "Save",
        key=f"{SAVE_TAGS_KEY}:{key}",
        on_click=_save_tags,
        args=(Tagging(key=key, box=box),),
        help=SAVE_TAGS_HELP,
        width="stretch",
    )
    box = f"{VALIDATED_KEY}:{key}"
    _seeded(box, value=bool(track[VALIDATED_COLUMN]))
    tick.checkbox("Validated", key=box, on_change=_validate, args=(key,), help=VALIDATED_HELP)
    _reassigning(track)


def _reference(track: pd.Series, *, points: pd.DataFrame) -> None:
    """Where this track is for someone without the corpus, and the same for
    every tagged track at once.

    A recording no ledger names has no link, which is the case for the
    five recordings the example set was built from by hand.
    """
    held = _track_reference(track, points=points)
    if held is None:
        st.caption(f"No ledger names {track['recording']}, so there is no link to it.")
        return

    st.caption("Reference", help=REFERENCE_HELP)
    st.code(reference_line(held), language=None, wrap_lines=True, width="content")
    _tuning(track, held=held)
    if not held.measured_rate:
        st.caption(f"{track['recording']} is not on disk, so the seconds stand on {NOMINAL_RATE:.0f} frames a second.")
    st.download_button(
        "Tagged tracks",
        data=_tagged_references(
            _track_labels_stamp(), MAP_PATH.stat().st_mtime, PATHS_PATH.stat().st_mtime, _ledger_stamp()
        ),
        file_name=REFERENCES_NAME,
        mime="text/csv",
        help=EXPORT_HELP,
    )


def _tuning(track: pd.Series, *, held: Reference) -> None:
    """The button that cuts this track's stretch out and opens it on the
    recordings page, where the settings can be moved.

    What is worth tuning is usually what the detector missed beside the
    track, which is why the stretch goes over rather than the track.
    """
    recording = _source(str(track["recording"]))
    if recording is None or recording.path is None:
        st.caption(TUNING_UNSOURCED_TEXT)
        return

    st.button(
        "Tune on this stretch",
        key=f"{TUNE_KEY}:{track['key']}",
        on_click=_tune,
        args=(recording.path, held),
        help=TUNING_HELP,
    )


def _tune(video: Path, held: Reference) -> None:
    """Cut the stretch, name it after what the track stands under, and take
    the person to it."""
    begin = max(held.begin_s - MARGIN_SECONDS, 0.0)
    end = held.end_s + MARGIN_SECONDS
    cut = cut_for_tuning(video, begin_s=begin, end_s=end, root=TUNING_DIR)
    note_label(
        TUNING_DIR,
        name=cut.name,
        label=", ".join(held.tags) or f"track {held.track_id}",
        begin_s=held.begin_s - begin,
        end_s=held.end_s - begin,
    )

    st.session_state[TUNING_PICK_KEY] = cut.name
    page = st.session_state.get(RECORDINGS_PAGE_KEY)
    if page is not None:
        st.switch_page(page)


def _track_reference(track: pd.Series, *, points: pd.DataFrame) -> Reference | None:
    """The reference to one track, and nothing where no ledger names its
    recording."""
    recording = str(track["recording"])
    entry = _ledger_entries(_ledger_stamp()).get(recording)
    if entry is None:
        return None

    frames = points["frame_number"]
    return reference(
        recording=recording,
        track_id=int(track["track_id"]),
        url=str(entry.get("url", "")),
        first_frame=int(frames.min()),
        last_frame=int(frames.max()),
        frames_per_second=_frame_rate(recording),
        tags=tags_of(_track_labels(_track_labels_stamp()), key=str(track["key"])),
    )


def _frame_rate(recording: str) -> float:
    """The rate the recording runs at, and 0 while the video is not on disk to
    be read."""
    for folder in (VIDEOS_DIR, FETCHED_DIR):
        video = folder / recording
        if video.is_file():
            return _probe_video(video).frames_per_second
    return 0.0


@st.cache_data(show_spinner=False)
def _tagged_references(tags_stamp: float, map_stamp: float, paths_stamp: float, ledger_stamp: tuple[float, ...]) -> str:
    """Every tagged track as a row of a CSV, built again whenever the tags, the
    map or the ledgers change.

    A tagged track the map no longer holds is left out, which is what a
    track dropped by a later analysis run looks like.

    The stamps are those files' modification times, and are what the
    cache is keyed on, which is why they are passed although the body
    never reads them.
    """
    labels = _track_labels(tags_stamp)
    entries = _ledger_entries(ledger_stamp)
    recordings = _map_frame(map_stamp).set_index("key")["recording"]
    stretches = _path_frame(paths_stamp).groupby("key")["frame_number"].agg(["min", "max"])

    references = []
    for key in sorted({key for keys in labels.values() for key in keys}):
        if key not in stretches.index or key not in recordings.index:
            continue
        recording = str(recordings[key])
        references.append(
            reference(
                recording=recording,
                track_id=int(key.rsplit("/", 1)[-1]),
                url=str(entries.get(recording, {}).get("url", "")),
                first_frame=int(stretches.loc[key, "min"]),
                last_frame=int(stretches.loc[key, "max"]),
                frames_per_second=_frame_rate(recording),
                tags=tags_of(labels, key=key),
            )
        )
    return references_csv(references)


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


def _save_tags(tagging: Tagging) -> None:
    """Keep the tags given in the box, unless one of them is close enough to a
    name already in use to be that name mistyped, which is put to the person
    first."""
    wanted = [canonical_label(str(name)) for name in st.session_state[tagging.box] or []]
    names = tuple(dict.fromkeys(name for name in wanted if name))
    known = _known_names()
    swaps = tuple((name, near_label(name, known=known)) for name in names if near_label(name, known=known))
    if swaps:
        st.session_state[SIMILAR_TAGS_KEY] = SimilarTags(tagging=tagging, names=names, swaps=swaps)
        return

    _keep_tags(tagging, names=names)


def _keep_tags(tagging: Tagging, *, names: tuple[str, ...]) -> None:
    """Put this track under these tags and no others, and confirm it.

    Tagging a track is someone looking at it and saying what it is,
    which is what confirming it says. Taking every tag off says nothing,
    and leaves the track as it was.
    """
    write_tags(TRACK_LABELS_PATH, read_labels(TRACK_LABELS_PATH), key=tagging.key, names=names)
    if names:
        _confirm(tagging.key, confirmed=True)


@st.dialog(SIMILAR_TAGS_TITLE)
def _similar_tags(similar: SimilarTags) -> None:
    """What to do about tags that are close to names already in use."""
    for name, near in similar.swaps:
        st.write(f"**{name}** is close to **{near}**, which is already in use.")
    st.caption(SIMILAR_TAGS_TEXT)

    use, keep = st.columns(2)
    swapped = dict(similar.swaps)
    if use.button("Use the names in use", type="primary", width="stretch"):
        _keep_tags(similar.tagging, names=tuple(dict.fromkeys(swapped.get(name, name) for name in similar.names)))
        st.rerun()
    if keep.button("Keep them as typed", width="stretch"):
        _keep_tags(similar.tagging, names=similar.names)
        st.rerun()


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


def _videos(*, picked: pd.Series | None, playing: pd.Series | None) -> None:
    """The video of the selected track, and under it the video of a track a
    gallery panel was clicked on to hold against it.

    One clip is built at a time, so the second waits while the first is
    being built. With no track selected the clicked one takes the first
    player, because there is nothing to hold it against.
    """
    first = picked if picked is not None else playing
    if first is None:
        return

    building = _video(first, player=VIDEO_PLAYER, may_build=True)
    compared = _compared(playing, picked=picked)
    if compared is not None:
        _video(compared, player=COMPARISON_PLAYER, may_build=not building)


def _compared(playing: pd.Series | None, *, picked: pd.Series | None) -> pd.Series | None:
    """The track held against the selected one, which is the track a gallery
    panel was clicked on while another track is selected."""
    if playing is None or picked is None or str(playing["key"]) == str(picked["key"]):
        return None
    return playing


def _video(track: pd.Series, *, player: str, may_build: bool) -> bool:
    """The video of one track under a line naming it, and whether a build was
    asked for."""
    title, note = VIDEO_TITLES[player]
    st.subheader(title, help=note)
    st.caption(f"Track {int(track['track_id'])} in {track['clip']}")
    key = str(track["key"])
    return _clip(
        key,
        recording=str(track["recording"]),
        stored=_stored_track(_track_points(PATHS_PATH.stat().st_mtime, key=key)),
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
        clips = _clips(source.path, track=stored, details=_probe_video(source.path))
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
            f"{(free - archived.size_bytes) / GIGABYTE:.1f} GB free on the disk, under the "
            f"{MIN_FREE_BYTES / GIGABYTE:.0f} GB this page keeps free. Make room and try again."
        )
        if source.link:
            st.link_button("Open on Drive", source.link)
        return False

    return archived.size_bytes <= AUTO_FETCH_BYTES or st.button(
        "Fetch video",
        key=f"{FETCH_KEY}:{player}",
        help=f"This {megabytes:.0f} MB recording is not on disk. Fetching it takes about "
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
        details = probe(video)
        clips = _clips(video, track=track, details=details)
        if not clips.built:
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


def _clips(video: Path, *, track: StoredTrack, details: VideoProbe) -> TrackClips:
    stretch = stretch_around(track, frames_per_second=details.frames_per_second, frame_count=details.frame_count)
    return track_clip_paths(video, track=track, stretch=stretch, settings=config().settings, output_dir=CLIPS_DIR)


def _probe_video(video: Path) -> VideoProbe:
    """What the recording's container says of itself, kept while the file
    stays as it is.

    Opening a recording to ask costs a tenth of a second, and the page
    asks again on every run for as long as one track stays selected. A
    fetch from the archive writes a recording where a part file stood,
    so the answer is kept against the file's size and the time it was
    written rather than against its name alone.
    """
    held = video.stat()
    return _probed(str(video), stamp=(held.st_size, held.st_mtime))


@st.cache_data(show_spinner=False, max_entries=PANEL_CACHE_ENTRIES)
def _probed(video: str, *, stamp: tuple[int, float]) -> VideoProbe:
    return probe(Path(video))


def _gallery(tracks: pd.DataFrame, *, cluster: str | None, sample: pd.DataFrame, playing: pd.Series | None) -> None:
    """The random sample of the cluster, framed in the colour that rings it on
    the map."""
    if cluster is None:
        st.caption("Select a track, or choose a cluster for the gallery in the sidebar.")
        return

    heading, sorting, shuffle = st.columns(GALLERY_ROW, vertical_alignment="bottom")
    heading.subheader(SAMPLE_TITLE, help=GALLERY_HELP)
    order = str(
        sorting.segmented_control(
            "Sort by", options=SAMPLE_ORDERS, default=BY_DISTANCE, key=SAMPLE_ORDER_KEY, help=SAMPLE_ORDER_HELP
        )
        or BY_DISTANCE
    )
    shuffle.button("Shuffle", on_click=_shuffle, help=SHUFFLE_HELP)
    members = int((tracks["cluster"] == cluster).sum())
    st.caption(f"{_cluster_title(cluster)} · {len(sample)} of {members} tracks")
    shown = in_order(sample, order=order)
    captions = [f"{place}. folder {folder}" for place, folder in enumerate(shown["label"], start=1)]
    _panels(
        shown,
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
    the cluster holds. The nearest tracks all take it, because they were
    picked by hand and each one stands under the name of whichever
    cluster it came from.

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
    members = _naming_members(tracks, cluster=cluster, nearest=nearest, given_to=given_to)
    group = named_group(labels, members=members, selected=selected, given_to=given_to)
    box = f"{given_to}:{cluster}" if given_to == CLUSTER_TRACKS else f"{given_to}:{selected}"
    naming = Naming(path=LABELS_PATH, keys=group.moving, box=f"{LABEL_KEY}:{box}", confirms=False)

    _name_box(chooser, label="Cluster label", given=group.given, naming=naming, help=LABEL_HELP)
    st.caption(_naming_caption(group, given_to=given_to))
    save.button(
        "Save",
        key=f"{SAVE_KEY}:{cluster}",
        on_click=_save_name,
        args=(naming,),
        help=SAVE_LABEL_HELP,
        width="stretch",
    )


def _naming_members(tracks: pd.DataFrame, *, cluster: str, nearest: pd.DataFrame, given_to: str) -> list[str]:
    """The tracks the chosen group is drawn from."""
    if given_to == NEIGHBOUR_TRACKS:
        return [str(key) for key in nearest["key"]]
    if given_to == PICKED_TRACKS:
        return list(_picked_keys())
    return [str(key) for key in tracks.loc[tracks["cluster"] == cluster, "key"]]


def named_group(labels: dict[str, list[str]], *, members: list[str], selected: str, given_to: str) -> Group:
    """The tracks a name is given to, out of the group it is being given to.

    The selected track takes the name of its cluster and of its
    neighbourhood, because it is the track the person is looking at
    while they name it. It stands at the middle of its neighbourhood,
    which is drawn as the tracks nearest it and holds none of itself.
    The tracks picked out of the galleries are named as they were
    picked, and the selected track is among them only where it was
    picked as well.

    Any other track of a cluster that stands under a name of its own is
    left alone, so that naming the cluster again does not take a
    reassignment back. The nearest tracks and the picked tracks all take
    the name, because they were picked by hand.
    """
    keys = [selected, *members] if given_to == NEIGHBOUR_TRACKS and selected else list(members)
    given = label_of(labels, keys=keys)
    held = _by_key(labels)
    whole = given_to in (NEIGHBOUR_TRACKS, PICKED_TRACKS)
    moving = keys if whole else [key for key in keys if key == selected or held.get(key, "") in ("", given)]
    return Group(
        keys=tuple(keys),
        moving=tuple(moving),
        given=given,
        under=sum(1 for key in keys if held.get(key) == given),
    )


def _naming_caption(group: Group, *, given_to: str) -> str:
    """How many of the tracks the name would be given to are under it
    already."""
    if not group.keys:
        return PICKING_TEXT if given_to == PICKED_TRACKS else "Select a track to name the ones nearest it."
    if not group.given:
        return f"None of these {len(group.keys)} tracks has a name yet."
    return f"{group.under} of {len(group.keys)} tracks under {group.given}"


def _neighbours(nearest: pd.DataFrame, *, radius: float, playing: pd.Series | None) -> None:
    """The tracks nearest the selected one, each captioned with the cluster it
    is in and framed in the colour that rings them on the map."""
    heading, sorting = st.columns(NEIGHBOUR_ROW, vertical_alignment="bottom")
    heading.subheader(NEAREST_TITLE, help=NEIGHBOURS_HELP)
    order = str(
        sorting.segmented_control(
            "Sort by", options=GALLERY_ORDERS, default=BY_DISTANCE, key=NEAREST_ORDER_KEY, help=NEIGHBOUR_ORDER_HELP
        )
        or BY_DISTANCE
    )
    st.caption(f"{len(nearest)} tracks within {radius:.2f}")
    shown = in_order(nearest, order=order)
    captions = [
        f"{place}. {_cluster_caption(cluster)} · {name}"
        for place, (cluster, name) in enumerate(zip(shown["cluster"], shown[NAME_COLUMN]), start=1)
    ]
    _panels(
        shown,
        captions=captions,
        frame=NEAREST_COLOR,
        playing=playing,
        key=NEAREST_GALLERY_KEY,
        on_click=_nearest_clicked,
        on_jump=_nearest_jumped,
    )


def _together(together: pd.DataFrame, *, track: pd.Series, playing: pd.Series | None) -> None:
    """The tracks of this recording that were running while the selected one
    was, each captioned with how much of its life ran alongside."""
    st.subheader(TOGETHER_TITLE, help=TOGETHER_HELP)
    st.caption(f"{len(together)} tracks in {track['clip']} overlap track {int(track['track_id'])} in time")
    captions = [
        f"{place}. track {int(number)} · {int(frames)} frames together"
        for place, (number, frames) in enumerate(zip(together["track_id"], together["overlap"]), start=1)
    ]
    _panels(
        together,
        captions=captions,
        frame=TOGETHER_COLOR,
        playing=playing,
        key=TOGETHER_GALLERY_KEY,
        on_click=_together_clicked,
        on_jump=_together_jumped,
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
    when it is clicked and selecting the track from its Jump button.

    The order the panels stand in is kept in the session, because a
    shift click picks the run between two of them and the click is
    answered before the page is drawn again.
    """
    played = "" if playing is None else str(playing["key"])
    picked = set(_picked_keys())
    st.session_state[_order_key(key)] = [str(track) for track in chosen["key"]]
    entries = tuple(
        GalleryEntry(
            key=str(track),
            caption=caption,
            cached=bool(cached),
            validated=bool(validated),
            playing=str(track) == played,
            picked=str(track) in picked,
        )
        for track, cached, validated, caption in zip(
            chosen["key"], chosen[CACHED_COLUMN], chosen[VALIDATED_COLUMN], captions
        )
    )
    track_gallery(
        _gallery_markup(PATHS_PATH.stat().st_mtime, entries, frame=frame, drawing=_drawing()),
        key=key,
        on_click=on_click,
        on_jump=on_jump,
    )


@st.cache_data(show_spinner=False, max_entries=PANEL_CACHE_ENTRIES)
def _gallery_markup(stamp: float, entries: tuple[GalleryEntry, ...], *, frame: str, drawing: str) -> str:
    """The panels of these tracks under these captions, kept so that a gallery
    that comes out the same on the next click is not drawn again."""
    if drawing == PATH_DRAWING:
        return gallery_html(_path_frame(stamp), entries=entries, frame=frame)
    return spectra_html(_path_frame(stamp), entries=entries, frame=frame, signal=drawing.lower())


def _drawing() -> str:
    """What the galleries and the selected track's rhythm chart draw."""
    return str(st.session_state.get(DRAWING_KEY) or PATH_DRAWING)


def charted_signal(drawing: str) -> str:
    """The signal the selected track's rhythm chart draws.

    The chart stands under the light curve whatever the galleries are
    drawing, so a track's rhythm can be read against what its blob was
    doing at the time without giving up the gallery of paths. It follows
    the chosen signal wherever one is chosen.
    """
    return SIGNALS[0] if drawing == PATH_DRAWING else drawing.lower()


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
        str(track[TAGS_COLUMN]),
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


@st.cache_resource(show_spinner=False, max_entries=1)
def _path_frame(stamp: float) -> pd.DataFrame:
    """Every mapped track frame by frame, keyed the way the map is, read again
    whenever the file changes.

    The frame is handed out as it stands and never copied, because it is
    millions of rows and a copy of it on every run of the page costs more
    than everything the page draws. Whatever takes it holds it read
    only: take a slice, and write into the slice.

    One is kept, so that a map written again while the page is open does
    not leave the one it replaces in memory.
    """
    frame = pq.read_table(PATHS_PATH).to_pandas()
    frame["key"] = _keys(frame)
    return frame


@st.cache_data(show_spinner=False, max_entries=PANEL_CACHE_ENTRIES)
def _track_points(stamp: float, *, key: str) -> pd.DataFrame:
    """One track's path, frame by frame, out of the frames of every track.

    Finding a track means reading every row of the corpus, so the tracks
    looked at lately are kept.

    The stamp is the paths file's modification time, and is what the
    cache is keyed on, which is why it is passed although the body never
    reads it.
    """
    return _points(_path_frame(stamp), key=key)


@st.cache_data(show_spinner=False, max_entries=PANEL_CACHE_ENTRIES)
def _stretches(stamp: float, *, clip: str) -> pd.DataFrame:
    """The frames every track of one recording ran over, read again whenever
    the paths file changes.

    One recording at a time, because the whole corpus is millions of
    frames and a selected track only ever asks about its own.

    The stamp is the file's modification time, and is what the cache is
    keyed on, which is why it is passed although the body never reads
    it.
    """
    return stretches(_path_frame(stamp), clip=clip)


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
