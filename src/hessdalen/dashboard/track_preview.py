"""Tracks drawn from their stored paths alone, with no video behind them.

A track's path is a few hundred numbers, and drawing it takes no time,
where drawing it on its video first has to pass every frame of the
recording ahead of it. These charts are what the track map shows at
once, and a gallery of them is how a cluster is judged at a glance.

A path is drawn through the blob's centre on each frame, which is what
the descriptors were computed from. The brightest pixel, which the
tracker matched on, hops about inside a large blob.
"""

from __future__ import annotations

import base64
import html
from collections.abc import Sequence
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale

FRAME_WIDTH_PIXELS = 420
CLOSE_UP_PIXELS = 240
PANEL_PIXELS = 120
PANEL_GAP_PIXELS = 8
PANEL_DOT_RADIUS = 1.8
CLOSE_UP_MARGIN = 0.55
"""Half the side of a fitted track's box, in units of the track's larger
extent, so a path fills most of its panel and never touches the edge."""

CAPTION_FONT_PIXELS = 10
FRAME_PIXELS = 2
PLAYING_FRAME_PIXELS = 3
CACHED_MARK = "🎥"
CACHED_TITLE = "The recording is on disk"
VALIDATED_MARK = "✓"
VALIDATED_TITLE = "Confirmed by a person"
VALIDATED_COLOUR = "#2e9e4f"
PLAYING_MARK = "▶"
PLAYING_TITLE = "Its video is the one playing"
PLAYING_COLOUR = "#7b3fe4"
"""Colour the panel of the track whose video plays is framed in.

It stands apart from the colours the galleries frame their panels in and
from the star over the selected track, because a track playing and a
track selected are two different things.
"""

TIME = alt.Color("phase:Q", scale=alt.Scale(scheme="viridis"), legend=None)
"""Colour along a path, from its first frame in dark blue to its last in
yellow, so direction and pace read off the spacing of the dots."""


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    """One panel of a gallery: the track it draws, what stands over it,
    whether the track's recording is on disk, whether the track itself is
    confirmed, and whether its video is the one playing."""

    key: str
    caption: str
    cached: bool
    validated: bool
    playing: bool


def frame_view(points: pd.DataFrame) -> alt.Chart:
    """The track where it was, in the whole frame."""
    width, height = int(points["frame_width"].iloc[0]), int(points["frame_height"].iloc[0])
    drawn = _with_phase(points)
    x = alt.X("centre_x:Q", scale=alt.Scale(domain=[0, width]), axis=None)
    y = alt.Y("centre_y:Q", scale=alt.Scale(domain=[height, 0]), axis=None)
    return (
        alt.layer(
            alt.Chart(drawn).mark_line(color="#9a9a9a", strokeWidth=1).encode(x=x, y=y, order="frame_number:Q"),
            alt.Chart(drawn).mark_circle(size=14).encode(x=x, y=y, color=TIME),
        )
        .properties(width=FRAME_WIDTH_PIXELS, height=int(FRAME_WIDTH_PIXELS * height / width), title="In the frame")
        .configure_view(stroke="#cfcfcf")
    )


def close_up(points: pd.DataFrame) -> alt.Chart:
    """The track fitted to its own box, keeping its proportions, so its shape
    can be seen however small it was in the frame."""
    return (
        _fitted_layers(fitted(points))
        .properties(width=CLOSE_UP_PIXELS, height=CLOSE_UP_PIXELS, title="Close up")
        .configure_view(stroke="#cfcfcf")
    )


def light_curve(points: pd.DataFrame) -> alt.Chart:
    """How bright and how large the blob was on each frame the track was
    matched on."""
    drawn = points.melt(
        id_vars=["frame_number"], value_vars=["brightness", "pixel_count"], var_name="measure", value_name="value"
    )
    drawn["measure"] = drawn["measure"].map({"brightness": "Brightness", "pixel_count": "Pixels"})
    return (
        alt.Chart(drawn)
        .mark_line(point=alt.OverlayMarkDef(size=12))
        .encode(
            x=alt.X("frame_number:Q", title="Frame"),
            y=alt.Y("value:Q", title=None),
            row=alt.Row("measure:N", title=None),
        )
        .properties(width=FRAME_WIDTH_PIXELS, height=90)
        .resolve_scale(y="independent")
    )


def gallery_html(paths: pd.DataFrame, *, entries: Sequence[GalleryEntry], frame: str) -> str:
    """Each track of these entries fitted to a small panel under its caption,
    framed in the given colour, in the order of the entries and in rows that
    wrap to the width they are given.

    The panels are SVG images in one block of HTML, so a gallery is a
    single element on the page however many tracks it holds, and
    Streamlit renders the page once for it. Each SVG goes in as an image
    because Streamlit's HTML sanitiser removes SVG written into the
    page. Each panel carries the key of its track, which is what a click
    on the gallery is read from.
    """
    keys = [entry.key for entry in entries]
    drawn = fitted(paths[paths["key"].isin(keys)])
    by_key = {key: group.sort_values("frame_number") for key, group in drawn.groupby("key", sort=False)}
    figures = "".join(_panel_svg(by_key[entry.key], entry=entry, frame=frame) for entry in entries)
    return f'<div style="display:flex;flex-wrap:wrap;gap:{PANEL_GAP_PIXELS}px">{figures}</div>'


def _panel_svg(points: pd.DataFrame, *, entry: GalleryEntry, frame: str) -> str:
    """One fitted track, its path in grey and a dot on each frame coloured by
    how far along the track it is."""
    x = (points["u"].to_numpy() + CLOSE_UP_MARGIN) / (2 * CLOSE_UP_MARGIN) * PANEL_PIXELS
    y = (points["v"].to_numpy() + CLOSE_UP_MARGIN) / (2 * CLOSE_UP_MARGIN) * PANEL_PIXELS
    colours = sample_colorscale("Viridis", points["phase"].tolist())
    line = " ".join(f"{px:.1f},{py:.1f}" for px, py in zip(x, y))
    dots = "".join(
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{PANEL_DOT_RADIUS}" fill="{colour}"/>'
        for px, py, colour in zip(x, y, colours)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PANEL_PIXELS}" height="{PANEL_PIXELS}">'
        f'<polyline points="{line}" fill="none" stroke="#9a9a9a" stroke-width="1"/>{dots}</svg>'
    )
    source = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    caption = html.escape(entry.caption)
    thickness = PLAYING_FRAME_PIXELS if entry.playing else FRAME_PIXELS
    return (
        f'<figure data-track="{html.escape(entry.key)}" '
        f'style="margin:0;width:{PANEL_PIXELS}px;cursor:pointer">'
        f'<figcaption style="display:flex;gap:4px;font-size:{CAPTION_FONT_PIXELS}px">'
        f'<span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{caption}</span>'
        f'<span style="margin-left:auto;white-space:nowrap">{_marks(entry)}</span></figcaption>'
        f'<img src="{source}" width="{PANEL_PIXELS}" height="{PANEL_PIXELS}" alt="{caption}" '
        f'style="border:{thickness}px solid {PLAYING_COLOUR if entry.playing else frame}"></figure>'
    )


def _marks(entry: GalleryEntry) -> str:
    """What a panel carries beside its caption: a camera while its recording is
    on disk, a tick once someone has confirmed the track, and a play mark while
    its video is the one playing."""
    marks = []
    if entry.playing:
        marks.append(f'<span title="{PLAYING_TITLE}" style="color:{PLAYING_COLOUR}">{PLAYING_MARK}</span>')
    if entry.cached:
        marks.append(f'<span title="{CACHED_TITLE}">{CACHED_MARK}</span>')
    if entry.validated:
        marks.append(f'<span title="{VALIDATED_TITLE}" style="color:{VALIDATED_COLOUR}">{VALIDATED_MARK}</span>')
    return "".join(marks)


def fitted(paths: pd.DataFrame) -> pd.DataFrame:
    """Each track's path moved to its own centre and scaled by its larger
    extent, as u and v, with the phase along the track added.

    Scaling both axes by the same extent keeps a straight line straight
    and a round loop round.
    """
    grouped = paths.groupby("key", sort=False)
    centre_x = (grouped["centre_x"].transform("min") + grouped["centre_x"].transform("max")) / 2.0
    centre_y = (grouped["centre_y"].transform("min") + grouped["centre_y"].transform("max")) / 2.0
    extent = np.maximum(
        grouped["centre_x"].transform("max") - grouped["centre_x"].transform("min"),
        grouped["centre_y"].transform("max") - grouped["centre_y"].transform("min"),
    ).clip(lower=1.0)

    drawn = _with_phase(paths)
    drawn["u"] = (paths["centre_x"] - centre_x) / extent
    drawn["v"] = (paths["centre_y"] - centre_y) / extent
    return drawn


def _fitted_layers(drawn: pd.DataFrame) -> alt.LayerChart:
    u = alt.X("u:Q", scale=alt.Scale(domain=[-CLOSE_UP_MARGIN, CLOSE_UP_MARGIN]), axis=None)
    v = alt.Y("v:Q", scale=alt.Scale(domain=[CLOSE_UP_MARGIN, -CLOSE_UP_MARGIN]), axis=None)
    return alt.LayerChart(
        layer=[
            alt.Chart()
            .mark_line(color="#9a9a9a", strokeWidth=1)
            .encode(x=u, y=v, order="frame_number:Q", detail="key:N"),
            alt.Chart().mark_circle(size=10).encode(x=u, y=v, color=TIME),
        ],
        data=drawn,
    )


def _with_phase(paths: pd.DataFrame) -> pd.DataFrame:
    """The paths with how far along its own track each frame sits, from 0 at
    the first frame to 1 at the last."""
    grouped = paths.groupby("key", sort=False)["frame_number"]
    first, last = grouped.transform("min"), grouped.transform("max")
    drawn = paths.copy()
    drawn["phase"] = (paths["frame_number"] - first) / np.maximum(last - first, 1)
    return drawn
