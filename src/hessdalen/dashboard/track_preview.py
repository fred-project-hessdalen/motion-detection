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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import altair as alt
import cv2
import numpy as np
import pandas as pd
from plotly.colors import sample_colorscale

from hessdalen.analysis.spectra import (
    DYNAMIC_RANGE_DB,
    HIGHEST_RATE,
    LOWEST_RATE,
    Spectrogram,
    spectrogram,
    track_signals,
)

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

PICKED_OUTLINE_PIXELS = 2
PICKED_OFFSET_PIXELS = 2
"""How a picked panel is boxed, in pixels.

The box is drawn in the page's own text colour, which stands apart from
every colour a gallery frames its panels in and reads in either theme.
An outline is drawn outside the panel and moves nothing beside it.
"""

JUMP_MARK = "↗"
JUMP_TITLE = "Jump to this track"

UNMEASURED_COLOUR = "#8a8a8a"
"""What a rhythm drawing shows where the track was too short to read a
rate, which is most of the slow rates on most tracks."""

RHYTHM_HEIGHT_PIXELS = 130
"""How tall the drawn part of a rhythm chart stands.

The chart asks to be padded out to this rather than fitted into it.
Fitted, the axes and the title are taken out of the height first, and
under the fonts the page is drawn with there is nothing left over: the
chart comes out as a pair of axes around an empty strip, and only
opening it full screen gives it the room to appear.
"""

RHYTHM_COLUMNS = 200
"""Frames a rhythm chart draws at most, one cell each.

A chart draws a rectangle per cell, and the longest track of the corpus
runs 2680 frames.
"""

TIME = alt.Color("phase:Q", scale=alt.Scale(scheme="viridis"), legend=None)
"""Colour along a path, from its first frame in dark blue to its last in
yellow, so direction and pace read off the spacing of the dots."""


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    """One panel of a gallery: the track it draws, what stands over it,
    whether the track's recording is on disk, whether the track itself is
    confirmed, whether its video is the one playing, and whether it is among
    the tracks picked out of the galleries."""

    key: str
    caption: str
    cached: bool
    validated: bool
    playing: bool
    picked: bool


@dataclass(frozen=True, slots=True)
class Rhythm:
    """One track's rhythm and the frames it was read over.

    The frames are every frame from the track's first to its last, which
    is more than the track was reported on.
    """

    frame_number: np.ndarray
    image: Spectrogram


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


def rhythm_chart(points: pd.DataFrame, *, signal: str) -> alt.Chart:
    """How strongly the chosen signal repeated, by rate and by frame.

    Drawn on the frame axis the light curve above it uses, so that a
    band can be read against what the blob was doing at the time. The
    rates are cycles per frame, which is 25 times the same number in
    hertz on these recordings.
    """
    found = rhythm(points, signal=signal)
    return (
        alt.Chart(_cells(found))
        .mark_rect()
        .encode(
            x=alt.X("frame:Q", title="Frame", scale=alt.Scale(domain=list(_span(found.frame_number)), nice=False)),
            x2="frame_end:Q",
            y=alt.Y(
                "rate:Q",
                title="Cycles a frame",
                scale=alt.Scale(type="log", domain=[LOWEST_RATE, HIGHEST_RATE], nice=False),
            ),
            y2="rate_above:Q",
            color=alt.Color(
                "power:Q",
                scale=alt.Scale(scheme="viridis", domain=[0.0, DYNAMIC_RANGE_DB], clamp=True),
                legend=None,
            ),
        )
        .properties(
            width=FRAME_WIDTH_PIXELS,
            height=RHYTHM_HEIGHT_PIXELS,
            title=f"{signal.capitalize()} rhythm",
            autosize=alt.AutoSizeParams(type="pad", contains="padding"),
        )
        .configure_view(stroke="#cfcfcf", fill=UNMEASURED_COLOUR)
    )


def rhythm(points: pd.DataFrame, *, signal: str) -> Rhythm:
    """One track's chosen signal read for rhythm, over the frames it spans."""
    ordered = points.sort_values("frame_number")
    reach = float(max(int(ordered["frame_width"].iloc[0]), int(ordered["frame_height"].iloc[0])))
    signals = track_signals(
        frame_number=ordered["frame_number"].to_numpy(),
        centre_x=ordered["centre_x"].to_numpy(),
        centre_y=ordered["centre_y"].to_numpy(),
        brightness=ordered["brightness"].to_numpy(),
        pixel_count=ordered["pixel_count"].to_numpy(),
        reach=reach,
    )
    return Rhythm(frame_number=signals.frame_number, image=spectrogram(signals.values[signal]))


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
    return _block(_panel(_path_svg(by_key[entry.key]), entry=entry, frame=frame) for entry in entries)


def spectra_html(paths: pd.DataFrame, *, entries: Sequence[GalleryEntry], frame: str, signal: str) -> str:
    """Each track of these entries drawn as its rhythm in the chosen signal,
    under its caption and framed in the given colour.

    A panel stands where the same track's path stands in the other
    gallery and answers a click the same way, so a cluster can be looked
    through by what its tracks repeated at as readily as by where they
    went. The rates run the same way in every panel, the slowest at the
    bottom, whatever the length of the track.
    """
    keys = [entry.key for entry in entries]
    held = paths[paths["key"].isin(keys)]
    by_key = {key: group for key, group in held.groupby("key", sort=False)}
    return _block(
        _panel(_rhythm_png(rhythm(by_key[entry.key], signal=signal).image), entry=entry, frame=frame)
        for entry in entries
    )


def _path_svg(points: pd.DataFrame) -> str:
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
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _rhythm_png(image: Spectrogram) -> str:
    """One track's rhythm as a picture, the fastest rate at the top and the
    first frame at the left.

    Enlarged to the panel by repeating whole cells, so that a track of
    forty frames plainly shows forty of them.
    """
    shades = np.clip(image.power / DYNAMIC_RANGE_DB, 0.0, 1.0) * 255.0
    drawn = cv2.applyColorMap(shades.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    drawn[~image.measurable] = _bgr(UNMEASURED_COLOUR)
    panel = cv2.resize(np.flipud(drawn), (PANEL_PIXELS, PANEL_PIXELS), interpolation=cv2.INTER_NEAREST)
    _, encoded = cv2.imencode(".png", panel)
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


def _bgr(colour: str) -> tuple[int, int, int]:
    red, green, blue = (int(colour[at : at + 2], 16) for at in (1, 3, 5))
    return blue, green, red


def _block(figures: Iterable[str]) -> str:
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:{PANEL_GAP_PIXELS}px;user-select:none">{"".join(figures)}</div>'
    )


def _panel(source: str, *, entry: GalleryEntry, frame: str) -> str:
    """One panel of a gallery: its picture under its caption and marks, framed
    in the gallery's colour and boxed while it is picked."""
    caption = html.escape(entry.caption)
    thickness = PLAYING_FRAME_PIXELS if entry.playing else FRAME_PIXELS
    boxed = (
        f";outline:{PICKED_OUTLINE_PIXELS}px solid currentColor;outline-offset:{PICKED_OFFSET_PIXELS}px"
        if entry.picked
        else ""
    )
    return (
        f'<figure data-track="{html.escape(entry.key)}" '
        f'style="margin:0;width:{PANEL_PIXELS}px;cursor:pointer{boxed}">'
        f'<figcaption style="display:flex;gap:4px;font-size:{CAPTION_FONT_PIXELS}px">'
        f'<span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{caption}</span>'
        f'<span style="margin-left:auto;white-space:nowrap">{_marks(entry)}</span></figcaption>'
        f'<img src="{source}" width="{PANEL_PIXELS}" height="{PANEL_PIXELS}" alt="{caption}" '
        f'style="border:{thickness}px solid {PLAYING_COLOUR if entry.playing else frame}"></figure>'
    )


def _marks(entry: GalleryEntry) -> str:
    """What a panel carries beside its caption: a play mark while its video is
    the one playing, a camera while its recording is on disk, a tick once
    someone has confirmed the track, and the arrow that jumps to it."""
    marks = []
    if entry.playing:
        marks.append(f'<span title="{PLAYING_TITLE}" style="color:{PLAYING_COLOUR}">{PLAYING_MARK}</span>')
    if entry.cached:
        marks.append(f'<span title="{CACHED_TITLE}">{CACHED_MARK}</span>')
    if entry.validated:
        marks.append(f'<span title="{VALIDATED_TITLE}" style="color:{VALIDATED_COLOUR}">{VALIDATED_MARK}</span>')
    marks.append(
        f'<span data-jump="{html.escape(entry.key)}" title="{JUMP_TITLE}" '
        f'style="border:1px solid currentColor;border-radius:3px;padding:0 2px">{JUMP_MARK}</span>'
    )
    return "".join(marks)


def _cells(found: Rhythm) -> pd.DataFrame:
    """The rhythm as one row per rectangle drawn, leaving out the rates the
    track was too short to carry.

    Thinned along the frames so that a long track does not ask for a
    rectangle on every one of them.
    """
    step = max(1, found.frame_number.size // RHYTHM_COLUMNS)
    at = np.arange(0, found.frame_number.size, step)
    frames = found.frame_number[at]
    edges = _rate_edges(found.image.rates)

    rate_at, frame_at = np.meshgrid(np.arange(found.image.rates.size), np.arange(at.size), indexing="ij")
    drawn = pd.DataFrame(
        {
            "frame": frames[frame_at.ravel()],
            "frame_end": frames[frame_at.ravel()] + step,
            "rate": edges[:-1][rate_at.ravel()],
            "rate_above": edges[1:][rate_at.ravel()],
            "power": found.image.power[:, at].ravel(),
        }
    )
    return drawn[found.image.measurable[:, at].ravel()]


def _rate_edges(rates: np.ndarray) -> np.ndarray:
    """Where one rate's row of the drawing gives way to the next."""
    apart = np.sqrt(rates[1] / rates[0])
    return np.concatenate(([rates[0] / apart], np.sqrt(rates[:-1] * rates[1:]), [rates[-1] * apart]))


def _span(frame_number: np.ndarray) -> tuple[int, int]:
    return int(frame_number[0]), int(frame_number[-1]) + 1


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
