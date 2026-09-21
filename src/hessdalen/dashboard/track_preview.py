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

import altair as alt
import numpy as np
import pandas as pd

FRAME_WIDTH_PIXELS = 420
CLOSE_UP_PIXELS = 240
PANEL_PIXELS = 130
GALLERY_COLUMNS = 6
CLOSE_UP_MARGIN = 0.55
"""Half the side of a fitted track's box, in units of the track's larger
extent, so a path fills most of its panel and never touches the edge."""

TIME = alt.Color("phase:Q", scale=alt.Scale(scheme="viridis"), legend=None)
"""Colour along a path, from its first frame in dark blue to its last in
yellow, so direction and pace read off the spacing of the dots."""


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


def gallery(paths: pd.DataFrame) -> alt.FacetChart:
    """Every given track in a panel of its own, fitted to the panel.

    The paths carry a panel column naming each one, and panels follow
    the order the paths arrive in.
    """
    drawn = fitted(paths)
    order = list(dict.fromkeys(drawn["panel"]))
    layered = _fitted_layers(drawn).properties(width=PANEL_PIXELS, height=PANEL_PIXELS)
    return layered.facet(
        facet=alt.Facet(
            "panel:N", sort=order, title=None, header=alt.Header(labelFontSize=10, labelLimit=PANEL_PIXELS)
        ),
        columns=GALLERY_COLUMNS,
    ).configure_view(stroke="#cfcfcf")


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
