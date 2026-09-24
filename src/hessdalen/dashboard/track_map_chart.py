"""The track map, drawn by plotly.js in a component of the page's own.

Streamlit replaces its own Plotly chart whenever the figure changes, and
the replacement starts zoomed out with nothing selected. The track map's
figure changes on every click, because it rings the tracks the galleries
show. This component keeps one plot for as long as the page is open and
hands it each new figure, so the zoom, the pan and the legend's hidden
entries all stay as they were.

What a point holds is written into a panel under the plot rather than
into a label over the point, because a label over the point covers the
points around it, which are what a point is read against. The panel
stands its values in fixed columns, so a value keeps its place from one
point to the next and a value too long for its column is cut off. The
point the panel holds is marked with an arrow above it. Both are drawn
by the browser from the point's own data, so moving the pointer over the
map costs no run of the page.

The pointer over the plot is the page's own, and a point under it turns
the pointer into a hand, because a point is there to be clicked.

Plotly reports a click on a point and says nothing about a click on the
empty space between them, so the plot's own mouse presses are watched to
tell a click apart from a pan. The press is watched on the plot and the
release on the whole page, because plotly lays a cover over the page
while a press is held and the release lands on that cover. The plot is
left alone where the release comes after a move of more than a few
pixels, where a point sits under the pointer, or where plotly has
reported that point itself, which it does a few milliseconds after the
release.

The press and the keypress are watched once for the page, and find the
plot when they fire, so that a plot drawn again leaves no listener
behind holding the one before it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import streamlit as st
from plotly.offline import get_plotlyjs_version

MAP_HEIGHT = 620
PLOTLY_URL = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{get_plotlyjs_version()}/plotly.min.js"
"""The plotly.js release the installed plotly package writes its figures
for, so the figures it writes are the ones the plot reads."""
CONFIG = {"scrollZoom": True, "displaylogo": False, "responsive": True}

DETAIL_COLUMNS = 4
"""How many columns the panel under the plot stands its values in.

Every value keeps its column whatever is under the pointer, so nothing
moves sideways as the pointer goes from one track to the next. Four
columns leave a recording's name the room it takes.
"""

HTML = f"""
<style>
.track-map-frame {{ position: relative; }}
.track-map-frame .track-map .nsewdrag {{ cursor: default !important; }}
.track-map-pointer {{
  position: absolute;
  display: none;
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: 7px solid currentColor;
  opacity: 0.55;
  transform: translate(-50%, -100%);
  pointer-events: none;
}}
.track-map-details {{
  display: grid;
  grid-template-columns: repeat({DETAIL_COLUMNS}, minmax(0, 1fr));
  gap: 0.35rem 1.1rem;
  margin-top: 0.4rem;
  font-size: 0.8rem;
  line-height: 1.3;
}}
.track-map-field {{ min-width: 0; }}
.track-map-name,
.track-map-value {{ display: block; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }}
.track-map-name {{ font-size: 0.72rem; opacity: 0.55; }}
.track-map-value {{ font-variant-numeric: tabular-nums; }}
@media (max-width: 760px) {{
  .track-map-details {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
<div class="track-map-frame">
  <div class="track-map" style="width:100%;height:{MAP_HEIGHT}px"></div>
  <div class="track-map-pointer"></div>
</div>
<div class="track-map-details"></div>
"""

JS = """
function loadPlotly(url) {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (!window.trackMapPlotly) {
    window.trackMapPlotly = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = url;
      script.onload = () => resolve(window.Plotly);
      script.onerror = () => reject(new Error(`plotly.js did not load from ${url}`));
      document.head.appendChild(script);
    });
  }
  return window.trackMapPlotly;
}

const CLICK_SLOP = 4;
const CLICK_SETTLE = 60;
const POINT_GAP = 6;
const MISSING = "\\u2014";

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const plot = parentElement.querySelector(".track-map");
  plot.report = setTriggerValue;
  plot.pointer = parentElement.querySelector(".track-map-pointer");
  plot.panel = parentElement.querySelector(".track-map-details");
  plot.details = data.details;
  if (!plot.dataset.described) {
    plot.dataset.described = "yes";
    describe(plot, null);
  }
  loadPlotly(data.plotly).then((Plotly) => {
    const style = getComputedStyle(plot);
    const font = { ...(data.figure.layout.font || {}), color: style.color, family: style.fontFamily };
    const layout = { ...data.figure.layout, font };
    return Plotly.react(plot, data.figure.data, layout, data.config).then(() => {
      mark(plot);
      if (plot.dataset.listening) return;
      plot.dataset.listening = "yes";
      plot.on("plotly_hover", (event) => {
        const point = event.points.find((candidate) => candidate.customdata);
        if (!point) return;
        plot.marked = { x: point.x, y: point.y };
        describe(plot, point.customdata.slice(1));
        mark(plot);
        pointing(plot, true);
      });
      plot.on("plotly_unhover", () => pointing(plot, false));
      plot.on("plotly_relayout", () => mark(plot));
      plot.on("plotly_relayouting", () => mark(plot));
      plot.on("plotly_click", (event) => {
        const point = event.points.find((candidate) => candidate.customdata);
        if (point) {
          plot.dataset.onTrack = "yes";
          plot.report("clicked", point.customdata[0]);
        }
      });
      plot.addEventListener("mousedown", (event) => {
        const own = event.button === 0 && !event.target.closest(".modebar, .legend");
        plot.pressedAt = own ? { x: event.clientX, y: event.clientY } : null;
      }, true);
      if (!window.trackMapRelease) {
        window.trackMapRelease = (event) => {
          const map = document.querySelector(".track-map");
          if (!map || !map.report) return;
          const from = map.pressedAt;
          map.pressedAt = null;
          if (event.button !== 0 || !from) return;
          if (Math.hypot(event.clientX - from.x, event.clientY - from.y) > CLICK_SLOP) return;
          setTimeout(() => {
            if (map.dataset.onTrack === "yes") {
              map.dataset.onTrack = "";
              return;
            }
            if ((map._hoverdata || []).length) return;
            map.report("cleared", String(Date.now()));
          }, CLICK_SETTLE);
        };
        document.addEventListener("mouseup", window.trackMapRelease, true);
      }
      if (!window.trackMapEscape) {
        window.trackMapEscape = (event) => {
          if (event.key !== "Escape") return;
          if (event.target.closest("input, textarea, [role='combobox'], [role='dialog'], [contenteditable]")) return;
          const map = document.querySelector(".track-map");
          if (map && map.report) map.report("cleared", String(Date.now()));
        };
        document.addEventListener("keydown", window.trackMapEscape);
      }
    });
  });
}

function describe(plot, values) {
  const fields = plot.details.map((detail, place) => {
    const field = document.createElement("div");
    field.className = "track-map-field";
    if (detail.wide) field.style.gridColumn = "span 2";
    const name = document.createElement("span");
    name.className = "track-map-name";
    name.textContent = detail.label;
    const value = document.createElement("span");
    value.className = "track-map-value";
    value.textContent = written(values ? values[place] : null, detail.digits);
    field.append(name, value);
    return field;
  });
  plot.panel.replaceChildren(...fields);
}

function written(value, digits) {
  if (value === null || value === undefined || value === "") return MISSING;
  return digits === null ? String(value) : Number(value).toFixed(digits);
}

function mark(plot) {
  const full = plot._fullLayout;
  if (!plot.marked || !full) {
    plot.pointer.style.display = "none";
    return;
  }
  const across = full.xaxis;
  const up = full.yaxis;
  const x = across._offset + across.d2p(plot.marked.x);
  const y = up._offset + up.d2p(plot.marked.y);
  const inside =
    x >= across._offset && x <= across._offset + across._length &&
    y >= up._offset && y <= up._offset + up._length;
  plot.pointer.style.display = inside ? "block" : "none";
  plot.pointer.style.left = `${x}px`;
  plot.pointer.style.top = `${y - POINT_GAP}px`;
}

function pointing(plot, over) {
  const dragger = plot.querySelector(".nsewdrag");
  if (!dragger) return;
  if (over) dragger.style.setProperty("cursor", "pointer", "important");
  else dragger.style.removeProperty("cursor");
}
"""

_component = st.components.v2.component("track_map", html=HTML, js=JS, isolate_styles=False)


def track_map_chart(
    figure: Mapping[str, Any],
    *,
    key: str,
    details: Sequence[Mapping[str, Any]],
    on_click: Callable[[], None],
    on_clear: Callable[[], None],
) -> None:
    """Draw the figure in the map's one plot, call on_click when a track is
    clicked, with the track's key under "clicked" in the component's state, and
    call on_clear when the plot is clicked away from every track or Escape is
    pressed.

    The figure holds the traces under "data" and the layout under
    "layout", as plotly.js reads them, which is what the page hands over
    rather than building a figure of plotly's own to write out.

    Each detail names a line of the panel under the plot, the digits its
    number is shown to, with no digits where the value is shown as it
    stands, and whether the line takes two of the panel's columns. The
    values follow the track's key in a point's customdata, in the order
    the details are given in.

    A press that moves the pointer more than a few pixels is a pan and
    clears nothing. A press that lands on the modebar or the legend is
    that control's own. Escape is taken from the whole page, apart from
    the presses that a box, a drop-down or a dialog has to answer for
    itself.

    The key names the one plot the page keeps, so it has to stay the
    same from one run of the page to the next.
    """
    _component(
        key=key,
        data={
            "figure": dict(figure),
            "config": CONFIG,
            "plotly": PLOTLY_URL,
            "details": list(details),
        },
        on_clicked_change=on_click,
        on_cleared_change=on_clear,
    )
