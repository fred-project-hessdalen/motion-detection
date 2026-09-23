"""The track map, drawn by plotly.js in a component of the page's own.

Streamlit replaces its own Plotly chart whenever the figure changes, and
the replacement starts zoomed out with nothing selected. The track map's
figure changes on every click, because it rings the tracks the galleries
show. This component keeps one plot for as long as the page is open and
hands it each new figure, so the zoom, the pan and the legend's hidden
entries all stay as they were.

Plotly reports a click on a point and says nothing about a click on the
empty space between them, so the plot's own mouse presses are watched to
tell a click apart from a pan.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import plotly.graph_objects as go
import streamlit as st
from plotly.offline import get_plotlyjs_version

MAP_HEIGHT = 620
PLOTLY_URL = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{get_plotlyjs_version()}/plotly.min.js"
"""The plotly.js release the installed plotly package writes its figures
for, so the figures it writes are the ones the plot reads."""
CONFIG = {"scrollZoom": True, "displaylogo": False, "responsive": True}

HTML = f'<div class="track-map" style="width:100%;height:{MAP_HEIGHT}px"></div>'

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

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const plot = parentElement.querySelector(".track-map");
  plot.report = setTriggerValue;
  loadPlotly(data.plotly).then((Plotly) => {
    const style = getComputedStyle(plot);
    const font = { ...(data.figure.layout.font || {}), color: style.color, family: style.fontFamily };
    const layout = { ...data.figure.layout, font };
    return Plotly.react(plot, data.figure.data, layout, data.config).then(() => {
      if (plot.dataset.listening) return;
      plot.dataset.listening = "yes";
      plot.on("plotly_click", (event) => {
        const point = event.points.find((candidate) => candidate.customdata);
        if (point) {
          plot.dataset.onTrack = "yes";
          plot.report("clicked", point.customdata[0]);
        }
      });
      let down = null;
      plot.addEventListener("mousedown", (event) => {
        down = event.button === 0 ? { x: event.clientX, y: event.clientY } : null;
      });
      plot.addEventListener("mouseup", (event) => {
        const from = down;
        down = null;
        if (event.button !== 0 || from === null) return;
        if (Math.hypot(event.clientX - from.x, event.clientY - from.y) > CLICK_SLOP) return;
        if (event.target.closest(".modebar, .legend")) return;
        setTimeout(() => {
          if (plot.dataset.onTrack === "yes") {
            plot.dataset.onTrack = "";
            return;
          }
          plot.report("cleared", String(Date.now()));
        }, 0);
      });
      document.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") return;
        if (event.target.closest("input, textarea, [role='combobox'], [role='dialog'], [contenteditable]")) return;
        plot.report("cleared", String(Date.now()));
      });
    });
  });
}
"""

_component = st.components.v2.component("track_map", html=HTML, js=JS, isolate_styles=False)


def track_map_chart(figure: go.Figure, *, key: str, on_click: Callable[[], None], on_clear: Callable[[], None]) -> None:
    """Draw the figure in the map's one plot, call on_click when a track is
    clicked, with the track's key under "clicked" in the component's state, and
    call on_clear when the plot is clicked away from every track or Escape is
    pressed.

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
        data={"figure": json.loads(figure.to_json()), "config": CONFIG, "plotly": PLOTLY_URL},
        on_clicked_change=on_click,
        on_cleared_change=on_clear,
    )
