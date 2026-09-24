"""A gallery of drawn tracks that selects the track a panel is clicked on.

The panels are one block of markup rather than an element each, because
Streamlit renders the whole page for every element it holds and a
gallery holds dozens. That block takes no clicks of its own, so it is
handed to a component of the page's own, which reports the track behind
the panel the click landed in.

Each panel carries its track under a data-track attribute, and the arrow
that jumps to the track carries it under a data-jump attribute, which is
what a click on either is read from and what the markup is written with.

A click on a panel reports how far it reaches as well as the track: one
panel on its own, one panel taken in or out while Ctrl is held, or the
run up to it while Shift is held. The moment of the click goes with it,
because two clicks on one panel report the same track and the page is
only handed what has changed.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

HTML = '<div class="track-gallery"></div>'

JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const gallery = parentElement.querySelector(".track-gallery");
  gallery.innerHTML = data.markup;
  gallery.report = setTriggerValue;
  if (gallery.dataset.listening) return;
  gallery.dataset.listening = "yes";
  gallery.addEventListener("click", (event) => {
    const jump = event.target.closest("[data-jump]");
    if (jump) {
      gallery.report("jumped", jump.getAttribute("data-jump"));
      return;
    }
    const panel = event.target.closest("[data-track]");
    if (!panel) return;
    const reach = event.shiftKey ? "range" : event.ctrlKey || event.metaKey ? "add" : "one";
    gallery.report("clicked", [reach, Date.now(), panel.getAttribute("data-track")].join(" "));
  });
}
"""

ONE, ADD, RANGE = "one", "add", "range"
"""How far a click on a panel reaches: the one panel, the panel in or out of
the tracks picked so far, or the run up to it."""

_component = st.components.v2.component("track_gallery", html=HTML, js=JS, isolate_styles=False)


def track_gallery(markup: str, *, key: str, on_click: Callable[[], None], on_jump: Callable[[], None]) -> None:
    """Draw this gallery markup, and call on_click when a panel is clicked and
    on_jump when the arrow on one is pressed.

    A jump reports the track's key. A click reports how far it reaches,
    the moment it was made and the track's key, in that order with a
    space between, so that the key keeps whatever it holds.

    The key names the one gallery the page keeps in that place, so it
    has to stay the same from one run of the page to the next.
    """
    _component(key=key, data={"markup": markup}, on_clicked_change=on_click, on_jumped_change=on_jump)
