"""The detector settings a run starts from, kept in a file.

Every value the dashboard's sidebar offers is read from
config/detector.toml, so the numbers a run starts from sit in one place
that can be edited by hand and written back from the page. The settings
that have no slider keep their values with the settings class they
belong to.

The file is read once and the reading is held, so a page that writes it
clears that hold and the next reader sees what was written.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import tomli_w

from hessdalen.dashboard.panels import PANEL_CHOICES, Panels
from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.detection import DetectionSettings
from hessdalen.processing.devices import DEVICES
from hessdalen.processing.movement import MovementSettings, TrackingSettings

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "detector.toml"

FRAME_HEIGHTS = (270, 540, 720, 1080, 1440, 2160)
"""Heights the dashboard offers to resize frames to before detection."""

HEADING = """\
# What the detector starts from, and what the dashboard's sidebar opens on.
# Save on the dashboard writes this file. Every value here has a slider or a
# control of its own on the page. The settings with no control, such as the
# closing size and the noise floor, stay with the settings they belong to.

"""


@dataclass(frozen=True, slots=True)
class Config:
    """What the dashboard opens on, and what a run starts from."""

    frame_height: int
    panels: Panels
    settings: MovementSettings


@cache
def config() -> Config:
    """The config as the file holds it."""
    return read_config(CONFIG_PATH)


def read_config(path: Path) -> Config:
    """The config a file holds.

    A value the file does not name, or names as something the detector
    cannot use, raises here rather than at the frame that first reads
    it.
    """
    held = tomllib.loads(path.read_text())
    detection = _table(held, "detection")
    background = _table(held, "background")
    tracking = _table(held, "tracking")

    return Config(
        frame_height=_one_of(held, "frame_height", FRAME_HEIGHTS),
        panels=_one_of(held, "panels", PANEL_CHOICES),
        settings=MovementSettings(
            background=BackgroundSettings(
                mean_alpha=float(background["mean_alpha"]),
                variance_alpha=float(background["variance_alpha"]),
            ),
            detection=DetectionSettings(
                foreground_sigma=float(detection["foreground_sigma"]),
                detection_sigma=float(detection["detection_sigma"]),
                min_pixels=int(detection["min_pixels"]),
            ),
            tracking=TrackingSettings(
                min_consecutive_frames=int(tracking["min_consecutive_frames"]),
                max_movement_ratio=float(tracking["max_movement_ratio"]),
                min_movement_ratio=float(tracking["min_movement_ratio"]),
                max_missed_frames=int(tracking["max_missed_frames"]),
                min_trajectory_span_ratio=float(tracking["min_trajectory_span_ratio"]),
            ),
            device=_one_of(held, "device", DEVICES),
        ),
    )


def write_config(held: Config, path: Path) -> None:
    """Write the config out, and let go of the reading held from before.

    The heading is written again each time, because what writes the
    tables carries no comment across and the file would otherwise stop
    saying what it is the first time the dashboard saved it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADING + tomli_w.dumps(_as_tables(held)))
    config.cache_clear()


def _as_tables(held: Config) -> dict[str, Any]:
    settings = held.settings
    return {
        "frame_height": held.frame_height,
        "panels": held.panels,
        "device": settings.device,
        "detection": {
            "foreground_sigma": settings.detection.foreground_sigma,
            "detection_sigma": settings.detection.detection_sigma,
            "min_pixels": settings.detection.min_pixels,
        },
        "background": {
            "mean_alpha": settings.background.mean_alpha,
            "variance_alpha": settings.background.variance_alpha,
        },
        "tracking": {
            "min_consecutive_frames": settings.tracking.min_consecutive_frames,
            "max_movement_ratio": settings.tracking.max_movement_ratio,
            "min_movement_ratio": settings.tracking.min_movement_ratio,
            "max_missed_frames": settings.tracking.max_missed_frames,
            "min_trajectory_span_ratio": settings.tracking.min_trajectory_span_ratio,
        },
    }


def _table(held: dict[str, Any], name: str) -> dict[str, Any]:
    table = held.get(name)
    if not isinstance(table, dict):
        raise ValueError(f"The config has no {name} table.")
    return table


def _one_of[Choice](held: dict[str, Any], name: str, choices: tuple[Choice, ...]) -> Choice:
    """The named value, which has to be one the detector knows."""
    value = held.get(name)
    if value not in choices:
        raise ValueError(f"The config gives {name} as {value!r}, which is not one of {list(choices)}.")
    return value  # type: ignore[return-value]
