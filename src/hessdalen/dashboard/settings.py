"""What the detector settings may be set to, and how they travel as a file.

The dashboard offers one slider per setting, and a file written here
carries the position of every one of them. The ends a slider offers are
kept here as well, so a file written by hand is measured against the
same numbers the sliders draw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.detection import DetectionSettings
from hessdalen.processing.movement import TrackingSettings

FILE_NAME = "detector-settings.json"

BACKGROUND_DEFAULTS = BackgroundSettings()
DETECTION_DEFAULTS = DetectionSettings()
TRACKING_DEFAULTS = TrackingSettings()


@dataclass(frozen=True, slots=True)
class Setting:
    """What one setting may be set to, and what it opens on."""

    lowest: float
    highest: float
    default: float

    def held(self, value: float) -> float:
        """This value brought inside the range the slider offers."""
        return type(self.default)(min(self.highest, max(self.lowest, value)))


SETTINGS: dict[str, Setting] = {
    "foreground_sigma": Setting(1.0, 15.0, DETECTION_DEFAULTS.foreground_sigma),
    "detection_sigma": Setting(5.0, 40.0, DETECTION_DEFAULTS.detection_sigma),
    "min_pixels": Setting(1, 50, DETECTION_DEFAULTS.min_pixels),
    "min_consecutive_frames": Setting(1, 20, TRACKING_DEFAULTS.min_consecutive_frames),
    "max_movement_ratio": Setting(0.002, 0.100, TRACKING_DEFAULTS.max_movement_ratio),
    "min_movement_ratio": Setting(0.0000, 0.0100, TRACKING_DEFAULTS.min_movement_ratio),
    "max_missed_frames": Setting(0, 30, TRACKING_DEFAULTS.max_missed_frames),
    "min_trajectory_span_ratio": Setting(0.000, 0.100, TRACKING_DEFAULTS.min_trajectory_span_ratio),
    "mean_alpha": Setting(0.05, 1.00, BACKGROUND_DEFAULTS.mean_alpha),
    "variance_alpha": Setting(0.005, 0.500, BACKGROUND_DEFAULTS.variance_alpha),
}


@dataclass(frozen=True, slots=True)
class Reading:
    """What reading a settings file produced.

    A problem is the one thing wrong with a file that held nothing
    usable, and is empty when there was none.
    """

    settings: dict[str, float]
    held: int
    problem: str


def defaults() -> dict[str, float]:
    """The value every setting opens on."""
    return {key: setting.default for key, setting in SETTINGS.items()}


def as_file(values: dict[str, float]) -> bytes:
    """The settings as the contents of a file to keep."""
    return json.dumps({key: values[key] for key in SETTINGS}, indent=2, sort_keys=True).encode()


def from_file(raw: bytes) -> Reading:
    """The settings a file holds, and what there was to say about reading it.

    A setting the file does not name is left out, so the slider holding
    it stays where it stands. A value outside what its slider offers is
    brought to the nearest end, because a file may have been written by
    hand and a slider cannot be drawn standing outside its own ends.
    """
    try:
        holding = json.loads(raw)
    except json.JSONDecodeError:
        return Reading(settings={}, held=0, problem="That file does not hold JSON.")
    if not isinstance(holding, dict):
        return Reading(settings={}, held=0, problem="That file does not hold a set of settings.")

    named = {key: value for key, value in holding.items() if key in SETTINGS and _is_number(value)}
    if not named:
        return Reading(settings={}, held=0, problem="That file names none of the settings.")

    settings = {key: SETTINGS[key].held(value) for key, value in named.items()}
    held = sum(1 for key, value in named.items() if settings[key] != value)
    return Reading(settings=settings, held=held, problem="")


def _is_number(value: object) -> bool:
    """Whether a value read out of a file can sit on a slider.

    A boolean is an integer in Python and is not a setting.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)
