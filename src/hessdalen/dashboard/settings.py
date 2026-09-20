"""What the detector settings may be set to, and how they travel as a file.

The dashboard offers one slider per setting, and a file written here
carries the position of every one of them. The ends a slider offers are
kept here as well, so a file written by hand is measured against the
same numbers the sliders draw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from hessdalen.config import config
from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.detection import DetectionSettings
from hessdalen.processing.devices import Device
from hessdalen.processing.movement import MovementSettings, TrackingSettings

FILE_NAME = "detector-settings.json"


@dataclass(frozen=True, slots=True)
class Setting:
    """The range one slider offers.

    A range written with whole ends counts something and cannot take a
    fraction, which is what a value read out of a file is brought to.
    """

    lowest: float
    highest: float

    @property
    def whole(self) -> bool:
        return isinstance(self.lowest, int) and isinstance(self.highest, int)

    def held(self, value: float) -> float:
        """This value brought inside the range the slider offers."""
        inside = min(self.highest, max(self.lowest, value))
        return int(inside) if self.whole else float(inside)


SETTINGS: dict[str, Setting] = {
    "foreground_sigma": Setting(1.0, 15.0),
    "detection_sigma": Setting(5.0, 40.0),
    "min_pixels": Setting(1, 50),
    "min_consecutive_frames": Setting(1, 20),
    "max_movement_ratio": Setting(0.002, 0.100),
    "min_movement_ratio": Setting(0.0000, 0.0100),
    "max_missed_frames": Setting(0, 30),
    "min_trajectory_span_ratio": Setting(0.000, 0.100),
    "mean_alpha": Setting(0.05, 1.00),
    "variance_alpha": Setting(0.005, 0.500),
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
    """The value every setting opens on, read from the config each time.

    Saving the page writes that file, so a later read has to see what
    was written rather than what the sliders were first drawn from.
    """
    return as_values(config().settings)


def as_values(settings: MovementSettings) -> dict[str, float]:
    """The settings under the names their sliders go by."""
    return {
        "foreground_sigma": settings.detection.foreground_sigma,
        "detection_sigma": settings.detection.detection_sigma,
        "min_pixels": settings.detection.min_pixels,
        "min_consecutive_frames": settings.tracking.min_consecutive_frames,
        "max_movement_ratio": settings.tracking.max_movement_ratio,
        "min_movement_ratio": settings.tracking.min_movement_ratio,
        "max_missed_frames": settings.tracking.max_missed_frames,
        "min_trajectory_span_ratio": settings.tracking.min_trajectory_span_ratio,
        "mean_alpha": settings.background.mean_alpha,
        "variance_alpha": settings.background.variance_alpha,
    }


def as_settings(values: dict[str, float], *, device: Device) -> MovementSettings:
    """The settings the sliders are standing at."""
    return MovementSettings(
        background=BackgroundSettings(
            mean_alpha=values["mean_alpha"],
            variance_alpha=values["variance_alpha"],
        ),
        detection=DetectionSettings(
            foreground_sigma=values["foreground_sigma"],
            detection_sigma=values["detection_sigma"],
            min_pixels=int(values["min_pixels"]),
        ),
        tracking=TrackingSettings(
            min_consecutive_frames=int(values["min_consecutive_frames"]),
            max_movement_ratio=values["max_movement_ratio"],
            min_movement_ratio=values["min_movement_ratio"],
            max_missed_frames=int(values["max_missed_frames"]),
            min_trajectory_span_ratio=values["min_trajectory_span_ratio"],
        ),
        device=device,
    )


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
