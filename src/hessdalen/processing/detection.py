"""The stage that turns a grayscale frame into the blobs worth tracking.

The detector's tracking stage consumes detections and does not care how
the pixels were measured, so this stage is the seam a different
implementation plugs into.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Protocol

import cv2
import numpy as np

from hessdalen.domain.models import BlobMeasurement
from hessdalen.processing.background import BackgroundModel, BackgroundSettings

Centroid = tuple[float, float]

FOREGROUND = 255
"""Value a mask carries where the pixel belongs to a blob."""

MEASURED = 128
"""Value the fill leaves on a blob whose pixels have been counted."""

FILLED = 1
"""Value the fill leaves in its own mask, which holds one blob at a time."""

SCALE_LADDER = (1.0, 1.25, 1.5625, 1.953125, 2.44140625, 3.0517578125)
"""Multipliers a frame's deviation may be divided by, each a quarter up."""


@dataclass(frozen=True, slots=True)
class Detection:
    centroid: Centroid
    blob: BlobMeasurement


@dataclass(frozen=True, slots=True)
class BlobAxes:
    major: float
    minor: float


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    """What a blob has to be before it is reported.

    The first three are set from the dashboard and come from the config,
    so they are asked of the caller. The ones below them have no control
    of their own and keep their values here.
    """

    foreground_sigma: float
    detection_sigma: float
    min_pixels: int
    close_size: int = 5
    foreground_budget: float = 1.5
    budget_rate: float = 0.05


class DetectionStage(Protocol):
    def detections(self, gray: np.ndarray) -> list[Detection]:
        """The blobs this frame holds, after folding it into the background."""
        ...

    def deviation_image(self) -> np.ndarray:
        """The deviation of the last frame, scaled for the debug video."""
        ...


class CpuDetectionStage:
    """Measures the frame with numpy and OpenCV on the host."""

    def __init__(
        self,
        *,
        background: BackgroundSettings,
        detection: DetectionSettings,
        timestamp_mask: np.ndarray | None,
    ) -> None:
        self.settings = detection
        self.timestamp_mask = timestamp_mask
        self._background = BackgroundModel(background)
        self._budget = ForegroundBudget(detection)
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (detection.close_size, detection.close_size))
        self._deviation = np.zeros((1, 1), dtype=np.float32)

    def detections(self, gray: np.ndarray) -> list[Detection]:
        deviation = self._background.deviation(gray)
        scale = self._budget.scale(partial(count_over, deviation))

        if scale != 1.0:
            deviation = np.divide(deviation, np.float32(scale), dtype=np.float32)
        self._deviation = deviation
        return self._extract_detections(gray, self._deviation)

    def deviation_image(self) -> np.ndarray:
        """Scale the deviation for the debug video, with the detection
        threshold at full white."""
        scaled = self._deviation * (255.0 / self.settings.detection_sigma)
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    def _extract_detections(self, gray: np.ndarray, deviation: np.ndarray) -> list[Detection]:
        peaks = cv2.findNonZero(self._peak_mask(deviation))
        if peaks is None:
            return []

        rows = peaks[:, 0, 1]
        columns = peaks[:, 0, 0]
        return blobs_around_peaks(
            gray=gray,
            foreground=self._foreground_mask(deviation),
            rows=rows,
            columns=columns,
            deviations=deviation[rows, columns],
            min_pixels=self.settings.min_pixels,
        )

    def _peak_mask(self, deviation: np.ndarray) -> np.ndarray:
        return threshold_mask(deviation, self.settings.detection_sigma, cv2.CMP_GE)

    def _foreground_mask(self, deviation: np.ndarray) -> np.ndarray:
        mask = threshold_mask(deviation, self.settings.foreground_sigma, cv2.CMP_GT)
        if self.timestamp_mask is not None:
            mask = cv2.bitwise_and(mask, self.timestamp_mask)
        if self.settings.close_size <= 1:
            return mask
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)


class ForegroundBudget:
    """How far a frame's deviation is divided down to hold its foreground count
    near the level the recording keeps returning to.

    An H.264 encoder repeats more than half the picture verbatim between
    keyframes, and a pixel it repeated carries a residual of exactly
    zero, which no threshold reaches. Every keyframe hands all of those
    pixels a residual again, so the count of pixels over the threshold
    jumps several times over however the deviation is scaled. Dividing
    the frame down until the count is back near that level is the one
    thing that holds it, because the count is what the keyframe changes.

    The level is tracked by stepping up when a frame is above it and
    down when it is below, so it settles in the middle of recent frames
    and one frame in twenty-five cannot carry it.
    """

    def __init__(self, settings: DetectionSettings) -> None:
        self.settings = settings
        self._typical: float | None = None

    def scale(self, count_above: Callable[[float], int]) -> float:
        plain = count_above(self.settings.foreground_sigma)
        if self._typical is None:
            if plain > 0:
                self._typical = float(plain)
            return 1.0

        allowed = self.settings.foreground_budget * self._typical
        rate = self.settings.budget_rate
        self._typical *= 1.0 + (rate if plain > self._typical else -rate)
        if plain <= allowed:
            return 1.0

        for step in SCALE_LADDER[1:]:
            if count_above(self.settings.foreground_sigma * step) <= allowed:
                return step
        return SCALE_LADDER[-1]


def count_over(deviation: np.ndarray, threshold: float) -> int:
    return cv2.countNonZero(threshold_mask(deviation, threshold, cv2.CMP_GT))


def blobs_around_peaks(
    *,
    gray: np.ndarray,
    foreground: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    deviations: np.ndarray,
    min_pixels: int,
) -> list[Detection]:
    """Report the blob around every pixel that reaches the detection threshold.

    A blob holding no such pixel cannot be reported, so filling out from
    the pixels that did leaves a handful of blobs to measure out of the
    hundreds the foreground threshold raises on a textured scene.

    The fill writes MEASURED over the blob it has just counted, so the
    peaks standing on MEASURED are that blob's own. Every earlier blob
    took its peaks with it when it was filled, and the peaks arrive in
    row order, so a blob whose strongest value appears more than once is
    placed at the first of them.

    The fill also marks its own pixels in a mask of its own, because a
    blob's bounding box can hold pixels an earlier blob left MEASURED,
    and those would otherwise be weighed and summed as this blob's.
    """
    found: list[Detection] = []
    claimed = np.zeros(rows.size, dtype=bool)
    fill = np.zeros((foreground.shape[0] + 2, foreground.shape[1] + 2), dtype=np.uint8)

    for index in range(rows.size):
        if claimed[index]:
            continue
        claimed[index] = True
        if foreground[rows[index], columns[index]] != FOREGROUND:
            continue

        pixel_count, _image, _mask, rect = cv2.floodFill(
            foreground,
            fill,
            (int(columns[index]), int(rows[index])),
            MEASURED,
            flags=cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (FILLED << 8) | 8,
        )
        left, top, width, height = rect
        blob = fill[top + 1 : top + 1 + height, left + 1 : left + 1 + width] == FILLED
        fill[top + 1 : top + 1 + height, left + 1 : left + 1 + width] = 0
        foreground[top : top + height, left : left + width][blob] = MEASURED

        members = [index] + [
            other
            for other in range(index + 1, rows.size)
            if not claimed[other] and foreground[rows[other], columns[other]] == MEASURED
        ]
        claimed[members] = True
        if pixel_count < min_pixels:
            continue

        strongest = max(members, key=lambda member: deviations[member])
        axes = blob_axes(blob)
        found.append(
            Detection(
                centroid=(float(columns[strongest]), float(rows[strongest])),
                blob=BlobMeasurement(
                    pixel_count=pixel_count,
                    peak_deviation=float(deviations[strongest]),
                    brightness=float(gray[top : top + height, left : left + width][blob].sum(dtype=np.float64)),
                    major_axis=axes.major,
                    minor_axis=axes.minor,
                ),
            )
        )
    return found


def blob_axes(blob: np.ndarray) -> BlobAxes:
    """The axis lengths of the ellipse with the blob's second moments.

    A meteor's streak and a bird of the same area part company here and
    nowhere else in the detection, because a pixel count cannot tell a
    long thin shape from a round one.
    """
    moments = cv2.moments(blob.astype(np.uint8), binaryImage=True)
    area = moments["m00"]
    if area <= 0.0:
        return BlobAxes(major=0.0, minor=0.0)

    across = moments["mu20"] / area
    down = moments["mu02"] / area
    diagonal = moments["mu11"] / area
    spread = math.sqrt(max(4.0 * diagonal * diagonal + (across - down) ** 2, 0.0))
    return BlobAxes(
        major=2.0 * math.sqrt(max(2.0 * (across + down + spread), 0.0)),
        minor=2.0 * math.sqrt(max(2.0 * (across + down - spread), 0.0)),
    )


def threshold_mask(image: np.ndarray, threshold: float, comparison: int) -> np.ndarray:
    """Pixels standing in the given relation to a threshold, as 0 and 255.

    OpenCV reads the threshold as a scalar, which its type stub does not
    admit.
    """
    mask: np.ndarray = cv2.compare(image, threshold, comparison)  # type: ignore[call-overload]
    return mask
