"""The stage that turns a grayscale frame into the blobs worth tracking.

The detector's tracking stage consumes detections and does not care how
the pixels were measured, so this stage is the seam a different
implementation plugs into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from hessdalen.processing.background import BackgroundModel, BackgroundSettings

Centroid = tuple[float, float]

FOREGROUND = 255
"""Value a mask carries where the pixel belongs to a blob."""

MEASURED = 128
"""Value the fill leaves on a blob whose pixels have been counted."""


@dataclass(frozen=True, slots=True)
class Detection:
    centroid: Centroid
    pixel_count: int
    peak_deviation: float


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    """What a blob has to be before it is reported.

    The first three are set from the dashboard and come from the config,
    so they are asked of the caller. The closing size has no control of
    its own and keeps its value here.
    """

    foreground_sigma: float
    detection_sigma: float
    min_pixels: int
    close_size: int = 5


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
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (detection.close_size, detection.close_size))
        self._deviation = np.zeros((1, 1), dtype=np.float32)

    def detections(self, gray: np.ndarray) -> list[Detection]:
        self._deviation = self._background.deviation(gray)
        return self._extract_detections(self._deviation)

    def deviation_image(self) -> np.ndarray:
        """Scale the deviation for the debug video, with the detection
        threshold at full white."""
        scaled = self._deviation * (255.0 / self.settings.detection_sigma)
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    def _extract_detections(self, deviation: np.ndarray) -> list[Detection]:
        peaks = cv2.findNonZero(self._peak_mask(deviation))
        if peaks is None:
            return []

        rows = peaks[:, 0, 1]
        columns = peaks[:, 0, 0]
        return blobs_around_peaks(
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


def blobs_around_peaks(
    *,
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
    """
    found: list[Detection] = []
    claimed = np.zeros(rows.size, dtype=bool)

    for index in range(rows.size):
        if claimed[index]:
            continue
        claimed[index] = True
        if foreground[rows[index], columns[index]] != FOREGROUND:
            continue

        pixel_count, _image, _mask, _rect = cv2.floodFill(
            foreground, None, (int(columns[index]), int(rows[index])), MEASURED, flags=cv2.FLOODFILL_FIXED_RANGE | 8
        )
        members = [index] + [
            other
            for other in range(index + 1, rows.size)
            if not claimed[other] and foreground[rows[other], columns[other]] == MEASURED
        ]
        claimed[members] = True
        if pixel_count < min_pixels:
            continue

        strongest = max(members, key=lambda member: deviations[member])
        found.append(
            Detection(
                centroid=(float(columns[strongest]), float(rows[strongest])),
                pixel_count=pixel_count,
                peak_deviation=float(deviations[strongest]),
            )
        )
    return found


def threshold_mask(image: np.ndarray, threshold: float, comparison: int) -> np.ndarray:
    """Pixels standing in the given relation to a threshold, as 0 and 255.

    OpenCV reads the threshold as a scalar, which its type stub does not
    admit.
    """
    mask: np.ndarray = cv2.compare(image, threshold, comparison)  # type: ignore[call-overload]
    return mask
