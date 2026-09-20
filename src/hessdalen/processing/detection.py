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


@dataclass(frozen=True, slots=True)
class Detection:
    centroid: Centroid
    pixel_count: int
    peak_deviation: float


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    foreground_sigma: float = 5.0
    detection_sigma: float = 15.0
    min_pixels: int = 3
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
        """Report the blob around every pixel that reaches the detection
        threshold.

        A blob holding no such pixel cannot be reported, so finding the
        pixels first leaves a handful of blobs to measure out of the
        hundreds the foreground threshold raises on a textured scene.
        """
        peaks = cv2.findNonZero(self._peak_mask(deviation))
        if peaks is None:
            return []

        foreground = self._foreground_mask(deviation)
        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        return [
            detection
            for component in np.unique(labels[peaks[:, 0, 1], peaks[:, 0, 0]])
            if 0 < component < component_count
            and (detection := self._component_to_detection(int(component), labels, stats, deviation)) is not None
        ]

    def _peak_mask(self, deviation: np.ndarray) -> np.ndarray:
        return threshold_mask(deviation, self.settings.detection_sigma, cv2.CMP_GE)

    def _foreground_mask(self, deviation: np.ndarray) -> np.ndarray:
        mask = threshold_mask(deviation, self.settings.foreground_sigma, cv2.CMP_GT)
        if self.timestamp_mask is not None:
            mask = cv2.bitwise_and(mask, self.timestamp_mask)
        if self.settings.close_size <= 1:
            return mask
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)

    def _component_to_detection(
        self,
        component: int,
        labels: np.ndarray,
        stats: np.ndarray,
        deviation: np.ndarray,
    ) -> Detection | None:
        x, y, width, height, pixel_count = (int(value) for value in stats[component])
        if pixel_count < self.settings.min_pixels:
            return None

        window = np.where(
            labels[y : y + height, x : x + width] == component, deviation[y : y + height, x : x + width], 0.0
        )
        peak_deviation = float(window.max())
        if peak_deviation < self.settings.detection_sigma:
            return None

        peak_y, peak_x = np.unravel_index(int(np.argmax(window)), window.shape)
        return Detection(
            centroid=(float(x + int(peak_x)), float(y + int(peak_y))),
            pixel_count=pixel_count,
            peak_deviation=peak_deviation,
        )


def threshold_mask(image: np.ndarray, threshold: float, comparison: int) -> np.ndarray:
    """Pixels standing in the given relation to a threshold, as 0 and 255.

    OpenCV reads the threshold as a scalar, which its type stub does not
    admit.
    """
    mask: np.ndarray = cv2.compare(image, threshold, comparison)  # type: ignore[call-overload]
    return mask
