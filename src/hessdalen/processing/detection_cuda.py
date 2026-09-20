"""The per-pixel stage of the detector, run on an NVIDIA GPU.

One kernel carries the whole per-pixel chain, so a frame is read once
and the background state never leaves the card. The host receives the
foreground mask and the few pixels that reached the detection threshold,
and never the deviation frame.
"""

from __future__ import annotations

import cupy as cp  # type: ignore[import-not-found]
import cv2
import numpy as np
from cupyx.scipy.ndimage import binary_dilation, binary_erosion  # type: ignore[import-not-found]

from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.detection import Detection, DetectionSettings

_BOX = cp.ElementwiseKernel(
    "raw uint8 source, int32 height, int32 width, int32 radius",
    "float32 smoothed",
    """
    const int row = i / width;
    const int column = i % width;
    float total = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        int y = row + dy;
        if (y < 0) { y = -y; }
        if (y >= height) { y = 2 * (height - 1) - y; }
        for (int dx = -radius; dx <= radius; ++dx) {
            int x = column + dx;
            if (x < 0) { x = -x; }
            if (x >= width) { x = 2 * (width - 1) - x; }
            total += (float)source[y * width + x];
        }
    }
    const int side = 2 * radius + 1;
    smoothed = total / (float)(side * side);
    """,
    "movement_box_filter",
)
"""Box filter reading the frame as it was uploaded, in bytes.

Out-of-frame reads mirror without repeating the edge pixel, which is
what the host filter does.
"""

_STEP = cp.ElementwiseKernel(
    "float32 smoothed, float32 variance_alpha, float32 mean_alpha, float32 noise_floor_variance, "
    "float32 outlier_sigma, float32 foreground_sigma, float32 detection_sigma, uint8 timestamp_mask",
    "float32 mean, float32 variance, float32 deviation, uint8 foreground, uint8 peaks",
    """
    const float value = smoothed;
    const float residual = value - mean;
    const float clamped = variance < noise_floor_variance ? noise_floor_variance : variance;
    const float distance = fabsf(residual) / sqrtf(clamped);

    deviation = distance;
    foreground = (distance > foreground_sigma && timestamp_mask != 0) ? 255 : 0;
    peaks = distance >= detection_sigma ? 255 : 0;

    if (distance <= outlier_sigma) {
        variance += variance_alpha * (residual * residual - variance);
    }
    mean += mean_alpha * (value - mean);
    """,
    "movement_step",
)


class CudaDetectionStage:
    """Measures the frame on the GPU and reports the blobs it holds."""

    def __init__(
        self,
        *,
        background: BackgroundSettings,
        detection: DetectionSettings,
        timestamp_mask: np.ndarray | None,
    ) -> None:
        self.settings = detection
        self.background = background
        self.timestamp_mask = timestamp_mask
        shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (detection.close_size, detection.close_size))
        self._close_kernel = cp.asarray(shape.astype(bool))
        self._noise_floor_variance = float(background.noise_floor) ** 2
        self._updates = 0
        self._buffers: _FrameBuffers | None = None
        self._state: _DeviceState | None = None

    def detections(self, gray: np.ndarray) -> list[Detection]:
        smoothed = self._smoothed(gray)
        state = self._state
        if state is None:
            self._state = _DeviceState.first(
                smoothed,
                shape=gray.shape,
                timestamp_mask=self.timestamp_mask,
                noise_floor_variance=self._noise_floor_variance,
            )
            return []

        self._updates += 1
        _STEP(
            smoothed,
            cp.float32(max(self.background.variance_alpha, 1.0 / self._updates)),
            cp.float32(self.background.mean_alpha),
            cp.float32(self._noise_floor_variance),
            cp.float32(self.background.outlier_sigma),
            cp.float32(self.settings.foreground_sigma),
            cp.float32(self.settings.detection_sigma),
            state.timestamp_mask,
            state.mean,
            state.variance,
            state.deviation,
            state.foreground,
            state.peaks,
        )

        rows, columns = cp.nonzero(state.peaks)
        if rows.size == 0:
            return []
        return self._blobs(state, rows=rows, columns=columns)

    def deviation_image(self) -> np.ndarray:
        """Scale the deviation for the debug video, with the detection
        threshold at full white."""
        state = self._state
        if state is None:
            return np.zeros((1, 1), dtype=np.uint8)

        scaled = state.deviation * cp.float32(255.0 / self.settings.detection_sigma)
        return cp.asnumpy(cp.clip(scaled, 0.0, 255.0).astype(cp.uint8))

    def _blobs(self, state: _DeviceState, *, rows: cp.ndarray, columns: cp.ndarray) -> list[Detection]:
        """One detection per blob, placed at the strongest of its peak pixels.

        A blob is reported only when its strongest pixel reaches the
        detection threshold, so that pixel is always in the peak mask
        and the blob's maximum is found among those pixels alone.
        """
        deviations = cp.asnumpy(state.deviation[rows, columns])
        peak_rows = cp.asnumpy(rows)
        peak_columns = cp.asnumpy(columns)
        foreground = self._closed(state.foreground)

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        strongest = self._strongest_per_component(
            labels=labels,
            component_count=component_count,
            peak_rows=peak_rows,
            peak_columns=peak_columns,
            deviations=deviations,
        )
        return [
            Detection(
                centroid=(float(peak_columns[index]), float(peak_rows[index])),
                pixel_count=int(stats[component][cv2.CC_STAT_AREA]),
                peak_deviation=float(deviations[index]),
            )
            for component, index in sorted(strongest.items())
            if int(stats[component][cv2.CC_STAT_AREA]) >= self.settings.min_pixels
        ]

    def _strongest_per_component(
        self,
        *,
        labels: np.ndarray,
        component_count: int,
        peak_rows: np.ndarray,
        peak_columns: np.ndarray,
        deviations: np.ndarray,
    ) -> dict[int, int]:
        """Index of the strongest peak pixel of each blob that holds one.

        The peaks arrive in row order, and a later pixel has to beat the
        one held to replace it, so a blob whose strongest value appears
        more than once keeps the first of them.
        """
        strongest: dict[int, int] = {}
        for index, component in enumerate(labels[peak_rows, peak_columns]):
            blob = int(component)
            if blob == 0 or blob >= component_count:
                continue
            leader = strongest.get(blob)
            if leader is None or deviations[index] > deviations[leader]:
                strongest[blob] = index
        return strongest

    def _closed(self, foreground: cp.ndarray) -> np.ndarray:
        """Close the mask on the card and hand the host the bytes it labels."""
        if self.settings.close_size <= 1:
            return cp.asnumpy(foreground)

        filled = binary_dilation(foreground > 0, structure=self._close_kernel, border_value=0)
        closed = binary_erosion(filled, structure=self._close_kernel, border_value=1)
        return cp.asnumpy(closed.astype(cp.uint8) * 255)

    def _smoothed(self, gray: np.ndarray) -> cp.ndarray:
        """Upload the frame as bytes and smooth it into floats on the card.

        Asking CuPy for floats would cast the frame on the host and send
        four times the bytes.
        """
        if self._buffers is None:
            self._buffers = _FrameBuffers(gray.shape)
        source = self._buffers.source
        source.set(np.ascontiguousarray(gray))

        radius = int(self.background.smoothing_size) // 2
        if radius < 1:
            return source.astype(cp.float32)

        height, width = gray.shape
        _BOX(source, np.int32(height), np.int32(width), np.int32(radius), self._buffers.smoothed)
        return self._buffers.smoothed


class _FrameBuffers:
    """Device memory the frame is read into and smoothed in.

    A buffer the driver has already mapped takes the frame half again as
    fast as one allocated for it, and the frame moves every time the
    detector is handed one.
    """

    __slots__ = ("smoothed", "source")

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.source = cp.empty(shape, dtype=cp.uint8)
        self.smoothed = cp.empty(shape, dtype=cp.float32)


class _DeviceState:
    """The background model and the frames derived from it, held on the
    card."""

    __slots__ = ("deviation", "foreground", "mean", "peaks", "timestamp_mask", "variance")

    def __init__(
        self,
        *,
        mean: cp.ndarray,
        variance: cp.ndarray,
        deviation: cp.ndarray,
        foreground: cp.ndarray,
        peaks: cp.ndarray,
        timestamp_mask: cp.ndarray,
    ) -> None:
        self.mean = mean
        self.variance = variance
        self.deviation = deviation
        self.foreground = foreground
        self.peaks = peaks
        self.timestamp_mask = timestamp_mask

    @classmethod
    def first(
        cls,
        smoothed: cp.ndarray,
        *,
        shape: tuple[int, ...],
        timestamp_mask: np.ndarray | None,
        noise_floor_variance: float,
    ) -> _DeviceState:
        mask = timestamp_mask if timestamp_mask is not None else np.full(shape, 255, dtype=np.uint8)
        return cls(
            mean=smoothed.copy(),
            variance=cp.full(shape, noise_floor_variance, dtype=cp.float32),
            deviation=cp.zeros(shape, dtype=cp.float32),
            foreground=cp.empty(shape, dtype=cp.uint8),
            peaks=cp.empty(shape, dtype=cp.uint8),
            timestamp_mask=cp.asarray(mask),
        )
