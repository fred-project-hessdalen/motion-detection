"""The per-pixel stage of the detector, run on an NVIDIA GPU.

The background state never leaves the card. The host receives the
foreground mask and the few pixels that reached the detection threshold,
and never the deviation frame.
"""

from __future__ import annotations

import math

import cupy as cp  # type: ignore[import-not-found]
import cv2
import numpy as np
from cupyx.scipy.ndimage import binary_dilation, binary_erosion  # type: ignore[import-not-found]

from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.detection import Detection, DetectionSettings, blobs_around_peaks

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

_SQUARES = cp.ElementwiseKernel(
    "float32 smoothed, float32 mean",
    "float32 squares",
    """
    const float residual = smoothed - mean;
    squares = residual * residual;
    """,
    "movement_squares",
)

_HALVE = cp.ElementwiseKernel(
    "raw float32 source, int32 source_rows, int32 source_columns, int32 columns",
    "float32 halved",
    """
    const int row = 2 * (i / columns);
    const int column = 2 * (i % columns);
    const int below = (row + 1 < source_rows) ? row + 1 : row;
    const int right = (column + 1 < source_columns) ? column + 1 : column;
    const float total = (source[row * source_columns + column] + source[row * source_columns + right])
                      + (source[below * source_columns + column] + source[below * source_columns + right]);
    halved = total * 0.25f;
    """,
    "movement_halve",
)
"""Mean of four neighbours, added as the two pairs the host adds.

An odd side reads its last line twice, which is the line the host pads
with.
"""

_NEIGHBOURHOOD = cp.ElementwiseKernel(
    "raw float32 blocks, int32 rows, int32 columns, int32 reach, int32 inner, float32 cells",
    "float32 spread",
    """
    const int row = i / columns;
    const int column = i % columns;
    float total = 0.0f;
    for (int down = -reach; down <= reach; ++down) {
        for (int right = -reach; right <= reach; ++right) {
            if (abs(down) <= inner && abs(right) <= inner) { continue; }
            int y = row + down;
            if (y < 0) { y = -y; }
            if (y >= rows) { y = 2 * (rows - 1) - y; }
            int x = column + right;
            if (x < 0) { x = -x; }
            if (x >= columns) { x = 2 * (columns - 1) - x; }
            total += blocks[y * columns + x];
        }
    }
    const float mean = total / cells;
    spread = sqrtf(mean > 0.0f ? mean : 0.0f);
    """,
    "movement_neighbourhood",
)
"""Spread of the residual over the blocks around each one.

The blocks are added in the order the host adds them, and out-of-frame
blocks mirror without repeating the edge block.
"""

_STEP = cp.ElementwiseKernel(
    "float32 smoothed, raw float32 spread, int32 width, int32 shift, int32 blocks_columns, "
    "float32 variance_alpha, float32 mean_alpha, float32 noise_floor_variance, "
    "float32 outlier_sigma, float32 foreground_sigma, float32 detection_sigma, uint8 timestamp_mask",
    "float32 mean, float32 variance, float32 deviation, uint8 foreground, uint8 peaks",
    """
    const float value = smoothed;
    const float residual = value - mean;
    const float clamped = variance < noise_floor_variance ? noise_floor_variance : variance;
    const float tracked = sqrtf(clamped);
    const float around = spread[((i / width) >> shift) * blocks_columns + ((i % width) >> shift)];
    const float noise = tracked > around ? tracked : around;
    const float distance = fabsf(residual) / noise;

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
        self._halvings = round(math.log2(int(background.noise_block)))
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
                halvings=self._halvings,
            )
            return []

        self._updates += 1
        _SQUARES(smoothed, state.mean, state.squares)
        spread = self._neighbourhood_noise(state)
        _STEP(
            smoothed,
            spread,
            np.int32(gray.shape[1]),
            np.int32(self._halvings),
            np.int32(spread.shape[1]),
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
        return blobs_around_peaks(
            foreground=self._closed(state.foreground),
            rows=cp.asnumpy(rows),
            columns=cp.asnumpy(columns),
            deviations=cp.asnumpy(state.deviation[rows, columns]),
            min_pixels=self.settings.min_pixels,
        )

    def _closed(self, foreground: cp.ndarray) -> np.ndarray:
        """Close the mask on the card and hand the host the bytes it labels."""
        if self.settings.close_size <= 1:
            return cp.asnumpy(foreground)

        filled = binary_dilation(foreground > 0, structure=self._close_kernel, border_value=0)
        closed = binary_erosion(filled, structure=self._close_kernel, border_value=1)
        return cp.asnumpy(closed.astype(cp.uint8) * 255)

    def _neighbourhood_noise(self, state: _DeviceState) -> cp.ndarray:
        """Reduce the squared residual to blocks and spread each block over its
        neighbours."""
        blocks = state.squares
        for halved in state.pyramid:
            _HALVE(
                blocks,
                np.int32(blocks.shape[0]),
                np.int32(blocks.shape[1]),
                np.int32(halved.shape[1]),
                halved,
            )
            blocks = halved

        span = int(self.background.neighbourhood_blocks)
        guard = int(self.background.guard_blocks)
        _NEIGHBOURHOOD(
            blocks,
            np.int32(blocks.shape[0]),
            np.int32(blocks.shape[1]),
            np.int32(span // 2),
            np.int32(guard // 2),
            np.float32(span * span - guard * guard),
            state.spread,
        )
        return state.spread

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

    __slots__ = (
        "deviation",
        "foreground",
        "mean",
        "peaks",
        "pyramid",
        "spread",
        "squares",
        "timestamp_mask",
        "variance",
    )

    def __init__(
        self,
        *,
        mean: cp.ndarray,
        variance: cp.ndarray,
        deviation: cp.ndarray,
        foreground: cp.ndarray,
        peaks: cp.ndarray,
        squares: cp.ndarray,
        pyramid: list[cp.ndarray],
        spread: cp.ndarray,
        timestamp_mask: cp.ndarray,
    ) -> None:
        self.mean = mean
        self.variance = variance
        self.deviation = deviation
        self.foreground = foreground
        self.peaks = peaks
        self.squares = squares
        self.pyramid = pyramid
        self.spread = spread
        self.timestamp_mask = timestamp_mask

    @classmethod
    def first(
        cls,
        smoothed: cp.ndarray,
        *,
        shape: tuple[int, ...],
        timestamp_mask: np.ndarray | None,
        noise_floor_variance: float,
        halvings: int,
    ) -> _DeviceState:
        mask = timestamp_mask if timestamp_mask is not None else np.full(shape, 255, dtype=np.uint8)
        pyramid = _pyramid(shape, halvings=halvings)
        return cls(
            mean=smoothed.copy(),
            variance=cp.full(shape, noise_floor_variance, dtype=cp.float32),
            deviation=cp.zeros(shape, dtype=cp.float32),
            foreground=cp.empty(shape, dtype=cp.uint8),
            peaks=cp.empty(shape, dtype=cp.uint8),
            squares=cp.empty(shape, dtype=cp.float32),
            pyramid=pyramid,
            spread=cp.empty(pyramid[-1].shape, dtype=cp.float32),
            timestamp_mask=cp.asarray(mask),
        )


def _pyramid(shape: tuple[int, ...], *, halvings: int) -> list[cp.ndarray]:
    """A buffer for each halving, each side rounded up so an odd one keeps its
    last line."""
    rows, columns = int(shape[0]), int(shape[1])
    levels: list[cp.ndarray] = []
    for _ in range(halvings):
        rows, columns = (rows + 1) // 2, (columns + 1) // 2
        levels.append(cp.empty((rows, columns), dtype=cp.float32))
    return levels
