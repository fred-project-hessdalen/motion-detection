import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class BackgroundSettings:
    """How fast the background follows the frames it is measuring.

    The two rates are set from the dashboard and come from the config,
    so they are asked of the caller. The ones below them have no control
    of their own and keep their values here.
    """

    mean_alpha: float
    variance_alpha: float
    noise_floor: float = 1.0
    outlier_sigma: float = 5.0
    smoothing_size: int = 3
    noise_block: int = 16
    neighbourhood_blocks: int = 13
    guard_blocks: int = 7


@dataclass(slots=True)
class BackgroundState:
    mean: np.ndarray
    variance: np.ndarray


class BackgroundModel:
    """Running per-pixel mean and noise level of a grayscale video.

    A pixel's noise is measured both over the frames before it and over
    the pixels around it, and the frame is divided by whichever of the
    two is larger. Reporting in noise is what lets a bright star field,
    a moonlit slope and a dark sky read on one scale, so one threshold
    covers every scene.
    """

    def __init__(self, settings: BackgroundSettings):
        self.settings = settings
        self._state: BackgroundState | None = None
        self._noise_floor_variance = float(settings.noise_floor) ** 2
        self._halvings = round(math.log2(int(settings.noise_block)))
        self._updates = 0

    def deviation(self, gray_frame: np.ndarray) -> np.ndarray:
        """Return how far each pixel sits from the background, in noise
        sigma."""
        smoothed = self._smooth(gray_frame)
        state = self._state
        if state is None:
            self._state = BackgroundState(
                mean=smoothed,
                variance=np.full_like(smoothed, self._noise_floor_variance),
            )
            return np.zeros_like(smoothed)

        residual = smoothed - state.mean
        squares = residual * residual
        tracked = np.sqrt(np.maximum(state.variance, self._noise_floor_variance))
        deviation = np.abs(residual) / np.maximum(tracked, self._neighbourhood_noise(squares))

        self._update(state, smoothed=smoothed, squares=squares, deviation=deviation)
        return deviation

    def _update(
        self, state: BackgroundState, *, smoothed: np.ndarray, squares: np.ndarray, deviation: np.ndarray
    ) -> None:
        """Fold the frame into the background, keeping its noise estimate free
        of moving objects.

        A bright object raises the variance of every pixel it crosses,
        and that pixel then needs an even brighter object to register,
        so an object holding still for a few frames would erase itself.
        Pixels the current frame calls foreground are therefore left out
        of the noise estimate.

        The mean gets no such exemption. Holding foreground pixels out
        of it freezes the background under anything that moves
        repeatedly, and swaying branches then read as movement on every
        frame.
        """
        self._updates += 1
        background = (deviation <= self.settings.outlier_sigma).astype(np.uint8)
        cv2.accumulateWeighted(squares, state.variance, self._variance_alpha(), mask=background)
        cv2.accumulateWeighted(smoothed, state.mean, self.settings.mean_alpha)

    def _neighbourhood_noise(self, squares: np.ndarray) -> np.ndarray:
        """How far the residual spreads around each pixel, with the pixel's own
        block and the blocks touching it left out.

        A frame the encoder has coded from scratch departs from the
        background everywhere at once, and a pixel measured against the
        frames before it cannot tell that from an object. The pixels
        around it can, because an object covers a few blocks and a
        recoded frame covers all of them. The blocks left out are what
        stops an object from raising the level it is measured against.
        """
        span = int(self.settings.neighbourhood_blocks)
        guard = int(self.settings.guard_blocks)
        blocks = self._blocks(squares)
        spread = np.sqrt(np.maximum(self._around(blocks) / float(span * span - guard * guard), 0.0))

        block = int(self.settings.noise_block)
        covered = cv2.resize(
            spread, (blocks.shape[1] * block, blocks.shape[0] * block), interpolation=cv2.INTER_NEAREST
        )
        height, width = squares.shape
        expanded: np.ndarray = covered[:height, :width]
        return expanded

    def _blocks(self, squares: np.ndarray) -> np.ndarray:
        """The frame reduced to the mean of each block, by repeated halving.

        A halving pads an odd side by repeating its last line, so the
        blocks cover a whole number of them and a pixel finds its block
        by dividing. Halving by area takes the mean of four neighbours
        as two pairs, and the graphics card adds them the same way,
        which is what keeps the two devices on the same blocks. Summing
        the four in a row instead leaves them a part in 1e5 apart, and
        that is enough to move a pixel across a threshold.
        """
        blocks = squares
        for _ in range(self._halvings):
            odd_rows, odd_columns = blocks.shape[0] % 2, blocks.shape[1] % 2
            if odd_rows or odd_columns:
                blocks = np.pad(blocks, ((0, odd_rows), (0, odd_columns)), mode="edge")
            blocks = cv2.resize(blocks, (blocks.shape[1] // 2, blocks.shape[0] // 2), interpolation=cv2.INTER_AREA)
        return blocks

    def _around(self, blocks: np.ndarray) -> np.ndarray:
        """The blocks summed over each block's neighbourhood, with the block
        itself and the blocks touching it left out.

        The order the blocks are added in is fixed for the same reason
        the halving's is. Blocks outside the frame mirror without
        repeating the edge block.
        """
        reach = int(self.settings.neighbourhood_blocks) // 2
        inner = int(self.settings.guard_blocks) // 2
        padded = np.pad(blocks, reach, mode="reflect")

        rows, columns = blocks.shape
        total = np.zeros_like(blocks)
        for down in range(-reach, reach + 1):
            for right in range(-reach, reach + 1):
                if abs(down) <= inner and abs(right) <= inner:
                    continue
                total += padded[down + reach : down + reach + rows, right + reach : right + reach + columns]
        return total

    def _variance_alpha(self) -> float:
        """Weight for this frame in the noise estimate.

        At the configured rate the estimate needs about a hundred frames
        to reach the true noise of the scene, and until it does it sits
        too low and everything reads as a deviation. Averaging the
        frames seen so far gives the estimate its scene from the start,
        and the configured rate takes over once it is the slower of the
        two.
        """
        return max(self.settings.variance_alpha, 1.0 / self._updates)

    def _smooth(self, gray_frame: np.ndarray) -> np.ndarray:
        as_float = np.asarray(gray_frame, dtype=np.float32)
        size = int(self.settings.smoothing_size)
        if size <= 1:
            return as_float.copy()
        return cv2.blur(as_float, (size, size))
