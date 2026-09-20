from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class BackgroundSettings:
    """How fast the background follows the frames it is measuring.

    The two rates are set from the dashboard and come from the config,
    so they are asked of the caller. The three below them have no
    control of their own and keep their values here.
    """

    mean_alpha: float
    variance_alpha: float
    noise_floor: float = 1.0
    outlier_sigma: float = 5.0
    smoothing_size: int = 3


@dataclass(slots=True)
class BackgroundState:
    mean: np.ndarray
    variance: np.ndarray


class BackgroundModel:
    """Running per-pixel mean and noise level of a grayscale video.

    The deviation this reports is measured in each pixel's own noise, so
    a bright star field, a moonlit slope and a dark sky all read on one
    scale and one threshold covers every scene.
    """

    def __init__(self, settings: BackgroundSettings):
        self.settings = settings
        self._state: BackgroundState | None = None
        self._noise_floor_variance = float(settings.noise_floor) ** 2
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
        noise = np.sqrt(np.maximum(state.variance, self._noise_floor_variance))
        deviation = np.abs(residual) / noise

        self._update(state, smoothed=smoothed, residual=residual, deviation=deviation)
        return deviation

    def _update(
        self, state: BackgroundState, *, smoothed: np.ndarray, residual: np.ndarray, deviation: np.ndarray
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
        cv2.accumulateWeighted(residual * residual, state.variance, self._variance_alpha(), mask=background)
        cv2.accumulateWeighted(smoothed, state.mean, self.settings.mean_alpha)

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
