"""Choice of where the per-pixel stage of the detector runs.

The GPU stage needs CuPy and an NVIDIA card, neither of which every
machine that runs the detector has, so the choice is made once here and
the rest of the code holds a stage without knowing which one it is.
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np

from hessdalen.processing.background import BackgroundSettings
from hessdalen.processing.detection import CpuDetectionStage, DetectionSettings, DetectionStage

Device = Literal["auto", "cpu", "cuda"]
DEVICES: tuple[Device, ...] = get_args(Device)


def detection_stage(
    *,
    device: Device,
    background: BackgroundSettings,
    detection: DetectionSettings,
    timestamp_mask: np.ndarray | None,
) -> DetectionStage:
    """The stage the requested device asks for.

    "auto" takes the GPU when one answers and the host otherwise, so a
    machine without a card runs the same code without being configured
    for it. "cuda" raises instead of falling back, so a run meant for
    the GPU does not quietly become a slow one.
    """
    if device == "cuda" or (device == "auto" and cuda_available()):
        from hessdalen.processing.detection_cuda import CudaDetectionStage

        return CudaDetectionStage(background=background, detection=detection, timestamp_mask=timestamp_mask)

    return CpuDetectionStage(background=background, detection=detection, timestamp_mask=timestamp_mask)


def cuda_available() -> bool:
    """Whether a CuPy-visible GPU answers on this machine."""
    try:
        import cupy  # type: ignore[import-not-found]
    except ImportError:
        return False

    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
