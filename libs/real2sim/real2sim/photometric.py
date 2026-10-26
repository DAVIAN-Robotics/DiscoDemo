# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Measured camera response: per-channel gain, lens blur / sharpening (MTF) and a left-right gain ramp."""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from real2sim.workcell import knob


def camera_rgb_gain(camera_doc: dict[str, Any]) -> tuple[float, float, float]:
    """Per-channel gain of the camera."""
    gain = tuple(float(v) for v in knob(camera_doc, "rgb_gain"))
    if len(gain) != 3 or any(v <= 0.0 for v in gain):
        raise ValueError(f"rgb_gain must contain three positive values, got {gain}")
    return gain  # type: ignore[return-value]


def camera_horizontal_gain(camera_doc: dict[str, Any]) -> tuple[float, float]:
    """Gain at the left and right image edges (linear ramp between them); (1, 1) when not measured."""
    if "horizontal_gain" not in camera_doc:
        return 1.0, 1.0
    left, right = (float(v) for v in knob(camera_doc, "horizontal_gain"))
    if left <= 0.0 or right <= 0.0:
        raise ValueError(f"horizontal_gain must contain positive left/right values, got {(left, right)}")
    return left, right


def camera_mtf(camera_doc: dict[str, Any]) -> tuple[float, float]:
    """Gaussian blur sigma (px) and unsharp amount of the camera."""
    sigma = float(knob(camera_doc, "mtf.blur_sigma_px"))
    sharpen = float(knob(camera_doc, "mtf.unsharp_amount"))
    if sigma < 0.0 or sharpen < 0.0:
        raise ValueError("camera MTF blur sigma and unsharp amount must be non-negative")
    return sigma, sharpen


def apply_camera_response(
    frame: NDArray[np.uint8],
    gain: tuple[float, float, float],
    blur_sigma_px: float,
    unsharp_amount: float,
    horizontal_gain: tuple[float, float],
) -> NDArray[np.uint8]:
    """Apply gain, blur, unsharp mask and the left-right ramp to one ``H x W x 3`` uint8 frame."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {frame.shape}")
    scaled = frame.astype(np.float32) * np.asarray(gain, dtype=np.float32)[None, None, :]
    if blur_sigma_px > 0.0:
        scaled = cv2.GaussianBlur(scaled, (0, 0), blur_sigma_px)
    if unsharp_amount > 0.0:
        low = cv2.GaussianBlur(scaled, (0, 0), 1.0)
        scaled = scaled + unsharp_amount * (scaled - low)
    left, right = horizontal_gain
    scaled *= np.linspace(left, right, frame.shape[1], dtype=np.float32)[None, :, None]
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)
