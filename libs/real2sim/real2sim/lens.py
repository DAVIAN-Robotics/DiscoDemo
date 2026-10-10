# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Measured camera lens post-processing utilities."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def distortion_maps(
    K: Any, dist: Any, hw: tuple[int, int] | list[int]
) -> tuple[NDArray[np.float32], NDArray[np.float32]] | None:
    """Map an ideal pinhole render into the measured distorted image plane.

    Output pixels live in the real (distorted) image. ``undistortPoints`` tells us
    which ideal rendered pixel each output pixel should sample.
    """
    K_arr = np.asarray(K, dtype=np.float64)
    dist_arr = np.asarray(dist, dtype=np.float64).reshape(-1)
    height, width = (int(hw[0]), int(hw[1]))
    if K_arr.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K_arr.shape}")
    if dist_arr.size not in (4, 5, 8, 12, 14):
        raise ValueError(f"unsupported OpenCV distortion length: {dist_arr.size}")
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {(height, width)}")
    if np.allclose(dist_arr, 0.0):
        return None

    import cv2

    yy, xx = np.indices((height, width), dtype=np.float32)
    distorted_pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    ideal_pixels = cv2.undistortPoints(distorted_pixels, K_arr, dist_arr, P=K_arr)
    ideal_pixels = ideal_pixels.reshape(height, width, 2).astype(np.float32)
    return ideal_pixels[..., 0], ideal_pixels[..., 1]


def apply_distortion(
    frame: NDArray[np.uint8], maps: tuple[NDArray[np.float32], NDArray[np.float32]] | None
) -> NDArray[np.uint8]:
    """Apply precomputed maps to one RGB frame; zero-distortion maps are a no-op."""
    if maps is None:
        return frame
    import cv2

    map_x, map_y = maps
    if frame.shape[:2] != map_x.shape or map_x.shape != map_y.shape:
        raise ValueError(f"frame/map mismatch: frame={frame.shape[:2]} maps={map_x.shape}/{map_y.shape}")
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
