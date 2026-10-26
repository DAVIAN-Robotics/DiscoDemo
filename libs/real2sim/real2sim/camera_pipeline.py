# SPDX-License-Identifier: Apache-2.0
"""Measured camera pipeline (lens distortion + camera response) shared by every renderer.

The Isaac camera sensor returns an ideal RGB render; the workcell also holds the measured lens and camera
response. Building and applying them here keeps every consumer (data render, evaluation) on the same pixels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from real2sim.lens import apply_distortion, distortion_maps
from real2sim.photometric import apply_camera_response, camera_horizontal_gain, camera_mtf, camera_rgb_gain
from real2sim.workcell import knob

CameraPipeline = dict[str, Any]


def build_camera_pipeline(camera_doc: dict[str, Any]) -> CameraPipeline:
    """Compile one ``workcell["cameras"][<name>]`` block (rendered at its measured size)."""
    hw = tuple(int(value) for value in camera_doc["hw"])
    return {
        "hw": hw,
        "distortion_maps": distortion_maps(knob(camera_doc, "K"), knob(camera_doc, "dist"), hw),
        "rgb_gain": camera_rgb_gain(camera_doc),
        "mtf": camera_mtf(camera_doc),
        "horizontal_gain": camera_horizontal_gain(camera_doc),
    }


def apply_camera_pipeline(frame: NDArray[np.uint8], pipeline: CameraPipeline) -> NDArray[np.uint8]:
    """Apply the measured lens distortion and camera response to one RGB frame."""
    if frame.shape != (*pipeline["hw"], 3):
        raise ValueError(f"camera frame shape {frame.shape} does not match the measured size {pipeline['hw']}")
    blur_sigma, unsharp = pipeline["mtf"]
    return apply_camera_response(
        apply_distortion(frame, pipeline["distortion_maps"]),
        pipeline["rgb_gain"],
        blur_sigma,
        unsharp,
        pipeline["horizontal_gain"],
    )


def apply_camera_pipeline_batch(frames: NDArray[np.uint8], pipeline: CameraPipeline) -> NDArray[np.uint8]:
    """Apply one camera pipeline to an ``N x H x W x 3`` batch."""
    return np.stack([apply_camera_pipeline(np.ascontiguousarray(frame), pipeline) for frame in frames], axis=0)
