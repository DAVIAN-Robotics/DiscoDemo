# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Camera preset bundles for the policy's image observations.

Each preset is a list of camera config classes that feed both the scene
(``camera_cfg=``) and the image observation group (via
``generate_image_obs_from_cameras``). Viewport-only cameras (e.g., the
third-person mirrored view used for video recording) are attached separately
inside the registration function and are not listed here.

Callers pass one of these lists directly to the registration function:

    from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
    auto_register_droid_envs(cameras=WRIST_LEFT_RIGHT_HEAD)
"""

from robolab.robots.droid import WristCameraPandaCfg
from robolab.variations.camera import (
    HeadCameraCfg,
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
)

WRIST = [WristCameraPandaCfg]

WRIST_LEFT = [
    OverShoulderLeftCameraCfg,
    WristCameraPandaCfg,
]

WRIST_RIGHT = [
    OverShoulderRightCameraCfg,
    WristCameraPandaCfg,
]


WRIST_LEFT_RIGHT = [
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
    WristCameraPandaCfg,
]


WRIST_LEFT_RIGHT_HEAD = [
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
    HeadCameraCfg,
    WristCameraPandaCfg,
]

LEFT_RIGHT = [
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
]
