# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Isaac ``TiledCameraCfg`` objects for the workcell cameras.

Parameters live in ``assets/workcell.json``. This module converts each render pose (4x4) into Isaac
(pos, quat) and the OpenCV ``K`` into focal length / aperture:

    fx = focal_length x W / horizontal_aperture
    fy = focal_length x H / vertical_aperture

Only ratios matter, so ``horizontal_aperture`` is fixed and ``focal_length`` solved. Omniverse ignores
aperture offsets (the principal point is the sensor centre); ``principal_point_pose_compensation``
approximates the film shift with a small rotation. isaaclab is imported lazily so ``camera_specs`` works
without a SimulationApp.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from real2sim.workcell import knob

if TYPE_CHECKING:
    from isaaclab.sensors import TiledCameraCfg


# Isaac default horizontal aperture (mm); arbitrary since only ratios matter.
DEFAULT_HORIZONTAL_APERTURE_MM = 20.955


def k_to_isaac(
    K: NDArray[np.float64],
    hw: tuple[int, int],
    horizontal_aperture_mm: float = DEFAULT_HORIZONTAL_APERTURE_MM,
) -> dict[str, float]:
    """Convert K (3x3) to Isaac camera parameters; ``hw`` is (height, width)."""
    height, width = hw
    fx, fy = float(K[0, 0]), float(K[1, 1])
    focal_length = fx * horizontal_aperture_mm / width
    vertical_aperture = focal_length * height / fy
    return {
        "focal_length": focal_length,
        "horizontal_aperture": horizontal_aperture_mm,
        "vertical_aperture": vertical_aperture,
    }


def principal_point_offset_px(K: NDArray[np.float64], hw: tuple[int, int]) -> tuple[float, float]:
    """Return the principal point offset (dx, dy) from the sensor centre in pixels."""
    height, width = hw
    return float(K[0, 2]) - width / 2.0, float(K[1, 2]) - height / 2.0


def principal_point_pose_compensation(K: NDArray[np.float64], hw: tuple[int, int]) -> NDArray[np.float64]:
    """Local camera rotation that shifts a centred Isaac render to the measured principal point."""
    dx, dy = principal_point_offset_px(K, hw)
    rx = np.arctan2(dy, float(K[1, 1]))
    ry = -np.arctan2(dx, float(K[0, 0]))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return rot_x @ rot_y


def transform_to_pose(T: NDArray[np.float64]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Convert a parent->camera 4x4 transform to (pos xyz, quat wxyz).

    Uses Shepperd's method (largest component first), which is stable when the trace
    is close to -1.
    """
    M = np.asarray(T, dtype=float)
    R = M[:3, :3]
    pos = tuple(float(v) for v in M[:3, 3])
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z])
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return pos, tuple(float(v) for v in quat)


def camera_specs(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten the workcell camera blocks into per-camera specs for Isaac cfgs."""
    specs: dict[str, dict[str, Any]] = {}
    for name, cam in doc["cameras"].items():
        hw = cam["hw"]
        K = np.asarray(knob(cam, "K"), dtype=float)
        optics = k_to_isaac(K, (hw[0], hw[1]))
        actual_mount = cam["mount"]
        actual_parent = cam["parent_frame"]
        expected_parent = "panda_hand" if actual_mount == "robot" else "base"
        if actual_parent != expected_parent:
            raise ValueError(
                f"{name}: mount={actual_mount!r} requires parent_frame={expected_parent!r}, got {actual_parent!r}"
            )
        # Render-only pose where one was fitted to the real images, else the calibrated extrinsic.
        key = "T_parent_cam_render" if "T_parent_cam_render" in cam else "T_parent_cam"
        T_parent_cam_render = np.asarray(knob(cam, key), dtype=float).copy()
        if cam.get("flip_180", False):
            # A 180-degree image flip is an optical-axis roll. Post-multiply so
            # translation stays in the parent frame while camera x/y both invert.
            T_parent_cam_render[:3, :3] = T_parent_cam_render[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
        # Omniverse ignores USD aperture offsets. A small local camera rotation
        # approximates the measured off-centre principal point in rendered RGB-D.
        T_parent_cam_render[:3, :3] = T_parent_cam_render[:3, :3] @ principal_point_pose_compensation(K, (hw[0], hw[1]))
        pos, quat = transform_to_pose(T_parent_cam_render)
        specs[name] = {
            "mount": cam["mount"],
            "parent_frame": cam["parent_frame"],
            "prim": cam["prim"],
            "hw": hw,
            "flip_180": bool(cam.get("flip_180", False)),
            "pos": pos,
            "rot": quat,
            "T_parent_cam_render": T_parent_cam_render,
            **optics,
        }
    return specs


def _make_cfg(spec: dict[str, Any]) -> "TiledCameraCfg":
    """Build one ``TiledCameraCfg`` from a spec."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    height, width = spec["hw"]
    # Only robot-mounted cameras live under the robot namespace.
    prim_path = (
        f"{{ENV_REGEX_NS}}/robot/{spec['prim']}" if spec["mount"] == "robot" else f"{{ENV_REGEX_NS}}/{spec['prim']}"
    )
    return TiledCameraCfg(
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(pos=spec["pos"], rot=spec["rot"], convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=spec["focal_length"],
            horizontal_aperture=spec["horizontal_aperture"],
            vertical_aperture=spec["vertical_aperture"],
            clipping_range=(0.01, 20.0),
        ),
        width=width,
        height=height,
    )


def obs_name(spec: dict[str, Any]) -> str:
    """Return the obs term name: the last segment of the camera prim path (e.g. ``wrist_cam``)."""
    return str(spec["prim"]).split("/")[-1]


def robot_camera_cfgs(doc: dict[str, Any], names: set[str] | None = None) -> dict[str, "TiledCameraCfg"]:
    """Robot-mounted cameras as an instance dict for ``extra_robot_cameras=``."""
    return {
        obs_name(s): _make_cfg(s)
        for name, s in camera_specs(doc).items()
        if s["mount"] == "robot" and (names is None or name in names)
    }


def scene_camera_cfgs(doc: dict[str, Any], names: set[str] | None = None) -> list[type]:
    """Fixed scene cameras as a list of wrapper cfg classes for ``cameras=``.

    RoboLab registration instantiates each entry, so classes (not instances) are returned.
    """
    from isaaclab.utils import configclass

    out: list[type] = []
    for camera_name, spec in camera_specs(doc).items():
        if spec["mount"] == "robot" or (names is not None and camera_name not in names):
            continue
        name = obs_name(spec)
        out.append(configclass(type(f"Real2SimSceneCamera_{name}", (), {name: _make_cfg(spec)})))
    return out
