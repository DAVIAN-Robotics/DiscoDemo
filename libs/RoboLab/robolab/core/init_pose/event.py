"""Turn an ``init_pose`` knob into IsaacLab EventTerm arguments."""

from __future__ import annotations

from typing import Any, Callable

from robolab.core.init_pose.box import BoxRange
from robolab.core.init_pose.container_policy import ALL_MOVABLE
from robolab.core.init_pose.knob import parse_init_pose

# Geometry keys a task must provide for ``pregrasp:``. The spec string only carries box widths.
PREGRASP_GEOMETRY_KEYS: frozenset[str] = frozenset(
    {
        "held",  # rigid object starting in the gripper
        "anchor",  # rigid object placed in the box (e.g. a board)
        "anchor_offset_xy_m",  # target point in the anchor frame, placed under the TCP (m)
        "held_length_m",  # held local z length (m)
        "grip_depth_m",  # finger depth from the held top (m)
        "finger_half_gap_m",  # per-finger joint target incl. preload (m)
        "hand_pos_m",  # measured hand origin at home, env-relative
        "hand_quat_wxyz",  # measured hand world quat at home
        "home_joint_pos",  # {joint: rad} at measurement; checked against default_joint_pos
        "hand_to_tcp_m",  # hand origin -> fingertip face (m)
    }
)


def build_init_pose_params(
    init_pose: str, *, container: str, pregrasp_geometry: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]]:
    """Build ``(event name, params)`` from ``init_pose``. Does not import isaaclab.

    Parameters
    ----------
    init_pose : str
        ``box:`` or ``pregrasp:`` spec.
    container : str
        Container asset name (keeps its default yaw in a ``box:`` layout).
    pregrasp_geometry : dict[str, Any] | None
        Required for ``pregrasp:``; task geometry with keys ``PREGRASP_GEOMETRY_KEYS``.

    Returns
    -------
    tuple[str, dict[str, Any]]
        ``("box" | "pregrasp", params)``.
    """
    spec = parse_init_pose(init_pose)
    if isinstance(spec, BoxRange):
        # All movable assets share one box and keep a minimum center distance.
        return "box", {
            "box_xy": spec.xy,
            "yaw_range": (spec.yaw_center_rad - spec.yaw_half_rad, spec.yaw_center_rad + spec.yaw_half_rad),
            "yaw_flip": spec.yaw_flip,
            "min_center_dist": float(spec.min_center_dist_m),
            "asset_cfg": [ALL_MOVABLE],
            "no_yaw_assets": [container],
            "reset_to_default_otherwise": True,
            "max_retries": 100,
        }
    if pregrasp_geometry is None:
        raise ValueError(f"init_pose 'pregrasp:' requires Task.pregrasp; keys: {sorted(PREGRASP_GEOMETRY_KEYS)}")
    missing = PREGRASP_GEOMETRY_KEYS - set(pregrasp_geometry)
    if missing:
        raise ValueError(f"pregrasp_geometry is missing keys: {sorted(missing)}")
    return "pregrasp", {
        "xy_half_m": float(spec.xy_half_m),
        "yaw_half_rad": float(spec.yaw_half_rad),
        **{k: pregrasp_geometry[k] for k in PREGRASP_GEOMETRY_KEYS},
    }


def build_init_pose_event(
    init_pose: str, *, container: str, pregrasp_geometry: dict[str, Any] | None = None
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """``build_init_pose_params`` plus the matching reset event function (imports isaaclab)."""
    from robolab.core.events.reset_pose import reset_pose_box, reset_pose_pregrasp

    name, params = build_init_pose_params(init_pose, container=container, pregrasp_geometry=pregrasp_geometry)
    return {"box": reset_pose_box, "pregrasp": reset_pose_pregrasp}[name], params
