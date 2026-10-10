# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Peg-in-hole success predicate.

FMB defines success as "the bottom surface of the object is completely inserted in the
correct matching hole". It is evaluated in the board local frame as three conditions:

1. depth: the lowest peg OBB corner is ``insertion_fraction * thickness`` below the board top;
2. lateral: the peg centre is within ``lateral_tolerance_m`` of the target hole (rejects
   insertion into a different hole);
3. axis: the peg z axis is within ``axis_tolerance_deg`` of the board normal.

The success hold is applied by RoboLab.
"""

import math
from typing import Any

import torch


# Local quaternion helpers (wxyz, IsaacLab convention) so the module imports without Isaac.
def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by q: v' = v + 2w(u x v) + 2u x (u x v), u = vector part of q."""
    w = q[..., :1]
    u = q[..., 1:]
    uv = torch.cross(u, v, dim=-1)
    return v + 2.0 * (w * uv + torch.cross(u, uv, dim=-1))


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _quat_apply(_quat_conjugate(q), v)


def _to_local(points_w: torch.Tensor, origin_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    """Express world points (N, P, 3) in the frame (origin (N, 3), quat wxyz (N, 4))."""
    rel = points_w - origin_w.unsqueeze(1)
    n, p, _ = rel.shape
    flat_quat = quat_w.unsqueeze(1).expand(n, p, 4).reshape(-1, 4)
    return _quat_apply_inverse(flat_quat, rel.reshape(-1, 3)).reshape(n, p, 3)


def peg_in_hole(
    env: Any,
    peg: str,
    board: str,
    hole_offset_m: tuple[float, float],
    insertion_fraction: float,
    lateral_tolerance_m: float,
    axis_tolerance_deg: float,
    depth_tolerance_m: float,
) -> torch.Tensor:
    """Return whether the peg is fully inserted in the given hole of the board.

    Parameters
    ----------
    env : Any
        IsaacLab ManagerBasedRLEnv.
    peg, board : str
        Scene object names.
    hole_offset_m : tuple[float, float]
        Hole centre (x, y) in the board local frame.
    insertion_fraction : float
        Required depth as a fraction of the board thickness (1.0 = through).
    lateral_tolerance_m : float
        Allowed lateral error from the hole centre.
    axis_tolerance_deg : float
        Allowed angle between the peg axis and the board normal.
    depth_tolerance_m : float
        Depth slack.

    Returns
    -------
    torch.Tensor
        (num_envs,) bool.
    """
    from robolab.core.world.world_state import get_world  # needs a running Isaac app

    world = get_world(env)

    peg_corners_w, peg_centroid_w = world.get_bbox(peg, env_id=None)  # (N, 8, 3), (N, 3)
    board_corners_w, _ = world.get_bbox(board, env_id=None)
    board_pos_w, board_quat_w = world.get_pose(board, env_id=None)  # (N, 3), (N, 4)

    peg_local = _to_local(peg_corners_w, board_pos_w, board_quat_w)
    board_local = _to_local(board_corners_w, board_pos_w, board_quat_w)

    board_top_z = board_local[..., 2].max(dim=1).values
    board_bottom_z = board_local[..., 2].min(dim=1).values
    thickness = board_top_z - board_bottom_z
    peg_min_z = peg_local[..., 2].min(dim=1).values

    depth_ok = peg_min_z <= board_top_z - insertion_fraction * thickness + depth_tolerance_m

    peg_centroid_local = _to_local(peg_centroid_w.unsqueeze(1), board_pos_w, board_quat_w).squeeze(1)
    hole = torch.tensor(hole_offset_m, device=peg_centroid_local.device, dtype=peg_centroid_local.dtype)
    lateral_ok = torch.linalg.norm(peg_centroid_local[:, :2] - hole, dim=-1) <= lateral_tolerance_m

    _, peg_quat_w = world.get_pose(peg, env_id=None)
    rel_quat = _quat_mul(_quat_conjugate(board_quat_w), peg_quat_w)
    up = torch.tensor([0.0, 0.0, 1.0], device=rel_quat.device, dtype=rel_quat.dtype).expand(rel_quat.shape[0], 3)
    peg_axis_local = _quat_apply(rel_quat, up)
    cos_tilt = peg_axis_local[:, 2].abs().clamp(max=1.0)  # an upside-down peg also counts as aligned
    axis_ok = cos_tilt >= math.cos(math.radians(axis_tolerance_deg))

    result: torch.Tensor = depth_ok & lateral_ok & axis_ok
    return result
