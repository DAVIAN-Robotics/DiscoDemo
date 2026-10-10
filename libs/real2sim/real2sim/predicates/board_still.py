# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Board-still condition of the peg-in-hole success: the board must not be pushed or turned."""

from __future__ import annotations

from typing import Any


def board_still(env: Any, board_name: str, max_disp_m: float, max_rot_deg: float) -> Any:
    """Whether the board is within ``max_disp_m`` (xy) and ``max_rot_deg`` of its pose at the episode start.

    The reference is the board pose at the first step of each episode (``episode_length_buf == 1`` in Isaac
    Lab), i.e. after the reset events (initial-state sampling, reverse-curriculum teleport).

    Returns
    -------
    torch.Tensor
        (num_envs,) bool.
    """
    import torch

    board = env.scene[board_name]
    pose = torch.cat([board.data.root_pos_w, board.data.root_quat_w], dim=-1)  # (N, 7), wxyz
    if not hasattr(env, "_board_start_pose"):
        env._board_start_pose = pose.clone()
    ref = env._board_start_pose
    first = env.episode_length_buf <= 1
    ref[first] = pose[first]
    disp = torch.linalg.norm(pose[:, :2] - ref[:, :2], dim=-1)
    rot = torch.rad2deg(2.0 * torch.acos((pose[:, 3:7] * ref[:, 3:7]).sum(-1).abs().clamp(max=1.0)))
    return (disp <= max_disp_m) & (rot <= max_rot_deg)
