# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Peg-in-hole success predicate wrapped in the RL registry convention.

The insertion test is ``fmb.predicates.peg_in_hole``; in addition the board must not have been pushed or
turned. Registers itself with the RL layer on import.
"""

from __future__ import annotations

from typing import Any

from real2sim.predicates.board_still import board_still
from robolab.robots.rl_obs import register_success_predicate


def peg_inserted(
    env: Any,
    object: str,  # noqa: A002  # RL predicate convention
    container: str,
    hole_offset_m: tuple[float, float],
    insertion_fraction: float,
    lateral_tolerance_m: float,
    axis_tolerance_deg: float,
    depth_tolerance_m: float,
    board_max_disp_m: float,
    board_max_rot_deg: float,
) -> Any:
    """Return whether the peg (``object``) is fully inserted in its hole and the board (``container``) stayed put.

    Parameters
    ----------
    env : Any
        IsaacLab ManagerBasedRLEnv.
    object : str
        Peg.
    container : str
        Board.
    hole_offset_m : tuple[float, float]
        Target point in the board local frame (hole centre, or midpoint of two holes).
    insertion_fraction, lateral_tolerance_m, axis_tolerance_deg, depth_tolerance_m
        Passed to ``peg_in_hole`` (values set by the task).
    board_max_disp_m, board_max_rot_deg
        The board must stay within this xy distance and rotation of its pose at the start of the episode, so
        insertions that push or turn the board onto the peg do not count.

    Returns
    -------
    torch.Tensor
        (num_envs,) bool.
    """
    from fmb.predicates.peg_in_hole import peg_in_hole

    inserted = peg_in_hole(
        env,
        peg=object,
        board=container,
        hole_offset_m=tuple(hole_offset_m),
        insertion_fraction=float(insertion_fraction),
        lateral_tolerance_m=float(lateral_tolerance_m),
        axis_tolerance_deg=float(axis_tolerance_deg),
        depth_tolerance_m=float(depth_tolerance_m),
    )
    return inserted & board_still(env, container, float(board_max_disp_m), float(board_max_rot_deg))


register_success_predicate("peg_inserted", peg_inserted)
