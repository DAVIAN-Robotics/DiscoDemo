# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Peg-in-hole success predicate wrapped in the RL registry convention.

The insertion test is ``fmb.predicates.peg_in_hole``. Registers itself with the RL layer on import.
"""

from __future__ import annotations

from typing import Any

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
) -> Any:
    """Return whether the peg (``object``) is fully inserted in its hole in the board (``container``).

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

    Returns
    -------
    torch.Tensor
        (num_envs,) bool.
    """
    from fmb.predicates.peg_in_hole import peg_in_hole

    return peg_in_hole(
        env,
        peg=object,
        board=container,
        hole_offset_m=tuple(hole_offset_m),
        insertion_fraction=float(insertion_fraction),
        lateral_tolerance_m=float(lateral_tolerance_m),
        axis_tolerance_deg=float(axis_tolerance_deg),
        depth_tolerance_m=float(depth_tolerance_m),
    )


register_success_predicate("peg_inserted", peg_inserted)
