# SPDX-License-Identifier: Apache-2.0
"""Cube-stack success predicate: the top cube rests on the bottom cube.

Three conditions in the env-local frame:

1. height: top centre z minus bottom centre z equals one edge within tolerance;
2. overlap: horizontal centre distance within the xy tolerance;
3. rest: top cube linear speed below ``settle_speed`` (passing through is not success).

Release is not required. The success hold is applied by RoboLab. Argument names follow the RL
predicate convention (``object``/``container``). Registers itself on import.
"""

from typing import Any

import torch

from robolab.robots.rl_obs import register_success_predicate


def stack_state(env: Any, top: str, bottom: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (dz, dxy, top speed) in the env-local frame."""
    origins = env.scene.env_origins
    top_pos = env.scene[top].data.root_pos_w - origins.to(env.scene[top].data.root_pos_w.dtype)
    bottom_pos = env.scene[bottom].data.root_pos_w - origins.to(env.scene[bottom].data.root_pos_w.dtype)
    dz = top_pos[:, 2] - bottom_pos[:, 2]
    dxy = torch.linalg.norm(top_pos[:, :2] - bottom_pos[:, :2], dim=-1)
    speed = torch.linalg.norm(env.scene[top].data.root_lin_vel_w, dim=-1)
    return dz, dxy, speed


def cube_stacked(
    env: Any,
    object: str,  # noqa: A002  # RL predicate convention
    container: str,
    stack_height_m: float = 0.05,
    height_tolerance_m: float = 0.01,
    xy_tolerance_m: float = 0.025,
    settle_speed: float = 0.02,
) -> torch.Tensor:
    """Return whether ``object`` (top cube) rests on ``container`` (bottom cube).

    Parameters
    ----------
    env : Any
        IsaacLab ManagerBasedRLEnv.
    object : str
        Top cube.
    container : str
        Bottom cube.
    stack_height_m, height_tolerance_m, xy_tolerance_m, settle_speed
        Success tolerances from ``assets/tasks/fr3_cube_stack.json``.

    Returns
    -------
    torch.Tensor
        (num_envs,) bool.
    """
    dz, dxy, speed = stack_state(env, object, container)
    return (torch.abs(dz - stack_height_m) <= height_tolerance_m) & (dxy <= xy_tolerance_m) & (speed <= settle_speed)


register_success_predicate("cube_stacked", cube_stacked)
