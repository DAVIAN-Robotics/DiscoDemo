"""Demonstration loader and transition extraction for the reverse curriculum.

Demos are RoboLab ``StateRecorder`` .h5 files (frame convention
``env_relative_pos+world_quat;world_frame_velocity``):

    traj_<n>/
      actions  [T-1, 8]   absolute arm joint target (joint 7 in the command frame, pi/4 below the physical joint)
                          + gripper command in [0, 1]
      success  [T]  bool
      states/
        robot/{joint_pos[T,nj], joint_vel[T,nj]}
        objects/<name>/{root_pose[T,7], root_vel[T,6]}
        articulations/<name>/{joint_pos, joint_vel, root_pose, root_vel}

``rc_teleport`` writes the ``states`` of a frame into the simulator.
"""

from __future__ import annotations

import os
from typing import Any

import h5py
import numpy as np
import torch

from robolab.robots.action_norm import normalize_gripper, relative_delta_demo_arm


def _load_nested_states(group: Any) -> Any:
    """Recursively load an h5 group into a nested dict of [T, D] tensors."""
    if hasattr(group, "shape"):
        return torch.as_tensor(np.asarray(group[:]))
    return {k: _load_nested_states(group[k]) for k in group}


def _truncate_state_dict(state_dict: Any, new_len: int) -> Any:
    """Truncate every leaf tensor to ``new_len`` along dim 0."""
    if isinstance(state_dict, dict):
        return {k: _truncate_state_dict(v, new_len) for k, v in state_dict.items()}
    return state_dict[:new_len]


def load_demo_states_robolab(h5_path: str, truncate_after_success: int) -> list[dict[str, Any]]:
    """Load every demonstration of a bank (all must end in success).

    Parameters
    ----------
    h5_path : str
        StateRecorder .h5.
    truncate_after_success : int
        Keep at most this many trailing success frames (states, actions and success are cut
        together).

    Returns
    -------
    list[dict]
        Per demo: ``state_dict`` (nested ``[T, D]``), ``actions`` ``[T-1, 8]``, ``success_mask``
        ``[T]`` bool and ``total_steps`` ``T``, in ``traj_<n>`` order.
    """
    demos: list[dict[str, Any]] = []
    with h5py.File(os.path.expanduser(h5_path), "r") as hf:
        keys = sorted((k for k in hf if k.startswith("traj_")), key=lambda k: int(k.split("_", 1)[1]))
        for key in keys:
            traj = hf[key]
            success = np.asarray(traj["success"][:]).astype(bool)
            assert success[-1], f"{h5_path}:{key} does not end in success"
            state_dict = _load_nested_states(traj["states"])
            actions = torch.as_tensor(np.asarray(traj["actions"][:]))
            total = int(success.shape[0])
            # Drop excess trailing success frames; keep T states / T-1 actions aligned.
            excess = int(success.sum()) - truncate_after_success
            if excess > 0:
                total -= excess
                state_dict = _truncate_state_dict(state_dict, total)
                actions = actions[: total - 1]
            demos.append(
                {
                    "state_dict": state_dict,
                    "actions": actions,
                    "success_mask": torch.as_tensor(success[:total]),
                    "total_steps": total,
                }
            )
    assert demos, f"no demonstrations in {h5_path}"
    return demos


def _slice_stacked_demo_observations(
    frame_observations: list[torch.Tensor], trajectory_lengths: list[int]
) -> list[torch.Tensor]:
    """Convert vectorized ``[time, demo, obs]`` captures into per-demo views."""
    if not frame_observations:
        raise ValueError("frame_observations must not be empty")
    stacked = torch.stack(frame_observations, dim=0)
    if stacked.ndim < 3 or stacked.shape[1] < len(trajectory_lengths):
        raise ValueError("frame observation batch does not cover every demo")
    if max(trajectory_lengths) > stacked.shape[0]:
        raise ValueError("trajectory length exceeds captured frame count")
    return [stacked[:length, demo_id] for demo_id, length in enumerate(trajectory_lengths)]


def extract_demo_transitions_robolab(demos: list[dict[str, Any]], env: Any, device: str) -> dict[str, torch.Tensor]:
    """Materialize demos as (obs, action, next_obs, reward, terminated) transitions.

    Every demo frame is reached by a teleport reset, so observations match the training
    observation exactly. ``env`` is a ``ReverseCurriculumIsaacEnv`` built with
    ``fixed_frontier=True`` and ``to_numpy=False`` (env ``i`` replays demo ``i``).

    - For each frame t, demo i gets ``start_step = min(t, T_i - 1)`` and one reset teleports env i.
    - Actions are the wrapper's policy-space demo actions; the trailing prev_action columns are
      a_{s-1} (zeros at s=0).
    - The reward is ``success_mask[1:]`` (1 on every success frame of the demonstration tail);
      only each demo's last transition is terminated.

    Returns
    -------
    dict[str, torch.Tensor]
        ``observation, next_observation, action, reward, terminated`` on ``device``.
    """
    dev = torch.device(device)
    core = env.core
    assert core.fixed_frontier and env.num_envs >= len(demos)
    lengths = [int(d["total_steps"]) for d in demos]
    frame_observations: list[torch.Tensor] = []
    for t in range(max(lengths)):
        for i, length in enumerate(lengths):
            core.demo_metadata[i].start_step = min(t, length - 1)
        obs, _ = env.reset()
        frame_observations.append(obs[: len(demos)].detach().clone())
    per_demo_obs = _slice_stacked_demo_observations(frame_observations, lengths)

    out: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("observation", "next_observation", "action", "reward", "terminated")
    }
    for i, demo in enumerate(demos):
        obs_full = per_demo_obs[i].to(dev)  # [T, D]
        actions = env.base_env._rc_demo_actions[i].to(dev)  # [T-1, 8]
        n = obs_full.shape[0]
        # obs(s).prev_action = a_{s-1} (zeros at s=0).
        obs_full[0, -actions.shape[-1] :] = 0.0
        obs_full[1:, -actions.shape[-1] :] = actions
        terminated = torch.zeros(n - 1, dtype=torch.bool, device=dev)
        terminated[-1] = True
        out["observation"].append(obs_full[:-1])
        out["next_observation"].append(obs_full[1:])
        out["action"].append(actions)
        out["reward"].append(demo["success_mask"][1:].to(dev).float())
        out["terminated"].append(terminated)
    return {k: torch.cat(v, dim=0) for k, v in out.items()}


def relative_demo_actions(actions: torch.Tensor, q_cur: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """Demo action ``[T-1, 8]`` -> relative policy action in ``[-1, 1]^8``.

    Arm = clipped ``(target - q_cur) / rate`` (joint 7 converted from the command frame); gripper =
    ``2 * g - 1``. Inverse of the env's relative action term ``q* = q_cur + a * rate``.
    ``q_cur`` is the arm joint position at each transition, ``[T-1, 7]``.
    """
    arm = relative_delta_demo_arm(actions[:, :7], q_cur, rate)
    grip = normalize_gripper(actions[:, 7:8])
    return torch.cat([arm, grip], dim=-1)
