# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reverse-curriculum teleport reset event and dynamic-timeout termination.

- ``rc_teleport`` (reset EventTerm): teleports each reset env to the complete demo state at a start frame
  sampled from its demo's curriculum frontier.
- ``rc_dynamic_timeout`` (time_out TerminationTerm): replaces the static time_out with a per-env budget set by
  ``rc_teleport`` from the horizon remaining after the start frame.

Both read attributes the curriculum wrapper sets on the unwrapped env before reset:
  - ``env._rc_core``: CurriculumCore (demo assignment, frontier sampling, time limit)
  - ``env._rc_demos``: list of nested demo state dicts (leaves ``[T, D]`` on device)
  - ``env._rc_total_steps``: list of demo lengths
  - ``env._rc_start_step``: long[num_envs], demo frame this episode starts from
  - ``env._rc_trunc_budget``: long[num_envs], dynamic truncation step

State conventions (same as ``StateRecorder``):
  - root_pose = env-relative pos (3) + world quat (4, wxyz); env_origin is added back before writing.
  - root_vel = world-frame lin (3) + ang (3); translation-invariant, written as is.
  - robot joint_pos/joint_vel are full joint vectors in articulation order, so no column remapping is needed.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


def rc_teleport(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Teleport reset envs to the complete state of their sampled demo frame.

    Envs are grouped by demo and written in batches per group. Demos may have different key sets, so there is
    no cross-demo stacking.
    """
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return

    core = env._rc_core
    demos = env._rc_demos
    robot = env.scene.articulations["robot"]

    # Per-env (demo, start) assignment is delegated to core.assign_reset.
    by_demo: dict[int, list[int]] = {}
    starts_by_eid: dict[int, int] = {}
    for eid in env_ids.tolist():
        d, s = core.assign_reset(eid)
        starts_by_eid[eid] = s
        env._rc_start_step[eid] = s
        env._rc_trunc_budget[eid] = core.compute_dynamic_timelimit_from_remaining(int(env._rc_total_steps[d]) - s)
        by_demo.setdefault(d, []).append(eid)

    for demo_idx, eids in by_demo.items():
        demo = demos[demo_idx]
        starts = [starts_by_eid[eid] for eid in eids]
        starts_t = torch.as_tensor(starts, dtype=torch.long, device=env.device)
        eids_t = torch.as_tensor(eids, dtype=torch.long, device=env.device)
        origins = env.scene.env_origins[eids_t]  # [k, 3] world-frame env origin

        # --- robot: full joint state (fixed base, no root) ---
        rjp = demo["robot"]["joint_pos"][starts_t]  # [k, nj]
        rjv = demo["robot"]["joint_vel"][starts_t]  # [k, nj]
        robot.write_joint_state_to_sim(rjp, rjv, env_ids=eids_t)

        # --- rigid objects: root pose (env-rel -> world) + root vel ---
        for name, leaf in demo["objects"].items():
            obj = env.scene.rigid_objects[name]
            pose = leaf["root_pose"][starts_t].clone()  # [k, 7]
            pose[:, :3] = pose[:, :3] + origins         # env-relative -> world
            vel = leaf["root_vel"][starts_t]            # [k, 6] world frame
            obj.write_root_pose_to_sim(pose, env_ids=eids_t)
            obj.write_root_velocity_to_sim(vel, env_ids=eids_t)

        # --- non-robot articulations: joint + root pose/vel ---
        for name, leaf in demo["articulations"].items():
            art = env.scene.articulations[name]
            pose = leaf["root_pose"][starts_t].clone()  # [k, 7]
            pose[:, :3] = pose[:, :3] + origins
            vel = leaf["root_vel"][starts_t]            # [k, 6]
            art.write_root_pose_to_sim(pose, env_ids=eids_t)
            art.write_root_velocity_to_sim(vel, env_ids=eids_t)
            ajp = leaf["joint_pos"][starts_t]
            ajv = leaf["joint_vel"][starts_t]
            art.write_joint_state_to_sim(ajp, ajv, env_ids=eids_t)


def rc_dynamic_timeout(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env dynamic truncation: ``episode_length_buf >= _rc_trunc_budget``."""
    return env.episode_length_buf >= env._rc_trunc_budget
