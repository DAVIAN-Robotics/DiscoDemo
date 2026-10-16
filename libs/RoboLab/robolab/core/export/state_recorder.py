# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teleport-complete state recorder for RoboLab (Isaac Lab) rollouts.

Captures the complete sim state at every frame so any frame can later be restored exactly by
teleport (e.g. for a reverse curriculum or offline rendering). Recorded per frame:

- robot joint positions / velocities (fixed base, so no root state);
- root pose and velocity of every rigid object;
- joint state, root pose and root velocity of every non-robot articulation.

Frame convention (also stored as the h5 root attr ``frame_convention``)
-----------------------------------------------------------------------
- Root position is env-relative (world_pos - env_origin); add env_origin back on teleport.
- Root quaternion is in the world frame (wxyz); invariant to the env-origin translation.
- Root velocity ``[lin(3), ang(3)]`` is in the world frame; a constant translation does not change
  it, so it is written back unchanged.

H5 layout
---------
    /                         (attrs: frame_convention)
      traj_<n>/               (attrs: traj_uid)
        actions   [T-1, 8]    # arm 7 + gripper 1
        success   [T]  bool   # True on the last frame
        skill_z   [d]         # skill-conditioned generators only
        extras/<key> [T-1, ...]  # per-transition records (contact forces)
        states/
          robot/
            joint_pos [T, nj]
            joint_vel [T, nj]
          objects/
            <obj_name>/
              root_pose [T, 7]   # env-relative pos(3) + world quat(4, wxyz)
              root_vel  [T, 6]   # world frame lin(3)+ang(3)
          articulations/
            <art_name>/          # non-robot articulation
              joint_pos [T, nj]
              joint_vel [T, nj]
              root_pose [T, 7]
              root_vel  [T, 6]

Frame / action alignment
------------------------
Applying action[t] in state[t] yields state[t+1], so there are T states and T-1 actions. The env
auto-resets when an episode ends, so the last recorded state is the one before the final step, and
the final action is dropped.
"""

import json
import os

import h5py
import numpy as np
import torch

from robolab.core.export.skill_z import resolve_episode_skill_z
from robolab.core.world.world_state import get_world
from robolab.robots.action_norm import drop_ineffective_gripper_requests

FRAME_CONVENTION = "env_relative_pos+world_quat;world_frame_velocity"


class StateRecorder:
    """Per-env, teleport-complete state recorder for multi-episode collection.

    Every step: ``capture_frame`` (state before the step), ``env.step``, ``attach_action`` (the command
    applied in that step). When an env's episode ends: ``mark_last_frame_success`` if it succeeded, then
    ``close_episode``. ``write`` stores the successful closed episodes in ``episode_<id>.h5`` and appends
    them to the sibling ``index.json``.

    Parameters
    ----------
    env : ManagerBasedRLEnv
        RoboLab env.
    out_dir : str
        Output directory (must exist).
    """

    def __init__(self, env, out_dir: str):
        self.env = env
        self.out_dir = out_dir
        self.world = get_world(env)
        self.num_envs = int(env.num_envs)
        # robot: joints only; other articulations: joints + root; rigid objects: root only.
        articulation_names = list(env.scene.articulations.keys())
        if "robot" not in articulation_names:
            raise ValueError(f"StateRecorder: no 'robot' articulation in the scene: {articulation_names}")
        self.nonrobot_articulations = [n for n in articulation_names if n != "robot"]
        self.object_names = list(env.scene.rigid_objects.keys())
        self._open = [self._new_episode() for _ in range(self.num_envs)]
        # Closed episodes (with env id and per-env sequence number) waiting for ``write``.
        self._pending: list[tuple[int, int, dict]] = []
        self._ep_seq = [0] * self.num_envs

    @staticmethod
    def _new_episode() -> dict:
        # frames: nested state dicts; actions: [8] tensors; grip_applied: bools; extras: {key: tensor}.
        return {"frames": [], "success": [], "actions": [], "grip_applied": [], "extras": [], "skill_z": []}

    def _build_frame(self, env_id: int) -> dict:
        """Nested state dict of one env (leaves are CPU tensors)."""
        w = self.world
        frame = {
            "robot": {
                "joint_pos": w.get_joint_positions("robot", env_id=env_id).detach().cpu(),
                "joint_vel": w.get_joint_velocity("robot", env_id=env_id).detach().cpu(),
            },
            "objects": {},
            "articulations": {},
        }
        for name in self.object_names:
            pos, quat = w.get_pose(name, is_relative=True, env_id=env_id)
            frame["objects"][name] = {
                "root_pose": torch.cat([pos.detach().cpu(), quat.detach().cpu()], dim=-1),  # [7]
                "root_vel": w.get_velocity(name, env_id=env_id).detach().cpu(),  # [6]
            }
        for name in self.nonrobot_articulations:
            pos, quat = w.get_pose(name, is_relative=True, env_id=env_id)
            frame["articulations"][name] = {
                "joint_pos": w.get_joint_positions(name, env_id=env_id).detach().cpu(),
                "joint_vel": w.get_joint_velocity(name, env_id=env_id).detach().cpu(),
                "root_pose": torch.cat([pos.detach().cpu(), quat.detach().cpu()], dim=-1),
                "root_vel": w.get_velocity(name, env_id=env_id).detach().cpu(),
            }
        return frame

    def capture_frame(self, skill_z=None) -> None:
        """Capture the current state of every env (before ``env.step``).

        Parameters
        ----------
        skill_z : array-like, optional
            ``(num_envs, z_dim)`` skill of each env; must be constant within an episode.
        """
        for env_id, ep in enumerate(self._open):
            ep["frames"].append(self._build_frame(env_id))
            ep["success"].append(False)
            if skill_z is not None:
                ep["skill_z"].append(np.asarray(skill_z[env_id], dtype=np.float32).copy())

    def attach_action(self, action, gripper_applied, extras=None) -> None:
        """Attach the command applied in the last step to the frame captured before it.

        IsaacLab processes a new action inside ``env.step``, so the applied command is only readable after
        the step.

        Parameters
        ----------
        action : torch.Tensor
            ``[num_envs, 8]``; the last column is the gripper request {0, 1}.
        gripper_applied : torch.Tensor
            ``[num_envs]`` bool, gripper close actually applied in this step. With the delayed gripper it
            can differ from the request (see ``_clean_gripper``).
        extras : dict[str, torch.Tensor], optional
            Per-transition records ``{key: [num_envs, ...]}``, written to ``traj_<n>/extras/<key>``.
        """
        for env_id, ep in enumerate(self._open):
            if len(ep["actions"]) != len(ep["frames"]) - 1:
                raise ValueError(f"env {env_id}: the last frame is not waiting for an action")
            ep["actions"].append(action[env_id].detach().cpu().clone())
            ep["grip_applied"].append(bool(gripper_applied[env_id]))
            if extras is not None:
                ep["extras"].append({k: v[env_id].detach().cpu().clone() for k, v in extras.items()})

    def mark_last_frame_success(self, env_id: int) -> None:
        """Flag the last captured frame of ``env_id``'s episode as successful.

        The env ends an episode on success after the step; the terminal state is gone by then (auto-reset),
        so the success is put on the state before the final step.
        """
        self._open[env_id]["success"][-1] = True

    def close_episode(self, env_id: int) -> None:
        """Queue the finished episode of ``env_id``; the env continues with its auto-reset episode."""
        self._pending.append((env_id, self._ep_seq[env_id], self._open[env_id]))
        self._ep_seq[env_id] += 1
        self._open[env_id] = self._new_episode()

    def pending_count(self) -> int:
        """Number of closed episodes not yet written."""
        return len(self._pending)

    def clear_pending(self) -> None:
        """Drop the closed episodes (after ``write``); open episodes are kept.

        Sequence numbers restart: uids stay unique because the write id differs.
        """
        self._pending = []
        self._ep_seq = [0] * self.num_envs

    @staticmethod
    def _stack_frames(frames: list) -> dict:
        """Stack T nested state dicts into a nested dict of ``[T, D]`` tensors."""
        first = frames[0]
        if isinstance(first, dict):
            return {k: StateRecorder._stack_frames([f[k] for f in frames]) for k in first}
        return torch.stack(frames, dim=0)

    @staticmethod
    def _clean_gripper(actions: list, gripper_applied: list) -> torch.Tensor:
        """``[T, 8]`` actions with gripper requests that were never applied removed.

        A delayed gripper actuator overwrites its pending slot whenever the request changes, so a request
        shorter than the delay is never applied. Such requests are jitter and are replaced by the previous
        label (``drop_ineffective_gripper_requests``).
        """
        stacked = torch.stack(actions, dim=0).clone()
        applied = torch.tensor(gripper_applied, dtype=stacked.dtype)
        stacked[:, -1] = drop_ineffective_gripper_requests(stacked[:, -1], applied)
        return stacked

    def write(self, episode_id: int) -> str:
        """Write the successful closed episodes to ``episode_<episode_id>.h5`` and update ``index.json``.

        Returns
        -------
        str
            Path of the written .h5 file.
        """
        h5_path = os.path.join(self.out_dir, f"episode_{episode_id}.h5")
        entries = []
        with h5py.File(h5_path, "w") as hf:
            hf.attrs["frame_convention"] = FRAME_CONVENTION
            for env_id, seq, ep in self._pending:
                if not ep["success"][-1]:
                    continue
                T = len(ep["frames"])
                if len(ep["actions"]) != T:
                    raise ValueError(f"env {env_id}: {len(ep['actions'])} actions for {T} frames")
                group = f"traj_{len(entries)}"
                grp = hf.create_group(group)
                # Identity survives repartitioning; seq 0 keeps the "<write>:<env>" form.
                traj_uid = f"{episode_id}:{env_id}" if seq == 0 else f"{episode_id}:{env_id}:{seq}"
                grp.attrs["traj_uid"] = traj_uid
                # The final action (out of the last recorded state) is dropped: T states, T-1 actions.
                grp.create_dataset("actions", data=self._clean_gripper(ep["actions"], ep["grip_applied"])[:-1].numpy())
                grp.create_dataset("success", data=np.asarray(ep["success"], dtype=bool))
                if ep["extras"]:
                    keys = list(ep["extras"][0])
                    for key in keys:
                        arr = torch.stack([e[key] for e in ep["extras"][:-1]], dim=0)
                        grp.create_dataset(f"extras/{key}", data=arr.numpy())
                self._write_states_group(grp.create_group("states"), self._stack_frames(ep["frames"]))
                z0 = resolve_episode_skill_z(ep["skill_z"], env_id)
                if z0 is not None:
                    grp.create_dataset("skill_z", data=z0)
                entries.append(
                    {"traj_uid": traj_uid, "h5_file": os.path.basename(h5_path), "h5_group": group, "num_frames": T}
                )
        self._update_index(entries)
        return h5_path

    @staticmethod
    def _write_states_group(grp, states: dict) -> None:
        """Recursively write a nested state dict into an h5 group."""
        for key, val in states.items():
            if isinstance(val, dict):
                StateRecorder._write_states_group(grp.create_group(key), val)
            else:
                grp.create_dataset(key, data=val.numpy())

    def _update_index(self, entries: list) -> None:
        """Append entries to ``index.json`` (atomic rewrite: the file is rewritten every batch)."""
        index_path = os.path.join(self.out_dir, "index.json")
        index = {"frame_convention": FRAME_CONVENTION, "episodes": []}
        if os.path.isfile(index_path):
            with open(index_path) as f:
                index = json.load(f)
        index["episodes"].extend(entries)
        tmp_path = index_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(index, f, indent=2)
        os.replace(tmp_path, index_path)
