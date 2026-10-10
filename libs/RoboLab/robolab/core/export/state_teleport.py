"""states.h5 trajectory loader and batched teleport to arbitrary (traj, frame) states.

Shared by the renderer and probes; do not duplicate it (the renderer's spot check validates this
implementation). Unlike ``rc_teleport`` it needs no curriculum state attached to the env.

Frame convention (see ``state_recorder``): root_pose = env-relative pos(3) + world quat(4, wxyz),
root_vel = world-frame lin(3) + ang(3). Only the position gets ``env_origins`` added on teleport.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import h5py
import numpy as np
import torch


def load_trajs(states_dir: str, device: str) -> list[dict[str, Any]]:
    """Load every ``episode_*.h5`` in ``states_dir`` into a flat trajectory list.

    Order is sorted file name, then ascending ``traj_`` index. ``traj_uid`` is the stable identity of a
    trajectory (repartitioning renumbers the groups).

    Args:
        states_dir: directory containing ``episode_*.h5``.
        device: device for the loaded tensors.

    Returns:
        List of trajectory dicts (states, ``actions``, ``success``, ``skill_z`` or None, ``traj_uid``).
    """
    paths = sorted(glob.glob(os.path.join(states_dir, "episode_*.h5")))
    if not paths:
        raise FileNotFoundError(f"no episode_*.h5 in states_dir: {states_dir}")

    trajs: list[dict[str, Any]] = []
    for path in paths:
        with h5py.File(path, "r") as hf:
            names = sorted((k for k in hf.keys() if k.startswith("traj_")), key=lambda k: int(k.split("_")[1]))
            for name in names:
                g = hf[name]
                st = g["states"]
                d: dict[str, Any] = {
                    "robot_jp": torch.as_tensor(np.asarray(st["robot"]["joint_pos"][:]), device=device),
                    "robot_jv": torch.as_tensor(np.asarray(st["robot"]["joint_vel"][:]), device=device),
                    "objects": {
                        n: {
                            "root_pose": torch.as_tensor(np.asarray(st["objects"][n]["root_pose"][:]), device=device),
                            "root_vel": torch.as_tensor(np.asarray(st["objects"][n]["root_vel"][:]), device=device),
                        }
                        for n in st["objects"].keys()
                    },
                    "arts": {},
                    "actions": np.asarray(g["actions"][:]).astype(np.float32),
                    "success": np.asarray(g["success"][:]).astype(bool),
                    "skill_z": np.asarray(g["skill_z"][:]).astype(np.float32) if "skill_z" in g else None,
                    "src_file": os.path.basename(path),
                    "src_group": name,
                    "traj_uid": str(g.attrs["traj_uid"]),
                }
                if "articulations" in st:
                    for n in st["articulations"].keys():
                        ag = st["articulations"][n]
                        d["arts"][n] = {k: torch.as_tensor(np.asarray(ag[k][:]), device=device) for k in ag.keys()}
                trajs.append(d)
    return trajs


def teleport_frames(
    scene: Any,
    robot: Any,
    origins: torch.Tensor,
    obj_names: list[str],
    art_names: list[str],
    slot_trajs: list[dict[str, Any]],
    slot_frames: list[int],
    slot_eids: list[int],
    device: str,
) -> None:
    """Write the full state of ``slot_trajs[i]`` at frame ``slot_frames[i]`` into env ``slot_eids[i]``.

    Frames are clamped to the trajectory length (short trajectories hold their last frame).
    """
    k = len(slot_eids)
    if not (len(slot_trajs) == len(slot_frames) == k):
        raise ValueError(f"slot length mismatch: trajs={len(slot_trajs)} frames={len(slot_frames)} eids={k}")
    eids_t = torch.as_tensor(slot_eids, dtype=torch.long, device=device)
    org = origins[eids_t]

    fr = [min(int(slot_frames[i]), int(slot_trajs[i]["robot_jp"].shape[0]) - 1) for i in range(k)]

    rjp = torch.stack([slot_trajs[i]["robot_jp"][fr[i]] for i in range(k)], dim=0)
    rjv = torch.stack([slot_trajs[i]["robot_jv"][fr[i]] for i in range(k)], dim=0)
    robot.write_joint_state_to_sim(rjp, rjv, env_ids=eids_t)

    for name in obj_names:
        obj = scene.rigid_objects[name]
        pose = torch.stack([slot_trajs[i]["objects"][name]["root_pose"][fr[i]] for i in range(k)], dim=0).clone()
        pose[:, :3] = pose[:, :3] + org
        vel = torch.stack([slot_trajs[i]["objects"][name]["root_vel"][fr[i]] for i in range(k)], dim=0)
        obj.write_root_pose_to_sim(pose, env_ids=eids_t)
        obj.write_root_velocity_to_sim(vel, env_ids=eids_t)

    for name in art_names:
        art = scene.articulations[name]
        pose = torch.stack([slot_trajs[i]["arts"][name]["root_pose"][fr[i]] for i in range(k)], dim=0).clone()
        pose[:, :3] = pose[:, :3] + org
        vel = torch.stack([slot_trajs[i]["arts"][name]["root_vel"][fr[i]] for i in range(k)], dim=0)
        ajp = torch.stack([slot_trajs[i]["arts"][name]["joint_pos"][fr[i]] for i in range(k)], dim=0)
        ajv = torch.stack([slot_trajs[i]["arts"][name]["joint_vel"][fr[i]] for i in range(k)], dim=0)
        art.write_root_pose_to_sim(pose, env_ids=eids_t)
        art.write_root_velocity_to_sim(vel, env_ids=eids_t)
        art.write_joint_state_to_sim(ajp, ajv, env_ids=eids_t)
