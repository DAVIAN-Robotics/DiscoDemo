# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the workcell joint gains, friction, limits and home pose into the articulation.

Applied at runtime right after the env is built, and verified by readback, so a value
in ``workcell.json`` either reaches the simulation or raises. These are simulator joint-drive
values; the real arm runs a task-space impedance controller.
"""

import math
from typing import Any

import torch

from real2sim.workcell import knob

# Workcell key -> articulation writer.
_WRITERS = {
    "joint_stiffness": "write_joint_stiffness_to_sim",
    "joint_damping": "write_joint_damping_to_sim",
    "joint_armature": "write_joint_armature_to_sim",
    "joint_friction": "write_joint_friction_coefficient_to_sim",
}


def resolve_joint(name: str, joint_names: list[str]) -> str:
    """Map a workcell joint name (``fr3_joint3``) onto the articulation (``panda_joint3``).

    Exact names win; otherwise the vendor prefix is stripped and the rest must match
    exactly one joint (so ``joint1`` never matches ``panda_finger_joint1``).
    """
    if name in joint_names:
        return name
    suffix = name.split("_", 1)[-1]  # "fr3_joint3" -> "joint3"
    hits = [j for j in joint_names if j.split("_", 1)[-1] == suffix]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"workcell joint {name!r} not found in the articulation (candidates {hits}, all {joint_names})")


def joint_gain_vector(doc: dict[str, Any], key: str, joint_names: list[str]) -> dict[str, float]:
    """Map a workcell ``{joint: value}`` table onto articulation joint names."""
    return {resolve_joint(n, joint_names): float(v) for n, v in knob(doc, f"robot.{key}").items()}


def home_joint_vector(doc: dict[str, Any], joint_names: list[str]) -> dict[str, float]:
    """Resolve the measured seven-joint FR3 home pose onto a live articulation."""
    try:
        home_knob = doc["robot"]["home_joint_pos"]
    except KeyError as exc:
        raise KeyError("SysID workcell is missing robot.home_joint_pos") from exc
    if not isinstance(home_knob, dict) or "value" not in home_knob or "source" not in home_knob:
        raise ValueError("robot.home_joint_pos must be a {value, source} knob")
    if home_knob["source"] != "measured":
        raise ValueError(f"robot.home_joint_pos must have source=measured for a SysID run; got {home_knob['source']!r}")
    table = home_knob["value"]
    if not isinstance(table, dict) or len(table) != 7:
        raise ValueError(
            "robot.home_joint_pos must contain exactly seven arm joints; "
            f"got {0 if not isinstance(table, dict) else len(table)}"
        )
    resolved: dict[str, float] = {}
    for source_name, raw_value in table.items():
        live_name = resolve_joint(source_name, joint_names)
        if "finger" in live_name:
            raise ValueError(f"robot.home_joint_pos contains gripper joint {source_name!r}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"robot.home_joint_pos.{source_name} is not finite: {raw_value!r}")
        if live_name in resolved:
            raise ValueError(f"multiple home joints map to {live_name!r}")
        resolved[live_name] = value
    if len(resolved) != 7:
        raise ValueError(f"robot.home_joint_pos resolved to {len(resolved)} joints, expected 7")
    return resolved


def apply_home_joint_pos(env: Any, doc: dict[str, Any], asset_name: str = "robot") -> dict[str, float]:
    """Install the workcell home as the articulation live default reset pose."""
    robot = env.scene[asset_name]
    joint_names: list[str] = list(robot.data.joint_names)
    resolved = home_joint_vector(doc, joint_names)
    ids = [joint_names.index(name) for name in resolved]
    values = torch.tensor([resolved[joint_names[i]] for i in ids], device=robot.device, dtype=torch.float32)
    robot.data.default_joint_pos[:, ids] = values.unsqueeze(0).expand(robot.num_instances, -1)
    landed = robot.data.default_joint_pos[0, ids]
    if not torch.allclose(landed, values, atol=1e-7, rtol=0.0):
        raise RuntimeError(
            "robot.home_joint_pos did not become the default reset pose: "
            f"wrote={values.tolist()} read={landed.tolist()}"
        )
    return resolved


def apply_joint_gains(env: Any, doc: dict[str, Any], asset_name: str = "robot") -> dict[str, int | str]:
    """Write the workcell joint gains, friction and limits into the articulation.

    Returns
    -------
    dict[str, int | str]
        Number of joints written per item.
    """
    robot = env.scene[asset_name]
    joint_names: list[str] = list(robot.data.joint_names)
    applied: dict[str, int | str] = {}
    for key, writer_name in _WRITERS.items():
        table = joint_gain_vector(doc, key, joint_names)
        ids = [joint_names.index(n) for n in table]
        tensor = torch.tensor([table[joint_names[i]] for i in ids], device=robot.device, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).repeat(robot.num_instances, 1)
        if key == "joint_friction":
            applied[key] = _write_friction(robot, ids, tensor[0])
            continue
        getattr(robot, writer_name)(tensor, joint_ids=ids)
        applied[key] = len(ids)

    # Limits are applied together with the gains so every caller gets the real bounds.
    from real2sim.joint_limits import apply_joint_limits

    applied["joint_limits"] = apply_joint_limits(env, doc, asset_name=asset_name)
    return applied


def _write_friction(robot: Any, ids: list[int], values: torch.Tensor) -> str:
    """Write joint friction through the PhysX view and verify it by readback.

    ``Articulation.write_joint_friction_coefficient_to_sim`` does not reach PhysX on
    Isaac Sim 5 (the >=5 branch never calls a setter).
    """
    view = robot.root_physx_view
    if not hasattr(view, "set_dof_friction_properties"):
        raise RuntimeError(
            "root_physx_view has no set_dof_friction_properties; joint friction cannot be "
            "written on this Isaac build and must not be reported as applied"
        )
    props = view.get_dof_friction_properties()
    if props.ndim != 3 or props.shape[2] < 2:
        raise RuntimeError(f"unexpected PhysX friction property shape {tuple(props.shape)}")
    # PhysX column 0 is static effort and column 1 is dynamic effort. The
    # workcell single vector means static=dynamic; column 0 alone makes moving joints frictionless.
    friction_cpu = values.detach().cpu().to(props.dtype)
    props[:, ids, 0] = friction_cpu
    props[:, ids, 1] = friction_cpu
    view.set_dof_friction_properties(props, indices=torch.arange(robot.num_instances, dtype=torch.int32))
    back = view.get_dof_friction_properties()[0, ids, :2].cpu()
    want = torch.stack((friction_cpu, friction_cpu), dim=1).to(back.dtype)
    if not torch.allclose(back, want, atol=1e-4, rtol=0.0):
        raise RuntimeError(f"joint friction did not stick: wrote {want.tolist()} read {back.tolist()}")
    # Keep IsaacLab mirror honest so anything reading robot.data sees the truth.
    robot._data.joint_friction_coeff[:, ids] = values.to(robot.device)
    return f"{len(ids)} static+dynamic (via physx view; IsaacLab writer is a no-op on Isaac Sim 5)"
