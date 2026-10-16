"""Physical / workspace bounds for normalizing the RoboLab policy observation.

The normalizer window comes from the task definition rather than demo statistics: object
positions use a task box, quaternions use ``[-1, 1]``, arm joints use the articulation soft limits
and the end effector uses a workspace box. Unknown term names, layout mismatches and
``low >= high`` raise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf

# Virtual term name for the 2-dim gripper-delay obs that the env wrapper inserts before
# ``prev_action`` (see ``RoboLabVectorEnv._augment_policy_obs``); it is not an obs-manager term.
GRIPPER_DELAY_TERM = "gripper_delay"
GRIPPER_DELAY_DIM = 2
_AXES = ("x", "y", "z")


def _pair(value: Any, what: str) -> tuple[float, float]:
    """Convert a two-element ``(lo, hi)`` sequence to floats; raise on bad length or order."""
    lo, hi = (float(v) for v in value)
    if not lo < hi:
        raise ValueError(f"{what}: requires lo({lo}) < hi({hi})")
    return lo, hi


@dataclass(frozen=True)
class ObsBoundsSpec:
    """The ``obs_bounds:`` config block.

    Parameters
    ----------
    object_xy : tuple[tuple[float, float], tuple[float, float]]
        Object xy box ``((x_lo, x_hi), (y_lo, y_hi))`` in the robot base frame (m).
    object_z_m : tuple[float, float]
        Object z range (m).
    eef_box_m : dict[str, tuple[float, float]]
        End-effector workspace box with keys ``x``, ``y`` and ``z``.
    """

    object_xy: tuple[tuple[float, float], tuple[float, float]]
    object_z_m: tuple[float, float]
    eef_box_m: dict[str, tuple[float, float]]

    @classmethod
    def from_config(cls, node: Any) -> ObsBoundsSpec:
        """Build from the OmegaConf node; missing or unknown keys raise."""
        doc = OmegaConf.to_container(node, resolve=True)
        assert isinstance(doc, dict) and set(doc) == {"object_xy", "object_z_m", "eef_box_m"}, doc
        x, y = doc["object_xy"]
        return cls(
            object_xy=(_pair(x, "obs_bounds.object_xy.x"), _pair(y, "obs_bounds.object_xy.y")),
            object_z_m=_pair(doc["object_z_m"], "obs_bounds.object_z_m"),
            eef_box_m={a: _pair(doc["eef_box_m"][a], f"obs_bounds.eef_box_m.{a}") for a in _AXES},
        )


def term_layout_from_env(env: Any) -> list[tuple[str, int]]:
    """Return ``(term name, flat dim)`` for the policy obs, in column order.

    Reads the Isaac ``ObservationManager`` policy group and inserts the 2-dim gripper-delay term
    right before ``prev_action``. The dims sum to the env obs dim.
    """
    om = env.unwrapped.observation_manager
    names = [str(n) for n in om.active_terms["policy"]]
    dims = [int(math.prod(tuple(d))) for d in om.group_obs_term_dim["policy"]]
    layout = list(zip(names, dims, strict=True))
    if not layout or layout[-1][0] != "prev_action":
        raise ValueError(f"the last policy obs term must be prev_action, got {names}")
    layout.insert(len(layout) - 1, (GRIPPER_DELAY_TERM, GRIPPER_DELAY_DIM))
    total = sum(d for _, d in layout)
    obs_dim = int(env.observation_space.shape[-1])
    if total != obs_dim:
        raise ValueError(f"obs term layout sum ({total}) != obs dim ({obs_dim}); layout={layout}")
    return layout


def arm_joint_limits_from_env(env: Any, asset_name: str = "robot") -> torch.Tensor:
    """Arm joint limits ``[7, 2]`` (lower, upper) in the same order and frame as ``arm_joint_pos``.

    Joints are taken in articulation order, and the joint-7 mount offset is subtracted because the
    obs reports joint 7 relative to that offset.
    """
    from robolab.robots.action_norm import ARM_J7_INDEX, PANDA_J7_MOUNT_OFFSET

    robot = env.unwrapped.scene[asset_name]
    wanted = {f"panda_joint{i}" for i in range(1, 8)}
    joint_ids = [i for i, name in enumerate(robot.data.joint_names) if name in wanted]
    if len(joint_ids) != 7:
        raise ValueError(f"could not find 7 arm joints: candidates {joint_ids}, all {list(robot.data.joint_names)}")
    limits: torch.Tensor = robot.data.soft_joint_pos_limits[0, joint_ids, :].detach().to("cpu", torch.float32)
    limits = limits.clone()
    limits[ARM_J7_INDEX, :] -= float(PANDA_J7_MOUNT_OFFSET)
    return limits


def resolve_obs_bounds(
    layout: Sequence[tuple[str, int]], spec: ObsBoundsSpec, *, joint_limits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalization bounds ``(low, high)``, each ``[D]`` float32, from a term layout.

    Parameters
    ----------
    layout : Sequence[tuple[str, int]]
        Output of ``term_layout_from_env``.
    spec : ObsBoundsSpec
        Task bounds.
    joint_limits : torch.Tensor
        ``[7, 2]`` arm joint limits in the obs frame (see ``arm_joint_limits_from_env``).
    """
    total = sum(int(d) for _, d in layout)
    low = torch.full((total,), float("nan"), dtype=torch.float32)
    high = torch.full((total,), float("nan"), dtype=torch.float32)
    (obj_x, obj_y), obj_z = spec.object_xy, spec.object_z_m
    obj_lo = (obj_x[0], obj_y[0], obj_z[0])
    obj_hi = (obj_x[1], obj_y[1], obj_z[1])
    eef_lo = tuple(spec.eef_box_m[a][0] for a in _AXES)
    eef_hi = tuple(spec.eef_box_m[a][1] for a in _AXES)
    # hand -> object = object - eef, by interval arithmetic.
    rel_lo = tuple(obj_lo[i] - eef_hi[i] for i in range(3))
    rel_hi = tuple(obj_hi[i] - eef_lo[i] for i in range(3))
    assert tuple(joint_limits.shape) == (7, 2), tuple(joint_limits.shape)

    # The target and the container share the object rules.
    rules: dict[str, tuple[Sequence[float], Sequence[float]]] = {
        "target_pos": (obj_lo, obj_hi),
        "target_quat": ([-1.0] * 4, [1.0] * 4),
        "hand_to_target": (rel_lo, rel_hi),
        "container_pos": (obj_lo, obj_hi),
        "container_quat": ([-1.0] * 4, [1.0] * 4),
        "hand_to_container": (rel_lo, rel_hi),
        "arm_joint_pos": (joint_limits[:, 0].tolist(), joint_limits[:, 1].tolist()),
        "gripper_pos": ([0.0], [1.0]),
        "eef_position": (eef_lo, eef_hi),
        "eef_orientation": ([-1.0] * 4, [1.0] * 4),
        GRIPPER_DELAY_TERM: ([0.0] * GRIPPER_DELAY_DIM, [1.0] * GRIPPER_DELAY_DIM),
    }
    offset = 0
    for name, dim_raw in layout:
        dim = int(dim_raw)
        if name == "prev_action":
            lo, hi = [-1.0] * dim, [1.0] * dim
        elif name in rules:
            lo, hi = rules[name]
        else:
            raise ValueError(f"unknown obs term {name!r}: no bounds rule")
        if len(lo) != dim:
            raise ValueError(f"obs term {name!r} has {dim} dims, bounds rule has {len(lo)}")
        low[offset : offset + dim] = torch.tensor(lo, dtype=torch.float32)
        high[offset : offset + dim] = torch.tensor(hi, dtype=torch.float32)
        offset += dim

    bad = (~(low < high)).nonzero(as_tuple=True)[0].tolist()
    if bad:
        raise ValueError(f"resolve_obs_bounds: dims {bad} have low >= high (low={low[bad].tolist()})")
    return low, high


def demo_outside_fraction(demo_obs: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Per-dim fraction ``[D]`` of raw demo obs ``[N, D]`` that fall outside ``[low, high]``."""
    if demo_obs.ndim != 2 or demo_obs.shape[-1] != low.numel() or low.numel() != high.numel():
        raise ValueError(
            f"demo_outside_fraction: shape mismatch, demo_obs={tuple(demo_obs.shape)} "
            f"low={tuple(low.shape)} high={tuple(high.shape)}"
        )
    x = demo_obs.float().detach().cpu()
    lo = low.float().detach().cpu()
    hi = high.float().detach().cpu()
    outside = (x < lo) | (x > hi)
    return outside.float().mean(dim=0)


def format_bounds_table(
    layout: Sequence[tuple[str, int]],
    low: torch.Tensor,
    high: torch.Tensor,
    outside: torch.Tensor | None = None,
) -> str:
    """Format per-term bounds (and optional demo-outside fraction) as a text table."""
    lines = [f"{'term':<20} {'dims':>7}  {'low':>10} {'high':>10}  {'demo_out%':>9}"]
    offset = 0
    for name, dim_raw in layout:
        dim = int(dim_raw)
        sl = slice(offset, offset + dim)
        out = "-" if outside is None else f"{float(outside[sl].max().item()) * 100.0:8.2f}%"
        lines.append(
            f"{name:<20} {f'{offset}:{offset + dim}':>7}  "
            f"{float(low[sl].min().item()):10.3f} {float(high[sl].max().item()):10.3f}  {out:>9}"
        )
        offset += dim
    return "\n".join(lines)
