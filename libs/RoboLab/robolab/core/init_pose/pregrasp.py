"""``pregrasp:`` init_pose form: start with an object in hand and an anchor object below the TCP.

Expresses insertion setups such as "hold the peg above the hole, board at a random pose". The held
object is fixed in the gripper; the anchor is placed so its target point (``anchor_offset``, e.g. the
hole center in the anchor's local frame) lies within ±xy_half of the point below the TCP, with a
random yaw within ±yaw_half.

Format: ``pregrasp:<xy_half_m>,<yaw_half_deg>``, e.g. ``pregrasp:0.03,15`` (±3 cm, ±15 deg). Which
objects are held and anchored is declared by the task (``Task.pregrasp`` ->
``build_init_pose_params(pregrasp_geometry=...)``), not by the string.

The sampler is torch-only and unit-testable without Isaac; the event function
``robolab.core.events.reset_pose.reset_pose_pregrasp`` calls it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch

_PREGRASP_RE = re.compile(r"^pregrasp:(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class PregraspRange:
    """Anchor randomization range: xy half-width (m) and yaw half-angle (rad)."""

    xy_half_m: float
    yaw_half_rad: float

    def __post_init__(self) -> None:
        if self.xy_half_m < 0.0 or self.yaw_half_rad < 0.0:
            raise ValueError("pregrasp half-width and half-angle must be non-negative")


def parse_pregrasp(text: str) -> PregraspRange | None:
    """Parse a ``pregrasp:...`` string into a PregraspRange; return None for other forms."""
    m = _PREGRASP_RE.match(str(text).strip())
    if m is None:
        return None
    xy_half, yaw_deg = (float(v) for v in m.groups())
    return PregraspRange(xy_half, math.radians(yaw_deg))


def sample_pregrasp_layout(
    n: int,
    tcp_xy: torch.Tensor,
    anchor_offset_xy: torch.Tensor,
    xy_half_m: float,
    yaw_range: tuple[float, float],
    *,
    device: Any = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the anchor root xy and yaw per env.

    The anchor-local target point ``anchor_offset_xy`` is placed at ``tcp_xy + U(±xy_half)^2`` in
    the world, so the root is ``target - R(yaw) @ offset``.

    Parameters
    ----------
    n : int
        Number of envs.
    tcp_xy : torch.Tensor
        (n, 2) env-relative TCP xy.
    anchor_offset_xy : torch.Tensor
        (2,) target point in the anchor's local frame.
    xy_half_m : float
        Target-point xy half-width.
    yaw_range : tuple[float, float]
        Uniform anchor yaw range (rad).
    device, generator
        Torch device / RNG.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(root_xy (n, 2), yaw (n,))``.
    """
    tcp = torch.as_tensor(tcp_xy, dtype=torch.float32, device=device).reshape(n, 2)
    off = torch.as_tensor(anchor_offset_xy, dtype=torch.float32, device=device).reshape(2)
    u = torch.rand((n, 2), generator=generator, device=device) * 2.0 - 1.0
    target = tcp + u * float(xy_half_m)
    lo, hi = float(yaw_range[0]), float(yaw_range[1])
    yaw = torch.rand((n,), generator=generator, device=device) * (hi - lo) + lo
    c, s = torch.cos(yaw), torch.sin(yaw)
    rotated = torch.stack([c * off[0] - s * off[1], s * off[0] + c * off[1]], dim=-1)
    return target - rotated, yaw
