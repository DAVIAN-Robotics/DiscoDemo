"""``box:`` init_pose form: uniform placement in an absolute box, yaw range, min center distance.

Unlike level/cm forms (±r around each object's nominal pose), every randomized object is drawn
uniformly from the same absolute box, rejecting draws that violate a minimum pairwise center
distance. The string fully specifies the distribution.

Format: ``box:<x_lo>,<x_hi>,<y_lo>,<y_hi>,<yaw_half_deg>,<min_center_dist_m>[,<yaw_center_deg>,<yaw_flip>]``
(robot base frame, m), e.g. ``box:0.30,0.75,-0.30,0.30,180,0.20``.

The two optional fields default to ``center=0, flip=0``. They constrain the long-axis direction
while leaving the 180-degree flip free: ``yaw ~ U(center ± half)``, then +180 deg with probability
1/2 when ``flip=1`` (e.g. ``center=90, flip=1`` for a banana lying along x with either bend side).

The sampler is torch-only and unit-testable without Isaac; the event function
``robolab.core.events.reset_pose.reset_pose_box`` calls it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch

_BOX_RE = re.compile(
    r"^box:(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)"
    r"(?:,(-?\d+(?:\.\d+)?),([01]))?$"
)


@dataclass(frozen=True)
class BoxRange:
    """Absolute box distribution. Lengths in m, yaw half-angle in rad."""

    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    yaw_half_rad: float
    min_center_dist_m: float
    yaw_center_rad: float = 0.0
    yaw_flip: bool = False

    def __post_init__(self) -> None:
        if not (self.x_lo < self.x_hi and self.y_lo < self.y_hi):
            raise ValueError(f"empty box range: x[{self.x_lo},{self.x_hi}] y[{self.y_lo},{self.y_hi}]")
        if self.yaw_half_rad < 0.0 or self.min_center_dist_m < 0.0:
            raise ValueError("yaw half-angle and min center distance must be non-negative")
        if self.yaw_half_rad > math.pi:
            raise ValueError(f"yaw half-angle exceeds 180 deg: {math.degrees(self.yaw_half_rad):.1f}")
        # No layout can satisfy a min distance >= the box diagonal; fail fast instead of exhausting retries.
        diag = math.hypot(self.x_hi - self.x_lo, self.y_hi - self.y_lo)
        if self.min_center_dist_m >= diag:
            raise ValueError(f"min_center_dist {self.min_center_dist_m} m is not smaller than the box diagonal {diag:.3f} m")

    @property
    def xy(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """``((x_lo, x_hi), (y_lo, y_hi))``."""
        return ((self.x_lo, self.x_hi), (self.y_lo, self.y_hi))


def parse_box(text: str) -> BoxRange | None:
    """Parse a ``box:...`` string into a BoxRange; return None for other forms."""
    m = _BOX_RE.match(str(text).strip())
    if m is None:
        return None
    x_lo, x_hi, y_lo, y_hi, yaw_deg, dist = (float(v) for v in m.groups()[:6])
    center_deg = 0.0 if m.group(7) is None else float(m.group(7))
    flip = False if m.group(8) is None else bool(int(m.group(8)))
    return BoxRange(x_lo, x_hi, y_lo, y_hi, math.radians(yaw_deg), dist, math.radians(center_deg), flip)


def sample_box_layout(
    n: int,
    n_assets: int,
    box_xy: tuple[tuple[float, float], tuple[float, float]],
    min_center_dist: float,
    yaw_range: tuple[float, float],
    yaw_mask: torch.Tensor,
    *,
    yaw_flip: bool = False,
    device: Any = "cpu",
    max_retries: int = 100,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sample uniform xy in the box and yaw per env, with pairwise center distance >= ``min_center_dist``.

    Parameters
    ----------
    n : int
        Number of envs.
    n_assets : int
        Number of objects (sharing the box).
    box_xy : ((x_lo, x_hi), (y_lo, y_hi))
        Absolute box (env-origin frame, m).
    min_center_dist : float
        Minimum pairwise center distance (m). Only violating envs are redrawn (rejection).
    yaw_range : (lo, hi)
        Uniform yaw range (rad); may be off-center.
    yaw_flip : bool
        If True, add 180 deg to each yaw with probability 1/2. Yaw is wrapped to (-pi, pi].
    yaw_mask : torch.Tensor
        ``(n_assets,)`` bool. Objects with False keep yaw 0 (default orientation).
    max_retries : int
        Rejection iteration cap. On exhaustion the last draw is kept.

    Returns
    -------
    xy : torch.Tensor
        ``(n, n_assets, 2)``.
    yaw : torch.Tensor
        ``(n, n_assets)``.
    leftover : int
        Envs still violating the min distance after all retries (normally 0).
    """
    (x_lo, x_hi), (y_lo, y_hi) = box_xy
    lo = torch.tensor([x_lo, y_lo], device=device, dtype=torch.float32)
    span = torch.tensor([x_hi - x_lo, y_hi - y_lo], device=device, dtype=torch.float32)

    def _draw(m: int) -> torch.Tensor:
        return lo + span * torch.rand((m, n_assets, 2), device=device, generator=generator)

    xy = _draw(n)
    leftover = 0
    if n_assets > 1 and min_center_dist > 0.0:
        eye = torch.eye(n_assets, device=device, dtype=torch.bool)
        for _ in range(max_retries):
            dist = torch.cdist(xy, xy)  # (n, A, A)
            bad = ((dist < min_center_dist) & ~eye).flatten(1).any(dim=1)
            if not bool(bad.any()):
                break
            xy[bad] = _draw(int(bad.sum()))
        else:
            dist = torch.cdist(xy, xy)
            leftover = int(((dist < min_center_dist) & ~eye).flatten(1).any(dim=1).sum())
    yaw = yaw_range[0] + (yaw_range[1] - yaw_range[0]) * torch.rand((n, n_assets), device=device, generator=generator)
    if yaw_flip:
        coin = torch.rand((n, n_assets), device=device, generator=generator) < 0.5
        yaw = yaw + coin.to(yaw.dtype) * math.pi
    # Used as absolute z-yaw: wrap to (-pi, pi] since center/flip can leave the range.
    yaw = torch.remainder(yaw + math.pi, 2.0 * math.pi) - math.pi
    yaw = yaw * yaw_mask.to(device=device, dtype=yaw.dtype)
    return xy, yaw, leftover
