# SPDX-License-Identifier: Apache-2.0
"""Action normalization between SAC actions in ``[-1, 1]`` and panda joint targets.

Pure functions, no Isaac dependency; joint limits are passed in by the caller.

Frame conventions (panda hand):
- Demo arm actions are in the *command frame*: joint7 is expressed relative to the +pi/4 hand-mount offset,
  i.e. it is 45 deg smaller than the physical joint value.
- SAC arm actions map per-joint affinely onto the physical joint limits, so the +pi/4 shift is already
  absorbed in the physical center. Do not add an extra joint7 offset to the SAC action term; the shift is only
  used when converting demo (command-frame) actions to physical joints.
- Gripper: demos use ``[0, 1]`` (threshold 0.5), SAC uses ``[-1, 1]`` (threshold 0) to use the full tanh
  range. ``a_sac = 2 * g_demo - 1`` keeps the two thresholds consistent.
"""

from __future__ import annotations

import numpy as np
import torch


class JerkLimitedJointCommand:
    """Stateful desired-velocity filter with per-joint velocity/acceleration/jerk caps.

    The relative arm action of the RL policies: arm channels in ``[-1, 1]`` are desired velocity
    fractions, and the filter state (velocity, acceleration) is not part of the policy observation.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        max_velocity: torch.Tensor,
        max_acceleration: torch.Tensor,
        max_jerk: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt = float(dt)
        self.max_velocity = self._limits(max_velocity, "max_velocity", device)
        self.max_acceleration = self._limits(max_acceleration, "max_acceleration", device)
        self.max_jerk = self._limits(max_jerk, "max_jerk", device)
        if not (self.max_velocity.shape == self.max_acceleration.shape == self.max_jerk.shape):
            raise ValueError("v/a/jerk limit vectors must have identical shapes")
        shape = (int(num_envs), int(self.max_velocity.numel()))
        self.velocity = torch.zeros(shape, dtype=torch.float32, device=self.max_velocity.device)
        self.acceleration = torch.zeros_like(self.velocity)

    @staticmethod
    def _limits(value: torch.Tensor, name: str, device: torch.device | str | None) -> torch.Tensor:
        out = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)
        if out.numel() == 0 or not bool(torch.isfinite(out).all()) or bool((out <= 0.0).any()):
            raise ValueError(f"{name} must be a finite positive vector, got {out}")
        return out

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        if env_ids is None:
            self.velocity.zero_()
            self.acceleration.zero_()
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.velocity.device)
        self.velocity[ids] = 0.0
        self.acceleration[ids] = 0.0

    def _max_safe_acceleration(self, velocity: torch.Tensor) -> torch.Tensor:
        """Largest acceleration that can still stop accelerating before ``+vmax``.

        A direct velocity clamp creates an acceleration discontinuity at the bound and
        therefore violates the jerk cap. This computes a discrete stopping margin for
        ramping acceleration to zero at ``max_jerk`` and solves the monotone bound with
        a short vectorized bisection.
        """
        jerk_step = self.max_jerk * self.dt

        def velocity_at_zero_acceleration(acceleration: torch.Tensor) -> torch.Tensor:
            positive = torch.clamp(acceleration, min=0.0)
            steps = torch.clamp(torch.ceil(positive / jerk_step) - 1.0, min=0.0)
            future_delta = self.dt * (steps * positive - jerk_step * steps * (steps + 1.0) * 0.5)
            return velocity + acceleration * self.dt + future_delta

        lo = -self.max_acceleration.expand_as(velocity)
        hi = self.max_acceleration.expand_as(velocity)
        for _ in range(20):
            mid = (lo + hi) * 0.5
            safe = velocity_at_zero_acceleration(mid) <= self.max_velocity
            lo = torch.where(safe, mid, lo)
            hi = torch.where(safe, hi, mid)
        return lo

    def step(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Return a v/a/jerk-limited relative joint displacement for one control step."""
        raw = torch.as_tensor(raw_action, dtype=torch.float32, device=self.velocity.device)
        if raw.shape != self.velocity.shape:
            raise ValueError(f"expected raw action shape {tuple(self.velocity.shape)}, got {tuple(raw.shape)}")
        desired_velocity = torch.clamp(raw, -1.0, 1.0) * self.max_velocity
        desired_acceleration = torch.clamp(
            (desired_velocity - self.velocity) / self.dt,
            -self.max_acceleration,
            self.max_acceleration,
        )
        acceleration = torch.clamp(
            desired_acceleration,
            self.acceleration - self.max_jerk * self.dt,
            self.acceleration + self.max_jerk * self.dt,
        )
        acceleration = torch.clamp(acceleration, -self.max_acceleration, self.max_acceleration)

        # Enforce both velocity bounds with enough margin to ramp acceleration back
        # to zero without exceeding the jerk envelope. The lower-bound constraint is
        # the sign-symmetric form of the upper-bound calculation.
        max_safe = self._max_safe_acceleration(self.velocity)
        min_safe = -self._max_safe_acceleration(-self.velocity)
        self.acceleration = torch.maximum(torch.minimum(acceleration, max_safe), min_safe)
        self.velocity = self.velocity + self.acceleration * self.dt
        return self.velocity * self.dt


# +45 deg hand-mount offset: command-frame joint7 -> physical panda_joint7.
PANDA_J7_MOUNT_OFFSET: float = np.pi / 4

# Index of joint7 in the arm 7-vector (panda_joint1..7, preserve_order=True).
ARM_J7_INDEX: int = 6


def command_to_physical(q_cmd: torch.Tensor) -> torch.Tensor:
    """Command-frame arm qpos -> physical panda joints (joint7 += offset)."""
    q = q_cmd.clone()
    q[..., ARM_J7_INDEX] = q[..., ARM_J7_INDEX] + PANDA_J7_MOUNT_OFFSET
    return q


def arm_scale_offset(qmin: torch.Tensor, qmax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(scale, offset) for Isaac ``JointPositionAction``: ``raw * scale + offset`` spans ``[qmin, qmax]``.

    The joint7 mount offset is already part of the physical center; do not add a separate joint7 offset.
    """
    half = 0.5 * (qmax - qmin)
    center = 0.5 * (qmin + qmax)
    return half, center


def rate_scale_offset(r: torch.Tensor) -> tuple[torch.Tensor, float]:
    """(scale, offset) for a relative rate action: ``processed = raw * r``.

    Offset is zero so that ``a = 0`` means zero displacement.
    """
    return r, 0.0


def relative_applied_target(
    processed: torch.Tensor, joint_pos: torch.Tensor, qmin: torch.Tensor, qmax: torch.Tensor
) -> torch.Tensor:
    """Applied target of a relative joint action: ``clamp(joint_pos + delta, qmin, qmax)``.

    Mirrors ``RateLimitedRelativeJointPositionAction.apply_actions``; recorded as the absolute commanded target.
    """
    return torch.clamp(processed + joint_pos, qmin, qmax)


def relative_delta_demo_arm(q_cmd: torch.Tensor, q_cur_phys: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Command-frame demo arm target + current physical qpos -> relative SAC action ``[-1,1]^7``.

    ``a = clip((command_to_physical(q_cmd) - q_cur_phys) / r, -1, 1)``: inverse of the relative action term.
    ``r`` is the per-axis max relative action rate (broadcastable).
    """
    delta = command_to_physical(q_cmd) - q_cur_phys
    return torch.clamp(delta / r, -1.0, 1.0)


def normalize_gripper(g_demo: torch.Tensor) -> torch.Tensor:
    """Demo gripper [0,1] -> SAC [-1,1]."""
    return 2.0 * g_demo - 1.0


def gripper_is_close(a_sac: torch.Tensor) -> torch.Tensor:
    """SAC gripper action -> close flag (threshold 0)."""
    return a_sac > 0.0


def drop_ineffective_gripper_requests(requested: torch.Tensor, applied: torch.Tensor) -> torch.Tensor:
    """Replace gripper requests that were never applied with the previous label.

    A delayed gripper actuator keeps a single pending slot, so a request run shorter than the delay is
    overwritten and never applied. If ``applied`` matched the request at least once during the run, the run is
    kept from the request time (the anticipating request is the label to learn); otherwise it is replaced by the
    previous label (or by ``applied`` for the first run). Without a delay this is the identity.

    Parameters
    ----------
    requested : torch.Tensor
        ``[T]`` gripper close requests in {0, 1}.
    applied : torch.Tensor
        ``[T]`` gripper close actually applied at each step, in {0, 1}.

    Returns
    -------
    torch.Tensor
        ``[T]`` cleaned requests (same dtype/device as ``requested``).

    Raises
    ------
    ValueError
        If the inputs are not 1-D or differ in length.
    """
    if requested.ndim != 1 or requested.shape != applied.shape:
        raise ValueError(
            f"requested {tuple(requested.shape)} and applied {tuple(applied.shape)} must be 1-D of equal length"
        )
    req = (requested > 0.5).tolist()
    app = (applied > 0.5).tolist()
    out = list(req)
    n = len(req)
    s = 0
    while s < n:
        e = s
        while e < n and req[e] == req[s]:
            e += 1
        if not any(app[t] == req[s] for t in range(s, e)):
            fill = out[s - 1] if s > 0 else app[s]
            for t in range(s, e):
                out[t] = fill
        s = e
    return torch.tensor(out, dtype=requested.dtype, device=requested.device)
