# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Measured gripper command-onset latency of the FR3 environment."""

from __future__ import annotations

import numpy as np


class TorchDelayedBinaryActuator:
    """Per-environment gripper command-onset delay (GPU-resident).

    Delay values are fixed per episode and can be reset for only the envs that
    auto-reset. Requested close is 1, open is 0; output follows the same binary
    convention and therefore never changes arm actions or finger travel speed.
    """

    def __init__(self, close_delay_steps, open_delay_steps, *, device) -> None:
        import torch

        close = torch.as_tensor(close_delay_steps, device=device, dtype=torch.long)
        opened = torch.as_tensor(open_delay_steps, device=device, dtype=torch.long)
        if close.ndim != 1 or opened.shape != close.shape or close.numel() == 0:
            raise ValueError("delay arrays must be non-empty 1-D arrays with equal shape")
        if bool(torch.any(close < 0)) or bool(torch.any(opened < 0)):
            raise ValueError("delay arrays must be non-negative")
        self.close_delay_steps = close.clone()
        self.open_delay_steps = opened.clone()
        self.requested = torch.zeros_like(close, dtype=torch.bool)
        self.applied = torch.zeros_like(close, dtype=torch.bool)
        self.pending_value = torch.zeros_like(close, dtype=torch.bool)
        self.pending_due = torch.full_like(close, -1)
        # Step of the last requested flip and whether one happened. Validity is a separate
        # flag because a teleport restore can make flip_step negative.
        self.flip_step = torch.zeros_like(close)
        self.flipped = torch.zeros_like(close, dtype=torch.bool)
        self.step_index = 0

    def set_episode_delays(self, env_ids, close_delay_steps, open_delay_steps, *, initial_close=None) -> None:
        import torch

        ids = torch.as_tensor(env_ids, device=self.requested.device, dtype=torch.long)
        close = torch.as_tensor(close_delay_steps, device=self.requested.device, dtype=torch.long)
        opened = torch.as_tensor(open_delay_steps, device=self.requested.device, dtype=torch.long)
        if close.shape != ids.shape or opened.shape != ids.shape:
            raise ValueError("episode delay values must match env_ids")
        self.close_delay_steps[ids] = close
        self.open_delay_steps[ids] = opened
        if initial_close is None:
            initial = torch.zeros_like(ids, dtype=torch.bool)
        else:
            initial = torch.as_tensor(initial_close, device=self.requested.device, dtype=torch.bool)
            if initial.shape != ids.shape:
                raise ValueError("initial_close must match env_ids")
        self.requested[ids] = initial
        self.applied[ids] = initial
        self.pending_value[ids] = initial
        self.pending_due[ids] = -1
        self.flipped[ids] = False

    def restore_pending_from_elapsed(self, env_ids, requested_close, elapsed_steps_since_edge) -> None:
        """Restore the hidden command queue after a state teleport.

        elapsed_steps_since_edge counts completed actuator calls since the
        most recent requested-command edge. -1 means no edge has occurred in
        the episode: ``requested_close`` has then been in effect since the
        start and is restored as both requested and applied. Used only by
        reverse-curriculum resets; ordinary episode resets use
        ``set_episode_delays``.
        """
        import torch

        ids = torch.as_tensor(env_ids, device=self.requested.device, dtype=torch.long)
        requested = torch.as_tensor(requested_close, device=self.requested.device, dtype=torch.bool)
        elapsed = torch.as_tensor(elapsed_steps_since_edge, device=self.requested.device, dtype=torch.long)
        if requested.shape != ids.shape or elapsed.shape != ids.shape:
            raise ValueError("restored gripper state must match env_ids")
        if bool(torch.any(elapsed < -1)):
            raise ValueError("elapsed_steps_since_edge must be -1 or non-negative")

        has_edge = elapsed >= 0
        delay = torch.where(requested, self.close_delay_steps[ids], self.open_delay_steps[ids])
        still_pending = has_edge & (elapsed <= delay)
        applied = torch.where(still_pending, ~requested, requested)

        self.requested[ids] = requested
        self.applied[ids] = applied
        self.pending_value[ids] = self.requested[ids]
        due = torch.full_like(elapsed, -1)
        due[still_pending] = self.step_index + delay[still_pending] - elapsed[still_pending]
        self.pending_due[ids] = due
        # Invert elapsed_steps so the restored state reads back the same elapsed count.
        self.flip_step[ids] = self.step_index - elapsed
        self.flipped[ids] = has_edge

    def remaining_steps(self):
        """Steps still queued before the pending command reaches the actuator.

        The value is ``pending_due - step_index`` while a command is queued and
        ``0`` when nothing is pending. Reading it right after ``step`` therefore
        counts down to ``0`` on the call *before* the command is applied, and it
        matches ``delay - elapsed_steps_since_edge`` after
        ``restore_pending_from_elapsed``. ``0`` is intentionally shared by "no
        pending command" and "applies on the next call" — the requested/applied
        pair disambiguates those two and the policy sees the requested bit.

        Returns
        -------
        torch.Tensor
            Long tensor shaped like ``requested``, always non-negative.
        """
        import torch

        remaining = self.pending_due - self.step_index
        return torch.where(self.pending_due >= 0, remaining, torch.zeros_like(remaining)).clamp_min(0)

    def elapsed_steps(self):
        """Steps completed since the most recent requested-command flip.

        This is the observable counterpart of ``remaining_steps``: it never
        needs the sampled per-episode delay, so a real robot can compute it from
        its own command history. The count keeps growing after the command has
        landed, which is why the observation layer saturates it — a large value
        means "this command has certainly taken effect".

        The convention matches ``restore_pending_from_elapsed`` exactly: the
        first reading after the flipping call is ``1``, and ``-1`` means no flip
        has happened yet in this episode.

        Returns
        -------
        torch.Tensor
            Long tensor shaped like ``requested``; ``-1`` where no flip occurred.
        """
        import torch

        elapsed = self.step_index - self.flip_step
        return torch.where(self.flipped, elapsed, torch.full_like(elapsed, -1))

    def step(self, requested):
        import torch

        values = torch.as_tensor(requested, device=self.requested.device, dtype=torch.bool)
        if values.shape != self.requested.shape:
            raise ValueError(f"requested must have shape {tuple(self.requested.shape)}")
        changed = values != self.requested
        if bool(torch.any(changed)):
            delays = torch.where(values, self.close_delay_steps, self.open_delay_steps)
            self.pending_value[changed] = values[changed]
            self.pending_due[changed] = self.step_index + delays[changed]
            self.requested[changed] = values[changed]
            self.flip_step[changed] = self.step_index
            self.flipped[changed] = True
        due = (self.pending_due >= 0) & (self.pending_due <= self.step_index)
        self.applied[due] = self.pending_value[due]
        self.pending_due[due] = -1
        self.step_index += 1
        return self.applied.clone()


def sample_delay_steps(
    rng: np.random.Generator, count: int, close_range: tuple[int, int], open_range: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive integer sampling of measured per-episode onset-delay ranges."""
    if count <= 0:
        raise ValueError("count must be positive")
    for name, bounds in (("close", close_range), ("open", open_range)):
        if len(bounds) != 2 or bounds[0] < 0 or bounds[0] > bounds[1]:
            raise ValueError(f"invalid {name} delay range {bounds}")
    return (
        rng.integers(close_range[0], close_range[1] + 1, size=count, dtype=np.int64),
        rng.integers(open_range[0], open_range[1] + 1, size=count, dtype=np.int64),
    )
