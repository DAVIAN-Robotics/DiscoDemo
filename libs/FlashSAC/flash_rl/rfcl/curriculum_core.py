"""Simulator-independent bookkeeping of the parallel reverse curriculum (RFCL).

Every demonstration ``i`` has a frontier ``c_i``, initialized at its last frame. All envs share one pool:
on each reset an env draws a demonstration state ``(i, t)`` with ``t`` in ``[c_i, T_i - 1]`` and weight
``0.5^(t - c_i)``, so start states concentrate at the frontiers. Once the last ``frontier_window``
episodes started exactly at ``c_i`` reach ``advance_threshold`` success, the frontier moves back by
``reverse_step_size``; a demonstration is solved when this test also passes at ``c_i = 0``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class DemoCurriculumMetadata:
    """Per-demo curriculum state."""

    start_step: int = 0
    total_steps: int = 0
    solved: bool = False


class CurriculumCore:
    """Reverse-curriculum frontiers, start-state sampling and episode time limits.

    Parameters
    ----------
    num_envs : int
        Number of parallel envs.
    demo_total_steps : list[int]
        Length ``T_i`` of each demonstration.
    device : Any
        Unused; kept for the wrapper interface.
    reverse_step_size : int
        Frames a frontier moves back when its demonstration is mastered.
    advance_threshold : float
        Success rate over the frontier window needed to move a frontier.
    frontier_window : int
        Number of most recent episodes started at a frontier that decide whether it moves.
    minimum_episode_steps : int
        Constant part of the time limit ``minimum_episode_steps + (T_i - t)``.
    fixed_frontier : bool
        Start every env ``k`` at the frontier of demonstration ``k % num_demos`` and never move
        frontiers (demo transition extraction).
    """

    def __init__(
        self,
        num_envs: int,
        demo_total_steps: list[int],
        device: Any,
        *,
        reverse_step_size: int,
        advance_threshold: float,
        frontier_window: int,
        minimum_episode_steps: int,
        fixed_frontier: bool = False,
    ):
        self.num_envs = num_envs
        self.num_demos = len(demo_total_steps)
        if self.num_demos == 0:
            raise ValueError("demo_total_steps must contain at least one demo")
        if not (0.0 < float(advance_threshold) <= 1.0):
            raise ValueError(f"advance_threshold must be in (0, 1]; got {advance_threshold}")
        self.reverse_step_size = int(reverse_step_size)
        self.advance_threshold = float(advance_threshold)
        self.frontier_window = int(frontier_window)
        self.minimum_episode_steps = int(minimum_episode_steps)
        self.fixed_frontier = bool(fixed_frontier)
        # Demonstration of each env (fixed-frontier mode keeps this round-robin assignment).
        self.env_to_demo = torch.arange(num_envs) % self.num_demos
        self.demo_metadata = [DemoCurriculumMetadata(max(t - 1, 0), t) for t in demo_total_steps]
        # Outcomes of episodes started at the current frontier of each demonstration.
        self._frontier_outcomes = [deque(maxlen=self.frontier_window) for _ in range(self.num_demos)]
        self._state_rng = np.random.RandomState(np.random.randint(2**32))
        self._active: list[tuple[int, int]] = []
        self._active_weights: np.ndarray | None = None
        self._active_dirty = True

    def _rebuild_active(self) -> None:
        """Collect the (demo, t) start states with t in [frontier, T-1] and their weights."""
        self._active = [(d, t) for d, md in enumerate(self.demo_metadata) for t in range(md.start_step, md.total_steps)]
        w = np.array([0.5 ** (t - self.demo_metadata[d].start_step) for (d, t) in self._active], dtype=float)
        self._active_weights = w / w.sum()
        self._active_dirty = False

    def mark_active_dirty(self) -> None:
        """Rebuild the start-state pool on the next reset (after frontiers are set from outside)."""
        self._active_dirty = True

    def assign_reset(self, env_idx: int) -> tuple[int, int]:
        """Return the (demo, start step) of an env that is being reset."""
        if self.fixed_frontier:
            d = int(self.env_to_demo[env_idx])
            return d, self.demo_metadata[d].start_step
        if self._active_dirty:
            self._rebuild_active()
        idx = int(self._state_rng.choice(len(self._active), p=self._active_weights))
        d, t = self._active[idx]
        self.env_to_demo[env_idx] = d
        return d, t

    def compute_dynamic_timelimit_from_remaining(self, remaining_steps: int) -> int:
        """Episode time limit for a start ``remaining_steps`` frames before the demonstration end."""
        return self.minimum_episode_steps + int(remaining_steps)

    def record_outcome(self, success: int, start_step: int, demo_idx: int) -> None:
        """Record one finished episode; only episodes started at the current frontier count."""
        if int(start_step) == self.demo_metadata[demo_idx].start_step:
            self._frontier_outcomes[demo_idx].append(int(success))

    @property
    def reverse_solved_frac(self) -> float:
        """Fraction of solved demonstrations."""
        return sum(int(md.solved) for md in self.demo_metadata) / len(self.demo_metadata)

    def frontier_success_rate(self, demo_idx: int) -> float:
        """Success rate over the frontier window (0 until the window is full)."""
        buf = self._frontier_outcomes[demo_idx]
        return float(np.mean(buf)) if len(buf) == self.frontier_window else 0.0

    def frontier_state(self) -> list[list[int]]:
        """Frontier-window outcomes of every demonstration (for checkpoints)."""
        return [list(buf) for buf in self._frontier_outcomes]

    def set_frontier_state(self, outcomes: list[list[int]]) -> None:
        """Restore the frontier-window outcomes saved by ``frontier_state``."""
        if len(outcomes) != self.num_demos:
            raise ValueError(f"frontier outcomes for {len(outcomes)} demos, expected {self.num_demos}")
        self._frontier_outcomes = [deque((int(v) for v in o), maxlen=self.frontier_window) for o in outcomes]

    def step_curriculum(self) -> bool:
        """Move every frontier whose window reached the threshold; return whether anything changed."""
        if self.fixed_frontier:
            return False
        changed = False
        for d, md in enumerate(self.demo_metadata):
            if self.frontier_success_rate(d) < self.advance_threshold:
                continue
            if md.start_step > 0:
                md.start_step = max(md.start_step - self.reverse_step_size, 0)
                self._frontier_outcomes[d] = deque(maxlen=self.frontier_window)
                print(f"[reverse] demo {d} frontier -> {md.start_step}", flush=True)
                changed = True
            elif not md.solved:
                md.solved = True
                print(f"[reverse] demo {d} solved", flush=True)
                changed = True
        if changed:
            self._active_dirty = True
        return changed
