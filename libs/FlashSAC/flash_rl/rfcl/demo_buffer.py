"""Immutable GPU storage of the demonstration transitions, with n-step returns precomputed.

Used only through ``MixedReplayBuffer``, which draws a fixed share of every batch from it.
"""

from __future__ import annotations

from collections.abc import MutableMapping

import gymnasium as gym
import torch

from flash_rl.buffers.base_buffer import BaseBuffer, Batch
from flash_rl.types import NDArray, Tensor


class DemoBuffer(BaseBuffer):
    """Fixed demonstration transitions sampled uniformly.

    Parameters
    ----------
    observation_space, action_space : gym.spaces.Space
        Spaces of the online buffer.
    sample_batch_size : int
        Rows per ``sample`` (``MixedReplayBuffer`` sets its demo share).
    device_type : str
        Storage device.
    n_step : int
        n-step return length.
    gamma : float
        Discount.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        sample_batch_size: int,
        device_type: str,
        *,
        n_step: int,
        gamma: float,
    ):
        super().__init__(observation_space, action_space, n_step, gamma, 0, 0, sample_batch_size)
        self._device = torch.device("cuda:0" if device_type == "cuda" else device_type)
        self._storage: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        """Number of demo transitions."""
        return int(self._storage["reward"].shape[0]) if self._storage else 0

    def reset(self) -> None:
        """Clear storage."""
        self._storage = {}

    def add(self, transition: MutableMapping[str, Tensor]) -> None:  # type: ignore[override]
        """Raise; demos are fixed after ``fill_from_transitions``."""
        raise RuntimeError("DemoBuffer is immutable; use fill_from_transitions.")

    def can_sample(self) -> bool:
        """True once filled."""
        return len(self) > 0

    def sample(self, sample_idxs: NDArray | None = None) -> Batch:
        """Return ``sample_batch_size`` uniformly drawn rows."""
        assert sample_idxs is None
        idxs = torch.randint(0, len(self), (self._sample_batch_size,), device=self._device)
        return {k: v[idxs] for k, v in self._storage.items()}

    def save(self, path: str) -> None:
        """No-op; demos are reloaded from disk, not checkpointed."""

    def get_observations(self) -> Tensor:
        """Return the demo observations."""
        return self._storage["observation"]

    def fill_from_transitions(self, transitions: MutableMapping[str, torch.Tensor]) -> None:
        """Store ``observation, action, next_observation, reward, terminated`` (leading dim N).

        ``terminated`` marks the last row of each demonstration, so n-step returns never cross
        demonstrations.
        """
        rew, nxt, term = self._n_step_targets(
            transitions["reward"].float().cpu(),
            transitions["next_observation"].cpu(),
            transitions["terminated"].bool().cpu(),
        )
        n = int(rew.shape[0])
        self._storage = {
            "observation": transitions["observation"].to(self._device),
            "action": transitions["action"].to(self._device),
            "next_observation": nxt.to(self._device),
            "reward": rew.to(self._device),
            "terminated": term.float().to(self._device),
            "truncated": torch.zeros(n, dtype=torch.float32, device=self._device),
        }

    def _n_step_targets(
        self, rewards: torch.Tensor, next_obs: torch.Tensor, terminated: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """n-step discounted reward, next observation and terminal flag of the last step used."""
        n_rows = rewards.shape[0]
        rows = torch.arange(n_rows)
        cum = torch.zeros(n_rows, dtype=torch.float64)
        last = rows.clone()
        alive = torch.ones(n_rows, dtype=torch.bool)  # no terminal row so far and still inside the array
        n_term = torch.zeros(n_rows, dtype=torch.bool)
        discount = 1.0
        for k in range(self._n_step):
            idx = rows + k
            alive &= idx < n_rows
            idx = idx.clamp(max=n_rows - 1)
            cum += torch.where(alive, discount * rewards[idx].double(), 0.0)
            last = torch.where(alive, idx, last)
            hit = alive & terminated[idx]
            n_term |= hit
            alive &= ~hit
            discount *= self._gamma
        return cum.float(), next_obs[last], n_term
