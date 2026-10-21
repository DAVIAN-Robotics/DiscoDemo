"""Replay buffer that mixes the online buffer with the demonstration buffer.

Each batch is ``[online rows | demo rows]`` with a fixed demo share. Everything except
``sample`` goes to the online buffer: demos are never added, saved or loaded.
"""

from __future__ import annotations

from collections.abc import MutableMapping

import torch

from flash_rl.buffers.base_buffer import BaseBuffer, Batch
from flash_rl.rfcl.demo_buffer import DemoBuffer
from flash_rl.types import NDArray, Tensor


class MixedReplayBuffer(BaseBuffer):
    """Online buffer plus a fixed demo share of every batch.

    Parameters
    ----------
    online : BaseBuffer
        The agent's replay buffer.
    demo : DemoBuffer
        Filled demonstration buffer.
    demo_ratio : float
        Share of each batch drawn from the demonstrations.
    """

    def __init__(self, online: BaseBuffer, demo: DemoBuffer, demo_ratio: float):
        assert 0.0 < demo_ratio < 1.0, demo_ratio
        self._online = online
        self._demo = demo
        # Keep the agent's batch size (shapes are fixed for torch.compile).
        self._batch_size = online._sample_batch_size
        self._demo_batch = max(1, int(round(self._batch_size * demo_ratio)))
        self._online_batch = self._batch_size - self._demo_batch
        self._observation_space = online._observation_space
        self._action_space = online._action_space
        self._n_step = online._n_step
        self._gamma = online._gamma
        self._max_length = online._max_length
        self._min_length = online._min_length
        self._sample_batch_size = self._batch_size
        self._demo._sample_batch_size = self._demo_batch
        self._online._sample_batch_size = self._online_batch

    def __len__(self) -> int:
        """Number of online transitions."""
        return len(self._online)

    def reset(self) -> None:
        """Reset the online buffer."""
        self._online.reset()

    def add(self, transition: MutableMapping[str, Tensor]) -> None:
        """Add a transition to the online buffer."""
        self._online.add(transition)

    def can_sample(self) -> bool:
        """Whether the online buffer can be sampled."""
        return self._online.can_sample()

    def sample(self, sample_idxs: NDArray | None = None) -> Batch:
        """Return ``_online_batch`` online rows followed by ``_demo_batch`` demo rows.

        Demos have no ``ep_uid``; their rows get uid 0, which carries no intrinsic reward.
        """
        assert sample_idxs is None
        online_batch = self._online.sample()
        demo_batch = self._demo.sample()
        assert set(online_batch) - set(demo_batch) <= {"ep_uid"}, sorted(online_batch)
        out: Batch = {}
        for k, v in online_batch.items():
            d = demo_batch[k] if k in demo_batch else torch.zeros((self._demo_batch,), dtype=v.dtype, device=v.device)
            out[k] = torch.cat([v, d], dim=0)
        return out

    def save(self, path: str) -> None:
        """Save the online buffer."""
        self._online.save(path)

    def load(self, path: str) -> None:
        """Load the online buffer."""
        self._online.load(path)

    def get_observations(self) -> Tensor:
        """Online observations."""
        return self._online.get_observations()
