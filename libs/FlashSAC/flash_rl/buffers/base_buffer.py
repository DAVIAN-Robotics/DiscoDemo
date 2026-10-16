"""Abstract replay buffer and batch type."""

from abc import ABC, abstractmethod
from collections.abc import MutableMapping

import gymnasium as gym

from flash_rl.types import NDArray, Tensor

Batch = MutableMapping[str, Tensor]


class BaseBuffer(ABC):
    """Base class for replay buffers."""

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
    ):
        """Store common buffer parameters.

        ``max_length`` is the capacity, ``min_length`` the number of stored transitions required
        before sampling, and ``sample_batch_size`` the batch size of one ``sample`` call.
        """
        self._observation_space = observation_space
        self._action_space = action_space
        self._max_length = max_length
        self._min_length = min_length
        self._n_step = n_step
        self._gamma = gamma
        self._sample_batch_size = sample_batch_size

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored transitions."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear storage and indices."""
        pass

    @abstractmethod
    def add(self, transition: MutableMapping[str, Tensor]) -> None:
        """Add a (batched) transition."""
        pass

    @abstractmethod
    def can_sample(self) -> bool:
        """Whether enough transitions are stored to sample."""
        pass

    @abstractmethod
    def sample(self, sample_idxs: NDArray | None = None) -> Batch:
        """Sample a batch."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the buffer contents to disk."""
        pass

    @abstractmethod
    def get_observations(self) -> Tensor:
        """Return all stored observations."""
        pass
