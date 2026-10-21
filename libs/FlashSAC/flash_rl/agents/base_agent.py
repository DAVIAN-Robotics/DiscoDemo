"""Abstract agent interface."""

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Generic, TypeVar

import gymnasium as gym

from flash_rl.types import NDArray, Tensor

Config = TypeVar("Config")


class BaseAgent(Generic[Config], ABC):
    """Common interface implemented by every agent."""

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: Config,
    ):
        """Store observation/action space and config on the agent."""
        self._observation_space = observation_space
        self._action_space = action_space
        self._cfg = cfg

    @abstractmethod
    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        """Sample the next actions given the previous transition."""
        pass

    @abstractmethod
    def process_transition(
        self,
        transition: MutableMapping[str, Tensor],
    ) -> None:
        """Handle interaction samples (e.g., add to replay buffer)."""
        pass

    @abstractmethod
    def can_start_training(self) -> bool:
        """Whether the agent is ready to update (e.g., enough samples in buffer)."""
        pass

    @abstractmethod
    def update(self) -> dict[str, Any]:
        """Run one update and return a metric dict for logging."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent checkpoint to ``path``."""
        pass

    @abstractmethod
    def save_replay_buffer(self, path: str) -> None:
        """Save the replay buffer to ``path``."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the agent checkpoint from ``path``."""
        pass

    @abstractmethod
    def load_replay_buffer(self, path: str) -> None:
        """Restore the replay buffer from ``path``."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Return internal-state metrics."""
        pass

    @property
    def observation_space(self) -> gym.spaces.Space[NDArray]:
        """Observation space."""
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space[NDArray]:
        """Action space."""
        return self._action_space

    @property
    def cfg(self) -> Config:
        """Agent config."""
        return self._cfg
