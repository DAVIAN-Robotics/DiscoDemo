"""Skill-latent observation wrapper: appends an episode-fixed skill z ~ N(0, I) to the observation.

z is part of the observation vector, so the buffer, agent and compiled update see only a larger
``obs_dim``.

Lifecycle:
  - reset(): sample z for every env and append it.
  - step(action): ``infos["final_obs"]`` gets the old z (bootstrap target of the finished
    episode); done envs then resample z, and the returned obs (already the first obs of the
    auto-reset episode) gets the new z.

The inner env must return numpy obs (``to_numpy=True``). gymnasium's VectorWrapper does not
forward attributes, so ``__getattr__`` does.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from flash_rl.types import NDArray


class SkillZConcatWrapper(gym.vector.VectorWrapper):
    """Append an episode-fixed skill z ~ N(0, I_{z_dim}) to the observation.

    Parameters
    ----------
    env : gym.vector.VectorEnv
        Inner env with numpy obs and a Box observation space.
    z_dim : int
        Skill dimension (> 0).
    seed : int
        Seed of the z sampler.
    """

    def __init__(self, env: gym.vector.VectorEnv[Any, Any, Any], z_dim: int, seed: int):
        super().__init__(env)
        if int(z_dim) <= 0:
            raise ValueError(f"z_dim must be > 0 (got {z_dim}); do not wrap when z_dim=0")
        self._z_dim = int(z_dim)
        self._np_rng = np.random.default_rng(int(seed))
        self._z_cur = np.zeros((int(env.num_envs), self._z_dim), dtype=np.float32)
        old_single = env.single_observation_space
        assert isinstance(old_single, gym.spaces.Box), type(old_single).__name__
        inf = np.full((self._z_dim,), np.inf, dtype=np.float32)
        self.single_observation_space = gym.spaces.Box(
            np.concatenate([np.asarray(old_single.low, dtype=np.float32), -inf]),
            np.concatenate([np.asarray(old_single.high, dtype=np.float32), inf]),
            dtype=np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    @property
    def z(self) -> NDArray:
        """Current per-env skill z, shape ``(num_envs, z_dim)``."""
        return self._z_cur

    def set_z(self, z: NDArray, env_ids: NDArray) -> None:
        """Overwrite z for the given envs (rollouts that replay a recorded episode's z)."""
        z_arr = np.asarray(z, dtype=np.float32)
        ids = np.asarray(env_ids, dtype=np.int64)
        if z_arr.shape != (ids.shape[0], self._z_dim):
            raise ValueError(f"set_z: z shape {z_arr.shape} != ({ids.shape[0]}, {self._z_dim})")
        self._z_cur[ids] = z_arr

    def _resample(self, mask: NDArray) -> None:
        n = int(mask.sum())
        if n:
            self._z_cur[mask] = self._np_rng.standard_normal((n, self._z_dim)).astype(np.float32)

    def _concat(self, obs: Any) -> NDArray:
        assert isinstance(obs, np.ndarray), f"expected numpy obs (build the env with to_numpy=True), got {type(obs)}"
        return np.concatenate([obs, self._z_cur], axis=-1)

    def reset(self, **kwargs: Any) -> tuple[NDArray, dict[str, Any]]:  # type: ignore[override]
        """Reset the inner env, resample z for all envs and append it."""
        obs, infos = self.env.reset(**kwargs)
        self._resample(np.ones(self.num_envs, dtype=bool))
        return self._concat(obs), infos

    def step(self, action: Any) -> tuple[NDArray, Any, Any, Any, dict[str, Any]]:  # type: ignore[override]
        """Step the inner env; ``final_obs`` keeps the old z, done envs resample before the returned obs."""
        obs, rew, term, trunc, infos = self.env.step(action)
        dones = np.asarray(term, dtype=bool) | np.asarray(trunc, dtype=bool)
        infos = dict(infos)
        infos["final_obs"] = self._concat(infos["final_obs"])
        self._resample(dones)
        return self._concat(obs), rew, term, trunc, infos

    def observe(self) -> NDArray:
        """Current observation of the inner env with the current z appended."""
        return self._concat(self.env.observe())

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped env."""
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)


def augment_demo_transitions_with_z(
    demo_transitions: dict[str, torch.Tensor], z_dim: int, seed: int
) -> dict[str, torch.Tensor]:
    """Append one z ~ N(0, I) per demonstration to ``observation`` and ``next_observation``.

    Demonstration boundaries are read from ``terminated`` (True on the last row of each demo).
    """
    term = demo_transitions["terminated"]
    assert term.dtype == torch.bool and term.ndim == 1 and bool(term[-1].item()), "bad demo boundary markers"
    # Row -> demo id: number of terminated rows strictly before this row.
    ep_id = torch.cumsum(term.long(), dim=0) - term.long()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    z_eps = torch.randn((int(term.sum().item()), int(z_dim)), generator=gen, dtype=torch.float32)
    obs = demo_transitions["observation"]
    z_rows = z_eps[ep_id.cpu()].to(device=obs.device, dtype=obs.dtype)
    out = dict(demo_transitions)
    out["observation"] = torch.cat([obs, z_rows], dim=-1)
    out["next_observation"] = torch.cat([demo_transitions["next_observation"], z_rows], dim=-1)
    return out
