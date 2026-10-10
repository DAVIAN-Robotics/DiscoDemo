"""Per-dimension observation normalization to [-1, 1] from explicit bounds.

Maps each dim with ``x_norm = clamp(2 * (x - low) / (high - low + eps) - 1, -1, 1)``. ``low`` /
``high`` are set once by the training script from the task's physical and workspace bounds
(``robolab_obs_bounds``) and live in registered buffers, so they are saved with the checkpoint
(``normalizer.pt``). Actions are already in [-1, 1] and are not normalized.
"""

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Bounds -> [-1, 1] affine map with clamping, for observations."""

    def __init__(self, obs_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self._obs_dim = obs_dim
        self._eps = eps
        self.register_buffer("obs_low", torch.full((obs_dim,), -1.0))
        self.register_buffer("obs_high", torch.full((obs_dim,), 1.0))
        # 1.0 once bounds are set; normalizing before that raises.
        self.register_buffer("_fitted_flag", torch.zeros(1))

    @property
    def is_fitted(self) -> bool:
        """Whether bounds have been set."""
        return bool(self._fitted_flag.item() > 0.5)

    def set_obs_bounds(self, low: torch.Tensor, high: torch.Tensor) -> None:
        """Set the obs window.

        Parameters
        ----------
        low, high : torch.Tensor
            ``[obs_dim]`` lower / upper bounds.

        Raises
        ------
        ValueError
            If the shapes are not ``[obs_dim]`` or any ``low >= high``.
        """
        if tuple(low.shape) != (self._obs_dim,) or tuple(high.shape) != (self._obs_dim,):
            raise ValueError(f"expected [{self._obs_dim}] bounds, got {tuple(low.shape)} / {tuple(high.shape)}")
        bad = (~(low < high)).nonzero(as_tuple=True)[0].tolist()
        if bad:
            raise ValueError(f"Normalizer.set_obs_bounds: low >= high at dims {bad}")
        self.obs_low.copy_(low.to(device=self.obs_low.device, dtype=self.obs_low.dtype))
        self.obs_high.copy_(high.to(device=self.obs_high.device, dtype=self.obs_high.dtype))
        self._fitted_flag.fill_(1.0)

    def normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize observations to [-1, 1]."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer.normalize_obs called before set_obs_bounds")
        y = 2.0 * (x - self.obs_low) / (self.obs_high - self.obs_low + self._eps) - 1.0
        return y.clamp(-1.0, 1.0)
