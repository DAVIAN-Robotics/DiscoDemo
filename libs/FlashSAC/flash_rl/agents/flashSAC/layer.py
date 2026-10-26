"""FlashSAC building blocks: unit-norm linear / norm layers and their ensemble (critic) variants."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_rl.agents.utils.distribution import safe_tanh_log_det_jacobian


class UnitLinear(nn.Module):
    """Bias-free orthogonal-init linear layer with unit-norm weights (per input feature)."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear map."""
        return self.w(x)  # type: ignore[no-any-return]

    def normalize_parameters(self) -> None:
        """Normalize the weight to unit norm along dim=-1; called after init and each optimizer step."""
        self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    """BatchNorm whose scale/bias are kept at norm sqrt(d)."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Apply ``F.batch_norm``."""
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def normalize_parameters(self) -> None:
        """Normalize scale and bias parameters to sqrt(d)."""
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
    """RMSNorm whose scale is kept at norm sqrt(d)."""

    def __init__(self, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``F.rms_norm``."""
        return F.rms_norm(x, self.weight.shape, self.weight, eps=self.eps)

    def normalize_parameters(self) -> None:
        """Normalize the scale to norm sqrt(d)."""
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class FlashSACEmbedder(nn.Module):
    """Input embedder: UnitBatchNorm followed by a UnitLinear projection to hidden_dim."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Return the embedding (norm -> linear)."""
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class FlashSACBlock(nn.Module):
    """Residual MLP block: two (linear, BatchNorm, ReLU) stages with a skip connection."""

    def __init__(self, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Apply w1 -> norm -> ReLU -> w2 -> norm -> ReLU and add the residual."""
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        x = x + residual
        return x


class NormalTanhPolicy(nn.Module):
    """Gaussian policy head with tanh squashing."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.mean_w = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))

        self.std_w = UnitLinear(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def get_mean_and_std(
        self,
        x: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the policy mean and std (bounded by log_std_min/max)."""
        # Use functional linear for AMP
        mean = F.linear(x, self.mean_w.w.weight, self.mean_bias)
        raw_log_std = F.linear(x, self.std_w.w.weight, self.std_bias)

        # normalize log-stds for stability
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(raw_log_std))
        std = torch.exp(log_std)

        return mean, std

    def forward(
        self,
        x: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample a tanh-squashed action; return it with the Jacobian-corrected log-prob."""
        mean, std = self.get_mean_and_std(x, training)

        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        tanh_action = torch.tanh(raw_action)

        # Compute log probability (accounting for tanh via Jacobian correction)
        log_prob = dist.log_prob(raw_action)  # type: ignore[no-untyped-call]
        log_prob = log_prob - safe_tanh_log_det_jacobian(raw_action)
        log_prob = log_prob.sum(1)

        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return tanh_action, info


# -------------------------------------
# Ensembled layers for Double Critic
# -------------------------------------


class EnsembleUnitLinear(nn.Module):
    """Ensemble of N orthogonal-init unit-norm linear layers (batched with einsum)."""

    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for i in range(num_ensemble):
            nn.init.orthogonal_(self.weight.data[i], gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each ensemble member's linear map."""
        # [N, B, in] @ [N, in, out] -> [N, B, out]
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    def normalize_parameters(self) -> None:
        """Normalize the weight to unit norm along dim=-1."""
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    """Ensemble BatchNorm with per-member statistics."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_ensemble: int, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Normalize with per-member batch statistics and update running stats."""
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                B = x.shape[1]
                # Cast to float32 for running stats (BatchNorm uses float32 even in AMP)
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_((var.squeeze(1) * (B / (B - 1))).float(), self.momentum)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(self.running_var.unsqueeze(1) + self.eps)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def normalize_parameters(self) -> None:
        """Normalize scale/bias to norm sqrt(d)."""
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
    """Ensemble RMSNorm."""

    def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by per-member RMS and apply the scale."""
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    def normalize_parameters(self) -> None:
        """Normalize the scale to norm sqrt(d)."""
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class EnsembleFlashSACEmbedder(nn.Module):
    """Ensemble input embedder: EnsembleUnitBatchNorm followed by EnsembleUnitLinear."""

    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Return the ensemble embedding (norm -> linear)."""
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class EnsembleFlashSACBlock(nn.Module):
    """Ensemble version of FlashSACBlock."""

    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Apply w1 -> norm -> ReLU -> w2 -> norm -> ReLU and add the residual."""
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        x = x + residual
        return x


class EnsembleCategoricalValue(nn.Module):
    """Ensemble categorical value head: bin logits give log-probs and the expected Q."""

    bin_values: torch.Tensor

    def __init__(
        self,
        num_ensemble: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
    ):
        super().__init__()
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
        self.register_buffer(
            "bin_values",
            torch.linspace(start=min_v, end=max_v, steps=num_bins, dtype=torch.float32).reshape(1, 1, -1),
        )

    def forward(
        self,
        x: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the expected Q from the bin softmax, plus log-prob info."""
        value = self.w(x) + self.bias.unsqueeze(1)
        log_prob = F.log_softmax(value, dim=-1)
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)
        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return value, info
