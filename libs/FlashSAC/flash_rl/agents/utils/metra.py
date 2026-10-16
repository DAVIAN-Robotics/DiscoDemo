"""METRA skill objective: representation phi, Lagrangian dual lambda, and a success-gated intrinsic reward.

The objective follows the official METRA implementation::

    cst_penalty = 1 - mean_j((phi(s') - phi(s))_j^2)        # temporal-distance constraint
    cst_penalty = clamp(cst_penalty, max=dual_slack)
    loss_te     = -(<phi(s') - phi(s), z> + lambda.detach() * cst_penalty).mean()
    loss_lambda = log_lambda * cst_penalty.detach().mean()  # lambda grows on violation

The intrinsic reward ``<phi(s') - phi(s), z>`` is credited only to transitions of successful
episodes (see `SkillGateTable`). Everything here runs eagerly, outside the compiled update.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MetraRepresentation(nn.Module):
    """phi MLP and log-parameterized lambda, each with its own optimizer."""

    def __init__(
        self,
        obs_sub_dim: int,
        z_dim: int,
        hidden: list[int],
        lr: float,
        dual_slack: float = 1e-3,
        lam_init: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.phi = _mlp(int(obs_sub_dim), [int(h) for h in hidden], int(z_dim)).to(device)
        # lambda is learned by dual ascent; lam_init only sets the early balance between
        # the constraint and the inner-product term.
        assert float(lam_init) > 0.0, f"lam_init must be positive: {lam_init}"
        self._log_lam = nn.Parameter(torch.full((), math.log(float(lam_init)), device=device))
        self._dual_slack = float(dual_slack)
        self._phi_opt = torch.optim.Adam(self.phi.parameters(), lr=float(lr))
        self._lam_opt = torch.optim.Adam([self._log_lam], lr=float(lr))

    @property
    def lam(self) -> float:
        """Current lambda = exp(log_lambda)."""
        return float(self._log_lam.detach().exp().item())

    @torch.no_grad()
    def r_int(self, obs_sub: torch.Tensor, next_obs_sub: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Intrinsic reward ``<phi(s') - phi(s), z>`` under the current phi."""
        out: torch.Tensor = ((self.phi(next_obs_sub) - self.phi(obs_sub)) * z).sum(dim=-1)
        return out

    def update(self, obs_sub: torch.Tensor, next_obs_sub: torch.Tensor, z: torch.Tensor) -> dict[str, float]:
        """One gradient step on phi and on lambda."""
        d_phi = self.phi(next_obs_sub) - self.phi(obs_sub)
        inner = (d_phi * z).sum(dim=-1)
        phi_dist = d_phi.square().mean(dim=-1)
        cst_penalty = (1.0 - phi_dist).clamp(max=self._dual_slack)

        lam = self._log_lam.exp()
        loss_te = -(inner + lam.detach() * cst_penalty).mean()
        self._phi_opt.zero_grad(set_to_none=True)
        loss_te.backward()
        self._phi_opt.step()

        # Use the detached penalty for the lambda step.
        loss_lam = self._log_lam * cst_penalty.detach().mean()
        self._lam_opt.zero_grad(set_to_none=True)
        loss_lam.backward()
        self._lam_opt.step()

        return {
            "loss_te": float(loss_te.item()),
            "lambda": self.lam,
            "cst_penalty_mean": float(cst_penalty.detach().mean().item()),
            "phi_dist_mean": float(phi_dist.detach().mean().item()),
            "r_int_mean": float(inner.detach().mean().item()),
        }

    def checkpoint_state_dict(self) -> dict[str, object]:
        """Resumable state: module weights plus both private Adam optimizers.

        Returns
        -------
        dict[str, object]
            ``version`` / ``module`` / ``phi_optimizer`` / ``lambda_optimizer`` / ``dual_slack``.
        """
        return {
            "version": 1,
            "module": self.state_dict(),
            "phi_optimizer": self._phi_opt.state_dict(),
            "lambda_optimizer": self._lam_opt.state_dict(),
            "dual_slack": self._dual_slack,
        }

    def load_checkpoint_state_dict(self, state: dict[str, object]) -> None:
        """Restore state from `checkpoint_state_dict`.

        Raises
        ------
        ValueError
            On a schema version or ``dual_slack`` mismatch.
        """
        if int(state.get("version", -1)) != 1:  # type: ignore[call-overload]
            raise ValueError(f"unsupported METRA checkpoint version: {state.get('version')!r}")
        if float(state["dual_slack"]) != self._dual_slack:  # type: ignore[arg-type]
            raise ValueError(f"METRA dual_slack mismatch: checkpoint={state['dual_slack']} current={self._dual_slack}")
        self.load_state_dict(state["module"])  # type: ignore[arg-type]
        self._phi_opt.load_state_dict(state["phi_optimizer"])  # type: ignore[arg-type]
        self._lam_opt.load_state_dict(state["lambda_optimizer"])  # type: ignore[arg-type]


class SkillGateTable:
    """Table mapping episode uid -> binary success gate.

    The training loop records each episode's success when it ends, and the gate is looked up at
    update time. uid 0 is reserved: demo transitions and unfinished episodes get gate 0. A run
    with more than ``capacity - 1`` episodes raises instead of reusing slots, so a replay row
    never reads another episode's gate.
    """

    def __init__(self, capacity: int = 1 << 20, device: str = "cpu"):
        self._cap = int(capacity)
        self._table = torch.zeros((self._cap,), dtype=torch.float32, device=device)

    def set(self, ep_uids: torch.Tensor, success: torch.Tensor) -> None:
        """Record finished episodes."""
        uid = ep_uids.to(dtype=torch.long, device=self._table.device)
        if int(uid.max()) >= self._cap:
            raise RuntimeError(f"episode uid {int(uid.max())} exceeds the gate table capacity {self._cap}")
        self._table[uid] = success.to(dtype=torch.float32, device=self._table.device)
        self._table[0] = 0.0  # keep the reserved slot at 0

    def lookup(self, ep_uids: torch.Tensor) -> torch.Tensor:
        """Return the (B,) float gate; uid 0 -> 0."""
        return self._table[ep_uids.to(dtype=torch.long, device=self._table.device)]

    def state_dict(self) -> dict[str, object]:
        """Snapshot with ``version`` / ``capacity`` / ``table`` (CPU copy)."""
        return {"version": 1, "capacity": self._cap, "table": self._table.detach().cpu()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore from `state_dict`.

        Raises
        ------
        ValueError
            On a version, capacity or tensor shape mismatch.
        """
        if int(state.get("version", -1)) != 1:  # type: ignore[call-overload]
            raise ValueError(f"unsupported skill gate checkpoint version: {state.get('version')!r}")
        if int(state["capacity"]) != self._cap:  # type: ignore[call-overload]
            raise ValueError(f"SkillGateTable capacity mismatch: checkpoint={state['capacity']} current={self._cap}")
        table = state["table"]
        if not isinstance(table, torch.Tensor) or tuple(table.shape) != (self._cap,):
            raise ValueError(f"invalid SkillGateTable tensor: {type(table).__name__}")
        self._table.copy_(table.to(dtype=self._table.dtype, device=self._table.device))
        self._table[0] = 0.0


def apply_skill_intrinsic(
    batch: dict[str, torch.Tensor],
    metra: MetraRepresentation,
    gate_table: SkillGateTable,
    alpha: float,
    z_dim: int,
    phi_dims: torch.Tensor,
) -> dict[str, float]:
    """Add the gated intrinsic reward to ``batch`` in place and take one phi/lambda step.

    ``reward += alpha * gate(ep_uid) * r_int``. ``r_int`` is recomputed with the current phi and
    ``z`` is read from the last ``z_dim`` observation dims. Call this after ``normalize_rewards``
    so the task-reward statistics are not affected. phi is trained on all transitions (including
    demos); only the intrinsic credit is gated.

    Parameters
    ----------
    batch : dict[str, torch.Tensor]
        Replay batch; ``reward`` is modified in place.
    metra : MetraRepresentation
        phi / lambda module.
    gate_table : SkillGateTable
        Episode uid -> success gate.
    alpha : float
        Diversity weight.
    z_dim : int
        Skill latent dimension.
    phi_dims : torch.Tensor
        Observation indices fed to phi.

    Returns
    -------
    dict[str, float]
        Diagnostic scalars for logging.
    """
    obs_sub = batch["observation"][:, phi_dims]
    next_obs_sub = batch["next_observation"][:, phi_dims]
    z = batch["observation"][:, -int(z_dim) :]
    r_int = metra.r_int(obs_sub, next_obs_sub, z)
    gate = gate_table.lookup(batch["ep_uid"])
    batch["reward"] = batch["reward"] + float(alpha) * gate * r_int
    rep_info = metra.update(obs_sub, next_obs_sub, z)
    return {
        "skill/r_int_mean": float(r_int.mean().item()),
        "skill/gate_frac": float(gate.mean().item()),
        "skill/lambda": rep_info["lambda"],
        "skill/phi_dist_mean": rep_info["phi_dist_mean"],
        "skill/loss_te": rep_info["loss_te"],
        "skill/cst_penalty_mean": rep_info["cst_penalty_mean"],
    }
