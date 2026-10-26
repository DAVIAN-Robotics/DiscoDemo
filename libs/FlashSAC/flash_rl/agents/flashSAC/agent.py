"""FlashSAC agent: config, network setup, action sampling, updates and checkpoints."""

import math
import os
from collections.abc import MutableMapping
from dataclasses import dataclass, replace
from typing import Any, cast

import gymnasium as gym
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler

from flash_rl.agents.base_agent import BaseAgent
from flash_rl.agents.flashSAC.network import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
)
from flash_rl.agents.flashSAC.update import (
    update_actor,
    update_critic,
    update_target_network,
    update_temperature,
)
from flash_rl.agents.utils.metra import MetraRepresentation, SkillGateTable, apply_skill_intrinsic
from flash_rl.agents.utils.network import Network
from flash_rl.agents.utils.normalizer import Normalizer
from flash_rl.agents.utils.reward_normalization import RewardNormalizer
from flash_rl.agents.utils.scheduler import warmup_cosine_decay_scheduler
from flash_rl.buffers.torch_buffer import TorchUniformBuffer
from flash_rl.types import NDArray, Tensor


@dataclass
class FlashSACConfig:
    """FlashSAC hyperparameters."""

    seed: int
    normalize_reward: bool
    normalized_G_max: float

    asymmetric_observation: bool
    device_type: str

    buffer_max_length: int
    buffer_min_length: int
    buffer_device_type: str
    sample_batch_size: int

    learning_rate_init: float
    learning_rate_peak: float
    learning_rate_end: float
    learning_rate_warmup_rate: float
    learning_rate_warmup_step: int
    learning_rate_decay_rate: float
    learning_rate_decay_step: int

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_bc_alpha: float
    actor_noise_zeta_mu: float
    actor_noise_zeta_max: int
    actor_update_period: int

    critic_num_blocks: int
    critic_hidden_dim: int
    critic_num_bins: int
    critic_min_v: float
    critic_max_v: float
    critic_target_update_tau: float

    temp_initial_value: float
    temp_target_sigma: float
    temp_target_entropy: float

    gamma: float
    n_step: int

    use_compile: bool
    compile_mode: str
    use_amp: bool

    load_optimizer: bool
    load_reward_normalizer: bool


def _init_flashsac_networks(
    actor_observation_dim: int,
    critic_observation_dim: int,
    action_dim: int,
    cfg: FlashSACConfig,
    device: torch.device,
) -> tuple[Network, Network, Network, Network]:
    # Create learning rate schedule
    warmup_cosine_decay_lr = warmup_cosine_decay_scheduler(
        init_value=cfg.learning_rate_init,
        peak_value=cfg.learning_rate_peak,
        end_value=cfg.learning_rate_end,
        warmup_steps=cfg.learning_rate_warmup_step,
        decay_steps=cfg.learning_rate_decay_step,
    )

    # Initialize actor
    actor_net = FlashSACActor(
        num_blocks=cfg.actor_num_blocks,
        input_dim=actor_observation_dim,
        hidden_dim=cfg.actor_hidden_dim,
        action_dim=action_dim,
    ).to(device)

    use_fused = device.type == "cuda" and torch.cuda.is_available()
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=cfg.learning_rate_peak, fused=use_fused)
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    actor = Network(
        network=actor_net,
        optimizer=actor_optimizer,
        scheduler=actor_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )
    # Manually compile `get_mean_and_std` function
    if cfg.use_compile:
        actor.network.get_mean_and_std = torch.compile(actor.network.get_mean_and_std, mode=cfg.compile_mode)  # type: ignore

    # Initialize critic
    critic_net = FlashSACDoubleCritic(
        num_blocks=cfg.critic_num_blocks,
        input_dim=critic_observation_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)

    critic_optimizer = optim.Adam(
        critic_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    critic = Network(
        network=critic_net,
        optimizer=critic_optimizer,
        scheduler=critic_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
    )

    # Initialize target critic (same as critic but no optimizer)
    target_critic_net = FlashSACDoubleCritic(
        num_blocks=cfg.critic_num_blocks,
        input_dim=critic_observation_dim + action_dim,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
    ).to(device)
    target_critic_net.load_state_dict(critic_net.state_dict())
    target_critic = Network(
        network=target_critic_net,
        optimizer=None,
        scheduler=None,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=True,
        ema_source=critic,  # wire EMA update source
        ema_tau=cfg.critic_target_update_tau,
    )

    # Initialize temperature
    temp_net = FlashSACTemperature(cfg.temp_initial_value).to(device)
    temp_optimizer = optim.Adam(
        temp_net.parameters(),
        lr=cfg.learning_rate_peak,
        fused=use_fused,
    )
    temp_scheduler = torch.optim.lr_scheduler.LambdaLR(
        temp_optimizer,
        lr_lambda=lambda step: warmup_cosine_decay_lr(step) / cfg.learning_rate_peak,
    )
    temperature = Network(
        network=temp_net,
        optimizer=temp_optimizer,
        scheduler=temp_scheduler,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        use_weight_normalization=False,
    )

    # normalize network parameters after initialization
    actor.normalize_parameters()
    critic.normalize_parameters()
    target_critic.normalize_parameters()

    return actor, critic, target_critic, temperature


@torch.compile
def _build_truncated_zeta_cdf(mu: float, max_n: int) -> torch.Tensor:
    """Build the truncated zeta CDF for the given mu and max_n."""
    ns = torch.arange(1, max_n + 1, dtype=torch.float32)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    cdf = torch.cumsum(pmf, dim=0)
    return cdf


@torch.compile
def _sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    """Sample an integer from the given CDF.

    Returns a 0-d int32 tensor.
    """
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)


def _sample_flashsac_actions(
    actor: Network,
    noise: torch.Tensor,
    observations: torch.Tensor,
    temperature: float,
    cur_count: torch.Tensor,
    cur_n: torch.Tensor,
    zeta_cdf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample actions with noise repeat logic fully in torch."""
    # forward actor → distribution mean and std
    mean, std = actor.apply(
        "get_mean_and_std",
        observations=observations,
        training=False,
    )
    # return deterministic actions without changing noise sampling params
    if temperature == 0.0:
        actions = torch.tanh(mean)
        return noise, actions, cur_count, cur_n

    # reinit noise after a certain number of steps (only during training)
    reinit = (cur_count == 0) | (cur_count >= cur_n)

    new_noise = torch.randn_like(mean)
    new_n = _sample_integer_from_cdf(zeta_cdf)

    noise = torch.where(reinit, new_noise, noise)
    cur_n = torch.where(reinit, new_n, cur_n)
    cur_count = torch.where(reinit, torch.zeros_like(cur_count), cur_count)

    # sample action
    actions = torch.tanh(mean + std * noise * temperature)

    return noise, actions, cur_count + 1, cur_n


def _update_networks(
    batch: dict[str, torch.Tensor],
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    cfg: FlashSACConfig,
    do_actor_update: bool,
    device: torch.device,
    grad_scaler: GradScaler | None,
    scheduler_step: int,
) -> dict[str, torch.Tensor]:
    if do_actor_update:
        # Update actor
        actor_info = update_actor(
            actor=actor,
            critic=critic,
            temperature=temperature,
            batch=batch,  # type: ignore
            bc_alpha=cfg.actor_bc_alpha,
            device=device,
            use_amp=cfg.use_amp,
            grad_scaler=grad_scaler,
            scheduler_step=scheduler_step,
        )

        # Update temperature
        temperature_info = update_temperature(
            temperature=temperature,
            entropy=actor_info["actor/entropy"],
            target_entropy=cfg.temp_target_entropy,
            scheduler_step=scheduler_step,
        )
    else:
        actor_info = {}
        temperature_info = {}

    # Update critic
    critic_info = update_critic(
        actor=actor,  # updated
        critic=critic,
        target_critic=target_critic,
        temperature=temperature,  # updated
        batch=batch,  # type: ignore
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
        num_bins=cfg.critic_num_bins,
        gamma=cfg.gamma,
        n_step=cfg.n_step,
        device=device,
        use_amp=cfg.use_amp,
        grad_scaler=grad_scaler,
        scheduler_step=scheduler_step,
    )

    target_critic_info = update_target_network(
        target_network=target_critic,
    )

    # Merge all info dicts
    update_info = {
        **actor_info,
        **critic_info,
        **target_critic_info,
        **temperature_info,
    }

    return update_info


def _resolve_compile_mode(mode: str) -> str:
    """Resolve 'auto' compile mode based on the installed torch version."""
    if mode != "auto":
        return mode
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        return "max-autotune"
    return "reduce-overhead"


class FlashSACAgent(BaseAgent[FlashSACConfig]):
    """FlashSAC agent implemented in PyTorch."""

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: FlashSACConfig,
    ):
        """FlashSAC agent implementation in PyTorch."""
        self._critic_observation_dim: int = observation_space.shape[-1]  # type: ignore
        self._action_dim: int = action_space.shape[-1]  # type: ignore
        if cfg.asymmetric_observation:
            self._actor_observation_dim = env_info["actor_observation_size"][-1]
        else:
            self._actor_observation_dim = self._critic_observation_dim

        temp_target_entropy = 0.5 * self._action_dim * math.log(2 * math.pi * math.e * cfg.temp_target_sigma**2)
        compile_mode = _resolve_compile_mode(cfg.compile_mode)
        cfg = replace(cfg, temp_target_entropy=temp_target_entropy, compile_mode=compile_mode)

        super().__init__(
            observation_space,
            action_space,
            env_info,
            cfg,
        )
        self._cfg = cfg

        device_type = cfg.device_type
        device_type = (
            device_type
            if device_type.startswith("cuda") and ":" in device_type
            else ("cuda:0" if device_type.startswith("cuda") else "cpu")
        )
        self._device = torch.device(device_type)

        # Initialize networks
        (
            self._actor,
            self._critic,
            self._target_critic,
            self._temperature,
        ) = _init_flashsac_networks(
            actor_observation_dim=self._actor_observation_dim,
            critic_observation_dim=self._critic_observation_dim,
            action_dim=self._action_dim,
            cfg=self._cfg,
            device=self._device,
        )
        self._update_step = 0

        # Grad scaler for FP16 AMP
        self._grad_scaler = GradScaler(device=self._device.type, enabled=self._cfg.use_amp)

        # Noise repetition (zeta distribution)
        self._zeta_cdf = _build_truncated_zeta_cdf(
            mu=self._cfg.actor_noise_zeta_mu, max_n=self._cfg.actor_noise_zeta_max
        ).to(self._device)
        self._cur_noise_repeat_n = torch.tensor(1, dtype=torch.int32, device=self._device)
        self._cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=self._device)
        action_shape = tuple(action_space.shape) if action_space.shape is not None else ()
        self._cached_noise = torch.randn(action_shape, device=self._device)

        # METRA skill modules; enabled only via set_skill_modules.
        self._skill_module: MetraRepresentation | None = None
        self._skill_gate_table: SkillGateTable | None = None
        self._skill_alpha: float = 0.0
        self._skill_z_dim: int = 0
        self._skill_phi_dims: torch.Tensor | None = None
        self._skill_ep_uid_next: int = 1

        # Reward normalizer
        self.reward_normalizer = None
        if self._cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=self._cfg.gamma,
                G_max=self._cfg.normalized_G_max,
                load_rms=self._cfg.load_reward_normalizer,
                device=self._device,
            )

        # Observation normalizer; the training script sets its bounds.
        self._normalizer = Normalizer(obs_dim=self._critic_observation_dim).to(self._device)

        # Replay buffer
        self._replay_buffer = self._make_replay_buffer(store_ep_uid=False)

    def _make_replay_buffer(self, store_ep_uid: bool) -> TorchUniformBuffer:
        return TorchUniformBuffer(
            observation_space=self._observation_space,
            action_space=self._action_space,
            n_step=self._cfg.n_step,
            gamma=self._cfg.gamma,
            max_length=self._cfg.buffer_max_length,
            min_length=self._cfg.buffer_min_length,
            sample_batch_size=self._cfg.sample_batch_size,
            device_type=self._cfg.buffer_device_type,
            store_ep_uid=store_ep_uid,
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        """Sample actions for the previous transition's next observation.

        Training uses zeta-repeated exploration noise; otherwise actions are deterministic
        (``tanh(mean)``).
        """
        temperature = 1.0 if training else 0.0
        observations = torch.as_tensor(prev_transition["next_observation"], dtype=torch.float32).to(self._device)
        # Normalize the full critic observation, then slice the actor part (asymmetric only).
        observations = self._normalizer.normalize_obs(observations)
        if self._cfg.asymmetric_observation:
            observations = observations[:, : self._actor_observation_dim]

        with torch.no_grad():
            (
                self._cached_noise,
                actions,
                self._cur_noise_repeat_count,
                self._cur_noise_repeat_n,
            ) = _sample_flashsac_actions(
                actor=self._actor,
                noise=self._cached_noise,
                observations=observations,
                temperature=temperature,
                cur_count=self._cur_noise_repeat_count,
                cur_n=self._cur_noise_repeat_n,
                zeta_cdf=self._zeta_cdf,
            )

        return actions.cpu().numpy()

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None:
        """Normalize the observations, update reward statistics and add the transition to the buffer."""
        # The buffer stores normalized observations; actions stay in the actor's tanh space.
        transition = dict(transition)
        for key in ("observation", "next_observation"):
            obs = torch.as_tensor(transition[key], dtype=torch.float32, device=self._device)
            transition[key] = self._normalizer.normalize_obs(obs)
        self._replay_buffer.add(transition)

        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            self.reward_normalizer.update_reward_stats(
                reward=torch.as_tensor(transition["reward"], device=self._device),
                terminated=torch.as_tensor(transition["terminated"], device=self._device),
                truncated=torch.as_tensor(transition["truncated"], device=self._device),
            )

    def can_start_training(self) -> bool:
        """Whether the replay buffer holds enough transitions to sample."""
        return self._replay_buffer.can_sample()

    def update(self, env_step: int) -> dict[str, Any]:
        """Run one SAC update step.

        The LR schedulers are stepped with ``env_step`` (cumulative env steps), so the cosine decay
        spans ``num_env_steps`` even though the reverse and forward phases use different env counts.
        """
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())

        for k, v in batch.items():
            batch[k] = v.to(self._device, non_blocking=True)

        if self._cfg.asymmetric_observation:
            batch["actor_observation"] = batch["observation"][:, : self._actor_observation_dim]
            batch["actor_next_observation"] = batch["next_observation"][:, : self._actor_observation_dim]
        else:
            batch["actor_observation"] = batch["observation"]
            batch["actor_next_observation"] = batch["next_observation"]

        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            # batch["unnormalized_reward"] = batch["reward"].clone()
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

        # METRA intrinsic reward, added after reward normalization (eager, outside the
        # compiled update), plus one phi/lambda step.
        _skill_info: dict[str, float] = {}
        if self._skill_module is not None:
            assert self._skill_gate_table is not None and self._skill_phi_dims is not None
            _skill_info = apply_skill_intrinsic(
                batch,
                self._skill_module,
                self._skill_gate_table,
                alpha=self._skill_alpha,
                z_dim=self._skill_z_dim,
                phi_dims=self._skill_phi_dims,
            )

        # Update step
        _update_info = _update_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            cfg=self._cfg,
            do_actor_update=(self._update_step % self._cfg.actor_update_period == 0),
            device=self._device,
            grad_scaler=self._grad_scaler,
            scheduler_step=env_step,
        )
        self._update_step += 1

        # Convert tensors to floats
        update_info: dict[str, float] = {}
        for key, value in _update_info.items():
            if isinstance(value, torch.Tensor):
                update_info[key] = value.item()
            elif not isinstance(value, dict):
                update_info[key] = float(value)

        update_info.update(_skill_info)

        return update_info

    def set_skill_modules(
        self,
        module: MetraRepresentation,
        gate_table: SkillGateTable,
        alpha: float,
        z_dim: int,
        phi_dims: torch.Tensor,
    ) -> None:
        """Enable the METRA skill objective (alpha > 0 only).

        Call before any transition is added: the replay buffer is rebuilt to also store episode uids.
        """
        assert float(alpha) > 0.0, "do not attach skill modules when alpha <= 0"
        assert int(z_dim) > 0 and phi_dims.numel() > 0
        assert len(self._replay_buffer) == 0, "attach skill modules before training starts"
        self._replay_buffer = self._make_replay_buffer(store_ep_uid=True)
        self._skill_module = module
        self._skill_gate_table = gate_table
        self._skill_alpha = float(alpha)
        self._skill_z_dim = int(z_dim)
        self._skill_phi_dims = phi_dims.to(dtype=torch.long, device=self._device)
        self._skill_ep_uid_next = 1

    @property
    def skill_episode_uid_next(self) -> int:
        """Next globally unique episode id used by the METRA gate lookup."""
        return self._skill_ep_uid_next

    def set_skill_episode_uid_next(self, value: int) -> None:
        """Publish the train-loop UID allocator so checkpoints capture it atomically."""
        if self._skill_module is None:
            raise RuntimeError("cannot set skill episode UID while skill modules are disabled")
        if int(value) < 1:
            raise ValueError(f"skill episode UID must be positive, got {value}")
        self._skill_ep_uid_next = int(value)

    def reset_action_noise_state(self, action_space: gym.spaces.Space) -> None:
        """Reset the cached per-env noise and zeta-repeat counters for a new action space.

        Call when the number of envs changes (reverse -> forward transition).
        """
        action_shape = tuple(action_space.shape) if action_space.shape is not None else ()
        self._cached_noise = torch.randn(action_shape, device=self._device)
        self._cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=self._device)
        self._cur_noise_repeat_n = torch.tensor(1, dtype=torch.int32, device=self._device)

    def save(self, path: str) -> None:
        """Save networks, normalizers and agent state under ``path``."""
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "actor.pt"))
        self._critic.save(os.path.join(path, "critic.pt"))
        self._target_critic.save(os.path.join(path, "target_critic.pt"))
        self._temperature.save(os.path.join(path, "temperature.pt"))
        if self.reward_normalizer is not None:
            self.reward_normalizer.save(os.path.join(path, "reward_normalizer.pt"))
        torch.save(self._normalizer.state_dict(), os.path.join(path, "normalizer.pt"))

        agent_state: dict[str, Any] = {
            "update_step": self._update_step,
            "grad_scaler_state_dict": self._grad_scaler.state_dict(),
        }
        torch.save(agent_state, os.path.join(path, "agent_state.pt"))
        if self._skill_module is not None:
            assert self._skill_gate_table is not None and self._skill_phi_dims is not None
            skill_state: dict[str, Any] = {
                "version": 1,
                "skill_type": "metra",
                "alpha": self._skill_alpha,
                "z_dim": self._skill_z_dim,
                "phi_dims": self._skill_phi_dims.detach().cpu(),
                "episode_uid_next": self._skill_ep_uid_next,
                "representation": self._skill_module.checkpoint_state_dict(),
                "gate_table": self._skill_gate_table.state_dict(),
            }
            torch.save(skill_state, os.path.join(path, "skill_state.pt"))

        print(f"\033[32m[FlashSAC]\033[0m Successfully saved checkpoint {self._update_step} at {path}.")

    def save_replay_buffer(self, path: str) -> None:
        """Save the replay buffer to ``path/replay_buffer.pt``."""
        self._replay_buffer.save(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Successfully saved replay buffer at {path}.")

    def load_actor(self, path: str) -> None:
        """Inference-only restore: load the actor and the input normalizer from ``path``.

        Works on checkpoints without critic files; do not use for resuming training.
        """
        self._actor.load(os.path.join(path, "actor.pt"), load_optimizer=False)
        self._load_normalizer(path)

    def _load_normalizer(self, path: str) -> None:
        self._normalizer.load_state_dict(torch.load(os.path.join(path, "normalizer.pt"), map_location=self._device))
        assert self._normalizer.is_fitted, "loaded normalizer was saved without bounds"

    def load(self, path: str) -> None:
        """Restore a full training checkpoint under ``path`` (resume)."""
        load_optimizer = self._cfg.load_optimizer
        self._actor.load(os.path.join(path, "actor.pt"), load_optimizer=load_optimizer)
        self._critic.load(os.path.join(path, "critic.pt"), load_optimizer=load_optimizer)
        self._target_critic.load(os.path.join(path, "target_critic.pt"), load_optimizer=False)
        self._temperature.load(os.path.join(path, "temperature.pt"), load_optimizer=load_optimizer)

        # Load agent-level optimizer state
        if load_optimizer:
            agent_state = torch.load(os.path.join(path, "agent_state.pt"), map_location=self._device)
            self._update_step = agent_state["update_step"]
            self._grad_scaler.load_state_dict(agent_state["grad_scaler_state_dict"])

        if self._cfg.load_reward_normalizer:
            assert self.reward_normalizer is not None
            self.reward_normalizer.load(os.path.join(path, "reward_normalizer.pt"))
        self._load_normalizer(path)

        # Replay entries refer to episode ids in this gate table; restore it with the module.
        if self._skill_module is not None:
            assert self._skill_gate_table is not None and self._skill_phi_dims is not None
            skill_state = torch.load(os.path.join(path, "skill_state.pt"), map_location=self._device)
            if int(skill_state["version"]) != 1 or skill_state["skill_type"] != "metra":
                raise ValueError(f"invalid METRA skill checkpoint metadata in {path}")
            for key, current in (("z_dim", self._skill_z_dim), ("alpha", self._skill_alpha)):
                if skill_state[key] != current:
                    raise ValueError(f"METRA {key} mismatch: checkpoint={skill_state[key]} current={current}")
            loaded_phi_dims = torch.as_tensor(skill_state["phi_dims"], dtype=torch.long)
            if not torch.equal(loaded_phi_dims.cpu(), self._skill_phi_dims.detach().cpu()):
                raise ValueError("METRA phi_dims mismatch between checkpoint and current environment")
            self._skill_module.load_checkpoint_state_dict(skill_state["representation"])
            self._skill_gate_table.load_state_dict(skill_state["gate_table"])
            self.set_skill_episode_uid_next(int(skill_state["episode_uid_next"]))

        print(f"\033[32m[FlashSAC]\033[0m Successfully loaded checkpoint from {path}.")

    def load_replay_buffer(self, path: str) -> None:
        """Restore the replay buffer from ``path/replay_buffer.pt``."""
        self._replay_buffer.load(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[FlashSAC]\033[0m Successfully loaded replay buffer from {path}.")

    def get_metrics(self) -> dict[str, Any]:
        """No internal metrics."""
        return {}
