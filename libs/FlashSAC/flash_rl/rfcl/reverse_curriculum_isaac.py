"""Reverse-curriculum wrapper for the RoboLab (Isaac Lab) GPU-batched RL env.

Teleport and auto-reset happen inside the env (``rc_teleport`` reset event, ``rc_dynamic_timeout``
termination). This wrapper owns the bookkeeping (``CurriculumCore``), attaches ``_rc_*`` attributes to
the inner env, and records episode outcomes after each step. Because auto-reset has already teleported
done envs when ``inner.step`` returns, outcomes use the start step snapshotted before the step.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from flash_rl.envs.robolab import recursive_to_numpy
from flash_rl.rfcl.curriculum_core import CurriculumCore, DemoCurriculumMetadata


def demo_gripper_schedule(close_cmd: torch.Tensor, total: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-state gripper (requested_close, elapsed_since_edge) from a demo's commands.

    Used to restore the delayed gripper actuator on teleport. The request at state
    ``s`` is ``close_cmd[s-1]``; state 0 takes the first action's command, so a demo
    that starts already grasping has no spurious close edge. ``elapsed`` follows the
    actuator convention: 1 at the edge state, +1 per step after it, -1 before any edge.

    Parameters
    ----------
    close_cmd : torch.Tensor
        ``[T-1]`` bool, close request of action ``t``.
    total : int
        Number of dense states ``T``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``requested`` ``[T]`` bool, ``elapsed`` ``[T]`` long.

    Raises
    ------
    ValueError
        If ``close_cmd`` length is not ``total - 1``.
    """
    device = close_cmd.device
    if int(close_cmd.shape[0]) != max(int(total) - 1, 0):
        raise ValueError(f"close_cmd length {int(close_cmd.shape[0])} != total-1 ({int(total) - 1})")
    requested = torch.zeros(int(total), dtype=torch.bool, device=device)
    elapsed = torch.full((int(total),), -1, dtype=torch.long, device=device)
    if total > 1:
        requested[0] = bool(close_cmd[0])
        requested[1:] = close_cmd
        prior = requested[:-1]
        edge_states = torch.nonzero(requested[1:] != prior, as_tuple=False).flatten() + 1
        last_edge = -1
        edge_set = set(int(value) for value in edge_states.tolist())
        for state_idx in range(1, int(total)):
            if state_idx in edge_set:
                last_edge = state_idx
            if last_edge >= 0:
                elapsed[state_idx] = state_idx - last_edge + 1
    return requested, elapsed


class ReverseCurriculumIsaacEnv(gym.vector.VectorWrapper):
    """Parallel reverse curriculum over the demonstrations for the RoboLab RL vector env.

    Parameters
    ----------
    env : gym.Env
        RoboLabVectorEnv built with ``curriculum=True`` (``rc_teleport`` / ``rc_dynamic_timeout``).
    states_dataset : list[dict]
        Output of ``load_demo_states_robolab``.
    max_relative_action_rate : torch.Tensor
        Per-joint rate of the relative arm action (converts demo actions to policy actions).
    to_numpy : bool
        Convert step/reset outputs to numpy.
    **core_kwargs
        ``CurriculumCore`` arguments (reverse_step_size, advance_threshold, frontier_window,
        minimum_episode_steps, fixed_frontier).
    """

    def __init__(
        self,
        env: gym.Env,
        states_dataset: list[dict[str, Any]],
        *,
        max_relative_action_rate: torch.Tensor,
        to_numpy: bool = False,
        **core_kwargs: Any,
    ):
        super().__init__(env)
        self.inner = env
        self.to_numpy = to_numpy
        if len(states_dataset) == 0:
            raise ValueError("states_dataset must contain at least one demo")
        self.states_dataset = states_dataset
        # The wrapper works on torch outputs and converts to numpy itself.
        self.inner.to_numpy = False
        base = env.unwrapped  # RobolabEnv
        self._base = base
        self._device = base.device
        if env.num_envs < len(states_dataset):
            raise ValueError(f"num_envs ({env.num_envs}) < num_demos ({len(states_dataset)})")
        totals = [int(d["total_steps"]) for d in states_dataset]
        self.core = CurriculumCore(env.num_envs, totals, self._device, **core_kwargs)

        # Attributes read by the teleport/timeout events (set before the first reset).
        base._rc_core = self.core
        base._rc_demos = [self._move_state_to_device(d["state_dict"]) for d in states_dataset]
        base._rc_total_steps = totals
        base._rc_start_step = torch.zeros(env.num_envs, dtype=torch.long, device=self._device)
        base._rc_trunc_budget = torch.zeros(env.num_envs, dtype=torch.long, device=self._device)
        base._rl_no_freeze = True

        # Demo actions a_{s-1} in the policy action space, injected as prev_action of teleport-start
        # observations. Isaac Lab reset zeroes the action buffer; without this, live obs (prev_action=0)
        # would differ from demo-buffer obs (a_{s-1}).
        from flash_rl.rfcl.demo_dataset_robolab import relative_demo_actions

        rate = torch.as_tensor(max_relative_action_rate, dtype=torch.float32, device=self._device)
        base._rc_demo_actions = []
        for demo in states_dataset:
            actions = demo["actions"].to(self._device)
            q_cur = demo["state_dict"]["robot"]["joint_pos"][:, :7].to(self._device)
            base._rc_demo_actions.append(relative_demo_actions(actions, q_cur[: actions.shape[0]], rate))
        self._action_dim = int(base._rc_demo_actions[0].shape[-1])
        self._demo_gripper_requested: list[torch.Tensor] = []
        self._demo_gripper_elapsed: list[torch.Tensor] = []
        for actions, total in zip(base._rc_demo_actions, totals, strict=True):
            requested, elapsed = demo_gripper_schedule(actions[:, -1] > 0, int(total))
            self._demo_gripper_requested.append(requested)
            self._demo_gripper_elapsed.append(elapsed)

    def _move_state_to_device(self, sd: Any) -> Any:
        if isinstance(sd, dict):
            return {k: self._move_state_to_device(v) for k, v in sd.items()}
        return sd.to(self._device)

    def _demo_prev_action(self, eids: list[int]) -> torch.Tensor:
        """Demo action a_{s-1} at each env's teleport frame (zeros when s == 0)."""
        pa = torch.zeros((len(eids), self._action_dim), device=self._device)
        for i, eid in enumerate(eids):
            d = int(self.core.env_to_demo[eid])
            s = int(self._base._rc_start_step[eid])
            if s > 0:
                pa[i] = self._base._rc_demo_actions[d][s - 1]
        return pa

    def _restore_demo_gripper_delay(self, eids: list[int]) -> None:
        if not eids:
            return
        requested = torch.zeros(len(eids), dtype=torch.bool, device=self._device)
        elapsed = torch.full((len(eids),), -1, dtype=torch.long, device=self._device)
        for i, eid in enumerate(eids):
            d = int(self.core.env_to_demo[eid])
            s = int(self._base._rc_start_step[eid])
            requested[i] = self._demo_gripper_requested[d][s]
            elapsed[i] = self._demo_gripper_elapsed[d][s]
        self.inner.restore_gripper_delay_from_demo(eids, requested, elapsed)

    @property
    def num_demos(self) -> int:
        """Number of demos."""
        return self.core.num_demos

    @property
    def demo_metadata(self) -> list[DemoCurriculumMetadata]:
        """Per-demo curriculum metadata."""
        return self.core.demo_metadata

    @property
    def reverse_solved_frac(self) -> float:
        """Fraction of solved demos."""
        return self.core.reverse_solved_frac

    @property
    def base_env(self) -> Any:
        """Unwrapped inner env."""
        return self._base

    def reset(self, *, seed: Any = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset all envs; every env is teleported to a curriculum start state."""
        # episode_length_buf must start at 0; the dynamic time limit manages the horizon.
        obs, info = self.inner.reset(random_start_init=False)
        eids = list(range(self.num_envs))
        self._restore_demo_gripper_delay(eids)
        # The gripper-delay obs was computed before the restore above; refresh it.
        self.inner.refresh_gripper_delay_obs(obs, eids)
        obs[:, -self._action_dim :] = self._demo_prev_action(eids)
        if self.to_numpy:
            obs = recursive_to_numpy(obs)
            info = recursive_to_numpy(info)
        return obs, info

    def step(self, action: Any) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the inner env, record outcomes of finished episodes and fix the obs of reset envs."""
        # Auto-reset inside inner.step overwrites the start step and the env's demo; snapshot them.
        prev_start = self._base._rc_start_step.clone()
        prev_demo = self.core.env_to_demo.clone()

        obs, rew, term, trunc, infos = self.inner.step(action)
        dones = term | trunc
        if bool(dones.any()):
            done_eids = dones.nonzero(as_tuple=True)[0].tolist()
            self._restore_demo_gripper_delay(done_eids)
            # Sparse reward: success iff reward > 0 (the safety penalty is bounded far below 1).
            success = rew > 0
            for k in done_eids:
                self.core.record_outcome(int(success[k].item()), int(prev_start[k].item()), int(prev_demo[k].item()))
            self.core.step_curriculum()
            # Done envs were teleported by auto-reset; set their prev_action to the demo action.
            # Clone first: obs may alias infos["final_obs"] (the bootstrap target).
            obs = obs.clone()
            obs[done_eids, -self._action_dim :] = self._demo_prev_action(done_eids)
            self.inner.refresh_gripper_delay_obs(obs, done_eids)
        if self.to_numpy:
            obs = recursive_to_numpy(obs)
            rew = recursive_to_numpy(rew)
            term = recursive_to_numpy(term)
            trunc = recursive_to_numpy(trunc)
            infos = recursive_to_numpy(infos)
        return obs, rew, term, trunc, infos
