# SPDX-License-Identifier: Apache-2.0
"""State-vector RL layer: observation terms, normalized actions and success reward.

Normalization follows `robolab.robots.action_norm`: arm joints map affinely onto their physical
limits (the joint7 +pi/4 hand-mount offset is already the limit center, so no extra offset), and
the gripper uses `[-1,1]` with threshold 0.
"""

from __future__ import annotations

from typing import Any

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, RelativeJointPositionAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from robolab.core.task.conditionals import object_in_container
from robolab.robots.action_norm import JerkLimitedJointCommand, arm_scale_offset, rate_scale_offset
from robolab.robots.droid import (
    arm_joint_pos_panda,
    eef_pos,
    eef_quat,
    gripper_pos_panda,
)

# Success predicates by name. Dispatch by name because the predicate travels in termination params
# across the reverse -> forward re-registration. Every predicate is called as
# ``fn(env, object=<target>, container=<anchor>, **task_kwargs)``.
SUCCESS_PREDICATES: dict[str, Any] = {"object_in_container": object_in_container}


def register_success_predicate(name: str, fn: Any) -> None:
    """Register an RL success predicate by name (entry point for packages outside RoboLab).

    Parameters
    ----------
    name : str
        Must equal ``terminations.success.func.__name__``.
    fn : Callable
        The predicate.

    Raises
    ------
    ValueError
        If a different function is already registered under ``name``.
    """
    existing = SUCCESS_PREDICATES.setdefault(name, fn)
    if existing is not fn:
        raise ValueError(f"success predicate {name!r} is already registered to a different function: {existing}")


# Params owned by the RL layer (they must stay consistent with obs/reward/randomize wiring).
# All other predicate params (tolerances, contact requirements, ...) come from the task.
_RL_OWNED_PARAM_KEYS: frozenset[str] = frozenset(
    {"object", "object_name", "container", "container_name", "hold_steps", "predicate", "pred_kwargs"}
)


def predicate_kwargs_from_task(succ_params: dict[str, Any]) -> dict[str, Any]:
    """Extract the task-declared predicate kwargs from success termination params.

    Parameters
    ----------
    succ_params : dict
        The task's ``terminations.success.params``, or the params of a previously built
        ``succ_done_term``.

    Returns
    -------
    dict
        All keys except ``_RL_OWNED_PARAM_KEYS``; empty means the predicate's own defaults apply.

    Notes
    -----
    On re-registration (reverse -> forward) the params come from ``succ_done_term``, where the
    task kwargs are nested under ``pred_kwargs``. The nested dict is used as the base so they
    survive the round trip.
    """
    nested = succ_params.get("pred_kwargs")
    merged: dict[str, Any] = dict(nested) if isinstance(nested, dict) else {}
    merged.update({k: v for k, v in succ_params.items() if k not in _RL_OWNED_PARAM_KEYS})
    return merged


########################################################
# Object / relative observation terms (state-vector)
########################################################


def object_pos(env: ManagerBasedRLEnv, object_name: str) -> torch.Tensor:
    """Object position (x, y, z) relative to the env origin."""
    obj = env.scene.rigid_objects[object_name]
    return obj.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]


def object_quat(env: ManagerBasedRLEnv, object_name: str) -> torch.Tensor:
    """Object world-frame orientation quaternion (w, x, y, z)."""
    obj = env.scene.rigid_objects[object_name]
    return obj.data.root_quat_w


def hand_to_object(env: ManagerBasedRLEnv, object_name: str) -> torch.Tensor:
    """Vector from the hand frame (``eef_pos``) to the object (both env-relative)."""
    obj_envrel = object_pos(env, object_name)
    return obj_envrel - eef_pos(env)


def build_rl_obs_group(target: str, container: str) -> ObsGroup:
    """Build the flat state-vector policy obs group.

    Layout (dims)::

      target        pos(3) + quat(4) + hand->obj(3) = 10
      container     pos(3) + quat(4) + hand->obj(3) = 10
      arm_joint_pos(7) + gripper(1) + eef_pos(3) + eef_quat(4) = 15
      prev_action(8)   -- always last; demo reconstruction reads ``obs[:, -action_dim:]``

    Parameters
    ----------
    target : str
        Object of the success termination.
    container : str
        Its anchor (container, bottom cube or board).
    """
    attrs: dict = {}
    for role, obj in (("target", target), ("container", container)):
        attrs[f"{role}_pos"] = ObsTerm(func=object_pos, params={"object_name": obj})
        attrs[f"{role}_quat"] = ObsTerm(func=object_quat, params={"object_name": obj})
        attrs[f"hand_to_{role}"] = ObsTerm(func=hand_to_object, params={"object_name": obj})
    attrs["arm_joint_pos"] = ObsTerm(func=arm_joint_pos_panda)
    attrs["gripper_pos"] = ObsTerm(func=gripper_pos_panda)
    attrs["eef_position"] = ObsTerm(func=eef_pos)
    attrs["eef_orientation"] = ObsTerm(func=eef_quat)
    attrs["prev_action"] = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True  # flat float tensor

    attrs["__post_init__"] = __post_init__
    cls = configclass(type("RLObsGroup", (ObsGroup,), attrs))
    return cls()


########################################################
# Actions — normalized joint position + threshold-0 gripper
########################################################


class NormalizedJointPositionAction(JointPositionAction):
    """JointPositionAction mapping raw `[-1,1]` affinely onto each joint's soft limits.

    Scale/offset are overwritten from `soft_joint_pos_limits` via `action_norm.arm_scale_offset`.
    The joint7 +pi/4 hand-mount offset is already the limit center, so the cfg sets no offset.
    """

    def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        # [num_envs, num_joints, 2] (lower, upper); preserve_order=True keeps cfg joint order.
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        qmin = limits[..., 0]
        qmax = limits[..., 1]
        scale, offset = arm_scale_offset(qmin, qmax)
        self._scale = scale
        self._offset = offset


@configclass
class NormalizedJointPositionActionCfg(JointPositionActionCfg):
    """config for :class:`NormalizedJointPositionAction`."""

    class_type = NormalizedJointPositionAction


class RateLimitedRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint action: raw ``[-1,1]`` -> per-step displacement ``delta = a * r``.

    Scale is the per-axis rate ``r`` and offset 0 (``action_norm.rate_scale_offset``); the
    resulting target is clamped to the joint limits.
    """

    def __init__(self, cfg: RateLimitedRelativeJointPositionActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        rate = list(cfg.max_relative_action_rate)
        if len(rate) != len(self._joint_ids):
            raise ValueError(
                f"max_relative_action_rate length ({len(rate)}) != number of arm joints ({len(self._joint_ids)})"
            )
        r = torch.tensor(rate, device=self.device, dtype=torch.float32)  # [7]
        scale, offset = rate_scale_offset(r)
        self._scale = scale.expand(self.num_envs, -1).clone()  # [N, 7]
        self._offset = float(offset)
        # Joint limits for clamping in apply_actions; [N, J, 2].
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        self._qmin = limits[..., 0]  # [N, 7]
        self._qmax = limits[..., 1]

    def apply_actions(self) -> None:
        target = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        target = torch.clamp(target, self._qmin, self._qmax)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class RateLimitedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """config for :class:`RateLimitedRelativeJointPositionAction`.

    ``max_relative_action_rate`` is the per-axis max displacement per step (rad) for the 7 arm
    joints, filled in by ``register_rl_env``.
    """

    class_type = RateLimitedRelativeJointPositionAction
    max_relative_action_rate: list[float] = []


class JerkLimitedRelativeJointPositionAction(RateLimitedRelativeJointPositionAction):
    """Relative arm action of the RL policies, smoothed by velocity / acceleration / jerk limits.

    Raw policy channels in ``[-1, 1]`` are desired velocity fractions. The command velocity and
    acceleration are internal state of the term; they are not part of the policy observation.
    """

    def __init__(self, cfg: JerkLimitedRelativeJointPositionActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        vmax = torch.as_tensor(cfg.max_joint_velocity, dtype=torch.float32, device=self.device)
        expected_rate = vmax * float(cfg.control_dt)
        actual_rate = torch.as_tensor(cfg.max_relative_action_rate, dtype=torch.float32, device=self.device)
        if not torch.allclose(expected_rate, actual_rate, rtol=2e-4, atol=1e-7):
            raise ValueError(
                "max_joint_velocity * control_dt must match max_relative_action_rate "
                f"(expected={expected_rate.tolist()}, got={actual_rate.tolist()})"
            )
        self._command = JerkLimitedJointCommand(
            self.num_envs,
            float(cfg.control_dt),
            vmax,
            torch.as_tensor(cfg.max_joint_acceleration, dtype=torch.float32, device=self.device),
            torch.as_tensor(cfg.max_joint_jerk, dtype=torch.float32, device=self.device),
            device=self.device,
        )

    @property
    def command_velocity(self) -> torch.Tensor:
        return self._command.velocity

    @property
    def command_acceleration(self) -> torch.Tensor:
        return self._command.acceleration

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        super().reset(env_ids)
        self._command.reset(env_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._command.step(actions)


@configclass
class JerkLimitedRelativeJointPositionActionCfg(RateLimitedRelativeJointPositionActionCfg):
    """Configuration for :class:`JerkLimitedRelativeJointPositionAction`."""

    class_type = JerkLimitedRelativeJointPositionAction
    control_dt: float = 0.05
    max_joint_velocity: list[float] = []
    max_joint_acceleration: list[float] = []
    max_joint_jerk: list[float] = []


class BinaryJointPositionThreshZeroAction(BinaryJointPositionAction):
    """Binary gripper action with threshold 0 (`actions > 0` -> close)."""

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        binary_mask = actions > 0.0  # True: close
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class BinaryJointPositionThreshZeroActionCfg(BinaryJointPositionActionCfg):
    """config for :class:`BinaryJointPositionThreshZeroAction`."""

    class_type = BinaryJointPositionThreshZeroAction


@configclass
class NormalizedJointPositionActionCfgPanda:
    """Absolute `[-1,1]^8` action group (panda_hand): 7 normalized arm joints + binary gripper."""

    body = NormalizedJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
        # No offset: scale/offset come from the joint limits, whose center already includes
        # the joint7 +pi/4 hand-mount offset.
    )

    finger_joint = BinaryJointPositionThreshZeroActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


@configclass
class JerkLimitedRelativeJointPositionActionCfgPanda:
    """Jerk-limited relative ``[-1,1]^8`` action group (RL policies); the internal v/a state is not observed.

    ``body`` limits are filled in by ``register_rl_env``.
    """

    body = JerkLimitedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_zero_offset=True,
    )

    finger_joint = BinaryJointPositionThreshZeroActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


########################################################
# Reward — named success term as 0/1 float
########################################################


def _held_success(
    env: ManagerBasedRLEnv,
    object_name: str,
    container_name: str,
    hold_steps: int,
    predicate: str,
    pred_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Whether the task's success predicate has held for ``hold_steps`` consecutive steps.

    This is the single success definition shared by reward, termination and eval. A per-env
    streak counter filters out transient successes (e.g. an object briefly passing through the
    container); it restarts at each episode's first step (``episode_length_buf == 1`` in IsaacLab).

    Reward and termination both call this every step, so the streak advances once per
    ``common_step_counter`` and the raw streak is cached; the ``>= hold_steps`` threshold is
    applied per caller, which makes the result independent of manager execution order.
    """
    stamp = int(env.common_step_counter)
    cache = getattr(env, "_succ_streak_cache", None)
    if cache is not None and cache[0] == stamp:
        # Second call in the same step: reuse the cached streak.
        return cache[1] >= int(hold_steps)
    pred = SUCCESS_PREDICATES[predicate](env, object=object_name, container=container_name, **pred_kwargs)
    streak = getattr(env, "_succ_streak", None)
    if streak is None or int(streak.shape[0]) != int(env.num_envs):
        streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    zeros = torch.zeros_like(streak)
    first = env.episode_length_buf == 1  # IsaacLab: the first step is 1, not 0
    streak = torch.where(first, zeros, streak)
    streak = torch.where(pred, streak + 1, zeros)
    env._succ_streak = streak
    env._succ_streak_cache = (stamp, streak)
    return streak >= int(hold_steps)


def success_reward(
    env: ManagerBasedRLEnv,
    object_name: str,
    container_name: str,
    hold_steps: int,
    predicate: str,
    pred_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Per-step 0/1 success reward (held success, see :func:`_held_success`)."""
    val = _held_success(env, object_name, container_name, hold_steps, predicate, pred_kwargs)
    # IsaacLab's RewardManager multiplies each term by step_dt; divide here so a success is
    # worth net 1.0, matching the demo buffer's success reward.
    return val.float() / env.step_dt


def success_done(
    env: ManagerBasedRLEnv,
    object_name: str,
    container_name: str,
    hold_steps: int,
    predicate: str,
    pred_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Success termination; shares :func:`_held_success` with :func:`success_reward`."""
    return _held_success(env, object_name, container_name, hold_steps, predicate, pred_kwargs)


def success_params(target: str, container: str, hold_steps: int, predicate: str, pred_kwargs: dict[str, Any]) -> dict:
    """Params of :func:`success_reward` / :func:`success_done`."""
    return {
        "object_name": target,
        "container_name": container,
        "hold_steps": int(hold_steps),
        "predicate": predicate,
        # Task-declared predicate kwargs (see predicate_kwargs_from_task).
        "pred_kwargs": dict(pred_kwargs),
    }


def build_rewards_cfg(succ_params: dict[str, Any], safety_penalty: dict | None) -> type:
    """Build the RewardsCfg: success reward 1 on the step the success has held ``hold_steps`` steps,
    plus the safety penalty.

    Returns a class because ``generate_task_env_cfg`` instantiates ``task_class.rewards()``.

    Parameters
    ----------
    succ_params : dict
        :func:`success_params`.
    safety_penalty : dict | None
        ``illegal_contact`` (+ ``obj_press``) blocks of ``robolab.robots.rl_safety.safety_penalty_total``.
    """
    attrs: dict = {"success": RewTerm(func=success_reward, weight=1.0, params=dict(succ_params))}
    if safety_penalty is not None:
        from robolab.robots.rl_safety import safety_penalty_total

        safety_params = dict(safety_penalty)
        safety_params["target_object"] = succ_params["object_name"]
        safety_params["container"] = succ_params["container_name"]
        attrs["safety"] = RewTerm(func=safety_penalty_total, weight=1.0, params=safety_params)
    return configclass(type("RLRewardsCfg", (), attrs))
