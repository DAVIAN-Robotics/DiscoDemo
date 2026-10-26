"""Gymnasium VectorEnv wrapper and factory for the FR3 RoboLab (Isaac Lab) state-vector RL envs."""

from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import F32NDArray, NDArray

# Extra policy-obs dims of the gripper command queue (requested_close, steps since the request).
GRIPPER_DELAY_OBS_DIM = 2
# Normalization cap (steps) for the time column; above the largest measured delay.
GRIPPER_DELAY_OBS_CAP = 60.0
# Fully open finger position (m, per finger).
OPEN_PER_FINGER_M = 0.04
GRIPPER_LOCK_MODES = ("closed",)

# The Isaac AppLauncher can be created only once per process. The reverse->forward transition
# closes the reverse env and builds the forward env in the same process, reusing this app.
_APP_SINGLETON: Any = None

PREGRASP_INITIAL_CLOSE_MARGIN_M = 0.005
# Consecutive re-resets after which an env with a broken grasp is considered chronic.
PREGRASP_SEAT_RESEAT_LIMIT = 8
# Fail only if more than this fraction of envs is chronic (a few is negligible; many is a regression).
PREGRASP_SEAT_CHRONIC_FRAC = 0.02
PREGRASP_SEAT_TILT_TOL_DEG = 5.0
PREGRASP_SEAT_OFFSET_TOL_M = 0.005


def recursive_to_numpy(
    data: torch.Tensor | dict[str, Any] | list[Any] | tuple[Any, ...] | NDArray,
) -> NDArray | dict[str, Any] | list[Any] | tuple[Any, ...]:
    """Convert torch tensors to numpy, recursing into dicts, lists and tuples."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {k: recursive_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(v) for v in data)
    else:
        return data


def env_kwargs_from_cfg(cfg: Any) -> dict[str, Any]:
    """Env arguments of a training run, shared by training, evaluation, collection and rollouts.

    Parameters
    ----------
    cfg : Any
        Run config (``fr3.yaml`` composed with a task and a method, or a saved ``config.yaml``).

    Returns
    -------
    dict[str, Any]
        Keyword arguments for ``make_robolab_rl_env``.
    """
    from omegaconf import OmegaConf

    jerk = OmegaConf.to_container(cfg.jerk_limited_action, resolve=True)
    assert isinstance(jerk, dict)
    return {
        "task": str(cfg.env.task),
        "env_version": str(cfg.env.env_version),
        "workcell": str(cfg.env.workcell),
        "init_pose": str(cfg.rfcl.init_pose),
        "success_hold_steps": int(cfg.success_hold_steps),
        "jerk_limited_action": {k: [float(x) for x in v] for k, v in jerk.items()},
        "gripper_delay_obs": True,
        "gripper_lock": cfg.env.gripper_lock,
    }


def initial_close_margin_m(pregrasp_start: bool, open_per_finger_m: float, close_per_finger_m: float) -> float:
    """Margin of the "already closed at reset" check (``initial_close_from_finger_pos``).

    * Pre-grasped tasks (``init_pose=pregrasp:...``): fully open minus 5 mm, so a thick held peg
      is not misread as open.
    * Other tasks: midpoint ``(open + close) / 2``. This check runs after ``rc_teleport``, so it sees
      demo finger values; a larger threshold would flip partly open demo frames to closed.

    Parameters
    ----------
    pregrasp_start : bool
        Whether ``init_pose`` is a ``pregrasp:`` spec.
    open_per_finger_m, close_per_finger_m : float
        Fully open / closed value (per finger).

    Returns
    -------
    float
        ``margin_m``.
    """
    if pregrasp_start:
        return PREGRASP_INITIAL_CLOSE_MARGIN_M
    return (float(open_per_finger_m) - float(close_per_finger_m)) / 2.0


def _load_pregrasp_geometry(task: str) -> dict[str, Any] | None:
    """Return the ``Task.pregrasp`` geometry from the task file, or None for non-pregrasp tasks."""
    from robolab.constants import TASK_DIR
    from robolab.core.task.task_utils import load_task_from_file, resolve_task_path

    path, _ = resolve_task_path(task, TASK_DIR)
    geom = getattr(load_task_from_file(str(path)), "pregrasp", None)
    return dict(geom) if isinstance(geom, dict) else None


def _quat_conj(q: "torch.Tensor") -> "torch.Tensor":
    """Conjugate of a wxyz quaternion."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Wxyz quaternion product ``a * b``."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_rotate(q: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
    """Rotate a vector by a wxyz quaternion."""
    w = q[..., :1]
    u = q[..., 1:]
    return v + 2.0 * torch.cross(u, torch.cross(u, v, dim=-1) + w * v, dim=-1)


def pregrasp_seating_error(
    env: Any, geom: dict[str, Any], hand_body: int, env_ids: "torch.Tensor"
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Pose error of the held object w.r.t. its expected in-hand pose (tilt in deg, lateral offset in m).

    The expected pose follows the reset event: held local +z equals hand local +z and the origin lies
    on the tool axis. Only axis tilt and the hand-frame xy offset are checked (z moves slightly under
    normal contact).

    Parameters
    ----------
    env : Any
        Unwrapped env (reads ``scene.articulations['robot']`` and ``scene[held]``).
    geom : dict[str, Any]
        ``Task.pregrasp`` geometry; only ``held`` is used.
    hand_body : int
        ``panda_hand`` body index.
    env_ids : torch.Tensor
        Env indices to check.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (tilt deg, xy offset norm m), each of length ``len(env_ids)``.
    """
    robot = env.scene.articulations["robot"]
    held = env.scene[str(geom["held"])]
    hq = robot.data.body_quat_w[env_ids, hand_body]
    hp = robot.data.body_pos_w[env_ids, hand_body]
    inv = _quat_conj(hq)
    dq = _quat_mul(inv, held.data.root_quat_w[env_ids])
    dp = _quat_rotate(inv, held.data.root_pos_w[env_ids] - hp)
    unit_z = torch.zeros_like(dp)
    unit_z[:, 2] = 1.0
    tilt = torch.rad2deg(torch.acos(torch.clamp(_quat_rotate(dq, unit_z)[:, 2], -1.0, 1.0)))
    return tilt, torch.linalg.norm(dp[:, :2], dim=-1)


def lock_gripper_channel(actions: torch.Tensor, lock: str | None) -> torch.Tensor:
    """Override the policy gripper channel (``action[:, 7]``).

    ``lock="closed"`` always requests close (+1). It is used for the pre-grasped insertion tasks,
    where opening is never needed but a policy can learn to drop the peg. The override is applied
    before the delay actuator, so the gripper dynamics are unchanged. ``None`` returns the input.

    Parameters
    ----------
    actions : torch.Tensor
        (n, 8) actions in [-1, 1].
    lock : str | None
        ``None`` or ``"closed"``.

    Returns
    -------
    torch.Tensor
        Actions with the gripper channel overridden.
    """
    if lock is None:
        return actions
    if lock not in GRIPPER_LOCK_MODES:
        raise ValueError(f"gripper_lock={lock!r} must be one of {list(GRIPPER_LOCK_MODES)} or None")
    out = actions.clone()
    out[:, 7] = 1.0
    return out


def initial_close_from_finger_pos(finger_pos: torch.Tensor, margin_m: float) -> torch.Tensor:
    """Return whether each env starts with the gripper already closed: ``finger < open - margin``.

    Parameters
    ----------
    finger_pos : torch.Tensor
        (n,) mean finger joint position (m, per finger).
    margin_m : float
        Minimum offset from open to count as closed (``initial_close_margin_m``).

    Returns
    -------
    torch.Tensor
        (n,) bool.
    """
    return finger_pos < (OPEN_PER_FINGER_M - float(margin_m))


class RoboLabVectorEnv(VectorEnv[torch.Tensor | F32NDArray, torch.Tensor | F32NDArray, torch.Tensor | F32NDArray]):
    """Gymnasium VectorEnv over one FR3 RoboLab task registered by ``register_rl_env``.

    Exposes the "policy" obs group as a flat float tensor and actions in ``[-1, 1]^8``. The plant
    (gains, gripper, materials, cameras) comes from the task's environment contract.

    Parameters
    ----------
    task : str
        Task file path.
    num_envs : int
        Number of parallel envs.
    seed : int
        Random seed.
    device : str
        Simulation device.
    env_version : str
        Environment contract ``fr3-<task>``.
    workcell : str
        Measured workcell json.
    init_pose : str
        ``box:...`` or ``pregrasp:...`` initial-state distribution. In curriculum mode the reverse
        teleport runs after it and overrides it.
    to_numpy : bool
        Convert outputs to numpy.
    curriculum : bool
        Register the reverse-curriculum variant (``rc_teleport`` reset event).
    cameras : bool
        Add the measured exterior and wrist cameras (evaluation videos and policy images). Without
        them the scene has no camera at all.
    success_terminates : bool
        End episodes on success (training, collection); evaluation runs to the time limit.
    success_hold_steps : int
        Control steps the success predicate must hold.
    safety_penalty : dict | None
        ``illegal_contact`` (+ ``obj_press``) blocks of the safety reward (training).
    safety_signals_only : dict | None
        ``illegal_contact`` block: contact sensors without a reward term (collection contact log).
    jerk_limited_action : dict | None
        Per-joint velocity / acceleration / jerk limits of the relative arm action (RL policies);
        None selects absolute joint-position targets (pi0.5 evaluation).
    gripper_delay_obs : bool
        Insert the gripper command-queue state (requested_close, steps since the request / CAP) right
        before ``prev_action`` in the obs. The sampled delay itself is never exposed.
    gripper_lock : str | None
        ``"closed"`` ignores the policy gripper channel (pre-grasped insertion tasks).
    """

    def __init__(
        self,
        task: str,
        num_envs: int,
        seed: int,
        device: str,
        *,
        env_version: str,
        workcell: str,
        init_pose: str,
        to_numpy: bool = True,
        curriculum: bool = False,
        cameras: bool = False,
        success_terminates: bool = True,
        success_hold_steps: int,
        safety_penalty: dict | None = None,
        safety_signals_only: dict | None = None,
        jerk_limited_action: dict[str, Any] | None = None,
        gripper_delay_obs: bool = False,
        gripper_lock: str | None = None,
    ):
        from real2sim.env_contract import CONTROL_DECIMATION, activate, gripper_contract, rendering_mode

        workcell_doc = activate(env_version, workcell)
        gripper = gripper_contract(workcell_doc)

        from isaaclab.app import AppLauncher

        # AppLauncher is created once per process; later envs (e.g. the forward phase) reuse it.
        # Cameras are fixed by the first build, so a conflicting request fails fast.
        global _APP_SINGLETON
        if _APP_SINGLETON is None:
            launcher_kwargs: dict[str, Any] = {"headless": True, "device": device, "enable_cameras": cameras}
            if cameras:
                launcher_kwargs["rendering_mode"] = rendering_mode(workcell_doc)
            _APP_SINGLETON = AppLauncher(**launcher_kwargs).app
            _APP_SINGLETON._discodemo_cameras = cameras
            _APP_SINGLETON._discodemo_env_version = env_version
        else:
            if _APP_SINGLETON._discodemo_cameras != cameras:
                raise AssertionError(
                    "the Isaac app is a per-process singleton and its camera setting is fixed by the first env: "
                    f"existing cameras={_APP_SINGLETON._discodemo_cameras}, requested {cameras}"
                )
            if _APP_SINGLETON._discodemo_env_version != env_version:
                raise AssertionError(
                    f"one Isaac app cannot host two environment contracts: existing "
                    f"{_APP_SINGLETON._discodemo_env_version!r}, requested {env_version!r}"
                )

        # These import isaaclab, so they must come after the AppLauncher.
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.rl import register_rl_env

        control_dt = CONTROL_DECIMATION / 120.0
        max_relative_action_rate = (
            None
            if jerk_limited_action is None
            else [float(v) * control_dt for v in jerk_limited_action["max_joint_velocity"]]
        )
        env_name, _ = register_rl_env(
            task=task,
            num_envs=num_envs,
            seed=seed,
            curriculum=curriculum,
            init_pose=init_pose,
            camera_doc=workcell_doc if cameras else None,
            success_terminates=success_terminates,
            success_hold_steps=success_hold_steps,
            safety_penalty=safety_penalty,
            safety_signals_only=safety_signals_only,
            max_relative_action_rate=max_relative_action_rate,
            jerk_limited_action=jerk_limited_action,
            gripper_close_width_m=gripper["close_width_m"],
            gripper_stiffness=gripper["stiffness"],
            gripper_damping=gripper["damping"],
            control_decimation=CONTROL_DECIMATION,
        )
        self.seed = seed
        self.device = device
        self.max_relative_action_rate = max_relative_action_rate

        from real2sim.env_contract import apply_runtime

        env, _ = create_env(scene=env_name, device=device, seed=seed, num_envs=num_envs)
        applied = apply_runtime(env, workcell_doc, env_version, render=cameras)
        print(f"[RoboLabVectorEnv] {env_version}: {sorted(applied)} applied", flush=True)
        self.envs = env
        # Manipulated object and container, as named by the success reward term.
        succ = cast(Any, env).reward_manager.get_term_cfg("success").params
        self.target_object = str(succ["object_name"])
        self.container = str(succ["container_name"])

        # Pre-grasp invariant: right after reset the held object must be seated in the hand.
        # Checked only for non-curriculum builds; in reverse curriculum the reset teleports to demo
        # frames, where a slightly tilted held object is normal.
        self._pregrasp_geom = None if curriculum else _load_pregrasp_geometry(task)
        self._pregrasp_hand_body: int | None = None
        self._pregrasp_reseat_count: torch.Tensor | None = None
        self._pregrasp_chronic: torch.Tensor | None = None
        self.env_version = env_version

        # RL always auto-resets. RobolabEnv defaults to an eval "freeze" mode that holds done envs;
        # left on, done envs would stay frozen and fire time_out every step.
        cast(Any, env)._rl_no_freeze = True

        self.num_envs = cast(Any, env).num_envs
        self.max_episode_steps = cast(Any, env).max_episode_length
        self.to_numpy = to_numpy
        # Envs on their first control step after a reset (pre-grasp settle and seating check).
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=device)

        self.gripper_delay_obs = bool(gripper_delay_obs)
        self.gripper_lock: str | None = None if gripper_lock is None else str(gripper_lock)
        if self.gripper_lock is not None and self.gripper_lock not in GRIPPER_LOCK_MODES:
            raise ValueError(f"gripper_lock={gripper_lock!r} must be one of {list(GRIPPER_LOCK_MODES)} or None")

        _base_obs_shape = cast(Any, env).single_observation_space["policy"].shape
        self.obs_size = (
            (int(_base_obs_shape[0]) + GRIPPER_DELAY_OBS_DIM,) if self.gripper_delay_obs else _base_obs_shape
        )
        self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=self.obs_size, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.action_size = cast(Any, env).single_action_space.shape
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.action_size, dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # Measured gripper command delay: requested open/close takes effect after a random number of
        # steps per episode.
        from real2sim.gripper_delay import TorchDelayedBinaryActuator

        self._initial_close_margin_m = initial_close_margin_m(
            str(init_pose).startswith("pregrasp"), OPEN_PER_FINGER_M, float(gripper["close_width_m"]) / 2.0
        )
        self._delay_rng = np.random.default_rng(int(seed) + 9173)
        zeros = np.zeros(self.num_envs, dtype=np.int64)
        self._gripper_delay = TorchDelayedBinaryActuator(zeros, zeros.copy(), device=self.device)
        self._close_delay_range = tuple(gripper["close_delay_range_steps"])
        self._open_delay_range = tuple(gripper["open_delay_range_steps"])
        self._reset_gripper_delay()

    def _reset_gripper_delay(self, env_ids: Any = None) -> None:
        from real2sim.gripper_delay import sample_delay_steps

        if env_ids is None:
            ids = np.arange(self.num_envs, dtype=np.int64)
        elif isinstance(env_ids, torch.Tensor):
            ids = env_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            ids = np.asarray(env_ids, dtype=np.int64)
        close, opened = sample_delay_steps(self._delay_rng, len(ids), self._close_delay_range, self._open_delay_range)
        ids_t = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        robot = self.envs.scene["robot"]
        finger_ids, _ = robot.find_joints("panda_finger_joint.*")
        finger_pos = robot.data.joint_pos[ids_t][:, finger_ids].mean(dim=1)
        initial_close = initial_close_from_finger_pos(finger_pos, self._initial_close_margin_m)
        self._gripper_delay.set_episode_delays(ids, close, opened, initial_close=initial_close)

    def restore_gripper_delay_from_demo(
        self, env_ids: Any, requested_close: Any, elapsed_steps_since_edge: Any
    ) -> None:
        """Restore pending gripper commands after a reverse-curriculum teleport."""
        self._gripper_delay.restore_pending_from_elapsed(env_ids, requested_close, elapsed_steps_since_edge)

    def gripper_delay_obs_columns(self) -> torch.Tensor:
        """Build the ``[num_envs, 2]`` gripper-delay obs columns from the current actuator state.

        Column 0 is requested_close (0/1). Column 1 is the number of steps since the last requested
        flip / CAP, clipped to [0, 1]. With no flip yet it saturates at 1.0 (the episode started
        with this command, so it is already applied). The first observation after a flip is 1 step,
        matching the actuator and demo convention.
        """
        if not self.gripper_delay_obs:
            raise RuntimeError("gripper delay obs columns requested on an env with gripper_delay_obs disabled")
        actuator = self._gripper_delay
        requested = actuator.requested.to(torch.float32)
        raw = actuator.elapsed_steps()
        # -1 (no flip yet) saturates at CAP: already applied.
        raw = torch.where(raw >= 0, raw, torch.full_like(raw, int(GRIPPER_DELAY_OBS_CAP)))
        value = raw.to(torch.float32) / GRIPPER_DELAY_OBS_CAP
        return torch.stack((requested, value.clamp(0.0, 1.0)), dim=1)

    def gripper_delay_obs_slice(self) -> slice:
        """Column slice of the 2 gripper-delay obs dims (right before prev_action)."""
        action_dim = int(self.action_size[-1])
        return slice(-(action_dim + GRIPPER_DELAY_OBS_DIM), -action_dim)

    def _augment_policy_obs(self, obs: torch.Tensor, columns: torch.Tensor | None = None) -> torch.Tensor:
        """Insert the 2 gripper-delay dims right before ``prev_action`` in the policy obs.

        Relies on prev_action being the last term. Returns ``obs`` unchanged when gripper_delay_obs
        is off.

        Parameters
        ----------
        obs : torch.Tensor
            Base policy obs ``[num_envs, D]``.
        columns : torch.Tensor | None
            Precomputed columns; None computes them from the current actuator state. Pass them when
            the obs was taken at a different actuator state (e.g. done envs).

        Returns
        -------
        torch.Tensor
            ``[num_envs, D+2]`` when enabled, otherwise ``obs``.
        """
        if not self.gripper_delay_obs:
            return obs
        cols = self.gripper_delay_obs_columns() if columns is None else columns
        action_dim = int(self.action_size[-1])
        return torch.cat((obs[:, :-action_dim], cols.to(obs.dtype), obs[:, -action_dim:]), dim=1)

    def refresh_gripper_delay_obs(self, obs: torch.Tensor, env_ids: list[int]) -> None:
        """Overwrite the gripper-delay columns of ``env_ids`` with the current actuator state (in place).

        Called by the reverse curriculum after it restores the command queue to the demo frame,
        which happens after the obs was computed.
        """
        if not self.gripper_delay_obs or not env_ids:
            return
        obs[env_ids, self.gripper_delay_obs_slice()] = self.gripper_delay_obs_columns()[env_ids].to(obs.dtype)

    @property
    def unwrapped(self) -> Any:  # type: ignore[override]
        """The base RobolabEnv."""
        return self.envs

    def _reseat_broken_pregrasps(self, checked: "torch.Tensor") -> torch.Tensor | None:
        """Re-reset envs whose held object is not seated after their first control step.

        Pre-grasped tasks only. Later releases by the policy are legitimate, so only the first step
        after a reset is checked.

        Parameters
        ----------
        checked : torch.Tensor
            Env ids on their first step after a reset (and not done this step).

        Returns
        -------
        torch.Tensor | None
            Re-reset env ids, or ``None``.

        Raises
        ------
        RuntimeError
            If more than ``PREGRASP_SEAT_CHRONIC_FRAC`` of the envs failed to seat
            ``PREGRASP_SEAT_RESEAT_LIMIT`` times in a row.
        """
        geom = self._pregrasp_geom
        assert geom is not None
        base = cast(Any, self.envs)
        if self._pregrasp_hand_body is None:
            self._pregrasp_hand_body = int(base.scene.articulations["robot"].data.body_names.index("panda_hand"))
        if self._pregrasp_reseat_count is None or self._pregrasp_chronic is None:
            self._pregrasp_reseat_count = torch.zeros(int(base.num_envs), dtype=torch.long, device=checked.device)
            self._pregrasp_chronic = torch.zeros(int(base.num_envs), dtype=torch.bool, device=checked.device)
        count, chronic = self._pregrasp_reseat_count, self._pregrasp_chronic
        tilt, off = pregrasp_seating_error(base, geom, self._pregrasp_hand_body, checked)
        bad = (tilt > PREGRASP_SEAT_TILT_TOL_DEG) | (off > PREGRASP_SEAT_OFFSET_TOL_M)
        count[checked[~bad]] = 0
        if not bool(bad.any()):
            return None
        idx = checked[bad]
        count[idx] += 1
        over = count[idx] > PREGRASP_SEAT_RESEAT_LIMIT
        if bool(over.any()):
            chronic[idx[over]] = True
            n_chronic = int(chronic.sum())
            cap = max(4, int(PREGRASP_SEAT_CHRONIC_FRAC * int(base.num_envs)))
            if n_chronic > cap:
                raise RuntimeError(
                    "pre-grasp reset failures are widespread "
                    f"(held={geom['held']!r}, chronic {n_chronic}/{base.num_envs} > cap {cap}, "
                    f"tolerance {PREGRASP_SEAT_TILT_TOL_DEG}deg / {PREGRASP_SEAT_OFFSET_TOL_M * 1000}mm). "
                    "This indicates a physics/asset regression."
                )
        # _reset_idx overwrites the pre-reset obs capture of this step's done envs, which step()
        # still needs, so preserve it.
        saved = base._rl_pre_reset_obs
        base._reset_idx(idx)
        base._rl_pre_reset_obs = saved
        self._reset_gripper_delay(idx)
        return idx

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        random_start_init: bool = True,
    ) -> tuple[torch.Tensor | F32NDArray, dict[str, Any]]:
        """Reset all envs and return (policy obs, info).

        Drops the stale pre-reset capture and, with ``random_start_init``, randomizes
        ``episode_length_buf`` to spread out resets.
        """
        obs_dict, infos = self.envs.reset()
        self._reset_gripper_delay()
        self._fresh[:] = True
        # Augment after resetting the actuator so obs shows the new episode's command queue.
        obs = self._augment_policy_obs(obs_dict["policy"])
        # A manual reset's pre-reset capture is not a bootstrap target; drop it.
        cast(Any, self.envs)._rl_pre_reset_obs = None
        # Spread episode horizons (RSL-RL style) to avoid synchronized resets.
        if random_start_init:
            cast(Any, self.envs).episode_length_buf = torch.randint_like(
                cast(Any, self.envs).episode_length_buf, high=int(self.max_episode_steps)
            )
        if self.to_numpy:
            obs = obs.cpu().numpy()
            infos = recursive_to_numpy(infos)  # type: ignore
        infos.update({"actor_observation_size": self.obs_size})
        return obs, infos

    def step(
        self, actions: torch.Tensor | F32NDArray
    ) -> tuple[
        torch.Tensor | F32NDArray,
        torch.Tensor | F32NDArray,
        torch.Tensor | F32NDArray,
        torch.Tensor | F32NDArray,
        dict[str, Any],
    ]:
        """Clamp the action, step, and return (obs, rew, term, trunc, info).

        Carries the safety-penalty signals in info and fills ``final_obs`` of done envs with the
        pre-reset obs s_{T+1} captured before auto-reset (correct timeout bootstrap). Pre-grasped
        envs whose grasp broke on their first step are re-reset and reported as truncated.
        """
        if isinstance(actions, torch.Tensor):
            torch_actions = actions.to(self.device)
        else:
            torch_actions = torch.from_numpy(actions).to(self.device)
        torch_actions = lock_gripper_channel(torch.clamp(torch_actions, -1.0, 1.0), self.gripper_lock)
        fresh = self._fresh.clone()
        # Pre-grasp settle: hold the arm still (zero relative action) on the first step after a reset
        # while the gripper channel keeps closing.
        if self._pregrasp_geom is not None and bool(fresh.any()):
            torch_actions = torch_actions.clone()
            torch_actions[fresh, :7] = 0.0
        applied_close = self._gripper_delay.step(torch_actions[:, 7] > 0.0)
        torch_actions = torch_actions.clone()
        torch_actions[:, 7] = torch.where(applied_close, 1.0, -1.0)
        obs_dict, rew, terminations, truncations, infos = cast(Any, self.envs.step(torch_actions))
        obs = obs_dict["policy"]
        done_mask = terminations.bool() | truncations.bool()
        reseated = None
        if self._pregrasp_geom is not None:
            checked = torch.nonzero(fresh & ~done_mask, as_tuple=False).flatten()
            if checked.numel() > 0:
                reseated = self._reseat_broken_pregrasps(checked)
        # Keep only the safety-penalty signals from the inner extras (for training logs).
        _safety_sig = infos.get("safety_penalty") if isinstance(infos, dict) else None
        infos = {"time_outs": truncations}
        if _safety_sig is not None:
            infos["safety_penalty"] = _safety_sig
        # Capture gripper-delay columns before reset envs resample their delays.
        gripper_cols_pre = self.gripper_delay_obs_columns() if self.gripper_delay_obs else None
        final_obs = obs.clone()
        if bool(done_mask.any()):
            # final_obs = pre-reset obs s_{T+1} of done envs, captured by RobolabEnv._reset_idx before
            # auto-reset, so truncated transitions bootstrap from the correct state.
            pre = cast(Any, self.envs)._rl_pre_reset_obs
            if pre is None or int(pre["env_ids"].numel()) != int(done_mask.sum()):
                raise RuntimeError(
                    "pre-reset obs capture does not match done envs; check RobolabEnv._reset_idx "
                    f"(captured={None if pre is None else int(pre['env_ids'].numel())}, "
                    f"done={int(done_mask.sum())})"
                )
            final_obs[pre["env_ids"]] = pre["policy"].to(final_obs.dtype)
        cast(Any, self.envs)._rl_pre_reset_obs = None  # consumed
        if reseated is not None:
            # The broken-grasp state ends the episode (truncated); the next obs is the new reset.
            truncations = truncations.clone()
            truncations[reseated] = True
            done_mask = done_mask.clone()
            done_mask[reseated] = True
            obs = obs.clone()
            obs[reseated] = cast(Any, self.envs).observation_manager.compute()["policy"][reseated].to(obs.dtype)
            infos["time_outs"] = truncations
        infos["final_obs"] = self._augment_policy_obs(final_obs, gripper_cols_pre)
        self._fresh[:] = False
        if bool(done_mask.any()):
            reset_ids = torch.nonzero(done_mask, as_tuple=False).flatten()
            if reseated is not None:
                # Re-reset envs already resampled their gripper delay.
                keep = ~torch.isin(reset_ids, reseated)
                self._reset_gripper_delay(reset_ids[keep])
            else:
                self._reset_gripper_delay(reset_ids)
            self._fresh[reset_ids] = True
            # Reset envs must show the resampled command queue.
            obs = self._augment_policy_obs(obs, self.gripper_delay_obs_columns() if self.gripper_delay_obs else None)
        else:
            obs = self._augment_policy_obs(obs, gripper_cols_pre)

        if self.to_numpy:
            obs = obs.cpu().numpy()
            rew = rew.cpu().numpy()
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()
            infos = recursive_to_numpy(infos)
        return obs, rew, terminations, truncations, infos

    def observe(self) -> torch.Tensor | F32NDArray:
        """Policy observation of every env now (after a script wrote states outside ``reset``/``step``)."""
        obs = self._augment_policy_obs(cast(Any, self.envs).observation_manager.compute()["policy"])
        return obs.cpu().numpy() if self.to_numpy else obs

    def close(self, **kwargs: Any) -> None:
        """No-op; the env is not closed here to avoid SimulationApp teardown."""
        return

    def render(self) -> None:
        """Not supported."""
        raise NotImplementedError("RoboLab RL envs do not support rendering")


def make_robolab_rl_env(num_envs: int, seed: int, **kwargs: Any) -> RoboLabVectorEnv:
    """Create a ``RoboLabVectorEnv`` on ``cuda:0`` if available, else CPU (see the class for arguments)."""
    return RoboLabVectorEnv(
        num_envs=num_envs, seed=seed, device="cuda:0" if torch.cuda.is_available() else "cpu", **kwargs
    )


def assert_policy_action_space(env: Any, max_relative_action_rate: list[float]) -> None:
    """Fail fast if the live env action term does not match the policy's relative action rate.

    Parameters
    ----------
    env : Any
        Env from ``make_robolab_rl_env`` (wrappers allowed).
    max_relative_action_rate : list[float]
        Per-joint rate of the policy run.

    Raises
    ------
    AssertionError
        If the action term is not relative or its rate disagrees.
    """
    body = env.unwrapped.action_manager.get_term("body")
    name = type(body).__name__
    assert "Relative" in name, f"policy uses relative actions but the action term is {name}"
    built = [float(x) for x in getattr(body.cfg, "max_relative_action_rate", [])]
    expect = [float(x) for x in max_relative_action_rate]
    assert len(built) == len(expect) and all(abs(a - b) <= 1e-6 for a, b in zip(built, expect, strict=True)), (
        f"env rate {built} != policy rate {expect}: env construction arguments are inconsistent."
    )
