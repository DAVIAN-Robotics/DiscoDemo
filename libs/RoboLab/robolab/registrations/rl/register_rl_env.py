# SPDX-License-Identifier: Apache-2.0
"""Registration of the FR3 state-vector RL envs.

Each env has a single "policy" obs group (no images), ``[-1,1]^8`` actions and a 0/1 success
reward. Target/container names are read from the task's success termination params and shared by
obs and reward.
"""

from __future__ import annotations

import copy


def register_rl_env(
    task: str,
    num_envs: int,
    seed: int,
    *,
    curriculum: bool,
    init_pose: str,
    success_hold_steps: int,
    gripper_close_width_m: float,
    gripper_stiffness: float,
    gripper_damping: float,
    control_decimation: int,
    camera_doc: dict | None = None,
    success_terminates: bool = True,
    safety_penalty: dict | None = None,
    safety_signals_only: dict | None = None,
    max_relative_action_rate: list[float] | None = None,
    jerk_limited_action: dict | None = None,
) -> tuple[str, type]:
    """Register one task as a state-vector RL env.

    Parameters
    ----------
    task : str
        Task file path.
    curriculum : bool
        Register the reverse-curriculum variant: adds an ``rc_teleport`` reset event and replaces
        ``time_out`` with the per-env ``rc_dynamic_timeout``. The ``env._rc_*`` attributes are
        attached by ``ReverseCurriculumIsaacEnv`` before the first reset.
    init_pose : str
        ``box:`` or ``pregrasp:`` spec (``robolab.core.init_pose``). Reset events run in definition
        order, so in curriculum mode ``rc_teleport`` overrides the sampled pose.
    success_hold_steps : int
        Control steps the success predicate must hold before the success reward / termination.
    gripper_close_width_m, gripper_stiffness, gripper_damping : float
        Measured gripper controller (environment contract).
    control_decimation : int
        Physics steps (1/120 s) per control step.
    camera_doc : dict | None
        Workcell document: adds the measured exterior (``left_back``) and wrist cameras. ``None``
        strips every camera from the scene (state-only training).
    success_terminates : bool
        If False, success does not end the episode; used for evaluation to run to the time limit.
    safety_penalty : dict | None
        ``illegal_contact`` (+ ``obj_press``) blocks: adds the ``safety`` reward term and the
        per-link contact sensors.
    safety_signals_only : dict | None
        ``illegal_contact`` block: per-link contact sensors without a reward term (collection
        contact log). Mutually exclusive with ``safety_penalty``.
    max_relative_action_rate, jerk_limited_action : list[float] | None, dict | None
        Jerk-limited relative arm action (RL policies); both None select absolute joint-position
        targets (pi0.5 evaluation).

    Returns
    -------
    tuple[str, type]
        (env_name, env_cfg_class)
    """
    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils import configclass

    from robolab.constants import TASK_DIR
    from robolab.core.environments.config import (
        generate_scene_env_cfg,
        generate_task_env_cfg,
        register_generated_env,
    )
    from robolab.core.events.reverse_curriculum import rc_dynamic_timeout, rc_teleport
    from robolab.core.init_pose.event import build_init_pose_event
    from robolab.core.observations.observation_utils import generate_obs_cfg
    from robolab.core.task.task_utils import load_task_from_file, resolve_task_path
    from robolab.robots.droid import DroidFr3Cfg, contact_gripper_panda
    from robolab.robots.rl_obs import (
        SUCCESS_PREDICATES,
        JerkLimitedRelativeJointPositionActionCfgPanda,
        NormalizedJointPositionActionCfgPanda,
        build_rewards_cfg,
        build_rl_obs_group,
        predicate_kwargs_from_task,
        success_done,
        success_params,
    )
    from robolab.variations.lighting import SphereLightCfg

    assert not (safety_penalty is not None and safety_signals_only is not None), (
        "safety_penalty and safety_signals_only are mutually exclusive"
    )
    if (max_relative_action_rate is None) != (jerk_limited_action is None):
        raise ValueError("max_relative_action_rate and jerk_limited_action go together")

    task_path, _ = resolve_task_path(task, TASK_DIR)
    task_class = load_task_from_file(task_path)
    # load_task_from_file returns a cached, shared class. The class-level mutations below persist into
    # the next call in the same process (reverse -> forward), so the reads accept both the original
    # and the mutated success term.
    succ_term = task_class.terminations().success
    succ_params = succ_term.params
    # The task's own keys (object/container) or those of a success_done term left by a previous call
    # (object_name/container_name, predicate).
    target = str(succ_params["object"] if "object" in succ_params else succ_params["object_name"])
    container = str(succ_params["container"] if "container" in succ_params else succ_params["container_name"])
    predicate = str(succ_params.get("predicate", succ_term.func.__name__))
    if predicate not in SUCCESS_PREDICATES:
        raise ValueError(f"unknown success predicate {predicate!r}; supported: {sorted(SUCCESS_PREDICATES)}")
    succ_kw = success_params(
        target, container, success_hold_steps, predicate, predicate_kwargs_from_task(dict(succ_params))
    )
    succ_done_term = DoneTerm(func=success_done, params=dict(succ_kw), time_out=False)

    safety_ic = (
        safety_signals_only if safety_signals_only is not None else (safety_penalty or {}).get("illegal_contact")
    )
    contact_robot_bodies = None if safety_ic is None else list(safety_ic["robot_bodies"])
    contact_scene_bodies = None if safety_ic is None else list(safety_ic["scene_bodies"])

    _event_func, randomize_params = build_init_pose_event(
        init_pose, container=container, pregrasp_geometry=getattr(task_class, "pregrasp", None)
    )

    # Horizon exhaustion is reported as termination, so the critic does not bootstrap from it.
    if curriculum:

        @configclass
        class RLCurriculumTerminations:
            success = succ_done_term
            rc_timeout = DoneTerm(func=rc_dynamic_timeout, time_out=False)

        # Reset events run in definition order: reset -> randomize -> teleport, so the teleport to a
        # demo state overrides the sampled pose.
        @configclass
        class RLCurriculumEvents:
            reset = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
            randomize_init_pose = EventTerm(func=_event_func, mode="reset", params=randomize_params)
            teleport = EventTerm(func=rc_teleport, mode="reset")

        task_class.terminations = RLCurriculumTerminations
        task_class.events = RLCurriculumEvents
    else:
        if success_terminates:

            @configclass
            class RLPlainTerminations:
                success = succ_done_term
                time_out = DoneTerm(func=mdp.time_out, time_out=False)

        else:

            @configclass
            class RLPlainTerminations:
                time_out = DoneTerm(func=mdp.time_out, time_out=True)

        # Set events explicitly too, so no rc_teleport survives from a curriculum call.
        @configclass
        class RLPlainEvents:
            reset = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
            randomize_init_pose = EventTerm(func=_event_func, mode="reset", params=randomize_params)

        task_class.terminations = RLPlainTerminations
        task_class.events = RLPlainEvents

    observations_cfg = generate_obs_cfg({"policy": build_rl_obs_group(target, container)})()
    if jerk_limited_action is not None:
        actions_cfg = JerkLimitedRelativeJointPositionActionCfgPanda()
        actions_cfg.body.control_dt = control_decimation / 120.0
        actions_cfg.body.max_joint_velocity = list(jerk_limited_action["max_joint_velocity"])
        actions_cfg.body.max_joint_acceleration = list(jerk_limited_action["max_joint_acceleration"])
        actions_cfg.body.max_joint_jerk = list(jerk_limited_action["max_joint_jerk"])
        actions_cfg.body.max_relative_action_rate = list(max_relative_action_rate or [])
    else:
        actions_cfg = NormalizedJointPositionActionCfgPanda()
    actions_cfg.finger_joint.close_command_expr = {"panda_finger_joint.*": float(gripper_close_width_m) * 0.5}
    task_class.rewards = build_rewards_cfg(succ_kw, safety_penalty)

    robot = copy.deepcopy(DroidFr3Cfg().robot)
    robot.actuators["panda_hand"].stiffness = float(gripper_stiffness)
    robot.actuators["panda_hand"].damping = float(gripper_damping)
    robot_attrs: dict = {"robot": robot}
    camera_cfg = None
    from real2sim.render_contract import neutral_dome_cfg

    if camera_doc is not None:
        from real2sim.cameras import robot_camera_cfgs, scene_camera_cfgs

        camera_cfg = scene_camera_cfgs(camera_doc, names={"left_back"})
        wrist = robot_camera_cfgs(camera_doc, names={"wrist"})
        if len(camera_cfg) != 1 or set(wrist) != {"wrist_cam"}:
            raise RuntimeError("measured camera selection must produce left_back scene + wrist robot camera")
        robot_attrs.update(wrist)
    robot_cfg = configclass(type("Fr3RobotCfg", (DroidFr3Cfg,), robot_attrs))

    scene_cfg = generate_scene_env_cfg(
        task_class, robot_cfg, camera_cfg=camera_cfg, lighting_cfg=SphereLightCfg, background_cfg=neutral_dome_cfg()
    )
    if camera_doc is None:
        # Strip all camera sensors (including the robot's wrist_cam): override every field whose default
        # is a CameraCfg with None; InteractiveScene skips None entries. Avoids per-env render products,
        # which can crash the RTX renderer at large num_envs.
        import dataclasses

        from isaaclab.sensors import CameraCfg

        cam_fields = []
        for f in dataclasses.fields(scene_cfg):
            if f.default is not dataclasses.MISSING:
                val = f.default
            elif f.default_factory is not dataclasses.MISSING:
                val = f.default_factory()
            else:
                continue
            if isinstance(val, CameraCfg):
                cam_fields.append(f.name)
        assert cam_fields, "no CameraCfg default field found in the scene cfg"
        scene_cfg = configclass(type(scene_cfg.__name__, (scene_cfg,), {name: None for name in cam_fields}))

    env_cfg_cls = generate_task_env_cfg(
        task_class,
        scene_cfg,
        observations_cfg=observations_cfg,
        actions_cfg=actions_cfg,
        contact_gripper=contact_gripper_panda,
        dt=1 / 120,
        render_interval=int(control_decimation),
        decimation=int(control_decimation),
        seed=seed,
        num_envs=num_envs,
        ee_body_name="panda_hand",
        disable_recorders=True,
        contact_robot_bodies=contact_robot_bodies,
        contact_scene_bodies=contact_scene_bodies,
    )

    base = task_class.__name__.removesuffix("Task")
    env_name = f"{base}RLCurriculum" if curriculum else f"{base}RL"
    env_cfg_cls.__name__ = f"{env_name}EnvCfg"
    register_generated_env(env_cfg_cls, env_name)
    return env_name, env_cfg_cls
