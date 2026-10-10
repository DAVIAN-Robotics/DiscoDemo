# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import robolab.constants
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR

"""
Scene registration:

For the same task, we can register multiple variants. For example, the following script will register something like this:

Task Name                | Environment                       | Config Class                            | Reg | Tags
---------------------------------------------------------------------------------------------------------------------------------------------------
BagelOnPlateTableTask    | BagelOnPlateTableTaskHomeOffice   | BagelOnPlateTableTaskHomeOfficeEnvCfg   | ✓   | all, pick_place
BagelOnPlateTableTask    | BagelOnPlateTableTaskBilliardHall | BagelOnPlateTableTaskBilliardHallEnvCfg | ✓   | all, pick_place
BananaInBowlTableTask    | BananaInBowlTableTaskHomeOffice   | BananaInBowlTableTaskHomeOfficeEnvCfg   | ✓   | all, pick_place
BananaInBowlTableTask    | BananaInBowlTableTaskBilliardHall | BananaInBowlTableTaskBilliardHallEnvCfg | ✓   | all, pick_place

The columns are:
- Task Name: The base task class name (groups variants together)
- Environment: The registered environment name (also the Gymnasium ID)
- Config Class: The generated configuration class name
- Reg: Registration status (✓ = registered with Gymnasium)
- Tags: Tag names this environment belongs to

"""
def auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, lighting_intensity=None, task=None, cameras=None,
                             randomize_background=False, background_seed=None,
                             task_dir=None, decimation=8, extra_robot_cameras=None, background_cfg=None):
    """Discover task files and register one camera-equipped env per task.

    Args:
        task_dirs: Subdirectories of ``task_dir`` to search for tasks.
        lighting_intensity: Optional lighting intensity override.
        task: If provided, only register the specified task(s) instead of discovering
              all tasks. Accepts a single task name/filename/path (str) or a list of them.
              Significantly faster when running a subset of tasks.
        cameras: List of camera config classes observed by the policy. Pass one of
              the presets from ``camera_presets`` (e.g. ``WRIST_LEFT``,
              ``WRIST_LEFT_RIGHT_HEAD``) or your own list. Defaults to ``WRIST_LEFT``.
              The viewport camera is attached separately for video recording.
        randomize_background: If True, sample a random background per task at registration time
              (excluding the default home_office background). Each registered env gets one fixed
              background; assignments are reproducible when ``background_seed`` is provided.
        background_seed: Seed for reproducible per-task background sampling. Ignored if
              ``randomize_background`` is False.
        task_dir: Root directory to search for tasks. Defaults to ``TASK_DIR``; external
              packages pass their own task root.
        decimation: Physics steps per policy step (dt = 1/120 s); also used as render_interval.
        extra_robot_cameras: ``{obs_name: TiledCameraCfg}`` for cameras parented to the robot.
              They are added to the robot cfg (so they spawn after their parent prim) and to
              the ``image_obs`` group.
        background_cfg: Background config class; replaces the default home_office background.
    """
    from robolab.core.environments.factory import auto_discover_and_create_cfgs
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import (
        DroidFr3Cfg,
        DroidJointPositionActionCfgPanda,
        ProprioceptionObservationCfgPanda,
        WristCameraPandaCfg,
        contact_gripper_panda,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    robot_cfg = DroidFr3Cfg

    if cameras is None:
        cameras = WRIST_LEFT

    if extra_robot_cameras:
        # Subclass the robot cfg so the shared module-level cfg is not mutated.
        from isaaclab.utils import configclass

        robot_cfg = configclass(type(f"{robot_cfg.__name__}WithExtraCams", (robot_cfg,),
                                     dict(extra_robot_cameras)))

    ImageObsCfg = generate_image_obs_from_cameras(cameras, extra_cameras=extra_robot_cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])

    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfgPanda(),
        "viewport_cam": ViewportCameraCfg()})

    # The wrist cam is robot-mounted (attached via the robot cfg). Including it as a
    # scene mixin puts wrist_cam before robot in dataclass field order, causing the
    # camera to spawn before its parent prim exists.
    scene_cameras = [c for c in cameras if c is not WristCameraPandaCfg]

    if background_cfg is None and randomize_background:
        from robolab.variations.backgrounds import find_background_files, generate_background_config

        rng = random.Random(background_seed)
        all_bgs = find_background_files()
        # Exclude the default so this is genuinely "besides the default"
        default_bg_path = HomeOfficeBackgroundCfg.dome_light.spawn.texture_file
        all_bgs = [p for p in all_bgs if p != default_bg_path]
        if not all_bgs:
            raise FileNotFoundError(
                "No backgrounds available for randomization after excluding the default."
            )

        def _bg_factory():
            return generate_background_config(rng.choice(all_bgs))

        background_cfg = _bg_factory
    elif background_cfg is None:
        background_cfg = HomeOfficeBackgroundCfg

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR if task_dir is None else task_dir,
        task_subdirs=task_dirs,
        tasks=task,
        pattern="*.py",
        env_prefix="",
        env_postfix="",
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidJointPositionActionCfgPanda(),
        robot_cfg=robot_cfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=background_cfg,
        contact_gripper=contact_gripper_panda,
        dt=1 / (60 * 2),
        render_interval=decimation,
        decimation=decimation,
        seed=1,
        ee_body_name="panda_hand",
    )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
