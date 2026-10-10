# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Franka FR3 robot config (panda_* joint and link names), observation terms and action terms."""

import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass, noise

from robolab.constants import ROBOTS_DIR

# Stock wrist camera resolution. DiscoDemo replaces this camera with the measured wrist camera.
_CAM_H = 270
_CAM_W = 480

# Official Franka Robotics FR3 geometry, imported once into a portable Isaac USD.
# Link/joint identifiers are renamed to the existing panda_* interface so camera,
# observation, action, and SysID code can switch visuals without a second control stack.
_FR3_PANDA_COMPAT_USD = os.path.join(
    ROBOTS_DIR, "fr3_official", "fr3_panda_compatible.usd"
)

# Offset of the end-effector control frame ("eef_frame") relative to the gripper mount.
EEF_OFFSET_POS: tuple[float, float, float] = (0.0, 0.0, 0.0)
EEF_OFFSET_ROT: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5)

_frame_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TF")
_frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)


def eef_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    """Returns the eef_frame position (x, y, z) in the env-local frame."""
    frames = env.scene[asset_cfg.name]
    idx = frames.data.target_frame_names.index("eef_frame")
    return frames.data.target_pos_w[:, idx, :] - env.scene.env_origins[:, 0:3]


def eef_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    """Returns the eef_frame orientation as quaternion (w, x, y, z) in the world frame."""
    frames = env.scene[asset_cfg.name]
    idx = frames.data.target_frame_names.index("eef_frame")
    return frames.data.target_quat_w[:, idx, :]

########################################################
# Actions
########################################################

class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    """Binary gripper action: values > 0.5 close, otherwise open."""

    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for :class:`BinaryJointPositionZeroToOneAction`."""

    class_type = BinaryJointPositionZeroToOneAction


########################################################
# Panda / FR3 robot
########################################################

_WRIST_CAM_PANDA = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/robot/panda_hand/wrist_cam",
    height=_CAM_H,
    width=_CAM_W,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.8,
        focus_distance=28.0,
        horizontal_aperture=5.376,
        vertical_aperture=3.024,
    ),
    # Calibrated for the aligned home pose (see _PANDA_J7_MOUNT_OFFSET); the gripper is
    # slightly visible at the bottom of the frame.
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.07399, 0.03101, -0.01713),
        rot=(-0.11388, -0.70418, 0.69211, 0.11019),
        convention="opengl",
    ),
)


# The stock panda_hand is mounted with a +45 deg roll about the approach axis. Joint7 is
# commanded/observed relative to this offset, so it binds three coupled sites that must move
# together: the init_state home, the action offset (+) and the obs offset (-).
_PANDA_J7_MOUNT_OFFSET = np.pi / 4

_ROBOT_FR3 = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FR3_PANDA_COMPAT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1 / 5 * np.pi,
            "panda_joint3": 0.0,
            "panda_joint4": -4 / 5 * np.pi,
            "panda_joint5": 0.0,
            "panda_joint6": 3 / 5 * np.pi,
            "panda_joint7": _PANDA_J7_MOUNT_OFFSET,
            "panda_finger_joint.*": 0.04,  # open
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            # Measured gripper speed (100 mm/s jaw width = 0.05 m/s per finger).
            velocity_limit_sim=0.05,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)


@configclass
class DroidFr3Cfg:
    """Scene mixin: official FR3 arm/hand geometry with panda_* names, wrist camera and link frames."""

    robot = _ROBOT_FR3

    wrist_cam = _WRIST_CAM_PANDA

    frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_frame_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/robot/panda_link{i}",
                name=f"panda_link{i}",
            )
            for i in range(8)
        ] + [
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/panda_hand",
                name="gripper_base",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/panda_hand",
                name="eef_frame",
                offset=OffsetCfg(pos=EEF_OFFSET_POS, rot=EEF_OFFSET_ROT),
            ),
        ],
    )


@configclass
class WristCameraPandaCfg:
    """Introspection wrapper exposing the panda-mounted wrist camera (key 'wrist_cam')."""
    wrist_cam = _WRIST_CAM_PANDA


# Isaac Lab's ContactSensor needs exactly one prim per env for filtered contact
# (force_matrix_w), so each finger gets its own sensor and predicates read them as a group
# (core/task/gripper_groups.py). "Detached" means both fingers are clear.
contact_gripper_panda = {
    "gripper": "{ENV_REGEX_NS}/robot/panda_leftfinger",
    "gripper_right": "{ENV_REGEX_NS}/robot/panda_rightfinger",
}


def gripper_pos_panda(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Gripper closure in [0, 1] (0 = open, 1 = closed).

    panda_finger_joint1 is 0.04 when fully open and 0.0 when fully closed.
    """
    robot = env.scene[asset_cfg.name]
    joint_names = ["panda_finger_joint1"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    finger = robot.data.joint_pos[:, joint_indices]
    return (0.04 - finger) / 0.04


def ee_pos_panda(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """End-effector position (x, y, z) in env-local frame — panda_hand link."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("panda_hand")
    return robot.data.body_pos_w[:, body_idx, :] - env.scene.env_origins[:, 0:3]


def ee_quat_panda(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """End-effector orientation quaternion (w, x, y, z) in world frame — panda_hand link."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("panda_hand")
    return robot.data.body_quat_w[:, body_idx, :]


def arm_joint_pos_panda(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Arm joint positions with the +45 deg hand-mount offset removed from panda_joint7.

    Joint7 reads ~0 at home. The joint7 column is resolved by name.
    """
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices].clone()
    j7_col = joint_indices.index(robot.data.joint_names.index("panda_joint7"))
    joint_pos[:, j7_col] = joint_pos[:, j7_col] - _PANDA_J7_MOUNT_OFFSET
    return joint_pos


@configclass
class DroidJointPositionActionCfgPanda:
    """Absolute arm joint-position targets plus a binary gripper command."""

    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
        # +45 deg on joint7 cancels the obs offset (see _PANDA_J7_MOUNT_OFFSET). The
        # regex does not match the finger joints, which are a separate term.
        offset={"panda_joint7": _PANDA_J7_MOUNT_OFFSET},
    )

    # > 0.5 closes; panda fingers open at 0.04, close at 0.0.
    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )


########################################################
# Observations
########################################################

@configclass
class ProprioceptionObservationCfgPanda(ObsGroup):
    arm_joint_pos = ObsTerm(func=arm_joint_pos_panda)
    gripper_pos = ObsTerm(
        func=gripper_pos_panda, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
    )
    ee_pos = ObsTerm(func=ee_pos_panda)
    ee_quat = ObsTerm(func=ee_quat_panda)
    eef_pos = ObsTerm(func=eef_pos)
    eef_quat = ObsTerm(func=eef_quat)

    def __post_init__(self) -> None:
        self.enable_corruption = False # must include
        self.concatenate_terms = False # must include
