# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim.utils as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robolab.constants
import robolab.core.utils.usd_utils as usd_utils


########################################################
#  Config class for randomizing initial pose
########################################################
@configclass
class RandomizeInitPoseUniform:
    """Configuration for randomizing initial pose uniformly."""

    @classmethod
    def from_params(cls,
        objects: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str,
        pose_range: dict,
        velocity_range: dict = None,
        collision_margin: float = 0.0,
        max_retries: int = 100):
        """Create a RandomizeInitPoseUniform instance with custom parameters.

        Args:
            objects: The asset(s) to randomize.
            pose_range: Dictionary of pose ranges for each axis (x, y, z, roll, pitch, yaw).
            velocity_range: Dictionary of velocity ranges for each axis.
            collision_margin: Additional margin between objects beyond their bounding boxes.
                Set to > 0 to enable collision-aware sampling. The actual collision distance
                will be: radius1 + radius2 + collision_margin, where radii are computed from
                the objects' bounding boxes.
            max_retries: Maximum number of resampling attempts when collision checking is enabled.
        """
        if velocity_range is None:
            velocity_range = {}

        # Create a new class with the custom parameters
        class CustomRandomizeInitPoseUniform(cls):
            randomize_init_pose = EventTerm(
                func=reset_pose_uniform,
                mode="reset",
                params={
                    "pose_range": pose_range,
                    "velocity_range": velocity_range,
                    "asset_cfg": objects,
                    "collision_margin": collision_margin,
                    "max_retries": max_retries,
                }
            )

        return CustomRandomizeInitPoseUniform()

########################################################
#  Object pose initialization
########################################################

def _parse_asset_cfg(asset_cfg: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str) -> list[str]:
    """Parse asset_cfg into a list of asset names.

    Args:
        asset_cfg: Asset configuration - can be a SceneEntityCfg, string, or list of either.

    Returns:
        List of asset name strings.
    """
    if isinstance(asset_cfg, SceneEntityCfg):
        return [asset_cfg.name]
    elif isinstance(asset_cfg, str):
        return [asset_cfg]
    elif isinstance(asset_cfg, list):
        return [each.name if isinstance(each, SceneEntityCfg) else each for each in asset_cfg if isinstance(each, (SceneEntityCfg, str))]
    else:
        return list(asset_cfg)


def _get_object_radius(asset: RigidObject | Articulation, env_id: int = 0) -> float:
    """Get the bounding radius of an object from its USD prim dimensions.

    Args:
        asset: The asset to get radius for.
        env_id: The environment ID to get the correct prim.

    Returns:
        The bounding radius (half of max XY dimension).
    """
    try:
        prim_path = asset.cfg.prim_path
        prims = sim_utils.find_matching_prims(prim_path)
        env_id_str = f"env_{env_id}"
        for prim in prims:
            if env_id_str in str(prim.GetPath()):
                dims = usd_utils.get_dimensions(prim)
                # Use max of X and Y dimensions for the bounding circle radius
                return float(max(dims[0], dims[1]) / 2)
        # Fallback: use first prim found
        if prims:
            dims = usd_utils.get_dimensions(prims[0])
            return float(max(dims[0], dims[1]) / 2)
    except Exception as e:
        if robolab.constants.VERBOSE:
            print(f"Warning: Could not get object radius: {e}")
    return 0.0


def _check_collision_with_others(
    position: torch.Tensor,
    other_positions: list[tuple[torch.Tensor, float]],
    obj_radius: float,
    collision_margin: float = 0.0,
) -> bool:
    """Check if a position collides with any of the other positions using bounding circles.

    Args:
        position: The sampled position (3,) tensor.
        other_positions: List of (position, radius) tuples for other objects.
        obj_radius: Bounding radius of the object being placed.
        collision_margin: Additional margin to add between objects.

    Returns:
        True if collision detected, False otherwise.
    """
    if not other_positions:
        return False

    for other_pos, other_radius in other_positions:
        # Minimum distance is sum of radii plus margin
        min_dist = obj_radius + other_radius + collision_margin
        # Use XY distance for collision check (objects on the same surface)
        dist_xy = torch.sqrt((position[0] - other_pos[0]) ** 2 + (position[1] - other_pos[1]) ** 2)
        if dist_xy < min_dist:
            return True
    return False


def sample_pose_uniform(
    env: ManagerBasedEnv,
    asset: RigidObject | Articulation,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    other_positions: list[list[tuple[torch.Tensor, float]]] | None = None,
    obj_radius: float = 0.0,
    collision_margin: float = 0.0,
    max_retries: int = 100,
):
    """Sample a random pose uniformly within the given ranges, optionally avoiding collisions.

    Args:
        env: The environment instance.
        asset: The asset to sample pose for.
        env_ids: The environment indices to sample for.
        pose_range: Dictionary of pose ranges for each axis.
        velocity_range: Dictionary of velocity ranges for each axis.
        other_positions: List of placed positions per env_id. Each element is a list of
            (position, radius) tuples for that environment.
        obj_radius: Bounding radius of this object (from bounding box).
        collision_margin: Additional margin between objects.
        max_retries: Maximum number of resampling attempts per environment.

    Returns:
        Tuple of (positions, orientations, velocities) tensors.
    """
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)

    positions_before = root_states[:, 0:3] + env.scene.env_origins[env_ids]

    # Sample positions with collision checking if enabled
    if other_positions is not None and (obj_radius > 0 or collision_margin > 0):
        # Sample with collision avoidance - process each environment separately
        all_positions = []
        for idx, env_id in enumerate(env_ids):
            env_id_int = env_id.item() if isinstance(env_id, torch.Tensor) else env_id
            env_other_positions = other_positions[env_id_int] if env_id_int < len(other_positions) else []

            # Try sampling until collision-free or max retries
            for attempt in range(max_retries):
                rand_sample = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (1, 6), device=asset.device)
                position = positions_before[idx] + rand_sample[0, 0:3]

                if not _check_collision_with_others(position, env_other_positions, obj_radius, collision_margin):
                    break
                if attempt == max_retries - 1 and robolab.constants.VERBOSE:
                    print(f"Warning: Max retries ({max_retries}) reached for collision-free sampling")

            all_positions.append(position)

        positions = torch.stack(all_positions, dim=0)
        # Sample orientations separately
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    else:
        # Original behavior - sample all at once
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
        positions = positions_before + rand_samples[:, 0:3]

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    if robolab.constants.VERBOSE:
        print(f"positions: {positions} positions_before: {positions_before}")

    return positions, orientations, velocities

def reset_pose_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str,
    reset_to_default_otherwise: bool = True,
    use_collision_check: bool = True,
    collision_margin: float = 0.01,
    max_retries: int = 100,
):
    """Reset asset root state to a random position and velocity uniformly within the given ranges.
    Adapted from isaaclab.envs.mdp.events.reset_root_state_uniform

    This function randomizes the root position and velocity of the assets in the asset_cfg list.
    If reset_to_default_otherwise is True, assets NOT in the asset_cfg list are reset to their default pose as specified in the scene configuration.
    Otherwise, they are left unchanged at reset time.
    If use_collision_check is True, the function will check for collisions with other objects during sampling. This prevents objects from being placed inside each other.
    The actual collision distance will be: radius1 + radius2 + collision_margin, where radii are computed from
    the objects' bounding boxes.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        pose_range: Dictionary of pose ranges for each axis.
        velocity_range: Dictionary of velocity ranges for each axis.
        asset_cfg: The asset(s) to randomize.
        reset_to_default_otherwise: If True, reset all other assets to default.
        use_collision_check: If True, check for collisions with other objects.
        collision_margin: Additional margin between objects beyond their bounding boxes.
            The actual collision distance will be: radius1 + radius2 + collision_margin, where radii are computed from
            the objects' bounding boxes.
        max_retries: Maximum number of resampling attempts when collision checking is enabled.
    """
    asset_names = _parse_asset_cfg(asset_cfg)
    sampled_pose_assets = set(asset_names)

    # If reset_to_default_otherwise is True, reset all other assets to default first
    if reset_to_default_otherwise:
        all_asset_names = _get_all_asset_names(env)
        default_pose_assets = all_asset_names - sampled_pose_assets
        print(f"Resetting '{sampled_pose_assets}' via random uniform pose sampling and all other assets {default_pose_assets} to default")
        if default_pose_assets:
            _reset_assets_to_default(env, env_ids, default_pose_assets)

    # Track placed positions for collision avoidance (per environment)
    # Each entry is a list of (position, radius) tuples
    num_envs = env.num_envs
    placed_positions: list[list[tuple[torch.Tensor, float]]] = [[] for _ in range(num_envs)]

    # Randomize the specified assets
    for object_name in asset_names:
        asset: RigidObject | Articulation = env.scene[object_name]

        if robolab.constants.VERBOSE:
            print(f"Randomizing initial pose for {object_name} according to: {pose_range} and {velocity_range}")

        # Get bounding radius for this object if collision checking is enabled
        obj_radius = 0.0
        if use_collision_check:
            # Use env_id 0 for getting radius (geometry is same across envs)
            obj_radius = _get_object_radius(asset, env_id=0)
            if robolab.constants.VERBOSE:
                print(f"  Object {object_name} bounding radius: {obj_radius:.4f}")

        positions, orientations, velocities = sample_pose_uniform(
            env, asset, env_ids, pose_range, velocity_range,
            other_positions=placed_positions if use_collision_check else None,
            obj_radius=obj_radius,
            collision_margin=collision_margin,
            max_retries=max_retries,
        )

        # Track placed positions for subsequent objects
        if use_collision_check:
            for idx, env_id in enumerate(env_ids):
                env_id_int = env_id.item() if isinstance(env_id, torch.Tensor) else env_id
                placed_positions[env_id_int].append((positions[idx].clone(), obj_radius))

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def _reset_assets_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_names_set: set[str],
):
    """Internal helper to reset specified assets to their default pose.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        asset_names_set: Set of asset names to reset.
    """
    # Reset rigid bodies that are in the asset list
    for name, rigid_object in env.scene.rigid_objects.items():
        if name not in asset_names_set:
            continue
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    # Reset articulations that are in the asset list
    for name, articulation_asset in env.scene.articulations.items():
        if name not in asset_names_set:
            continue
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    # Reset deformable objects that are in the asset list
    for name, deformable_object in env.scene.deformable_objects.items():
        if name not in asset_names_set:
            continue
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def _get_all_asset_names(env: ManagerBasedEnv) -> set[str]:
    """Get all asset names in the scene."""
    all_names = set()
    all_names.update(env.scene.rigid_objects.keys())
    all_names.update(env.scene.articulations.keys())
    all_names.update(env.scene.deformable_objects.keys())
    return all_names


def reset_pose_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str,
):
    """Reset the specified assets' pose to the default state specified in the scene configuration.

    Only resets assets that are in the asset_cfg list.
    Every other asset is left unchanged.
    Adapted from isaaclab.envs.mdp.events.reset_scene_to_default.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        asset_cfg: The asset(s) to reset to default pose. Can be a single asset or list of assets.
    """
    asset_names_set = set(_parse_asset_cfg(asset_cfg))
    _reset_assets_to_default(env, env_ids, asset_names_set)


########################################################
#  Logging functions
########################################################

def log_init_poses(init_object_poses, output_dir, title=""):
    """Randomize the pose of a single object."""
    import os

    from robolab.core.utils.plot_utils import plot_objects

    title = title.lower().replace(' ', '_')
    plot_objects(init_object_poses, title=title, image_path=os.path.join(output_dir, f"{title}.png"))

    import json
    with open(os.path.join(output_dir, f"{title}.json"), "w") as f:
        json.dump(init_object_poses, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# ---------------------------------------------------------------------------------------------
# DiscoDemo initial-state distributions (init_pose='box:...' / 'pregrasp:...').
# ---------------------------------------------------------------------------------------------
import warnings  # noqa: E402

from robolab.core.init_pose.box import sample_box_layout  # noqa: E402
from robolab.core.init_pose.container_policy import ALL_MOVABLE, movable_asset_names  # noqa: E402
from robolab.core.init_pose.pregrasp import sample_pregrasp_layout  # noqa: E402


def reset_pose_box(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    box_xy: tuple[tuple[float, float], tuple[float, float]],
    yaw_range: tuple[float, float],
    asset_cfg: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str,
    min_center_dist: float,
    yaw_flip: bool = False,
    no_yaw_assets: tuple[str, ...] | list[str] = (),
    reset_to_default_otherwise: bool = True,
    max_retries: int = 100,
    table_name: str = "table",
):
    """``init_pose='box:...'``: place all randomized assets uniformly in one absolute XY box.

    Unlike ``reset_pose_uniform``, positions are drawn in a box relative to the env origin (not
    around each object's nominal pose), and instead of collision checks only a minimum pairwise
    center distance is enforced by rejection (``init_pose.box.sample_box_layout``). z and the
    remaining orientation come from the default root state.

    Args:
        box_xy: ``((x_lo, x_hi), (y_lo, y_hi))`` in meters, relative to the env origin.
        yaw_range: Uniform yaw range (rad), written as an absolute z-yaw. May be off-center.
        yaw_flip: If True, add 180 deg to the sampled yaw with probability 1/2.
        asset_cfg: Assets to randomize (``ALL_MOVABLE`` sentinel allowed).
        min_center_dist: Minimum pairwise center distance (m).
        no_yaw_assets: Assets whose yaw stays at the default (e.g. containers).
        reset_to_default_otherwise: Reset all other assets to their default state.
        max_retries: Rejection-sampling iteration cap.
        table_name: Support-surface name, used to resolve ``ALL_MOVABLE``.
    """
    asset_names = _parse_asset_cfg(asset_cfg)
    if asset_names == [ALL_MOVABLE]:
        asset_names = movable_asset_names(sorted(env.scene.rigid_objects.keys()), table_name=table_name)
    if not asset_names:
        raise ValueError("reset_pose_box: no assets to randomize")
    if reset_to_default_otherwise:
        others = _get_all_asset_names(env) - set(asset_names)
        if others:
            _reset_assets_to_default(env, env_ids, others)

    n = int(len(env_ids))
    device = env.device
    yaw_mask = torch.tensor([name not in set(no_yaw_assets) for name in asset_names], device=device)
    xy, yaw, leftover = sample_box_layout(
        n,
        len(asset_names),
        box_xy,
        float(min_center_dist),
        yaw_range,
        yaw_mask,
        yaw_flip=bool(yaw_flip),
        device=device,
        max_retries=max_retries,
    )
    if leftover:
        # Python shows the first occurrence only, so frequent resets do not flood the log.
        warnings.warn(
            f"reset_pose_box: {leftover}/{n} envs did not satisfy min_center_dist within {max_retries} retries",
            stacklevel=2,
        )
    origins = env.scene.env_origins[env_ids]
    zeros = torch.zeros(n, device=device)
    for i, name in enumerate(asset_names):
        asset: RigidObject | Articulation = env.scene[name]
        st = asset.data.default_root_state[env_ids].clone()
        pos = st[:, 0:3] + origins
        pos[:, 0:2] = origins[:, 0:2] + xy[:, i]
        if bool(yaw_mask[i]):
            # Absolute z-yaw.
            quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw[:, i])
        else:
            quat = st[:, 3:7]
        asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)


# The pregrasp home check and hand-pose tensors are cached once per robot object to avoid a
# device->host sync on every reset. The cache lives on the robot object (``robot._pregrasp_cache``)
# rather than in a dict keyed by ``id(robot)``, because a rebuilt env can reuse a freed object's id.


def reset_pose_pregrasp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    xy_half_m: float,
    yaw_half_rad: float,
    held: str,
    anchor: str,
    anchor_offset_xy_m: tuple[float, float],
    held_length_m: float,
    grip_depth_m: float,
    finger_half_gap_m: float,
    hand_pos_m: tuple[float, float, float],
    hand_quat_wxyz: tuple[float, float, float, float],
    home_joint_pos: dict[str, float],
    hand_to_tcp_m: float = 0.1034,
    robot_name: str = "robot",
    finger_joint_pattern: str = "panda_finger_joint.*",
    table_name: str = "table",
) -> None:
    """``init_pose='pregrasp:...'``: start with ``held`` grasped and ``anchor`` placed under the TCP.

    One function does everything: (a) arm at default (home), (b) finger preload, (c) ``held``
    placed relative to the hand, (d) ``anchor`` sampled in a box under the TCP. Splitting these
    into separate events would let them use inconsistent random draws / indices.

    The hand pose is a measured constant (``hand_pos_m`` / ``hand_quat_wxyz``) because body poses
    are not refreshed right after ``write_joint_state_to_sim``. If the current
    ``default_joint_pos`` differs from the measurement home (``home_joint_pos``) this raises,
    rather than placing objects at a stale hand pose.

    Held-object convention: held local +z = hand local +z (tool axis, pointing down). The held
    origin (local z=0 face) sits ``grip_depth_m`` above the TCP and local z extends downward.

    Parameters
    ----------
    env : ManagerBasedEnv
        Environment instance.
    env_ids : torch.Tensor
        Env indices to reset.
    xy_half_m : float
        Half-width (m) of the anchor target-point XY box, centered on the TCP XY.
    yaw_half_rad : float
        Half-range (rad) of the anchor absolute z-yaw.
    held : str
        Rigid object that starts in the gripper.
    anchor : str
        Rigid object placed in the box under the TCP.
    anchor_offset_xy_m : tuple[float, float]
        Target point in the anchor's local frame (m); this point is placed within
        TCP XY +/- ``xy_half_m``.
    held_length_m : float
        Length of the held object (m). Unused here (consumed by the oracle geometry).
    grip_depth_m : float
        Depth (m) the fingers reach into the held object; the held origin sits this far above the TCP.
    finger_half_gap_m : float
        Per-finger joint target (m), including preload.
    hand_pos_m : tuple[float, float, float]
        Measured hand origin position, env-relative (m).
    hand_quat_wxyz : tuple[float, float, float, float]
        Measured hand orientation, world wxyz quaternion.
    home_joint_pos : dict[str, float]
        Arm joint name -> value (rad) at measurement time; checked against ``default_joint_pos``.
    hand_to_tcp_m : float
        Distance (m) from the hand origin to the TCP (fingertip face) along the tool axis (hand local +z).
    robot_name : str
        Name of the robot in the scene.
    finger_joint_pattern : str
        Regex selecting the finger joints.
    table_name : str
        Unused; accepted for signature compatibility.
    """
    del held_length_m, table_name  # unused here; kept for signature compatibility

    asset_names = {held, anchor}
    others = _get_all_asset_names(env) - asset_names
    # (a) everything else (including the robot, at default_joint_pos = home) to default.
    _reset_assets_to_default(env, env_ids, others)
    robot: Articulation = env.scene[robot_name]
    n = int(len(env_ids))
    device = env.device

    source_pos = tuple(float(v) for v in hand_pos_m)
    source_quat = tuple(float(v) for v in hand_quat_wxyz)
    cache = getattr(robot, "_pregrasp_cache", None)
    if cache is not None:
        # A different pregrasp config on the same robot object must not silently reuse the cache.
        cached_pos, cached_quat, _cached_unit_z, cached_source_pos, cached_source_quat = cache
        if cached_source_pos != source_pos or cached_source_quat != source_quat:
            raise RuntimeError(
                f"pregrasp: hand pose constants cached on the robot (pos={cached_source_pos}, "
                f"quat={cached_source_quat}) differ from this call (pos={source_pos}, quat={source_quat}); "
                "the same robot object is shared by different pregrasp configs"
            )

    if cache is None:
        # The measured hand pose is only valid at the measurement home; check with a single transfer.
        joint_names = list(robot.data.joint_names)
        home_ids = [joint_names.index(name) for name in home_joint_pos]
        live_values = robot.data.default_joint_pos[env_ids[0], home_ids].cpu().tolist()
        for name, live in zip(home_joint_pos, live_values):
            value = float(home_joint_pos[name])
            if abs(live - value) > 1e-4:
                raise RuntimeError(
                    f"pregrasp: default_joint_pos[{name}]={live:.5f} differs from the home {value:.5f} used to "
                    "measure the hand pose; re-measure the hand pose and update the config"
                )

        # The quaternion is a config constant; check it is unit-norm.
        hand_quat_norm_sq = float(sum(q * q for q in hand_quat_wxyz))
        if abs(hand_quat_norm_sq - 1.0) > 1e-3:
            raise RuntimeError(
                f"pregrasp: hand_quat_wxyz norm^2={hand_quat_norm_sq:.5f} != 1; not a unit quaternion"
            )

        # Source values are cached too so later calls can be checked against them (above).
        hand_pos_const = torch.tensor(hand_pos_m, device=device, dtype=torch.float32)
        hand_quat_const = torch.tensor(hand_quat_wxyz, device=device, dtype=torch.float32)
        unit_z_const = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        robot._pregrasp_cache = (hand_pos_const, hand_quat_const, unit_z_const, source_pos, source_quat)
        cache = robot._pregrasp_cache

    hand_pos_const, hand_quat_const, unit_z_const, _cached_source_pos, _cached_source_quat = cache

    # (b) finger preload.
    finger_ids, _ = robot.find_joints(finger_joint_pattern)
    if not finger_ids:
        raise RuntimeError(f"no finger joints match '{finger_joint_pattern}': {robot.data.joint_names}")
    finger_q = torch.full((n, len(finger_ids)), float(finger_half_gap_m), device=device)
    robot.write_joint_state_to_sim(finger_q, torch.zeros_like(finger_q), joint_ids=finger_ids, env_ids=env_ids)

    # (c) held, relative to the measured hand pose.
    origins = env.scene.env_origins[env_ids]
    hand_pos = hand_pos_const.expand(n, 3) + origins
    hand_quat = hand_quat_const.expand(n, 4)
    down = math_utils.quat_apply(hand_quat, unit_z_const.expand(n, 3))
    held_pos = hand_pos + down * (float(hand_to_tcp_m) - float(grip_depth_m))
    held_obj: RigidObject = env.scene[held]
    held_obj.write_root_pose_to_sim(torch.cat([held_pos, hand_quat], dim=-1), env_ids=env_ids)
    held_obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)

    # (d) anchor: target point within TCP XY +/- xy_half, yaw +/- yaw_half.
    tcp_xy = (hand_pos + down * float(hand_to_tcp_m) - origins)[:, :2]
    root_xy, yaw = sample_pregrasp_layout(
        n, tcp_xy, torch.tensor(anchor_offset_xy_m, device=device), float(xy_half_m),
        (-float(yaw_half_rad), float(yaw_half_rad)), device=device,
    )
    anchor_obj: RigidObject = env.scene[anchor]
    st = anchor_obj.data.default_root_state[env_ids].clone()
    pos = st[:, 0:3] + origins
    pos[:, 0:2] = origins[:, 0:2] + root_xy
    zeros = torch.zeros(n, device=device)
    quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    anchor_obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    anchor_obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)
