# SPDX-License-Identifier: Apache-2.0
"""FR3 environment contracts: the scene and measured plant of each task.

Every consumer (training, evaluation, data collection, rendering) builds its env through
``activate`` and ``apply_runtime``, so all of them drive the same plant. A contract is named
``fr3-<task>``. The top half is Isaac-free; Isaac/PhysX imports stay inside the runtime functions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from real2sim.workcell import WORKCELL_JSON, knob, load_workcell

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SCENES = PACKAGE_ROOT / "assets" / "scenes"
TASKS = ("pnp_banana", "stack_cube", "fmb_round", "fmb_sqcircle")
# 1/120 s physics, 20 Hz control.
CONTROL_DECIMATION = 6


@dataclass(frozen=True)
class TaskProfile:
    """Scene, required prims and material targets of one task."""

    task: str
    required_prims: tuple[str, ...]
    #: Objects whose contact material comes from the workcell (others are baked into the scene).
    material_objects: tuple[str, ...]
    #: Whether the scene must author the board mass from ``peg_board.json``.
    expects_board_mass: bool = False

    @property
    def version(self) -> str:
        """Contract name ``fr3-<task>``."""
        return f"fr3-{self.task}"

    @property
    def scene(self) -> Path:
        """Scene file of the task."""
        return _SCENES / f"{self.task}.usda"

    @property
    def task_file(self) -> Path:
        """RoboLab task definition of the task."""
        return PACKAGE_ROOT / "real2sim" / "tasks" / f"{self.task}.py"


# Top of the 32 mm foam mat that every scene places on the table (m, robot base frame); objects are
# placed on it.
TABLE_TOP_Z_M = 0.032

_PROFILES = (
    TaskProfile("pnp_banana", ("banana", "bowl", "foam_tile_2_2", "wall"), ("banana", "bowl")),
    # Cube material, colour and size (``assets/tasks/stack_cube.json``) are baked into the scene.
    TaskProfile("stack_cube", ("cube_red", "cube_blue", "foam_tile_2_2", "wall"), ()),
    # FMB peg and board physics are baked into the scene.
    TaskProfile("fmb_round", ("peg", "board", "foam_tile_2_2", "wall"), (), expects_board_mass=True),
    TaskProfile("fmb_sqcircle", ("peg", "board", "foam_tile_2_2", "wall"), (), expects_board_mass=True),
)
PROFILES: dict[str, TaskProfile] = {p.version: p for p in _PROFILES}


def profile(version: str) -> TaskProfile:
    """Return the profile of a contract (``fr3-<task>``).

    Raises
    ------
    ValueError
        If the contract is not registered.
    """
    try:
        return PROFILES[version]
    except KeyError:
        raise ValueError(f"unknown environment contract {version!r}; known {sorted(PROFILES)}") from None


def rendering_mode(doc: dict[str, Any]) -> str:
    """Read the RTX rendering mode from the workcell.

    Parameters
    ----------
    doc : dict[str, Any]
        Loaded workcell document.

    Returns
    -------
    str
        One of ``performance`` / ``balanced`` / ``quality``.

    Raises
    ------
    ValueError
        For any other value.
    """
    mode = str(doc["render_appearance"]["rendering_mode"])
    if mode not in {"performance", "balanced", "quality"}:
        raise ValueError(f"unsupported rendering mode {mode!r}")
    return mode


def gripper_contract(doc: dict[str, Any]) -> dict[str, Any]:
    """Read the gripper half of the plant contract from the workcell.

    Parameters
    ----------
    doc : dict[str, Any]
        Loaded workcell document.

    Returns
    -------
    dict[str, Any]
        Close width, gains and the ranges of the command delays (control steps).
    """
    prefix = "robot.gripper_dynamics"
    return {
        "close_width_m": float(knob(doc, f"{prefix}.close_width_m")),
        "stiffness": float(knob(doc, f"{prefix}.stiffness")),
        "damping": float(knob(doc, f"{prefix}.damping")),
        "close_delay_range_steps": list(knob(doc, f"{prefix}.close_delay_range_steps")),
        "open_delay_range_steps": list(knob(doc, f"{prefix}.open_delay_range_steps")),
    }


def object_material_contract(doc: dict[str, Any], objects: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Read and validate the per-object contact material contract.

    Parameters
    ----------
    doc : dict[str, Any]
        Loaded workcell document.
    objects : tuple[str, ...]
        Objects to read. Empty for contracts without a workcell material contract.

    Returns
    -------
    dict[str, dict[str, Any]]
        Object name -> friction / restitution / combine mode.

    Raises
    ------
    ValueError
        If friction is not positive, restitution is outside [0, 1], or the combine
        mode is not ``max``.
    """
    result: dict[str, dict[str, Any]] = {}
    for name in objects:
        prefix = f"{name}.contact_material"
        item = {
            "static_friction": float(knob(doc, f"{prefix}.static_friction")),
            "dynamic_friction": float(knob(doc, f"{prefix}.dynamic_friction")),
            "restitution": float(knob(doc, f"{prefix}.restitution")),
            "friction_combine_mode": str(doc[name]["contact_material"]["friction_combine_mode"]),
        }
        if item["static_friction"] <= 0.0 or item["dynamic_friction"] <= 0.0:
            raise ValueError(f"{name} friction must be positive")
        if not 0.0 <= item["restitution"] <= 1.0:
            raise ValueError(f"{name} restitution must be in [0, 1]")
        if item["friction_combine_mode"] != "max":
            raise ValueError(f"{name} friction combine mode must be max")
        result[name] = item
    return result


def validate_scene_matches_workcell(
    doc: dict[str, Any], scene_path: str | Path, required_prims: tuple[str, ...], expects_board_mass: bool
) -> dict[str, float]:
    """Fail if the scene disagrees with the workcell geometry or the mat height.

    Parameters
    ----------
    doc : dict[str, Any]
        Loaded workcell document.
    scene_path : str | Path
        Scene to validate.
    required_prims : tuple[str, ...]
        Prim names that must appear in the scene text.
    expects_board_mass : bool
        Also check the board mass against ``peg_board.json``.

    Returns
    -------
    dict[str, float]
        Observed and expected curtain x, mat top z and (optionally) board mass.

    Raises
    ------
    ValueError
        If the curtain, mat top, board mass or a required prim disagrees.
    """
    text = Path(scene_path).read_text(encoding="utf-8")
    marker = 'def Mesh "curtain"'
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"scene {scene_path} has no curtain mesh")
    block = text[start : start + 4096]
    match = re.search(
        r"float3\[\] extent = \[\(([-+0-9.eE]+),[^\]]+\), \(([-+0-9.eE]+),",
        block,
    )
    if match is None:
        raise ValueError("cannot read the curtain extent")
    observed_min_x = min(float(match.group(1)), float(match.group(2)))
    expected_x = float(knob(doc, "backdrop.curtain_x_m"))
    if not math.isclose(observed_min_x, expected_x, abs_tol=1e-5, rel_tol=0.0):
        raise ValueError(f"scene/workcell drift: curtain min-x scene={observed_min_x:.9f}, workcell={expected_x:.9f}")
    for name in required_prims:
        if f'"{name}"' not in text:
            raise ValueError(f"scene is missing {name!r}")
    observed_mat_top = scene_foam_mat_top_z(text)
    if not math.isclose(observed_mat_top, TABLE_TOP_Z_M, abs_tol=1e-6, rel_tol=0.0):
        raise ValueError(f"scene foam mat top z {observed_mat_top:.6f} != {TABLE_TOP_Z_M}")
    payload = {
        "curtain_min_x_m": observed_min_x,
        "expected_curtain_x_m": expected_x,
        "foam_mat_top_z_m": observed_mat_top,
    }
    if expects_board_mass:
        from real2sim.peg_board import load_peg_board

        observed_board_mass = scene_board_mass_kg(text)
        expected_board_mass = float(knob(load_peg_board(), "peg_board.board_mass_kg"))
        if not math.isclose(observed_board_mass, expected_board_mass, abs_tol=1e-9, rel_tol=0.0):
            raise ValueError(
                "scene/peg_board drift: board mass "
                f"scene={observed_board_mass:.6f} kg, peg_board.board_mass_kg={expected_board_mass:.6f} kg"
            )
        payload["board_mass_kg"] = observed_board_mass
    return payload


def scene_board_mass_kg(text: str) -> float:
    """Read the ``physics:mass`` authored on ``/world/board`` in a scene.

    Parameters
    ----------
    text : str
        Scene ``.usda`` text.

    Returns
    -------
    float
        Board mass (kg).

    Raises
    ------
    ValueError
        If there is no board prim or no authored mass.
    """
    start = text.find('def Xform "board"')
    if start < 0:
        raise ValueError("scene has no board Xform")
    match = re.search(r"float physics:mass = ([-+0-9.eE]+)", text[start : start + 4096])
    if match is None:
        raise ValueError("scene board has no physics:mass")
    return float(match.group(1))


def _xform_z(block: str, op: str) -> float:
    match = re.search(rf"xformOp:{op} = \([^,]+, [^,]+, ([-+0-9.eE]+)\)", block)
    if match is None:
        raise ValueError(f"cannot read xformOp:{op} z")
    return float(match.group(1))


def scene_foam_mat_top_z(text: str) -> float:
    """Read the top z of the physics mat (``/world/table/foam_mat``) from a scene.

    The mat is a size-2 Cube under the ``table`` Xform, so its top is
    ``table_z + translate_z + scale_z``.

    Parameters
    ----------
    text : str
        usda text.

    Returns
    -------
    float
        Mat top z (m).

    Raises
    ------
    ValueError
        If the table Xform or the foam_mat Cube is missing.
    """
    table_start = text.find('def Xform "table"')
    if table_start < 0:
        raise ValueError("scene has no table Xform")
    foam_start = text.find('def Cube "foam_mat"', table_start)
    if foam_start < 0:
        raise ValueError("scene has no foam_mat Cube")
    table_z = _xform_z(text[table_start:foam_start], "translate")
    foam_block = text[foam_start : foam_start + 2048]
    return table_z + _xform_z(foam_block, "translate") + _xform_z(foam_block, "scale")


def activate(version: str, workcell_path: str | Path = WORKCELL_JSON) -> dict[str, Any]:
    """Load the workcell of a contract and check that its scene agrees with it.

    Parameters
    ----------
    version : str
        Contract name ``fr3-<task>``.
    workcell_path : str | Path
        Measured workcell json.

    Returns
    -------
    dict[str, Any]
        Workcell document.
    """
    prof = profile(version)
    doc = load_workcell(str(Path(workcell_path).resolve(strict=True)))
    validate_scene_matches_workcell(doc, prof.scene, prof.required_prims, prof.expects_board_mass)
    if int(knob(doc, "robot.control_hz")) != 120 // CONTROL_DECIMATION:
        raise ValueError("the FR3 environment requires 1/120 s physics and 20 Hz control")
    return doc


def apply_object_materials(doc: dict[str, Any], objects: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Apply the workcell contact material to the given objects, then verify it.

    Parameters
    ----------
    doc : dict[str, Any]
        Loaded workcell document.
    objects : tuple[str, ...]
        Objects to update. Empty means no-op.

    Returns
    -------
    dict[str, dict[str, Any]]
        Object name -> prim path and material settings.

    Raises
    ------
    RuntimeError
        If the materials found on the stage differ from ``objects``.
    """
    settings = object_material_contract(doc, objects)
    if not settings:
        return {}
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    applied: dict[str, dict[str, Any]] = {}
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        name = next(
            (candidate for candidate in settings if path.endswith(f"/{candidate}/physics_material")),
            None,
        )
        if name is None:
            continue
        cfg = settings[name]
        material = UsdPhysics.MaterialAPI.Apply(prim)
        material.CreateStaticFrictionAttr(cfg["static_friction"])
        material.CreateDynamicFrictionAttr(cfg["dynamic_friction"])
        material.CreateRestitutionAttr(cfg["restitution"])
        physx = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        physx.CreateFrictionCombineModeAttr(cfg["friction_combine_mode"])
        applied[name] = {"prim_path": path, **cfg}
    if set(applied) != set(objects):
        raise RuntimeError(f"expected {set(objects)} physics materials, got {applied}")
    return applied


def apply_gripper_runtime(env: Any, doc: dict[str, Any]) -> dict[str, Any]:
    """Install the measured partial-width gripper controller."""
    import torch

    cfg = gripper_contract(doc)
    robot = env.scene["robot"]
    ids = [index for index, name in enumerate(robot.data.joint_names) if "finger" in name]
    if len(ids) != 2:
        raise RuntimeError(f"expected two finger joints, got {robot.data.joint_names}")
    shape = (robot.num_instances, len(ids))
    robot.write_joint_stiffness_to_sim(torch.full(shape, cfg["stiffness"], device=robot.device), joint_ids=ids)
    robot.write_joint_damping_to_sim(torch.full(shape, cfg["damping"], device=robot.device), joint_ids=ids)
    env.action_manager.get_term("finger_joint")._close_command[:] = cfg["close_width_m"] * 0.5
    return {"close_width_m": cfg["close_width_m"], "stiffness": cfg["stiffness"], "damping": cfg["damping"]}


def apply_runtime(env: Any, doc: dict[str, Any], version: str, *, render: bool = False) -> dict[str, Any]:
    """Apply every post-spawn plant setting and return what was applied.

    Parameters
    ----------
    env : Any
        Built RoboLab/Isaac env.
    doc : dict[str, Any]
        Result of ``activate``.
    version : str
        Contract name; selects the material targets.
    render : bool
        Also apply the render contract (appearance and camera pipelines).

    Returns
    -------
    dict[str, Any]
        Home pose, joint physics, gripper, object materials (and render).
    """
    from real2sim.robot_gains import apply_home_joint_pos, apply_joint_gains

    result: dict[str, Any] = {
        "home": apply_home_joint_pos(env, doc),
        "joint_physics": apply_joint_gains(env, doc),
        "gripper": apply_gripper_runtime(env, doc),
        "object_materials": apply_object_materials(doc, profile(version).material_objects),
    }
    if render:
        from real2sim.render_contract import apply_render_runtime

        result["appearance"] = apply_render_runtime(env, doc)
    return result
