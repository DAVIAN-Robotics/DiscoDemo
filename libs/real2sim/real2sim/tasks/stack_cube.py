# SPDX-License-Identifier: Apache-2.0
"""Stack-Cube: stack the red cube on the blue cube (measured cubes of the FR3 cell).

Same cell as PnP-Banana with different objects. Size, mass, colour, success tolerances and
episode length come from ``assets/tasks/stack_cube.json``.
"""

import json
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import real2sim.predicates.cube_stacked  # noqa: F401  # registers the success predicate
from real2sim.predicates.cube_stacked import cube_stacked
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task

_PACKAGE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC_JSON = os.path.join(_PACKAGE, "assets", "tasks", "stack_cube.json")
SCENE_USD = os.path.join(_PACKAGE, "assets", "scenes", "stack_cube.usda")
CONTACT_OBJECTS = ["cube_red", "cube_blue", "table"]

with open(SPEC_JSON, encoding="utf-8") as _handle:
    _SPEC = json.load(_handle)
_SUCCESS = _SPEC["success"]


SUCCESS_PARAMS: dict[str, object] = {
    # RL predicate convention: ``object`` is the top cube, ``container`` the bottom one.
    "object": "cube_red",
    "container": "cube_blue",
    "stack_height_m": float(_SUCCESS["stack_height_m"]),
    "height_tolerance_m": float(_SUCCESS["height_tolerance_m"]),
    "xy_tolerance_m": float(_SUCCESS["xy_tolerance_m"]),
    "settle_speed": float(_SUCCESS["settle_speed_m_s"]),
}


@configclass
class StackCubeTerminations:
    """Success: red cube one edge above the blue one, within its top face, and at rest."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=cube_stacked, params=dict(SUCCESS_PARAMS))


@dataclass
class StackCubeTask(Task):  # type: ignore[misc]  # robolab Task is untyped
    """Pick up the red cube and place it on the blue cube."""

    contact_object_list = CONTACT_OBJECTS
    scene = import_scene(SCENE_USD, CONTACT_OBJECTS)
    terminations = StackCubeTerminations
    instruction = {
        "default": "stack the red cube on the blue cube",
        "vague": "Stack the blocks",
        "specific": "Pick up the red cube and set it down on top of the blue cube",
    }
    episode_length_s: int = int(_SPEC["episode_length_s"]["value"])
    attributes = ["pick_and_place", "stacking", "rigid_object"]
