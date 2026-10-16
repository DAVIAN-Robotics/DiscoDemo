# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""PnP-Banana: put the banana into the bowl (measured objects of the FR3 cell).

The banana does not lie flat in the bowl (its chord is longer than the inner rim), so
pose and rest are not checked: success is the banana inside the open-top bowl volume,
in contact with the bowl, and released by the gripper.
"""

import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container
from robolab.core.task.task import Task

# The fixed cameras are static colliders in a scene subLayer, not scene-config
# attributes, so they cannot be listed here: they block the arm but report no contact.
CONTACT_OBJECTS = ["banana", "bowl", "table"]
SCENE_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "scenes", "pnp_banana.usda"
)


@configclass
class PnPBananaTerminations:
    """Success: banana inside the bowl, touching it, and not held by the gripper."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "banana",
            "container": "bowl",
            "gripper_name": "gripper",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class PnPBananaTask(Task):  # type: ignore[misc]  # robolab Task is untyped
    """Pick up the banana on the table and place it in the bowl."""

    contact_object_list = CONTACT_OBJECTS
    scene = import_scene(SCENE_USD, CONTACT_OBJECTS)
    terminations = PnPBananaTerminations
    instruction = {
        "default": "Pick up the banana and place it in the bowl",
        "vague": "Put the fruit in the bowl",
        "specific": "Grasp the yellow banana and place it inside the red bowl on the table",
    }
    episode_length_s: int = 30  # 600 steps at 20 Hz
    attributes = ["pick_and_place", "rigid_object"]
