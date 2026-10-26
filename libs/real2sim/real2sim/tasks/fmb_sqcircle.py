# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Peg-Insert SqCircle: FMB shape 6 (23.76 mm cylinder + square prong) into board_0.

The two prongs are 39.6 mm apart and the peg only fits one way, so yaw must be aligned.
Starts pre-grasped at the FR3 home pose; only the board pose is randomized.
"""

from dataclasses import dataclass

import real2sim.predicates.peg_inserted  # noqa: F401  # registers the success predicate
from real2sim.peg_board import PEGS
from real2sim.tasks._peg_insert_common import (
    CONTACT_OBJECTS,
    EPISODE_LENGTH_S,
    instruction_for,
    make_pregrasp,
    make_terminations,
    scene_usd_for,
)
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task

_SPEC = PEGS["sqcircle"]


@dataclass
class FmbSqCircleTask(Task):  # type: ignore[misc]  # robolab Task is untyped
    """Insert the grasped two-pronged peg into its matching pair of holes."""

    contact_object_list = CONTACT_OBJECTS
    scene = import_scene(scene_usd_for(_SPEC), CONTACT_OBJECTS)
    terminations = make_terminations(_SPEC)
    pregrasp = make_pregrasp(_SPEC)
    instruction = instruction_for(_SPEC)
    episode_length_s: int = EPISODE_LENGTH_S
    attributes = ["insertion", "contact_rich"]
