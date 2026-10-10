# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Wiring shared by the two peg-in-hole tasks.

No Task subclass is defined here: the RoboLab loader picks the first Task subclass in a
task module's namespace, so each task file defines its own class.
"""

from __future__ import annotations

import os
from typing import Any

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from real2sim.peg_board import PegSpec, load_home_tcp, pregrasp_geometry
from real2sim.predicates.peg_inserted import peg_inserted

CONTACT_OBJECTS = ["peg", "board", "table"]
_SCENES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "scenes")

# FMB success definition: full insertion, 5 mm lateral, 10 deg axis, 2 mm depth slack.
INSERTION_FRACTION = 1.0
LATERAL_TOLERANCE_M = 0.005
AXIS_TOLERANCE_DEG = 10.0
DEPTH_TOLERANCE_M = 0.002
# The board must stay within 1 cm (xy) and 5 deg of its pose at the episode start.
BOARD_MAX_DISP_M = 0.01
BOARD_MAX_ROT_DEG = 5.0
EPISODE_LENGTH_S = 20  # 400 steps at 20 Hz


def scene_usd_for(spec: PegSpec) -> str:
    """Return the scene file of one peg task."""
    return os.path.join(_SCENES, f"{spec.task}.usda")


def make_terminations(spec: PegSpec) -> type:
    """Build the terminations configclass for one peg."""

    @configclass
    class PegInsertTerminations:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        success = DoneTerm(
            func=peg_inserted,
            params={
                "object": "peg",
                "container": "board",
                "hole_offset_m": tuple(spec.hole_offset_xy_m),
                "insertion_fraction": INSERTION_FRACTION,
                "lateral_tolerance_m": LATERAL_TOLERANCE_M,
                "axis_tolerance_deg": AXIS_TOLERANCE_DEG,
                "depth_tolerance_m": DEPTH_TOLERANCE_M,
                "board_max_disp_m": BOARD_MAX_DISP_M,
                "board_max_rot_deg": BOARD_MAX_ROT_DEG,
            },
        )

    return PegInsertTerminations


def make_pregrasp(spec: PegSpec) -> dict[str, Any]:
    """Return ``Task.pregrasp`` (geometry for the RoboLab ``pregrasp:`` init-pose event)."""
    return pregrasp_geometry(spec, load_home_tcp())


def instruction_for(spec: PegSpec) -> dict[str, str]:
    """Return the three language instructions."""
    shape = "cylindrical peg" if spec.key == "round" else "two-pronged peg (a cylinder and a square)"
    return {
        "default": f"Insert the {shape} into its matching hole on the board",
        "vague": "Put the part in place",
        "specific": f"Lower the grasped {shape} straight into the matching hole on the red board and keep it inserted",
    }
