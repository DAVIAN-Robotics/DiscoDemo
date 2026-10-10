# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""FMB board_0 and the two pegs: asset paths, dimensions, hole offsets, pregrasp geometry.

Isaac-free. Assets are the FMB STEP files converted to USD under ``libs/fmb/assets``
(board_0, shape 2 Round L and shape 6 Square+Circle L, length 100 mm). Hole
coordinates come from ``libs/fmb/assets/clearance.json``; shape numbers follow the
official FMB reference sheet.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from fmb.constants import ASSET_DIR as FMB_ASSET_DIR
from real2sim.workcell import load_workcell

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME_TCP_JSON = os.path.join(PACKAGE_ROOT, "assets", "fr3_home_tcp.json")
PEG_BOARD_JSON = os.path.join(PACKAGE_ROOT, "assets", "peg_board.json")


def load_peg_board(path: str = PEG_BOARD_JSON) -> dict[str, Any]:
    """Load the peg/board knobs (``{value, source}`` schema of ``workcell.json``).

    Parameters
    ----------
    path : str
        ``peg_board.json`` path.

    Returns
    -------
    dict[str, Any]
        Peg/board knob document.
    """
    doc: dict[str, Any] = load_workcell(path)
    return doc


# Centring shift applied by the STEP-to-mesh conversion (xy of board_0, mm).
BOARD0_TRANSLATION_MM = (-115.0, -150.0)

HAND_TO_TCP_M = 0.1034  # panda_hand origin -> fingertip face (Franka spec)
GRIP_DEPTH_M = 0.025  # grasp this far below the top of the peg
# Gap left between each finger and the peg surface at reset. Teleporting the peg into
# an already-closed grasp with any penetration ejects it on the next physics step.
FINGER_RESET_CLEARANCE_M = 0.0002

_CLEARANCE_JSON = os.path.join(FMB_ASSET_DIR, "clearance.json")


def _board0_holes() -> list[dict[str, Any]]:
    with open(_CLEARANCE_JSON) as fh:
        c = json.load(fh)
    return [h for h in c["holes"] if h["board_index"] == 0 and h["kind"] == "peg_hole"]


def hole_offset_from_clearance(long_mm: float, is_circle: bool | None, confident: bool = True) -> tuple[float, float]:
    """Find one board_0 hole and return its offset in the board local frame.

    Parameters
    ----------
    long_mm : float
        Hole size (mm).
    is_circle : bool | None
        True for a round hole, False for a square one, None for either.
    confident : bool
        ``confident`` flag in clearance.json (the shape-6 round hole is a low-confidence fit there).

    Returns
    -------
    tuple[float, float]
        Offset in the board local frame (m).

    Raises
    ------
    RuntimeError
        If not exactly one hole matches.
    """
    hits = [
        h
        for h in _board0_holes()
        if abs(float(h["long_mm"]) - long_mm) < 0.05
        and (is_circle is None or bool(h["is_circle"]) == is_circle)
        and bool(h["confident"]) == confident
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"board_0 has {len(hits)} holes with long={long_mm} circle={is_circle} confident={confident}"
        )
    cx, cy = hits[0]["center_xy_mm"]
    return (
        (cx + BOARD0_TRANSLATION_MM[0]) / 1000.0,
        (cy + BOARD0_TRANSLATION_MM[1]) / 1000.0,
    )


def sqcirc_hole_midpoint_from_clearance() -> tuple[float, float]:
    """Midpoint (m) of the shape-6 square and round holes, where the two-pronged peg centre goes."""
    sq = hole_offset_from_clearance(26.76, False, True)
    ci = hole_offset_from_clearance(26.76, True, False)
    return ((sq[0] + ci[0]) / 2.0, (sq[1] + ci[1]) / 2.0)


@dataclass(frozen=True)
class PegSpec:
    """One peg.

    Parameters
    ----------
    key : str
        ``"round"`` (FMB shape 2 Round, size L) or ``"sqcircle"`` (shape 6 Square+Circle, size L).
    grip_width_m : float
        Width between the fingers (round: diameter, sqcirc: short side of the handle).
    length_m : float
        Local z length.
    hole_offset_xy_m : tuple[float, float]
        Target point of the peg centroid in the board local frame.
    task : str
        Task id (``fmb_round`` / ``fmb_sqcircle``); also the scene file stem.
    """

    key: str
    grip_width_m: float
    length_m: float
    hole_offset_xy_m: tuple[float, float]
    task: str


PEGS: dict[str, PegSpec] = {
    "round": PegSpec(
        key="round",
        grip_width_m=0.04981,
        length_m=0.100,
        hole_offset_xy_m=hole_offset_from_clearance(52.81, True, True),
        task="fmb_round",
    ),
    "sqcircle": PegSpec(
        key="sqcircle",
        grip_width_m=0.02376,
        length_m=0.100,
        hole_offset_xy_m=sqcirc_hole_midpoint_from_clearance(),
        task="fmb_sqcircle",
    ),
}


def load_home_tcp(path: str = HOME_TCP_JSON) -> dict[str, Any]:
    """Load the measured home pose (``home_joint_pos``, ``hand_pos_m``, ``hand_quat_wxyz``)."""
    with open(path) as fh:
        doc: dict[str, Any] = json.load(fh)
    assert doc["home_joint_pos"] and doc["hand_pos_m"] and doc["hand_quat_wxyz"], path
    return doc


def pregrasp_geometry(spec: PegSpec, home_tcp: dict[str, Any]) -> dict[str, Any]:
    """Return ``Task.pregrasp`` (all RoboLab ``PREGRASP_GEOMETRY_KEYS``).

    Parameters
    ----------
    spec : PegSpec
        Selected peg.
    home_tcp : dict[str, Any]
        Result of ``load_home_tcp()``.

    Returns
    -------
    dict[str, Any]
        Pregrasp geometry.
    """
    return {
        "held": "peg",
        "anchor": "board",
        "anchor_offset_xy_m": tuple(spec.hole_offset_xy_m),
        "held_length_m": float(spec.length_m),
        "grip_depth_m": GRIP_DEPTH_M,
        "finger_half_gap_m": float(spec.grip_width_m) / 2.0 + FINGER_RESET_CLEARANCE_M,
        "hand_pos_m": tuple(float(v) for v in home_tcp["hand_pos_m"]),
        "hand_quat_wxyz": tuple(float(v) for v in home_tcp["hand_quat_wxyz"]),
        "home_joint_pos": {str(k): float(v) for k, v in home_tcp["home_joint_pos"].items()},
        "hand_to_tcp_m": HAND_TO_TCP_M,
    }
