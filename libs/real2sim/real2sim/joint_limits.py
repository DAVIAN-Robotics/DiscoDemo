# SPDX-License-Identifier: Apache-2.0
"""Write the real controller's joint limits into the articulation.

The robot USD bounds are not the arm's: the real arm runs a controller that
enforces its own joint limits (on j7, 2.8973 rad, tighter than the FR3 URDF). Without
them, planned trajectories can reach joint angles the real arm cannot, and the rest of
the arm contorts to compensate.

The limits are in the physical arm's frame; the sim's ``panda_joint7`` reads the same
value at home, so no conversion is needed (``apply_joint_limits`` checks that the home
pose lies inside the limits). The real controller's joint-avoidance potential is not
reproduced.
"""

from typing import Any

import torch

from real2sim.robot_gains import resolve_joint
from real2sim.workcell import knob


def apply_joint_limits(env: Any, doc: dict[str, Any], asset_name: str = "robot") -> dict[str, Any]:
    """Write `robot.joint_limits` from the workcell into the articulation.

    Returns a diagnostic dict. Raises if the knob is missing, if a joint cannot be
    resolved, if the home pose would fall outside, or if the readback disagrees.
    """
    try:
        lim = knob(doc, "robot.joint_limits")
    except KeyError as e:
        raise KeyError(
            "workcell has no robot.joint_limits. Without it the USD's own bounds "
            "apply, and they are not the arm's -- see the module docstring."
        ) from e
    if lim is None:
        raise KeyError("workcell robot.joint_limits is null")

    robot = env.scene[asset_name]
    names = list(robot.data.joint_names)
    ids, lows, highs, applied = [], [], [], {}
    for jname, (lo, hi) in lim.items():
        sim_name = resolve_joint(jname, names)
        if hi <= lo:
            raise ValueError(f"{jname}: upper {hi} is not above lower {lo}")
        ids.append(names.index(sim_name))
        lows.append(float(lo))
        highs.append(float(hi))
        applied[sim_name] = [float(lo), float(hi)]

    dev = robot.device
    limits = torch.tensor([[lo, hi] for lo, hi in zip(lows, highs, strict=True)], dtype=torch.float32, device=dev)
    limits = limits.unsqueeze(0).repeat(robot.num_instances, 1, 1)

    # The home pose has to survive the new bounds. If it does not, either the
    # limits or the frame is wrong, and clamping the home pose silently would
    # move the start of every demo.
    home = robot.data.default_joint_pos[:, ids]
    outside = (home < limits[..., 0]) | (home > limits[..., 1])
    if bool(outside.any()):
        bad = [names[ids[j]] for j in outside[0].nonzero().flatten().tolist()]
        raise ValueError(
            f"the home pose falls outside the limits about to be written: {bad}. "
            "Check the j7 frame convention before anything else -- the sim and the "
            "real arm agree at 0.7846 rad, and a mismatch there is exactly pi/4."
        )

    robot.write_joint_position_limit_to_sim(limits, joint_ids=ids)

    back = robot.data.joint_pos_limits[0, ids].detach().cpu().numpy()
    for k, (jid, lo, hi) in enumerate(zip(ids, lows, highs, strict=True)):
        if abs(float(back[k][0]) - lo) > 1e-5 or abs(float(back[k][1]) - hi) > 1e-5:
            raise RuntimeError(
                f"{names[jid]} limits did not stick: wrote [{lo}, {hi}], read back [{back[k][0]}, {back[k][1]}]"
            )

    return {
        "applied": applied,
        "source": doc["robot"]["joint_limits"]["source"],
    }
