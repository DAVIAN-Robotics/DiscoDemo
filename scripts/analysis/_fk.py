r"""Joint positions -> ``panda_hand`` pose without simulation (pyroki FK on the ``panda_description`` URDF).

Matches the simulated FR3 ``panda_hand`` pose (robot base at the env origin) up to ~2e-5 m in position, so the
hand pose can be recovered for datasets that store joint positions only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt

Arr = npt.NDArray[np.float64]


@lru_cache(maxsize=1)
def _robot() -> tuple[Any, int]:
    """Pyroki robot and the ``panda_hand`` link index (built once)."""
    import pyroki as pk
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    robot = pk.Robot.from_urdf(load_robot_description("panda_description"))
    return robot, robot.links.names.index("panda_hand")


def ee_pose(qpos: Arr) -> Arr:
    """Joint positions ``[T, >=8]`` (arm 7 + fingers) -> ``[T, 7]`` position + wxyz quaternion (``w >= 0``).

    Parameters
    ----------
    qpos : numpy.ndarray
        Arm 7 + finger joints; only the first finger joint is used (8 actuated joints in pyroki's panda).

    Returns
    -------
    numpy.ndarray
        ``panda_hand`` pose in the env frame.
    """
    robot, li = _robot()
    fk = np.asarray(robot.forward_kinematics(np.asarray(qpos[:, :8], dtype=np.float64)))[:, li]
    quat = fk[:, 0:4] * np.where(fk[:, 0:1] < 0, -1.0, 1.0)
    out: Arr = np.concatenate([fk[:, 4:7], quat], axis=1)
    return out


#: Rows per FK call; a fixed size avoids JAX recompiling for every trajectory length.
CHUNK = 16384


def ee_pose_many(qpos_list: list[Arr]) -> list[Arr]:
    """Batched ``ee_pose`` over many trajectories (fixed-size chunks), split back per trajectory.

    Parameters
    ----------
    qpos_list : list of numpy.ndarray
        Per-trajectory ``[T_i, >=8]``.

    Returns
    -------
    list of numpy.ndarray
        Per-trajectory ``[T_i, 7]``.
    """
    lens = [len(q) for q in qpos_list]
    allq = np.concatenate([np.asarray(q[:, :8], dtype=np.float64) for q in qpos_list])
    n = len(allq)
    pad = (-n) % CHUNK
    allq = np.concatenate([allq, np.repeat(allq[-1:], pad, axis=0)])
    outs = [ee_pose(allq[i : i + CHUNK]) for i in range(0, len(allq), CHUNK)]
    flat = np.concatenate(outs)[:n]
    return list(np.split(flat, np.cumsum(lens)[:-1]))
