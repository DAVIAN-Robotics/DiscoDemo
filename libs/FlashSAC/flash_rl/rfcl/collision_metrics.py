"""Safety signals of collected trajectories: the export filter and the IC / CD metrics.

Per trajectory (from ``collect_rl_states.py`` contact logs and object states):

* ``contact_peak`` / ``contact_sustained``: the largest ``C_contact`` (summed non-allowed robot-scene
  contact force, saturated at ``FORCE_CAP``) at one step and over ``SUSTAINED_STEPS`` consecutive steps.
* ``jam_norm``: insertion tasks, the largest norm of the deadband-adjusted (axial, lateral) peg-board force.
* ``illegal_contact`` (IC, N*s): the unsaturated non-allowed contact force summed over the trajectory times
  the control step.
* ``container_disp`` (CD, m): the largest xy displacement of the container from its initial position.

A trajectory is exported only if its contact peak, sustained contact and jam force stay below the
thresholds and the container moved less than ``CONTAINER_DISP``. IC is reported for the exported
trajectories; it does not filter. Forces are the simulator's reported contact forces.
"""

from typing import Any

import numpy as np

CONTACT_PEAK = 50.0
CONTACT_SUSTAINED = 20.0
SUSTAINED_STEPS = 3
FORCE_CAP = 100.0
JAM_NORM = 15.0
JAM_DEADBAND_AXIAL = 5.0
JAM_DEADBAND_LATERAL = 15.0
CONTAINER_DISP = 0.01
CONTROL_DT = 0.05


def pair_forces(sensors: Any, robot_bodies: list[str], scene_bodies: list[str]) -> Any:
    """Pairwise (robot link x scene body) contact forces ``[N, R, S]``.

    Sums the body-wise force norms of each ``<link>__safety`` sensor's ``force_matrix_w``
    ``(N, B, M, 3)`` (the source of the training contact penalty); the filter order M matches
    ``scene_bodies``.
    """
    import torch

    cols = []
    for link in robot_bodies:
        fn = torch.linalg.vector_norm(sensors[f"{link}__safety"].data.force_matrix_w, dim=-1).sum(dim=1)  # (N, M)
        assert fn.shape[1] == len(scene_bodies), f"{link}__safety has {fn.shape[1]} filters, expected {scene_bodies}"
        cols.append(fn)
    return torch.stack(cols, dim=1)


def illegal_pair_mask(
    robot_bodies: list[str], scene_bodies: list[str], allowed_pairs: set[tuple[str, str]]
) -> np.ndarray:
    """``[R, S]`` bool, True for (robot link, scene body) pairs outside the allowed grasp contacts."""
    return np.array([[(r, s) not in allowed_pairs for s in scene_bodies] for r in robot_bodies])


def trajectory_signals(
    pair_force: Any,
    illegal_mask: Any,
    container_xy: Any,
    press_axial: Any | None = None,
    press_lateral: Any | None = None,
) -> dict[str, float]:
    """Safety signals of one trajectory (see the module docstring).

    Parameters
    ----------
    pair_force : Any
        ``[T-1, R, S]`` pairwise contact forces per transition.
    illegal_mask : Any
        ``[R, S]`` output of ``illegal_pair_mask``.
    container_xy : Any
        ``[T, 2]`` container xy per state (row 0 = initial position).
    press_axial, press_lateral : Any or None
        ``[T-1]`` peg-board force components (insertion tasks only).
    """
    pf = np.asarray(pair_force, dtype=np.float64)
    total = pf[:, np.asarray(illegal_mask, dtype=bool)].sum(axis=1)  # [T-1] before saturation
    cc = np.minimum(total, FORCE_CAP)
    k = SUSTAINED_STEPS
    sustained = float(np.lib.stride_tricks.sliding_window_view(cc, k).min(axis=1).max()) if len(cc) >= k else 0.0
    jam = 0.0
    if press_axial is not None:
        ax = np.minimum(np.asarray(press_axial, dtype=np.float64), FORCE_CAP) - JAM_DEADBAND_AXIAL
        la = np.minimum(np.asarray(press_lateral, dtype=np.float64), FORCE_CAP) - JAM_DEADBAND_LATERAL
        jam = float(np.sqrt((np.clip(ax, 0, None) ** 2 + np.clip(la, 0, None) ** 2).max()))
    cxy = np.asarray(container_xy, dtype=np.float64)
    return {
        "contact_peak": float(cc.max()),
        "contact_sustained": sustained,
        "jam_norm": jam,
        "illegal_contact": float(total.sum() * CONTROL_DT),
        "container_disp": float(np.linalg.norm(cxy - cxy[0], axis=1).max()),
    }


def export_filter_drops(signals: dict[str, float]) -> bool:
    """Whether a trajectory with these ``trajectory_signals`` is dropped before export."""
    return (
        signals["contact_peak"] > CONTACT_PEAK
        or signals["contact_sustained"] > CONTACT_SUSTAINED
        or signals["jam_norm"] > JAM_NORM
        or signals["container_disp"] >= CONTAINER_DISP
    )
