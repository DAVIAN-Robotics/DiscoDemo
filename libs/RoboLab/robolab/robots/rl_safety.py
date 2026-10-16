# SPDX-License-Identifier: Apache-2.0
"""Safety penalty of the RL generator: ``C_contact`` (+ ``C_jam`` on the insertion tasks).

A single composite RewTerm (``safety_penalty_total``) returns ``-(C_contact + C_jam) / step_dt``, so
the penalty is on the same scale as the success reward of 1.0. It is applied in both curriculum
phases.

- ``C_contact``: ``weight * min(sum |F|, sat)`` over (robot link, scene body) contacts outside the
  allowed finger-object pairs.
- ``C_jam``: ``weight * (min(relu(F_axial - db_axial)^2, sat_a) + min(relu(F_lateral - db_lateral)^2, sat_l))``
  for the held peg pressing on the board.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
from robolab.core.world.world_state import get_world


def illegal_contact_force(
    env: ManagerBasedRLEnv,
    robot_bodies: list[str],
    scene_bodies: list[str],
    allowed_pairs: set[tuple[str, str]],
) -> torch.Tensor:
    """Sum of contact force norms (N) over (robot_link, scene_body) pairs not in ``allowed_pairs``.

    Each link has a ``{link}__safety`` contact sensor filtered on all ``scene_bodies`` (created at
    env registration), so intended grasps (finger, object) can be allowed.
    """
    sensors = get_contact_sensors(env.scene)
    total = torch.zeros(env.num_envs, device=env.device)
    for link in robot_bodies:
        sensor_name = f"{link}__safety"
        force_matrix = sensors[sensor_name].data.force_matrix_w  # (N, B, M, 3)
        assert force_matrix.ndim == 4 and force_matrix.shape[-1] == 3, (
            f"illegal_contact: expected force_matrix_w (N,B,M,3) for {sensor_name}, got {tuple(force_matrix.shape)}"
        )
        force_norm = torch.linalg.vector_norm(force_matrix, dim=-1).sum(dim=1)  # (N, M), summed over link bodies
        # Filter columns follow the order of scene_bodies (set at registration).
        assert force_norm.shape[1] == len(scene_bodies), (
            f"illegal_contact: filter count {force_norm.shape[1]} != scene_bodies {len(scene_bodies)} for {sensor_name}"
        )
        for m, scene_body in enumerate(scene_bodies):
            if (link, scene_body) not in allowed_pairs:
                total = total + force_norm[:, m]
    return total


def press_components(env: ManagerBasedRLEnv, body1: str, surfaces: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the ``body1``-surface contact force into axial (world z) and lateral (xy) magnitudes.

    A correct insertion is guided by lateral contact with the hole wall, while a jammed peg pushes
    axially, so each component gets its own deadband. The board is flat with yaw-only randomization,
    so its normal is world z; negative z (pulling away) is clipped to 0.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (axial, lateral) magnitudes, each (N,), summed over surfaces.
    """
    world = get_world(env)
    axial = torch.zeros(env.num_envs, device=env.device)
    lateral = torch.zeros(env.num_envs, device=env.device)
    for surface in surfaces:
        force = world.get_contact_force(body1, surface)  # (N, 3)
        axial = axial + force[:, 2].clamp(min=0.0)
        lateral = lateral + torch.linalg.vector_norm(force[:, :2], dim=-1)
    return axial, lateral


def _deadband_quadratic(signal: torch.Tensor, deadband: float) -> torch.Tensor:
    """relu(signal - deadband)^2."""
    excess = torch.clamp(signal - deadband, min=0.0)
    return excess * excess


def safety_penalty_total(
    env: ManagerBasedRLEnv,
    target_object: str,
    container: str,
    illegal_contact: dict,
    obj_press: dict | None = None,
) -> torch.Tensor:
    """Return ``-(C_contact + C_jam) / step_dt`` (N,) and log the raw signals in ``env.extras``.

    Parameters
    ----------
    target_object, container : str
        Manipulated object and container.
    illegal_contact : dict
        ``{weight, sat, robot_bodies, scene_bodies, allowed_pairs}``.
    obj_press : dict | None
        ``{weight, sat, deadband_axial, deadband_lateral, surfaces}`` (insertion tasks).
    """
    contact = illegal_contact_force(
        env,
        robot_bodies=list(illegal_contact["robot_bodies"]),
        scene_bodies=list(illegal_contact["scene_bodies"]),
        allowed_pairs={tuple(p) for p in illegal_contact["allowed_pairs"]},
    )
    total = float(illegal_contact["weight"]) * torch.clamp(contact, max=float(illegal_contact["sat"]))
    signals = {"illegal_contact_force": float(contact.mean())}
    if obj_press is not None:
        sat = float(obj_press["sat"])
        db_ax = float(obj_press["deadband_axial"])
        db_lat = float(obj_press["deadband_lateral"])
        axial, lateral = press_components(env, target_object, list(obj_press["surfaces"]))
        weight = float(obj_press["weight"])
        total = total + weight * torch.clamp(_deadband_quadratic(axial, db_ax), max=(sat - db_ax) ** 2)
        total = total + weight * torch.clamp(_deadband_quadratic(lateral, db_lat), max=(sat - db_lat) ** 2)
        signals["press_axial_n"] = float(axial.mean())
        signals["press_lateral_n"] = float(lateral.mean())
    env.extras["safety_penalty"] = signals
    return -(total / env.step_dt)
