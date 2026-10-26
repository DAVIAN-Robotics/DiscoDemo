# SPDX-License-Identifier: Apache-2.0
"""Per-step snapshots of the full scene state (all articulations and rigid objects) during rollouts.

Conventions
-----------
* Take the snapshot right before applying the action, so ``state[t]`` pairs with ``actions[t]`` and
  auto-reset states never leak into the finished episode.
* Positions are relative to the env origin; orientations are world quaternions ``wxyz`` (IsaacLab
  ``root_state_w`` convention).
* Keys are ``state/art/<name>/<field>`` and ``state/obj/<name>/<field>``; joint and body names are
  stored under ``meta/...``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]


def _np(x: Any) -> F32:
    """Convert a torch tensor to a float32 numpy array."""
    out: F32 = x.detach().cpu().numpy().astype(np.float32)
    return out


def snapshot(base: Any) -> dict[str, F32]:
    """Snapshot the scene state of all envs.

    Parameters
    ----------
    base : Any
        ``env.unwrapped`` (IsaacLab ``ManagerBasedRLEnv``).

    Returns
    -------
    dict
        Key -> ``[num_envs, ...]`` array. Articulations give ``joint_pos`` / ``joint_vel`` ``[N, J]`` and
        ``body_pos`` / ``body_quat`` / ``body_linvel`` / ``body_angvel`` ``[N, B, 3|4]``; rigid objects
        give ``pos`` / ``quat`` / ``linvel`` / ``angvel`` ``[N, 3|4]``.
    """
    scene = base.scene
    org = scene.env_origins
    out: dict[str, F32] = {}
    for name, art in scene.articulations.items():
        d = art.data
        bs = d.body_state_w
        out[f"state/art/{name}/joint_pos"] = _np(d.joint_pos)
        out[f"state/art/{name}/joint_vel"] = _np(d.joint_vel)
        out[f"state/art/{name}/body_pos"] = _np(bs[..., :3] - org[:, None, :])
        out[f"state/art/{name}/body_quat"] = _np(bs[..., 3:7])
        out[f"state/art/{name}/body_linvel"] = _np(bs[..., 7:10])
        out[f"state/art/{name}/body_angvel"] = _np(bs[..., 10:13])
    for name, obj in scene.rigid_objects.items():
        rs = obj.data.root_state_w
        out[f"state/obj/{name}/pos"] = _np(rs[:, :3] - org)
        out[f"state/obj/{name}/quat"] = _np(rs[:, 3:7])
        out[f"state/obj/{name}/linvel"] = _np(rs[:, 7:10])
        out[f"state/obj/{name}/angvel"] = _np(rs[:, 10:13])
    return out


def names(base: Any) -> dict[str, npt.NDArray[np.str_]]:
    """Joint and body names that label the columns of `snapshot`.

    Returns
    -------
    dict
        ``meta/art/<name>/joint_names`` and ``meta/art/<name>/body_names`` -> string arrays.
    """
    out: dict[str, npt.NDArray[np.str_]] = {}
    for name, art in base.scene.articulations.items():
        out[f"meta/art/{name}/joint_names"] = np.asarray(list(art.data.joint_names), dtype=np.str_)
        out[f"meta/art/{name}/body_names"] = np.asarray(list(art.data.body_names), dtype=np.str_)
    return out


def env_slice(snaps: list[dict[str, F32]], i: int, t0: int, t1: int) -> dict[str, F32]:
    """Stack env ``i``'s snapshots over steps ``[t0, t1)`` along a time axis.

    Returns
    -------
    dict
        Key -> ``[t1 - t0, ...]`` array.
    """
    assert 0 <= t0 < t1 <= len(snaps), f"range [{t0}, {t1}) is outside the {len(snaps)} snapshots"
    return {k: np.stack([s[k][i] for s in snaps[t0:t1]]) for k in snaps[t0]}
