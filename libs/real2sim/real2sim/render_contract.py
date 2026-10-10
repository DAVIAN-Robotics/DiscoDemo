# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Render settings shared by every consumer of the FR3 environment (data render and evaluation).

:func:`render_profile` compiles the measured ``workcell.json`` into the RTX rendering mode and the measured
camera pipelines; :func:`apply_render_runtime` installs the measured appearance and the profile on a live
environment; :func:`render_cameras` renders the current scene; :func:`read_camera` / :func:`read_camera_tiles`
read camera frames through the pipelines. :func:`neutral_dome_cfg` is the background of every rendered env.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from real2sim.camera_pipeline import CameraPipeline, apply_camera_pipeline_batch, build_camera_pipeline
from real2sim.env_contract import rendering_mode

#: Isaac sensor -> workcell camera of the rendered views (exterior and wrist).
RENDER_CAMERAS: Mapping[str, str] = {"left_back_cam": "left_back", "wrist_cam": "wrist"}

# Render-only passes: once at session start, after each large teleport (DLSS converges), and per frame
# (one pass suffices for continuous 20 Hz trajectories).
WARMUP_RENDERS = 48
RESET_SETTLE = 16
FRAME_SETTLE = 1

#: Attribute holding the installed profile on a live env.
ENV_PROFILE_ATTR = "_real2sim_render_profile"
#: Attribute holding the wrist-cable updater on a live env.
ENV_TETHER_ATTR = "_real2sim_wrist_tether_update"


def neutral_dome_cfg() -> Any:
    """Scene background of the data render and the evaluation: an untextured dome light, visible to the cameras.

    :func:`apply_render_runtime` sets its intensity from the workcell.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.utils import configclass

    @configclass
    class NeutralDomeCfg:
        dome_light = AssetBaseCfg(
            prim_path="/World/background",
            spawn=sim_utils.DomeLightCfg(intensity=50.0, visible_in_primary_ray=True),
        )

    return NeutralDomeCfg


@dataclass(frozen=True)
class RenderProfile:
    """RTX rendering mode and the measured pipeline of each rendered camera."""

    rendering_mode: str
    camera_pipelines: Mapping[str, CameraPipeline]


def render_profile(doc: dict[str, Any]) -> RenderProfile:
    """Compile the workcell into the render profile."""
    return RenderProfile(
        rendering_mode=rendering_mode(doc),
        camera_pipelines={sensor: build_camera_pipeline(doc["cameras"][cam]) for sensor, cam in RENDER_CAMERAS.items()},
    )


def apply_render_runtime(env: Any, doc: dict[str, Any]) -> dict[str, Any]:
    """Author the measured appearance (lights, materials, cable) and install the render profile on ``env``.

    Returns
    -------
    dict
        Manifest of the authored appearance.
    """
    from real2sim.appearance import apply_render_appearance, wrist_tether_updater

    manifest = apply_render_appearance(doc)
    # Readers (e.g. the eval worker) recover the profile from the env alone.
    setattr(env, ENV_PROFILE_ATTR, render_profile(doc))
    setattr(env, ENV_TETHER_ATTR, wrist_tether_updater(doc))
    return manifest


def render_cameras(base: Any) -> None:
    """Move the wrist cable to the current robot pose, then render the cameras (``env.step`` does not render them)."""
    getattr(base, ENV_TETHER_ATTR)(base.scene.articulations["robot"], base.scene.env_origins)
    base.sim.render()


def render_profile_of(env: Any) -> RenderProfile:
    """Profile installed by :func:`apply_render_runtime`."""
    return getattr(env, ENV_PROFILE_ATTR)


def read_camera(sensor: Any, name: str, indices: Sequence[int], profile: RenderProfile) -> NDArray[np.uint8]:
    """Read one camera as ``N x H x W x 3`` uint8 RGB through its measured pipeline.

    TiledCamera data is lazy: ``sim.render()`` advances RTX, but the sensor tensor may still hold a
    partially refreshed frame until it is recomputed, so the read forces a refresh first.
    """
    import torch

    sensor.update(0.0, force_recompute=True)
    rgb = sensor.data.output["rgb"][list(indices), :, :, :3].detach()
    assert rgb.dtype == torch.uint8, rgb.dtype
    return apply_camera_pipeline_batch(np.ascontiguousarray(rgb.cpu().numpy()), profile.camera_pipelines[name])


def read_camera_tiles(base: Any, names: Sequence[str], indices: Sequence[int]) -> NDArray[np.uint8]:
    """Read several cameras and tile them left to right: ``N x H x (W * len(names)) x 3`` uint8."""
    profile = render_profile_of(base)
    return np.concatenate([read_camera(base.scene.sensors[n], n, indices, profile) for n in names], axis=2)
