# SPDX-FileCopyrightText: Copyright (c) 2026 The DiscoDemo Authors
# SPDX-License-Identifier: Apache-2.0

"""Render-only appearance calibration for the measured FR3 workcell.

This module deliberately imports Isaac/Usd only inside the applying function so its
configuration can be unit-tested on machines without a running SimulationApp.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from real2sim.workcell import knob


def _is_robot_visual_path(path: str) -> bool:
    """Return whether a robot prim path belongs to render-only visual geometry."""
    return any("visual" in segment.lower() for segment in path.split("/") if segment)


def _robot_material_role(path: str, source_material_path: str) -> str:
    """Map official sub-materials, trusting subset names over importer bindings."""

    def classify(token: str) -> str | None:
        token = token.lower()
        if "color_0_0_0" in token:
            return "seam_black"
        if "color_192_192_192" in token:
            return "polished_metal"
        if "color_64_64_64" in token:
            # The vendor reuses this material for wide decorative side trims and
            # the final wrist rim. Real FR3 has white link shells; retain gray only
            # on the link7/flange assembly next to the end effector.
            if any(name in token for name in ("panda_link7", "_08_flange")):
                return "joint_metal"
            return "white"
        if any(name in token for name in ("part__feature_009", "part__feature001_008", "part__feature_007")):
            return "eef_gray"
        if any(
            name in token
            for name in (
                "color_8_159_255",
                "color_255_0_0",
                "color_128_255_128",
                "color_255_66_6",
                "color_0_174_255",
            )
        ):
            return "source"
        if any(
            name in token
            for name in (
                "color_202_209_238",
                "color_255_255_255",
                "color_210_210_239",
                "part__feature002",
                "part__feature005",
                "part__feature001_006",
            )
        ):
            return "white"
        return None

    return classify(path) or classify(source_material_path) or "white"


def sample_cubic_bezier(points, steps_per_segment: int = 6):
    """Sample one or more connected cubic Bezier segments (4+3n controls)."""
    controls = [tuple(float(v) for v in point) for point in points]
    if steps_per_segment < 1:
        raise ValueError("steps_per_segment must be positive")
    if len(controls) < 4 or (len(controls) - 4) % 3 != 0:
        raise ValueError("Bezier controls must contain 4+3n points")
    sampled = []
    for start in range(0, len(controls) - 1, 3):
        p0, p1, p2, p3 = controls[start : start + 4]
        for index in range(steps_per_segment + 1):
            if start and index == 0:
                continue
            t = index / steps_per_segment
            omt = 1.0 - t
            sampled.append(
                tuple(
                    omt**3 * p0[axis] + 3.0 * omt**2 * t * p1[axis] + 3.0 * omt * t**2 * p2[axis] + t**3 * p3[axis]
                    for axis in range(3)
                )
            )
    return tuple(sampled)


def sample_anchored_tether(start, anchor, bow, steps: int = 24):
    """Sample a render-only flexible tether between a moving and fixed endpoint."""
    if steps < 2:
        raise ValueError("tether steps must be at least 2")
    p0 = tuple(float(v) for v in start)
    p1 = tuple(float(v) for v in anchor)
    bend = tuple(float(v) for v in bow)
    if any(len(point) != 3 for point in (p0, p1, bend)):
        raise ValueError("tether start, anchor and bow must be vec3")
    return tuple(
        tuple((1.0 - t) * p0[axis] + t * p1[axis] + 4.0 * t * (1.0 - t) * bend[axis] for axis in range(3))
        for t in (index / steps for index in range(steps + 1))
    )


def sample_guided_tether(start, guide, anchor, bow, steps: int = 24):
    """Sample a moving-to-fixed tether routed through an off-raster service guide."""
    if steps < 4:
        raise ValueError("guided tether steps must be at least 4")
    visible_steps = max(2, int(round(steps * 0.75)))
    hidden_steps = steps - visible_steps
    first = sample_anchored_tether(start, guide, bow, visible_steps)
    second = sample_anchored_tether(guide, anchor, (0.0, 0.0, 0.0), hidden_steps)
    return first + second[1:]


def sample_departing_guided_tether(start, departure_control, guide, anchor, bow, steps: int = 24):
    """Sample a smooth cable that leaves the hand along a measured outer-surface tangent."""
    if steps < 4:
        raise ValueError("departing guided tether steps must be at least 4")
    visible_steps = max(2, int(round(steps * 0.75)))
    hidden_steps = steps - visible_steps
    control2 = tuple(
        float(guide[axis]) + 0.35 * (float(departure_control[axis]) - float(guide[axis])) + float(bow[axis])
        for axis in range(3)
    )
    first = sample_cubic_bezier(
        (start, departure_control, control2, guide),
        steps_per_segment=visible_steps,
    )
    second = sample_anchored_tether(guide, anchor, (0.0, 0.0, 0.0), hidden_steps)
    return first + second[1:]


def wrist_rig_profile(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the validated render-only wrist cable: a hand-local cable, a bridge to the bridge link and a
    service tether to the base (no collision)."""
    block = doc["render_appearance"]["wrist_rig"]

    def vec3(name: str) -> tuple[float, float, float]:
        values = tuple(float(v) for v in knob(block, name))
        if len(values) != 3:
            raise ValueError(f"wrist_rig.{name} must contain three values")
        return values  # type: ignore[return-value]

    matrix = np.asarray(knob(block, "T_hand_cable_frame"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("wrist_rig.T_hand_cable_frame must be a homogeneous 4x4 transform")
    if not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-6):
        raise ValueError("wrist_rig.T_hand_cable_frame rotation must be orthonormal")
    profile = {
        "camera_prim": str(block["camera_prim"]),
        "T_hand_cable_frame": tuple(tuple(float(value) for value in row) for row in matrix),
        "face_band_center_hand_m": vec3("face_band_center_hand_m"),
        "face_band_size_m": vec3("face_band_size_m"),
        "cable_points_cam_m": tuple(tuple(float(v) for v in point) for point in knob(block, "cable_points_cam_m")),
        "cable_steps_per_span": int(knob(block, "cable_steps_per_span")),
        "cable_width_m": float(knob(block, "cable_width_m")),
        "tether_start_cam_m": vec3("tether_start_cam_m"),
        "bridge_end_link": str(block["bridge_end_link"]),
        "bridge_departure_control_cam_m": vec3("bridge_departure_control_cam_m"),
        "bridge_approach_control_link_m": vec3("bridge_approach_control_link_m"),
        "bridge_attachment_link_m": vec3("bridge_attachment_link_m"),
        "bridge_camera_clearance_m": float(knob(block, "bridge_camera_clearance_m")),
        "service_departure_control_link_m": vec3("service_departure_control_link_m"),
        "bridge_steps": int(knob(block, "bridge_steps")),
        "tether_guide_base_m": vec3("tether_guide_base_m"),
        "tether_stow_guide_base_m": vec3("tether_stow_guide_base_m"),
        "tether_stow_anchor_base_m": vec3("tether_stow_anchor_base_m"),
        "tether_deploy_hand_z_low_m": float(knob(block, "tether_deploy_hand_z_low_m")),
        "tether_deploy_hand_z_high_m": float(knob(block, "tether_deploy_hand_z_high_m")),
        "tether_visible_deploy_min": float(knob(block, "tether_visible_deploy_min")),
        "tether_anchor_base_m": vec3("tether_anchor_base_m"),
        "tether_bow_base_m": vec3("tether_bow_base_m"),
        "tether_steps": int(knob(block, "tether_steps")),
    }
    if any(v <= 0.0 for v in profile["face_band_size_m"]):
        raise ValueError("wrist_rig.face_band_size_m values must be positive")
    points = profile["cable_points_cam_m"]
    if any(len(point) != 3 for point in points) or len(points) < 4 or (len(points) - 4) % 3 != 0:
        raise ValueError("wrist_rig.cable_points_cam_m must contain 4+3n Bezier vec3 points")
    if profile["cable_width_m"] <= 0.0:
        raise ValueError("wrist rig cable width must be positive")
    if profile["cable_steps_per_span"] < 2 or profile["tether_steps"] < 4 or profile["bridge_steps"] < 2:
        raise ValueError("wrist rig cable / tether / bridge step counts are too small")
    if profile["tether_deploy_hand_z_high_m"] <= profile["tether_deploy_hand_z_low_m"]:
        raise ValueError("wrist rig tether deploy high z must exceed low z")
    if not 0.0 <= profile["tether_visible_deploy_min"] <= 1.0:
        raise ValueError("wrist rig tether visible deploy minimum must be in [0,1]")
    if profile["tether_start_cam_m"] != points[-1]:
        raise ValueError("dynamic bridge start must equal the hand-local cable endpoint")
    return profile


def appearance_profile(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the validated render appearance of the workcell (materials, light levels, wrist cable)."""
    block = doc["render_appearance"]
    role_specs: dict[str, dict[str, Any]] = {}
    for role in ("joint_metal", "polished_metal", "eef_gray", "seam_black"):
        raw = knob(block, f"robot_materials.{role}")
        role_specs[role] = {
            "rgb": tuple(float(v) for v in raw["rgb"]),
            "roughness": float(raw["roughness"]),
            "metallic": float(raw["metallic"]),
        }
    emission = knob(block, "banana_emission")
    preview = knob(block, "banana_preview_override")
    profile = {
        "robot_rgb": tuple(float(v) for v in knob(block, "robot_rgb")),
        "robot_roughness": float(knob(block, "robot_roughness")),
        "robot_materials": role_specs,
        "sphere_intensity": float(knob(block, "sphere_intensity")),
        "sphere_radius_m": float(knob(block, "sphere_radius_m")),
        "sphere_position_base_m": tuple(float(v) for v in knob(block, "sphere_position_base_m")),
        "dome_intensity": float(knob(block, "dome_intensity")),
        "banana_diffuse_tint": tuple(float(v) for v in knob(block, "banana_diffuse_tint")),
        "banana_diffuse_texture": str(knob(block, "banana_diffuse_texture")),
        "banana_roughness": float(knob(block, "banana_roughness")),
        "banana_emission": {
            "color": tuple(float(v) for v in emission["color"]),
            "intensity": float(emission["intensity"]),
        },
        "banana_preview_emission_scale": float(preview["emission_scale"]),
        "bowl_materials": knob(block, "bowl_materials"),
        "wrist_rig": wrist_rig_profile(doc),
    }
    assert preview["enabled"], "the banana preview material is part of the render appearance"
    if len(profile["robot_rgb"]) != 3 or any(not 0.0 <= v <= 1.0 for v in profile["robot_rgb"]):
        raise ValueError(f"robot_rgb must contain three values in [0,1], got {profile['robot_rgb']}")
    if not 0.0 <= profile["robot_roughness"] <= 1.0:
        raise ValueError(f"robot_roughness must be in [0,1], got {profile['robot_roughness']}")
    for role, spec in profile["robot_materials"].items():
        if len(spec["rgb"]) != 3 or any(not 0.0 <= v <= 1.0 for v in spec["rgb"]):
            raise ValueError(f"{role}.rgb must contain three values in [0,1]")
        if not 0.0 <= spec["roughness"] <= 1.0 or not 0.0 <= spec["metallic"] <= 1.0:
            raise ValueError(f"{role} roughness/metallic must be in [0,1]")
    if profile["sphere_intensity"] <= 0.0 or profile["dome_intensity"] <= 0.0 or profile["sphere_radius_m"] <= 0.0:
        raise ValueError("light intensities and the sphere radius must be positive")
    if len(profile["sphere_position_base_m"]) != 3:
        raise ValueError("sphere light position must contain three values")
    if len(profile["banana_diffuse_tint"]) != 3 or any(v <= 0.0 for v in profile["banana_diffuse_tint"]):
        raise ValueError("banana diffuse tint must contain three positive values")
    if not 0.0 <= profile["banana_roughness"] <= 1.0:
        raise ValueError("banana roughness must be in [0,1]")
    texture_path = Path(profile["banana_diffuse_texture"])
    if texture_path.is_absolute() or ".." in texture_path.parts:
        raise ValueError("banana diffuse texture must be package-relative")
    if len(emission["color"]) != 3 or any(not 0.0 <= v <= 1.0 for v in emission["color"]) or emission["intensity"] < 0:
        raise ValueError("banana emission color must be in [0,1] and its intensity non-negative")
    if not 0.0 <= profile["banana_preview_emission_scale"] <= 1.0:
        raise ValueError("banana preview emission scale must be in [0,1]")
    if set(profile["bowl_materials"]) != {"red", "black"}:
        raise ValueError("bowl_materials must contain red and black")
    for role, spec in profile["bowl_materials"].items():
        if len(spec["rgb"]) != 3 or any(not 0.0 <= float(v) <= 1.0 for v in spec["rgb"]):
            raise ValueError(f"bowl {role} RGB must contain three values in [0,1]")
        for key in ("roughness", "clearcoat", "clearcoat_roughness"):
            if key in spec and not 0.0 <= float(spec[key]) <= 1.0:
                raise ValueError(f"bowl {role} {key} must be in [0,1]")
        if "specular_rgb" in spec and (
            len(spec["specular_rgb"]) != 3 or any(not 0.0 <= float(v) <= 1.0 for v in spec["specular_rgb"])
        ):
            raise ValueError(f"bowl {role} specular_rgb must contain three values in [0,1]")
    return profile


def apply_render_appearance(doc: dict[str, Any]) -> dict[str, Any]:
    """Author the measured render appearance on the stage.

    Matte-white robot materials, calibrated dome and sphere light levels, the render-only wrist cable
    (hand-local cable, bridge and service tether; ``update_dynamic_wrist_tether`` moves the last two) and
    the banana and bowl materials (only in scenes that have them).

    Returns
    -------
    dict[str, Any]
        Short manifest: robot roots, material bindings per role and light levels.
    """
    profile = appearance_profile(doc)

    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()

    def define_material(role: str, spec: dict[str, Any]):
        base = f"/World/Looks/real2sim_fr3_{role}"
        material = UsdShade.Material.Define(stage, base)
        shader = UsdShade.Shader.Define(stage, base + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*spec["rgb"]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(spec["roughness"])
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(spec["metallic"])
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    material_specs = {
        "white": {"rgb": profile["robot_rgb"], "roughness": profile["robot_roughness"], "metallic": 0.0},
        **profile["robot_materials"],
    }
    materials = {role: define_material(role, spec) for role, spec in material_specs.items()}

    robot_roots = [str(prim.GetPath()) for prim in stage.TraverseAll() if str(prim.GetPath()).endswith("/robot")]
    if not robot_roots:
        raise RuntimeError("could not find spawned robot root for visual material override")

    # ---- render-only wrist cable
    rig = profile["wrist_rig"]
    from real2sim.cameras import camera_specs, transform_to_pose

    wrist_spec = camera_specs(doc)["wrist"]
    if rig["camera_prim"] != wrist_spec["prim"]:
        raise ValueError(f"wrist rig camera_prim {rig['camera_prim']!r} != calibrated {wrist_spec['prim']!r}")
    cam_pos, cam_quat = transform_to_pose(np.asarray(rig["T_hand_cable_frame"]))
    cable_samples = sample_cubic_bezier(rig["cable_points_cam_m"], steps_per_segment=rig["cable_steps_per_span"])
    cable_leaf_names = tuple(f"usb_cable_seg_{index:02d}" for index in range(len(cable_samples) - 1))

    def render_only(geom, material, *, cast_shadows: bool = True) -> None:
        prim = geom.GetPrim()
        UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.render)
        if not cast_shadows:
            prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"render-only wrist rig unexpectedly has collision: {prim.GetPath()}")

    def add_cylinder_between(path: str, start, end) -> None:
        a = Gf.Vec3d(*start)
        b = Gf.Vec3d(*end)
        delta = b - a
        length = delta.GetLength()
        if length <= 0.0:
            raise ValueError(f"zero-length cable segment: {path}")
        cylinder = UsdGeom.Cylinder.Define(stage, path)
        cylinder.CreateAxisAttr().Set(UsdGeom.Tokens.z)
        cylinder.CreateRadiusAttr(rig["cable_width_m"] * 0.5)
        cylinder.CreateHeightAttr(length)
        xf = UsdGeom.Xformable(cylinder)
        xf.AddTranslateOp().Set((a + b) * 0.5)
        quat = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), delta / length).GetQuat()
        imag = quat.GetImaginary()
        xf.AddOrientOp().Set(Gf.Quatf(float(quat.GetReal()), Gf.Vec3f(float(imag[0]), float(imag[1]), float(imag[2]))))
        # The cable casts no shadow (it is thin and the light is diffuse).
        render_only(cylinder, materials["seam_black"], cast_shadows=False)

    for robot_root in robot_roots:
        if not stage.GetPrimAtPath(robot_root + "/" + rig["camera_prim"]).IsValid():
            raise RuntimeError(f"calibrated wrist camera prim is missing: {robot_root}/{rig['camera_prim']}")
        rig_path = f"{robot_root}/panda_hand/real2sim_wrist_rig"
        # IsaacLab clones env_0 USD opinions into later environments. Once the source rig is authored,
        # cloned rig prims already exist; validate them instead of adding duplicate xformOps.
        if stage.GetPrimAtPath(rig_path).IsValid():
            for leaf in cable_leaf_names:
                cloned = stage.GetPrimAtPath(rig_path + "/" + leaf)
                if not cloned.IsValid() or cloned.HasAPI(UsdPhysics.CollisionAPI):
                    raise RuntimeError(f"cloned wrist rig prim is missing or has collision: {rig_path}/{leaf}")
            continue
        rig_xf = UsdGeom.Xform.Define(stage, rig_path)
        rig_xf.AddTranslateOp().Set(Gf.Vec3d(*cam_pos))
        rig_xf.AddOrientOp().Set(Gf.Quatf(cam_quat[0], Gf.Vec3f(cam_quat[1], cam_quat[2], cam_quat[3])))
        for index, (start, end) in enumerate(zip(cable_samples[:-1], cable_samples[1:], strict=True)):
            add_cylinder_between(rig_path + f"/usb_cable_seg_{index:02d}", start, end)
        # The narrow grey stripe on the real Franka Hand face, in panda_hand coordinates so a wrist-camera
        # recalibration cannot move it.
        face_band_path = f"{robot_root}/panda_hand/real2sim_eef_face_band"
        if not stage.GetPrimAtPath(face_band_path).IsValid():
            band = UsdGeom.Cube.Define(stage, face_band_path)
            band.CreateSizeAttr(1.0)
            band_xf = UsdGeom.Xformable(band)
            band_xf.AddTranslateOp().Set(Gf.Vec3d(*rig["face_band_center_hand_m"]))
            band_xf.AddScaleOp().Set(Gf.Vec3f(*rig["face_band_size_m"]))
            render_only(band, materials["eef_gray"])

    # Bridge (hand -> bridge link) and service tether (bridge link -> base) start as placeholders.
    UsdGeom.Xform.Define(stage, "/World/real2sim_cable_bridges")
    UsdGeom.Xform.Define(stage, "/World/real2sim_tethers")
    placeholder_bridge = sample_cubic_bezier(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.03), (0.03, 0.0, 0.03), (0.03, 0.0, 0.0)),
        steps_per_segment=rig["bridge_steps"],
    )
    anchor = rig["tether_anchor_base_m"]
    placeholder_tether = sample_guided_tether(
        (anchor[0], anchor[1], anchor[2] + 0.10),
        rig["tether_guide_base_m"],
        anchor,
        rig["tether_bow_base_m"],
        rig["tether_steps"],
    )
    for robot_root in robot_roots:
        env_name = robot_root.split("/")[-2]
        for root, leaf, samples in (
            (f"/World/real2sim_cable_bridges/{env_name}", "bridge_seg", placeholder_bridge),
            (f"/World/real2sim_tethers/{env_name}", "tether_seg", placeholder_tether),
        ):
            UsdGeom.Xform.Define(stage, root)
            for index, (start, end) in enumerate(zip(samples[:-1], samples[1:], strict=True)):
                add_cylinder_between(root + f"/{leaf}_{index:02d}", start, end)

    # ---- robot materials. The official FR3 asset keeps each link visual instanceable, and a material
    # authored on /robot cannot override bindings inside those read-only prototypes. Expand only robot
    # instances in this render stage, then bind every Gprim directly (render-only: collision,
    # articulation and dynamics stay intact).
    robot_prefixes = tuple(root + "/" for root in robot_roots)
    for _ in range(8):
        changed = False
        for prim in list(stage.TraverseAll()):
            path = str(prim.GetPath())
            if path.startswith(robot_prefixes) and _is_robot_visual_path(path) and prim.IsInstance():
                prim.SetInstanceable(False)
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError("robot visual de-instancing did not converge")

    lights: dict[str, float] = {}
    render_prims: list[tuple[Any, str]] = []
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if path == "/World/background":
            UsdLux.LightAPI(prim).GetIntensityAttr().Set(profile["dome_intensity"])
            lights[path] = profile["dome_intensity"]
        elif path.endswith("/sphere"):
            UsdLux.LightAPI(prim).GetIntensityAttr().Set(profile["sphere_intensity"])
            UsdLux.SphereLight(prim).GetRadiusAttr().Set(profile["sphere_radius_m"])
            xformable = UsdGeom.Xformable(prim)
            translate_ops = [
                op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ]
            translate = translate_ops[0] if translate_ops else xformable.AddTranslateOp()
            translate.Set(Gf.Vec3d(*profile["sphere_position_base_m"]))
            lights[path] = profile["sphere_intensity"]
        if not path.startswith(robot_prefixes) or not _is_robot_visual_path(path):
            continue
        if prim.IsInstanceProxy():
            if prim.IsA(UsdGeom.Gprim):
                raise RuntimeError(f"robot appearance still has an instance-proxy Gprim: {path}")
            continue
        is_subset = prim.GetTypeName() == "GeomSubset"
        if not prim.IsA(UsdGeom.Gprim) and not is_subset:
            continue
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        source_path = str(bound.GetPath()) if bound and bound.GetPrim().IsValid() else ""
        # Parent Gprims sometimes inherit an arbitrary child-subset binding from the DAE importer. Only
        # subsets may use that fallback; parent shells stay white.
        render_prims.append((prim, _robot_material_role(path, source_path if is_subset else "")))
    if "/World/background" not in lights or not any(path.endswith("/sphere") for path in lights):
        raise RuntimeError(f"expected dome+sphere lights, got {lights}")
    role_counts: dict[str, int] = {}
    for prim, role in render_prims:
        role_counts[role] = role_counts.get(role, 0) + 1
        if role != "source":
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[role])

    # ---- banana (PnP-Banana only): OmniPBR tuning plus a preview material on the banana root.
    banana_texture = Path(__file__).resolve().parents[1] / profile["banana_diffuse_texture"]
    if not banana_texture.is_file():
        raise FileNotFoundError(f"banana diffuse texture does not exist: {banana_texture}")
    banana_roots = [prim for prim in stage.TraverseAll() if str(prim.GetPath()).endswith("/scene/banana")]
    if banana_roots:
        tuned = 0
        for prim in stage.TraverseAll():
            if "/scene/banana/" not in str(prim.GetPath()) or not prim.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(prim)
            if str(shader.GetIdAttr().Get()) != "OmniPBR":
                continue
            shader.GetInput("diffuse_tint").Set(Gf.Vec3f(*profile["banana_diffuse_tint"]))
            texture_input = shader.GetInput("diffuse_texture")
            if not texture_input:
                texture_input = shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
            texture_input.Set(Sdf.AssetPath(str(banana_texture)))
            shader.GetInput("roughness").Set(profile["banana_roughness"])
            emission = profile["banana_emission"]
            shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission["color"]))
            shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(emission["intensity"])
            tuned += 1
        if tuned < 1:
            raise RuntimeError("expected at least one banana OmniPBR shader")
        base = "/World/Looks/real2sim_banana_preview"
        banana_material = UsdShade.Material.Define(stage, base)
        banana_shader = UsdShade.Shader.Define(stage, base + "/surface")
        banana_shader.CreateIdAttr("UsdPreviewSurface")
        banana_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(profile["banana_roughness"])
        reader = UsdShade.Shader.Define(stage, base + "/st_reader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        diffuse = UsdShade.Shader.Define(stage, base + "/diffuse")
        diffuse.CreateIdAttr("UsdUVTexture")
        diffuse.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(banana_texture)))
        diffuse.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        diffuse.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        banana_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            diffuse.ConnectableAPI(), "rgb"
        )
        glow = UsdShade.Shader.Define(stage, base + "/emission")
        glow.CreateIdAttr("UsdUVTexture")
        glow.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(banana_texture)))
        glow.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        scale = profile["banana_preview_emission_scale"]
        glow.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(scale, scale, scale, 1.0))
        glow.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        banana_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            glow.ConnectableAPI(), "rgb"
        )
        banana_material.CreateSurfaceOutput().ConnectToSource(banana_shader.ConnectableAPI(), "surface")
        for root in banana_roots:
            UsdShade.MaterialBindingAPI.Apply(root).Bind(
                banana_material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )

    # ---- bowl (PnP-Banana only): measured red / black materials.
    bowl_shaders = [
        prim
        for prim in stage.TraverseAll()
        if "/scene/bowl/" in str(prim.GetPath())
        and prim.IsA(UsdShade.Shader)
        and str(UsdShade.Shader(prim).GetIdAttr().Get()) == "UsdPreviewSurface"
    ]
    if any("/scene/bowl/" in str(prim.GetPath()) for prim in stage.TraverseAll()) and not bowl_shaders:
        raise RuntimeError("expected at least one bowl UsdPreviewSurface shader")
    for prim in bowl_shaders:
        path = str(prim.GetPath())
        role = "red" if "/mat_red/" in path else "black" if "/mat_black/" in path else None
        if role is None:
            continue
        spec = profile["bowl_materials"][role]
        shader = UsdShade.Shader(prim)
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(*(float(v) for v in spec["rgb"])))
        shader.GetInput("roughness").Set(float(spec["roughness"]))
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*(float(v) for v in spec["specular_rgb"]))
        )
        shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(spec["clearcoat"]))
        shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(float(spec["clearcoat_roughness"]))

    return {"robot_roots": len(robot_roots), "material_role_bindings": role_counts, "lights": lights}


def _quat_rotate_wxyz(quat, point):
    """Rotate a vec3 by a scalar-first quaternion without importing scipy."""
    w, x, y, z = (float(v) for v in quat)
    px, py, pz = (float(v) for v in point)
    uv = (
        y * pz - z * py,
        z * px - x * pz,
        x * py - y * px,
    )
    uuv = (
        y * uv[2] - z * uv[1],
        z * uv[0] - x * uv[2],
        x * uv[1] - y * uv[0],
    )
    return (
        px + 2.0 * (w * uv[0] + uuv[0]),
        py + 2.0 * (w * uv[1] + uuv[1]),
        pz + 2.0 * (w * uv[2] + uuv[2]),
    )


def wrist_tether_updater(doc: dict[str, Any]) -> Callable[[Any, Any], None]:
    """Return ``update(robot, env_origins)``, which moves the hand-to-link bridge and the link-to-base
    service tether of every env to the current robot pose (call after each teleport)."""
    import omni.usd
    from pxr import Gf, UsdGeom

    from real2sim.cameras import camera_specs, transform_to_pose

    rig = wrist_rig_profile(doc)
    stage = omni.usd.get_context().get_stage()
    cam_pos, cam_quat = transform_to_pose(np.asarray(rig["T_hand_cable_frame"]))
    exo_camera_world = tuple(float(v) for v in camera_specs(doc)["left_back"]["T_parent_cam_render"][:3, 3])

    def camera_point_to_hand(point):
        rotated = _quat_rotate_wxyz(cam_quat, point)
        return tuple(cam_pos[axis] + rotated[axis] for axis in range(3))

    bridge_start_hand = camera_point_to_hand(rig["tether_start_cam_m"])
    bridge_departure_hand = camera_point_to_hand(rig["bridge_departure_control_cam_m"])
    hand_local_segment_count = ((len(rig["cable_points_cam_m"]) - 1) // 3) * rig["cable_steps_per_span"]

    def pull_toward_exo(point):
        """Move a control point toward the exterior camera so the bridge stays in front of the arm."""
        delta = tuple(exo_camera_world[axis] - point[axis] for axis in range(3))
        norm = sum(value * value for value in delta) ** 0.5
        clearance = rig["bridge_camera_clearance_m"]
        if norm <= clearance + 1e-6:
            return point
        return tuple(point[axis] + clearance * delta[axis] / norm for axis in range(3))

    def update_cylinders(root, leaf_prefix, samples, *, visible: bool) -> None:
        visibility = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
        for segment_index, (start, end) in enumerate(zip(samples[:-1], samples[1:], strict=True)):
            path = root + f"/{leaf_prefix}_{segment_index:02d}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"dynamic wrist cable prim is missing: {path}")
            a = Gf.Vec3d(*start)
            b = Gf.Vec3d(*end)
            delta = b - a
            length = delta.GetLength()
            if length <= 0.0:
                raise RuntimeError(f"dynamic wrist cable segment collapsed: {path}")
            UsdGeom.Cylinder(prim).GetHeightAttr().Set(length)
            prim.GetAttribute("xformOp:translate").Set((a + b) * 0.5)
            quat = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), delta / length).GetQuat()
            imag = quat.GetImaginary()
            prim.GetAttribute("xformOp:orient").Set(
                Gf.Quatf(float(quat.GetReal()), Gf.Vec3f(float(imag[0]), float(imag[1]), float(imag[2])))
            )
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(visibility)

    def update(robot, env_origins) -> None:
        hand_id = robot.data.body_names.index("panda_hand")
        link_id = robot.data.body_names.index(rig["bridge_end_link"])
        hand_positions = robot.data.body_pos_w[:, hand_id].detach().cpu().tolist()
        hand_quats = robot.data.body_quat_w[:, hand_id].detach().cpu().tolist()
        link_positions = robot.data.body_pos_w[:, link_id].detach().cpu().tolist()
        link_quats = robot.data.body_quat_w[:, link_id].detach().cpu().tolist()
        origins = env_origins.detach().cpu().tolist()

        def on_hand(env_index, point):
            offset = _quat_rotate_wxyz(hand_quats[env_index], point)
            return tuple(float(hand_positions[env_index][axis]) + offset[axis] for axis in range(3))

        def on_link(env_index, point):
            offset = _quat_rotate_wxyz(link_quats[env_index], point)
            return tuple(float(link_positions[env_index][axis]) + offset[axis] for axis in range(3))

        for env_index in range(min(len(hand_positions), len(origins))):
            bridge_start_world = on_hand(env_index, bridge_start_hand)
            departure_world = on_hand(env_index, bridge_departure_hand)
            approach_world = on_link(env_index, rig["bridge_approach_control_link_m"])
            attachment_world = on_link(env_index, rig["bridge_attachment_link_m"])
            service_departure_world = on_link(env_index, rig["service_departure_control_link_m"])
            # The tether is deployed (pulled out of its stow) as the hand goes down.
            deploy = min(
                1.0,
                max(
                    0.0,
                    (rig["tether_deploy_hand_z_high_m"] - float(hand_positions[env_index][2]))
                    / (rig["tether_deploy_hand_z_high_m"] - rig["tether_deploy_hand_z_low_m"]),
                ),
            )
            anchor_world = tuple(
                float(origins[env_index][axis])
                + (1.0 - deploy) * rig["tether_stow_anchor_base_m"][axis]
                + deploy * rig["tether_anchor_base_m"][axis]
                for axis in range(3)
            )
            guide_world = tuple(
                float(origins[env_index][axis])
                + (1.0 - deploy) * rig["tether_stow_guide_base_m"][axis]
                + deploy * rig["tether_guide_base_m"][axis]
                for axis in range(3)
            )
            hand_local_visible = deploy >= rig["tether_visible_deploy_min"]
            rig_root = f"/World/envs/env_{env_index}/robot/panda_hand/real2sim_wrist_rig"
            visibility = UsdGeom.Tokens.inherited if hand_local_visible else UsdGeom.Tokens.invisible
            for segment_index in range(hand_local_segment_count):
                path = rig_root + f"/usb_cable_seg_{segment_index:02d}"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise RuntimeError(f"hand-local wrist cable prim is missing: {path}")
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set(visibility)

            # In the real high-wrist poses the dynamic cable is fully occluded: contract the bridge into the
            # off-raster stow anchor instead of drawing an attachment-to-stow-guide diagonal.
            hidden_controls = (
                anchor_world,
                (anchor_world[0], anchor_world[1], anchor_world[2] + 0.004),
                (anchor_world[0] + 0.004, anchor_world[1], anchor_world[2] + 0.004),
                (anchor_world[0] + 0.004, anchor_world[1], anchor_world[2]),
            )
            visible_controls = (
                bridge_start_world,
                pull_toward_exo(departure_world),
                pull_toward_exo(approach_world),
                attachment_world,
            )
            render_controls = visible_controls if hand_local_visible else hidden_controls
            bridge_samples = sample_cubic_bezier(render_controls, steps_per_segment=rig["bridge_steps"])
            tether_samples = sample_departing_guided_tether(
                render_controls[-1],
                service_departure_world,
                guide_world,
                anchor_world,
                rig["tether_bow_base_m"],
                rig["tether_steps"],
            )
            update_cylinders(
                f"/World/real2sim_cable_bridges/env_{env_index}",
                "bridge_seg",
                bridge_samples,
                visible=hand_local_visible,
            )
            update_cylinders(
                f"/World/real2sim_tethers/env_{env_index}", "tether_seg", tether_samples, visible=hand_local_visible
            )

    return update
