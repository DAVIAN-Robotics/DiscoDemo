# Official FR3 Isaac asset

- Source: `https://github.com/frankarobotics/franka_description`
- Source commit: `7aeeddc449edf8d62b594f9e36a81da53e7796f9`
- Upstream license: Apache-2.0; the unmodified `LICENSE` and `NOTICE` are stored here.
- Model: `fr3` arm with the official white `franka_hand` end effector.

The upstream xacro was expanded with absolute paths, then only its interface identifiers
were renamed from `fr3_*` to `panda_*`. Geometry, joint origins and axes, limits,
inertials, collision meshes, colors, and the 103.4 mm hand TCP definition were not
changed. The rename keeps existing RoboLab camera, action, observation, contact, and
SysID code compatible while replacing the stock FER/Panda model with the actual FR3
description.

`fr3_panda_compatible.urdf` is the portable intermediate source; the USD and its `configuration/`
layers were generated from it with the Isaac Lab URDF importer (`config.yaml`). Runtime joint stiffness, damping, armature, and
friction are still overwritten from the measured real2sim workcell SysID profile.
