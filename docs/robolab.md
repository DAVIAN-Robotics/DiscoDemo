# RoboLab (`libs/RoboLab`)

Upstream: RoboLab by NVIDIA (Apache License 2.0, `libs/RoboLab/LICENSE`, third-party notices in
`libs/RoboLab/THIRD_PARTY_NOTICES.md`). DiscoDemo uses its Isaac Lab environment layer (scene and task generation,
world state, predicates, LeRobot export) and adds the FR3 robot, the RL environment registration and the
data-generation scripts.

## Changed upstream files

- `robolab/robots/droid.py`: the robot is the official FR3 (Franka description, `assets/robots/fr3_official`) with
  the `panda_*` joint and link names and a Franka hand instead of the DROID Robotiq gripper. Joint 7 carries the
  +45 deg hand-mount offset at home and is observed / commanded in the command frame (offset removed). The gripper
  speed is the measured 0.05 m/s per finger; both fingers have contact sensors. The IK action configs are removed.
- `robolab/registrations/droid/auto_env_registrations_jointpos.py`: registers FR3 environments; new arguments
  `task_dir` (external task root), `decimation`, `extra_robot_cameras` (the measured wrist camera) and `background_cfg`
  (the data render passes the untextured dome of `real2sim.render_contract.neutral_dome_cfg` instead of the HDR
  office background).
- `robolab/registrations/droid/camera_presets.py`: wrist camera class renamed with the robot.
- `robolab/variations/camera.py`: stock cameras render at 480x270 (the data uses the measured workcell cameras) with
  a 0.05-2 m clipping range.
- `robolab/core/environments/base.py`: env spacing 3 m (neighbouring envs stay outside the 2 m camera range), DLSS
  quality mode, larger PhysX GPU buffers for 2,048 envs.
- `robolab/core/environments/config.py`: `generate_task_env_cfg` takes the end-effector body, can disable the HDF5
  recorder, and adds per-link contact sensors for the contact penalty.
- `robolab/core/environments/env.py`: RL mode resets finished envs instead of freezing them and keeps the observation
  before the automatic reset for bootstrapping.
- `robolab/core/sensors/contact_sensor_utils.py`: one contact sensor per robot link, filtered on the scene bodies.
- `robolab/core/observations/observation_utils.py`: image observations for robot-mounted cameras.
- `robolab/core/task/conditionals.py`, `hull_check.py`, `predicate_logic.py`: "gripper detached" reads both finger
  sensors; the open-top container test closes the air column above the rim (`open_top_slack`).
- `robolab/core/task/task_utils.py`: task lookup in a flat external task directory, with the load error of each file
  reported.
- `robolab/core/events/reset_pose.py`: adds the `box:` and `pregrasp:` initial-state events (the upstream events are
  unchanged).
- `robolab/core/export/lerobot_exporter.py`: writes the skill latent and `meta/provenance.json` (source settings),
  computes image statistics in a streaming pass, re-encodes clips to H.264 before concatenation and fails on missing
  or malformed input.
- `robolab/core/export/__init__.py`, `robolab/__init__.py`: exports trimmed to what is shipped.
- Docstrings / examples only: `core/environments/factory.py`, `runtime.py`, `core/events/basic_recorders.py`.
- `pyproject.toml`, `THIRD_PARTY_NOTICES.md`: trimmed to the shipped subset.
- `assets/fixtures/franka_table.usd`, `assets/fixtures/Props/instaceable_meshes.usd`,
  `assets/objects/ycb/textures/obj_000010.png`: the Git LFS objects of upstream, stored as regular files.

Removed upstream content: policies and inference servers, the dashboard, analysis tools, examples, docs, the
benchmark task files and scene generation, other robot and IK registrations, and the assets not used by the four
tasks.

## Added (DiscoDemo)

- `robolab/registrations/rl/register_rl_env.py`, `robolab/robots/rl_obs.py`, `rl_safety.py`, `action_norm.py`: the
  state-based RL environment (observation, relative jerk-limited arm action, held-success reward, contact / jamming
  penalty).
- `robolab/core/init_pose/`: the `box:` and `pregrasp:` initial-state distributions.
- `robolab/core/events/reverse_curriculum.py`: reset to demonstration states for the reverse curriculum.
- `robolab/core/export/state_recorder.py`, `state_teleport.py`, `skill_z.py`, `render_resume.py`: camera-free
  rollout states and their rendering.
- `scripts/`: collection, export filter, sharding, rendering, conversion and merging (stage 2).
