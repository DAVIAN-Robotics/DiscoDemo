"""Render collected rollout states with the measured cameras (raw robolab schema).

Second half of the decoupled pipeline (``collect_rl_states.py`` is the first). Loads the camera-free
``episode_*.h5`` state files, builds a camera-enabled env of the task, teleports each env slot to every
saved frame's full state (robot joints, object poses; no physics step) and renders the exterior and wrist
cameras with the measured lens and camera response. Teleporting the saved state is exact, whereas action
replay would diverge.

Outputs (convert with ``convert_raw_to_lerobot.py``)::

    <out-dir>/<Task>/data.hdf5             data/demo_N: actions[T,8], joint_position[T,8], joint_velocity[T,8]
                                           (joint 7 in the command frame, gripper in [0, 1])
    <out-dir>/<Task>/<instr>_<N>__<camera>.mp4
    <out-dir>/episode_results.json
    <out-dir>/render_manifest.json         demo_N -> source episode file, traj group and traj_uid

Actions are stored as collected; the terminal action repeats the last one so actions and states both
have length T. Trajectories are rendered in groups of ``--num-envs`` sorted by length; ``--resume`` skips
groups finished by an interrupted run.

Usage::

    python libs/RoboLab/scripts/render_states_to_lerobot.py --task pnp_banana \\
        --states-dir <states_dir> --out-dir <raw_render_dir> --num-envs 32
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

if sys.path and sys.path[0].endswith(os.path.join("RoboLab", "scripts")):
    sys.path.pop(0)

import numpy as np

from real2sim.env_contract import CONTROL_DECIMATION, TASKS, activate, profile, rendering_mode
from real2sim.render_contract import FRAME_SETTLE, RENDER_CAMERAS, RESET_SETTLE, WARMUP_RENDERS, render_profile

#: Output camera name -> Isaac sensor of the measured workcell camera.
CAMERAS = {"over_shoulder_left_camera": "left_back_cam", "wrist_cam": "wrist_cam"}
J7_MOUNT_OFFSET = float(np.pi / 4.0)
FINGER_OPEN = 0.04  # panda_finger_joint1 open position
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAME = "panda_finger_joint1"
FPS = 20
# DLSS execMode 2 (Quality): less ghosting and flicker than Performance at this resolution.
DLSS_MODE = 2
PROGRESS_LOG_S = 30.0


def _state8_from_full(jp_full, jv_full, arm_cols, finger_col):
    """Full joint vector ``[k, nj]`` -> 8-dim state (joint 7 in the command frame, gripper in [0, 1])."""
    import torch

    j7 = ARM_JOINT_NAMES.index("panda_joint7")
    arm_state = jp_full[:, arm_cols].clone()
    arm_state[:, j7] = arm_state[:, j7] - J7_MOUNT_OFFSET
    finger = jp_full[:, finger_col : finger_col + 1]
    grip01 = (FINGER_OPEN - finger) / FINGER_OPEN  # 0 = open, 1 = closed
    state8 = torch.cat([arm_state, grip01], dim=-1).detach().cpu().numpy().astype(np.float32)
    grip_vel = -jv_full[:, finger_col : finger_col + 1] / FINGER_OPEN
    vel8 = torch.cat([jv_full[:, arm_cols], grip_vel], dim=-1).detach().cpu().numpy().astype(np.float32)
    return state8, vel8


class _VideoWriter:
    """Streaming H.264 mp4 writer (PyAV, yuv420p); frames are encoded as they are rendered."""

    def __init__(self, path: str, fps: float) -> None:
        import av

        self._av = av
        self._c = av.open(path, mode="w")
        self._st = self._c.add_stream("libx264", rate=int(round(fps)))
        self._st.pix_fmt = "yuv420p"
        self._init = False

    def append_data(self, frame) -> None:
        """Encode one RGB uint8 [H, W, 3] frame."""
        if not self._init:
            self._st.height, self._st.width = int(frame.shape[0]), int(frame.shape[1])
            self._init = True
        for pkt in self._st.encode(self._av.VideoFrame.from_ndarray(frame, format="rgb24")):
            self._c.mux(pkt)

    def close(self) -> None:
        """Flush the encoder and close the container."""
        if self._init:
            for pkt in self._st.encode():
                self._c.mux(pkt)
        self._c.close()


def main() -> int:
    """Render every trajectory of ``--states-dir``."""
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--states-dir", required=True, help="dir with episode_*.h5 from collect_rl_states.py")
    p.add_argument("--out-dir", required=True, help="raw robolab-schema output (data.hdf5 + mp4)")
    p.add_argument("--workcell", default="libs/real2sim/assets/workcell.json")
    p.add_argument("--num-envs", type=int, default=32, help="trajectories rendered in parallel (<=256)")
    p.add_argument("--teleport-tol", type=float, default=1e-4, help="tolerance of the final teleport spot check")
    p.add_argument("--resume", action="store_true", help="skip groups finished by an interrupted run")
    args = p.parse_args()
    assert args.num_envs <= 256, "--num-envs must be <= 256"

    states_dir = os.path.abspath(args.states_dir)
    out_root = os.path.abspath(args.out_dir)
    version = f"fr3-{args.task}"
    task_profile = profile(version)
    workcell_doc = activate(version, args.workcell)

    from isaaclab.app import AppLauncher

    # RTX rendering mode is an AppLauncher argument, fixed by the environment contract.
    AppLauncher(headless=True, enable_cameras=True, device="cuda:0", rendering_mode=rendering_mode(workcell_doc))

    import h5py
    import torch

    from real2sim.camera_pipeline import apply_camera_pipeline
    from real2sim.cameras import robot_camera_cfgs, scene_camera_cfgs
    from real2sim.render_contract import neutral_dome_cfg
    from real2sim.robot_gains import apply_joint_gains
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.export import render_resume
    from robolab.core.export.state_teleport import load_trajs, teleport_frames
    from robolab.core.task.task_utils import resolve_task_path
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    pipelines = {c: render_profile(workcell_doc).camera_pipelines[s] for c, s in CAMERAS.items()}
    task_dir = str(task_profile.task_file.parent)
    _, task = resolve_task_path(str(task_profile.task_file), task_dir)
    auto_register_droid_envs(
        task_dirs=None,
        task_dir=task_dir,
        task=task,
        decimation=CONTROL_DECIMATION,
        cameras=scene_camera_cfgs(workcell_doc, names={RENDER_CAMERAS["left_back_cam"]}),
        extra_robot_cameras=robot_camera_cfgs(workcell_doc),
        background_cfg=neutral_dome_cfg(),
    )
    env_cfg = parse_env_cfg(get_envs(task=[task])[0], device="cuda:0", num_envs=args.num_envs, use_fabric=True)
    env_cfg.sim.render.dlss_mode = DLSS_MODE
    env, env_cfg = create_env(env_cfg, device="cuda:0", num_envs=args.num_envs, use_fabric=True)
    apply_joint_gains(env, workcell_doc)
    from real2sim.render_contract import apply_render_runtime

    apply_render_runtime(env, workcell_doc)
    env.reset()

    base = env.unwrapped
    sim, scene = base.sim, base.scene
    robot = scene.articulations["robot"]
    n_envs = base.num_envs
    dev = env.device
    origins = scene.env_origins[:n_envs]
    names = list(robot.data.joint_names)
    arm_cols = [names.index(j) for j in ARM_JOINT_NAMES]
    finger_col = names.index(FINGER_JOINT_NAME)
    from real2sim.appearance import wrist_tether_updater

    update_wrist_tether = wrist_tether_updater(workcell_doc)

    trajs = load_trajs(states_dir, dev)
    with h5py.File(os.path.join(states_dir, trajs[0]["src_file"]), "r") as hf:
        instruction = str(hf.attrs["instruction"])
        src_attrs = dict(hf.attrs)
    # Group by length: a group renders all slots for its longest trajectory. Output demo order follows
    # length, so prefix subsets of the output are biased toward short episodes.
    trajs.sort(key=lambda t: int(t["robot_jp"].shape[0]))
    print(f"[render] {task}: {len(trajs)} trajectories, {n_envs} per group", flush=True)
    cleaned_instr = re.sub(r"[^\w\s]", "", instruction).replace(" ", "_")

    plan = render_resume.group_plan(len(trajs), n_envs)
    progress = render_resume.RenderProgress.open(
        out_root,
        fingerprint=render_resume.render_fingerprint(
            args={k: v for k, v in vars(args).items() if k != "resume"},
            env={},
            uids=[f"{t['src_file']}:{t['src_group']}" for t in trajs],
            n_envs=n_envs,
        ),
        n_groups=len(plan),
        resume=args.resume,
    )
    if progress.done:
        print(f"[render] resume: {len(progress.done)}/{len(plan)} groups already done", flush=True)

    task_out = os.path.join(out_root, task)
    os.makedirs(task_out, exist_ok=True)
    h5_path = os.path.join(task_out, "data.hdf5")
    # Written to a temp file and swapped in after the merge, so an interrupted run keeps the previous output.
    hf_out = h5py.File(h5_path + ".tmp", "w")
    # The collection provenance attrs (policy checkpoint, initial states, ...) are carried over.
    for k, v in src_attrs.items():
        hf_out.attrs[k] = v
    hf_out.attrs["state_layout"] = "panda_joint1..7 (joint 7 in the command frame), gripper_close"
    hf_out.attrs["action_layout"] = "panda_joint1..7 absolute target (joint 7 in the command frame), gripper_close"
    data_grp = hf_out.create_group("data")
    obj_names = list(trajs[0]["objects"].keys())
    art_names = list(trajs[0]["arts"].keys())

    def teleport(slot_trajs, slot_frames, slot_eids):
        teleport_frames(scene, robot, origins, obj_names, art_names, slot_trajs, slot_frames, slot_eids, dev)
        # write_*_to_sim writes PhysX only; push to fabric so the cameras see it.
        scene.write_data_to_sim()
        sim.forward()
        update_wrist_tether(robot, origins)

    # writer.close flushes the encoder (expensive); PyAV releases the GIL, so close in threads.
    close_pool = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 8))
    results = []
    n_demos = 0

    # The renderer is cold after env.reset: warm it up on frame 0 of the first group still to render.
    first = next((g0 for gi, g0, _ in plan if gi not in progress.done), 0)
    warm = trajs[first : first + n_envs]
    teleport(warm, [0] * len(warm), list(range(len(warm))))
    for _ in range(WARMUP_RENDERS):
        sim.render()

    for gi, g0, g1 in plan:
        group = trajs[g0:g1]
        gk = len(group)
        if gi in progress.done:
            for i in range(gk):
                for c in CAMERAS:
                    mp4 = os.path.join(task_out, f"{cleaned_instr}_{n_demos + i}__{c}.mp4")
                    if not os.path.isfile(mp4):
                        raise FileNotFoundError(f"group {gi} is marked done but {mp4} is missing")
                results.append({"task": task, "instruction": instruction, "success": True})
            n_demos += gk
            continue
        slots = list(range(gk))
        lengths = [int(t["robot_jp"].shape[0]) for t in group]
        acc_state8: list[list] = [[] for _ in range(gk)]
        acc_vel8: list[list] = [[] for _ in range(gk)]
        writers = {
            c: [_VideoWriter(os.path.join(task_out, f"{cleaned_instr}_{n_demos + i}__{c}.mp4"), FPS) for i in slots]
            for c in CAMERAS
        }
        n_written = [0] * gk
        futures = []

        # Group start is a large jump: teleport to frame 0 and let DLSS converge.
        t0 = last_log = time.perf_counter()
        teleport(group, [0] * gk, slots)
        for _ in range(RESET_SETTLE):
            sim.render()
        for f in range(max(lengths)):
            if time.perf_counter() - last_log >= PROGRESS_LOG_S:
                last_log = time.perf_counter()
                print(f"[render] group {gi + 1}/{len(plan)} frame {f}/{max(lengths)}", flush=True)
            teleport(group, [min(f, n - 1) for n in lengths], slots)
            for _ in range(FRAME_SETTLE):
                sim.render()
            # TiledCamera updates lazily; force a recompute.
            for sensor in CAMERAS.values():
                scene.sensors[sensor].update(0.0, force_recompute=True)
            st8, vel8 = _state8_from_full(
                robot.data.joint_pos[slots], robot.data.joint_vel[slots], arm_cols, finger_col
            )
            rgb = {
                c: scene.sensors[s].data.output["rgb"][:gk, :, :, :3].detach().to(torch.uint8).cpu().numpy()
                for c, s in CAMERAS.items()
            }
            for i in slots:
                if f >= lengths[i]:
                    continue
                acc_state8[i].append(st8[i])
                acc_vel8[i].append(vel8[i])
                for c in CAMERAS:
                    writers[c][i].append_data(np.ascontiguousarray(apply_camera_pipeline(rgb[c][i], pipelines[c])))
                n_written[i] += 1
                if n_written[i] == lengths[i]:
                    # Close finished slots in the background while the rest of the group renders.
                    futures.append(close_pool.submit(lambda i=i, w=writers: [w[c][i].close() for c in CAMERAS]))
        # Join before writing the states so they are never paired with incomplete videos.
        for fu in futures:
            fu.result()

        part = []
        for i, traj in enumerate(group):
            n = lengths[i]
            act = traj["actions"]  # [T-1, 8]
            assert act.shape[0] == n - 1 and n_written[i] == n, (act.shape, n, n_written[i])
            z = traj["skill_z"]
            part.append(
                {
                    "num": n_demos,
                    "T": n,
                    "actions": np.concatenate([act, act[-1:]], axis=0),
                    "joint_position": np.stack(acc_state8[i]),
                    "joint_velocity": np.stack(acc_vel8[i]),
                    # The episode skill z, repeated per frame (generators with a skill z only).
                    "skill_z": None if z is None else np.tile(z.reshape(1, -1), (n, 1)),
                }
            )
            results.append({"task": task, "instruction": instruction, "success": True})
            n_demos += 1
        render_resume.write_part(render_resume.part_path(out_root, gi), part)
        progress.mark_done(gi)
        print(
            f"[render] group {gi + 1}/{len(plan)}: {gk} demos ({n_demos}/{len(trajs)}) "
            f"in {time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    # Merge the group parts in order (resumed runs produce the same data.hdf5).
    render_resume.merge_parts(data_grp, [render_resume.part_path(out_root, gi) for gi, _, _ in plan])
    data_grp.attrs["total"] = sum(int(data_grp[k].attrs["num_samples"]) for k in data_grp)
    hf_out.close()
    os.replace(h5_path + ".tmp", h5_path)

    manifest = {
        "demos": [
            {
                "demo": f"demo_{i}",
                "src_file": t["src_file"],
                "src_group": t["src_group"],
                "traj_uid": t["traj_uid"],
                "num_frames": int(t["robot_jp"].shape[0]),
            }
            for i, t in enumerate(trajs)
        ]
    }
    with open(os.path.join(out_root, "render_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out_root, "episode_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Teleport spot check: re-teleport demo 0 at its middle frame and compare robot joints and object poses
    # with the saved state (objects separately, since only they add env_origins; quaternions up to sign).
    mid = int(trajs[0]["robot_jp"].shape[0]) // 2
    teleport([trajs[0]], [mid], [0])
    for _ in range(2):
        sim.render()
    dev_max = float((robot.data.joint_pos[0] - trajs[0]["robot_jp"][mid]).abs().max())
    origin0 = origins[0]
    for name in obj_names:
        saved = trajs[0]["objects"][name]["root_pose"][mid]
        real = scene.rigid_objects[name].data.root_pose_w[0]
        dev_max = max(dev_max, float((real[:3] - origin0 - saved[:3]).abs().max()))
        dev_max = max(dev_max, float(min((real[3:7] - saved[3:7]).abs().max(), (real[3:7] + saved[3:7]).abs().max())))
    print(f"[render] teleport spot check: max |realized - saved| = {dev_max:.3e}", flush=True)
    assert dev_max <= args.teleport_tol, f"teleport spot check failed: {dev_max:.3e} > {args.teleport_tol:.3e}"
    print(f"[render] done: {n_demos} demos -> {h5_path}", flush=True)
    return 0


if __name__ == "__main__":
    code = 0
    try:
        code = main()
    except SystemExit as e:  # argparse --help / usage errors
        code = e.code if isinstance(e.code, int) else int(e.code is not None)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
