"""Collect successful rollouts of an RL generator as camera-free simulator states.

The generator is state-based, so collection needs no renderer: the env is built without cameras and
with many parallel envs. Every episode an env runs is recorded (envs auto-reset); only successful ones
are written, as full simulator states plus the joint-target actions that were applied. Rendering is a
separate step (``render_states_to_lerobot.py``) that teleports through these states with cameras.

The environment (task, initial-state distribution, success hold, plant, action space) comes from the
training run's ``config.yaml``, so collection runs the same env the generator was trained in.

Output (``--out-dir``)::

    episode_<batch>.h5               StateRecorder traj_ schema; root attrs = instruction + provenance
      traj_<n>/
        actions   [T-1, 8]           arm joint target applied at the step (joint 7 in the command frame,
                                     pi/4 below the physical joint) + gripper close {0, 1}
        success   [T]                bool
        skill_z   [T, d]             skill-conditioned generators only
        states/robot/joint_pos|joint_vel, states/objects/<name>/root_pose|root_vel, ...
        extras/pair_force [T-1, R, S]               robot link x scene body contact force
        extras/press_axial|press_lateral [T-1]      insertion tasks: peg-board force components
    index.json                       trajectory index (traj_uid, h5 file/group, num_frames)
    collect_progress.json            finished batches, for --resume-from-existing
    collect_budget.json              simulated steps and attempted / accepted episodes (generation yield)

usage::

    python libs/RoboLab/scripts/collect_rl_states.py --ckpt-dir <run>/step<N> --config-yaml <run>/config.yaml \\
        --num-envs 2048 --target-k 10000 --out-dir <states_dir> --seed 0
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback
from typing import Any

import numpy as np

# Seed offset per finished batch when resuming, so a resumed run does not replay the initial states and
# skill z of the batches it already collected.
RESUME_SEED_STRIDE = 10007
PROGRESS_FILENAME = "collect_progress.json"
BUDGET_FILENAME = "collect_budget.json"


def _write_json_atomic(path: str, obj: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _provenance(cfg: Any, args: argparse.Namespace, env: Any) -> dict[str, Any]:
    """Collection settings stamped into every h5; a resumed run must match them exactly."""
    return {
        "task": str(cfg.env.task),
        "env_version": str(cfg.env.env_version),
        "policy_ckpt": os.path.abspath(args.ckpt_dir),
        "policy_config_yaml": os.path.abspath(args.config_yaml),
        "init_pose": str(cfg.rfcl.init_pose),
        "success_hold_steps": int(cfg.success_hold_steps),
        "skill_z_dim": int(cfg.skill.z_dim),
        "target_object": str(env.target_object),
        "container": str(env.container),
    }


def _check_resume(out_dir: str, num_envs: int, seed: int, prov: dict[str, Any]) -> int:
    """Finished batches of an existing output dir (0 if empty); refuses to mix different settings."""
    shards = sorted(glob.glob(os.path.join(out_dir, "episode_*.h5")))
    prog = os.path.join(out_dir, PROGRESS_FILENAME)
    if not shards and not os.path.exists(prog):
        return 0
    d = _read_json(prog)
    assert (int(d["num_envs"]), int(d["seed"])) == (num_envs, seed), f"{out_dir} was collected with {d}"
    import h5py

    for shard in shards:
        with h5py.File(shard, "r") as hf:
            for key, cur in prov.items():
                old = hf.attrs[key]
                old = old.decode() if isinstance(old, bytes) else old
                assert type(cur)(old) == cur, f"{shard} was collected with {key}={old!r}, now {cur!r}"
    return int(d["batches"])


def _success_count(out_dir: str) -> int:
    """Successful trajectories recorded in ``index.json``."""
    path = os.path.join(out_dir, "index.json")
    if not os.path.exists(path):
        return 0
    return len({e["traj_uid"] for e in _read_json(path)["episodes"]})


def main() -> int:
    """Collect until ``--target-k`` successful trajectories are written."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt-dir", required=True, help="generator checkpoint step dir (actor.pt + normalizer.pt)")
    p.add_argument("--config-yaml", required=True, help="config.yaml of the training run")
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--target-k", type=int, required=True, help="successful trajectories to collect")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume-from-existing",
        action="store_true",
        help="continue an interrupted collection in --out-dir (same settings required)",
    )
    args = p.parse_args()

    import h5py
    import torch
    from omegaconf import OmegaConf

    from flash_rl.agents import create_agent
    from flash_rl.envs.robolab import env_kwargs_from_cfg, lock_gripper_channel, make_robolab_rl_env
    from flash_rl.envs.skill_z import SkillZConcatWrapper
    from flash_rl.rfcl.eval_request import shrink_replay_buffer_for_inference

    cfg = OmegaConf.load(args.config_yaml)
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)
    batches_done = 0
    if args.resume_from_existing and os.path.exists(os.path.join(out_root, PROGRESS_FILENAME)):
        batches_done = int(_read_json(os.path.join(out_root, PROGRESS_FILENAME))["batches"])
    seed = args.seed + RESUME_SEED_STRIDE * batches_done

    illegal_contact = OmegaConf.to_container(cfg.safety_penalty.illegal_contact, resolve=True)
    # Contact sensors for the safety signals, without a reward term.
    env: Any = make_robolab_rl_env(args.num_envs, seed, safety_signals_only=illegal_contact, **env_kwargs_from_cfg(cfg))
    if int(cfg.skill.z_dim) > 0:
        env = SkillZConcatWrapper(env, int(cfg.skill.z_dim), seed=seed)
    _, env_info = env.reset()
    cfg.agent.use_compile = False
    shrink_replay_buffer_for_inference(cfg)
    agent = create_agent(env.observation_space, env.action_space, env_info, cfg.agent)
    agent.load_actor(args.ckpt_dir)

    prov = _provenance(cfg, args, env)
    if args.resume_from_existing:
        assert _check_resume(out_root, args.num_envs, args.seed, prov) == batches_done
    else:
        assert not glob.glob(os.path.join(out_root, "episode_*.h5")), f"{out_root} is not empty"

    from flash_rl.rfcl.collision_metrics import pair_forces
    from robolab.core.export.state_recorder import StateRecorder
    from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
    from robolab.robots.action_norm import PANDA_J7_MOUNT_OFFSET, gripper_is_close, relative_applied_target
    from robolab.robots.rl_safety import press_components

    base = env.unwrapped
    n = int(env.num_envs)
    instruction = str(base.cfg.instruction)
    arm_term = base.action_manager.get_term("body")
    grip_term = base.action_manager.get_term("finger_joint")
    j7_col = list(arm_term._joint_names).index("panda_joint7")
    sensors = get_contact_sensors(base.scene)
    robot_bodies, scene_bodies = list(illegal_contact["robot_bodies"]), list(illegal_contact["scene_bodies"])
    # Insertion tasks also record the peg-board force (C_jam).
    press = cfg.safety_penalty.get("obj_press")
    press_surfaces = None if press is None else [str(s) for s in press.surfaces]
    skill_z = env if isinstance(env, SkillZConcatWrapper) else None

    n_success = _success_count(out_root)
    budget_path = os.path.join(out_root, BUDGET_FILENAME)
    budget = (
        _read_json(budget_path)
        if batches_done
        else {"sim_steps_total": 0, "episodes_attempted": 0, "episodes_accepted": 0}
    )
    print(f"[collect] {instruction!r}: {n_success}/{args.target_k} successes, batch {batches_done}", flush=True)

    obs, _ = env.reset(random_start_init=False)
    recorder = StateRecorder(base, out_dir=out_root)
    batch_id = batches_done
    t_start = time.time()
    while n_success < args.target_k:
        # One batch = until n finished episodes are pending; unfinished episodes continue in the next batch.
        n_closed = n_closed_succ = steps = 0
        while recorder.pending_count() < n:
            steps += 1
            actions = np.array(agent.sample_actions(0, prev_transition={"next_observation": obs}, training=False))
            raw = lock_gripper_channel(torch.as_tensor(actions), env.gripper_lock)
            # Every env is captured every step; after a done the next frame is the new episode's first.
            recorder.capture_frame(skill_z=None if skill_z is None else skill_z.z)
            q_pre = arm_term._asset.data.joint_pos[:, arm_term._joint_ids].clone()
            obs, rewards, terms, truncs, _ = env.step(actions)
            # Arm label = target applied in this step (processed actions are updated inside env.step).
            arm = relative_applied_target(arm_term.processed_actions, q_pre, arm_term._qmin, arm_term._qmax).clone()
            arm[:, j7_col] -= PANDA_J7_MOUNT_OFFSET  # recorded in the command frame
            grip = gripper_is_close(raw[:, 7:8]).to(arm.device, arm.dtype)
            extras = {"pair_force": pair_forces(sensors, robot_bodies, scene_bodies)}
            if press_surfaces is not None:
                extras["press_axial"], extras["press_lateral"] = press_components(
                    base, env.target_object, press_surfaces
                )
            applied_close = (grip_term.processed_actions == grip_term._close_command).all(dim=-1)
            recorder.attach_action(torch.cat([arm, grip], dim=-1), applied_close, extras=extras)
            done = np.asarray(terms).astype(bool) | np.asarray(truncs).astype(bool)
            success = np.asarray(rewards) > 0
            # The env ends an episode on success after the hold; mark its last pre-step frame.
            for eid in np.nonzero(done)[0]:
                if success[eid]:
                    recorder.mark_last_frame_success(int(eid))
                recorder.close_episode(int(eid))
                n_closed += 1
                n_closed_succ += int(success[eid])

        budget["sim_steps_total"] += n * steps
        budget["episodes_attempted"] += n_closed
        budget["episodes_accepted"] += n_closed_succ
        if n_closed_succ:
            # Unfinished episodes at the batch boundary are not written.
            h5_path = recorder.write(batch_id)
            with h5py.File(h5_path, "a") as hf:
                hf.attrs["instruction"] = instruction
                for key, value in prov.items():
                    hf.attrs[key] = value
            n_success += n_closed_succ
        recorder.clear_pending()
        batch_id += 1
        _write_json_atomic(
            os.path.join(out_root, PROGRESS_FILENAME), {"batches": batch_id, "num_envs": n, "seed": args.seed}
        )
        _write_json_atomic(budget_path, budget)
        elapsed = time.time() - t_start
        print(
            f"[collect] batch {batch_id - 1}: +{n_closed_succ}/{n_closed} -> {n_success}/{args.target_k} "
            f"({elapsed:.0f}s)",
            flush=True,
        )
    print(f"[collect] done: {n_success} successful trajectories in {out_root}; budget {budget}", flush=True)
    return 0


if __name__ == "__main__":
    # Exit with os._exit: Isaac Sim's atexit teardown can hang.
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
