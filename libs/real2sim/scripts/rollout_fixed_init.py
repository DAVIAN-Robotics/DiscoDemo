r"""Roll out an FR3 policy from fixed initial states and record end-effector trajectories.

All episodes start from the same object layout (or from a small bank of layouts), so the
spread of the recorded trajectories reflects the policy alone. The layout is pinned by
overwriting the object (and arm) state after reset. Each episode is saved as
``ep_XXXXX.npz``; ``render_eef_2d.py`` draws them.

The fixed scenes of the paper figures are ``libs/real2sim/assets/fixed_scenes/<task>.json``. A record
pins the listed objects and, if it has an ``arm`` entry, the arm (otherwise the arm keeps its reset
state, the home pose). ``--dump-inits N`` writes the layouts sampled by the first N envs as a new bank.

``--export-view <dir>`` instead renders the pinned scene once from the figure camera (square 640 px,
44 deg field of view, looking from the front) and writes ``background.png``, ``camera.json`` and
``layout.json`` for ``render_eef_2d.py``.

Usage
-----
    python libs/real2sim/scripts/rollout_fixed_init.py --ckpt <run>/step<N> --out-dir <dir> --n-env 64 \
        --bank-json libs/real2sim/assets/fixed_scenes/<task>.json [--export-view <view_dir>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

# Record field -> (policy obs term, expected width).
_FIELD_TO_TERM = {
    "qpos": ("arm_joint_pos", 7),
    "gripper": ("gripper_pos", 1),
    "ee_pos": ("eef_position", 3),
    "ee_quat": ("eef_orientation", 4),
    "obj_pos": ("target_pos", 3),
    "obj_quat": ("target_quat", 4),
    "container_pos": ("container_pos", 3),
}


def _build_term_slices(base: Any) -> dict[str, tuple[int, int]]:
    """Locate each record field in the flat policy observation.

    Parameters
    ----------
    base : Any
        Unwrapped Isaac env with an ``observation_manager``.

    Returns
    -------
    dict[str, tuple[int, int]]
        Record field -> (start, end) column range.
    """
    import numpy as np

    obs_mgr = base.observation_manager
    names = list(obs_mgr.active_terms["policy"])
    dims = list(obs_mgr.group_obs_term_dim["policy"])
    assert len(names) == len(dims), f"active_terms ({len(names)}) vs group_obs_term_dim ({len(dims)})"
    term_span: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, dim in zip(names, dims, strict=True):
        width = int(np.prod(dim))
        term_span[name] = (offset, offset + width)
        offset += width
    field_slices: dict[str, tuple[int, int]] = {}
    for field, (term, want_dim) in _FIELD_TO_TERM.items():
        assert term in term_span, f"obs term {term!r} (field {field!r}) missing; active_terms={names}"
        start, end = term_span[term]
        assert end - start == want_dim, f"obs term {term!r} has width {end - start}, expected {want_dim}"
        field_slices[field] = (start, end)
    return field_slices


def _episode_arrays(
    rows: list[Any], field_slices: dict[str, tuple[int, int]], init_seed: int, skill_z: Any, success: bool
) -> dict[str, Any]:
    """Slice per-step flat observations into the per-episode record arrays.

    Parameters
    ----------
    rows : list
        Flat observations, one per step.
    field_slices : dict[str, tuple[int, int]]
        Result of ``_build_term_slices``.
    init_seed : int
        Seed of the rollout.
    skill_z : numpy.ndarray or None
        Episode skill latent, if any.
    success : bool
        Episode success.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arrays to store in the episode npz.
    """
    import numpy as np

    arr = np.stack(rows).astype(np.float32)
    out: dict[str, Any] = {
        "success": np.array(bool(success)),
        "ep_len": np.array(int(arr.shape[0])),
        "init_seed": np.array(int(init_seed)),
        **{field: arr[:, s:e].astype(np.float32) for field, (s, e) in field_slices.items()},
    }
    if skill_z is not None:
        out["skill_z"] = np.asarray(skill_z, dtype=np.float32)
    return out


def main() -> None:
    """Build env and agent from a checkpoint and roll out ``n_env`` episodes from fixed inits."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (actor.pt + normalizer.pt); config.yaml next to it")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-env", type=int, default=64, help="number of parallel episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bank-json", default="", help="init bank json; env i starts from bank[i %% n_init]")
    ap.add_argument("--export-view", default="", help="render the figure background and camera into this dir and exit")
    ap.add_argument(
        "--dump-inits",
        type=int,
        default=0,
        help="if N>0, write the object poses and arm state of envs 0..N-1 after reset as a bank json and exit",
    )
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from omegaconf import OmegaConf

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ck = Path(a.ckpt)
    cfg_path = next(p for p in (ck / "config.yaml", ck.parent / "config.yaml") if p.exists())
    cfg = OmegaConf.load(cfg_path)
    z_dim = int(cfg.skill.z_dim)
    print(f"[roll] ckpt={ck} z_dim={z_dim} alpha={cfg.skill.alpha} n_env={a.n_env}", flush=True)

    from flash_rl.envs.robolab import env_kwargs_from_cfg, make_robolab_rl_env

    env_kwargs = env_kwargs_from_cfg(cfg)
    view_dir = Path(a.export_view) if a.export_view else None
    if view_dir is not None:
        # The figure camera replaces the exterior camera of a copy of the workcell (raw render: no distortion,
        # unit gain, no blur).
        from render_eef_2d import FIGURE_CAMERA, cam_params

        from real2sim.workcell import load_workcell

        view_dir.mkdir(parents=True, exist_ok=True)
        cam = cam_params(**FIGURE_CAMERA)
        T = np.eye(4)
        T[:3, :3] = np.asarray(cam["R_ros"])
        T[:3, 3] = np.asarray(cam["pos_w"])
        doc = load_workcell(str(env_kwargs["workcell"]))
        lb = doc["cameras"]["left_back"]
        side = int(FIGURE_CAMERA["side"])
        lb.pop("T_parent_cam_render")
        lb.update(
            {
                "hw": [side, side],
                "K": {"value": cam["K"], "source": "reference"},
                "dist": {"value": [0.0] * 5, "source": "reference"},
                "T_parent_cam": {"value": T.tolist(), "source": "reference"},
                "flip_180": False,
                "rgb_gain": {"value": [1.0, 1.0, 1.0], "source": "reference"},
                "mtf": {
                    "blur_sigma_px": {"value": 0.0, "source": "reference"},
                    "unsharp_amount": {"value": 0.0, "source": "reference"},
                },
            }
        )
        lb.pop("horizontal_gain", None)
        env_kwargs["workcell"] = str(view_dir / "workcell_figure.json")
        Path(env_kwargs["workcell"]).write_text(json.dumps(doc, indent=1))
        (view_dir / "camera.json").write_text(json.dumps(cam, indent=1))
    env: Any = make_robolab_rl_env(a.n_env, a.seed, cameras=view_dir is not None, **env_kwargs)

    import torch

    from flash_rl.agents import create_agent
    from flash_rl.diversity import scene_state
    from flash_rl.envs.skill_z import SkillZConcatWrapper
    from flash_rl.rfcl.eval_request import shrink_replay_buffer_for_inference

    if z_dim > 0:
        env = SkillZConcatWrapper(env, z_dim, seed=a.seed)
    base = env.unwrapped
    slices = _build_term_slices(base)
    hold = int(cfg.success_hold_steps)

    _obs0, env_info = env.reset(random_start_init=False)
    cfg.agent.use_compile = False
    shrink_replay_buffer_for_inference(cfg)
    agent = create_agent(env.observation_space, env.action_space, env_info, cfg.agent)
    agent.load_actor(str(ck))

    scene = base.scene
    origins = scene.env_origins
    robot = scene["robot"]

    from fixed_init_bank import assign_envs, load_bank, object_names

    def _capture(i: int) -> dict[str, Any]:
        """Capture the current state of env ``i`` as an init record (all rigid objects and the arm)."""
        o = origins[i].detach().cpu().numpy()
        objs = {}
        for nm in scene.rigid_objects:
            st = scene.rigid_objects[nm].data.root_state_w[i, :7].detach().cpu().numpy()
            objs[nm] = (st[:3] - o).tolist() + st[3:7].tolist()
        return {
            "objects": objs,
            "arm": {
                "joint_pos": robot.data.joint_pos[i].detach().cpu().numpy().tolist(),
                "joint_vel": robot.data.joint_vel[i].detach().cpu().numpy().tolist(),
            },
        }

    # The skill z of the measured episodes is the one the wrapper sampled at the first reset.
    z = np.random.default_rng(a.seed).standard_normal((a.n_env, z_dim)).astype(np.float32) if z_dim > 0 else None
    obs, _ = env.reset(random_start_init=False)

    # Dump after reset, when the sampler has placed the objects.
    if a.dump_inits > 0:
        assert a.dump_inits <= a.n_env, f"--dump-inits ({a.dump_inits}) exceeds --n-env ({a.n_env})"
        recs = [_capture(i) for i in range(a.dump_inits)]
        Path(out_dir, "init_bank.json").write_text(json.dumps(recs, indent=1))
        print(f"[roll] dumped {len(recs)} inits -> {out_dir}/init_bank.json", flush=True)
        sys.stdout.flush()
        os._exit(0)

    bank = load_bank(Path(a.bank_json)) if a.bank_json else [_capture(0)]
    names = object_names(bank)
    env_init = assign_envs(a.n_env, len(bank))
    print(f"[roll] bank: {len(bank)} init x {a.n_env // len(bank)} env, objects={names}", flush=True)

    # Pin the layout (and the arm where the record has one).
    ids = torch.arange(a.n_env, device=origins.device)
    if all("arm" in bank[k] for k in env_init):

        def _arm(field: str) -> torch.Tensor:
            return torch.stack(
                [torch.as_tensor(bank[k]["arm"][field], dtype=torch.float32, device=origins.device) for k in env_init]
            )

        robot.write_joint_state_to_sim(_arm("joint_pos"), _arm("joint_vel"), env_ids=ids)
    for nm in names:
        o = scene.rigid_objects[nm]
        st = o.data.default_root_state[: a.n_env].clone()
        t = torch.stack(
            [torch.as_tensor(bank[k]["objects"][nm], dtype=torch.float32, device=origins.device) for k in env_init]
        )
        st[:, :3] = t[:, :3] + origins[: a.n_env]
        st[:, 3:7] = t[:, 3:7]
        st[:, 7:] = 0.0
        o.write_root_state_to_sim(st, env_ids=ids)
    scene.write_data_to_sim()
    base.sim.forward()
    scene.update(base.sim.get_physics_dt())
    if view_dir is not None:
        from real2sim.appearance import wrist_tether_updater
        from real2sim.render_contract import WARMUP_RENDERS, read_camera_tiles

        wrist_tether_updater(json.loads(Path(env_kwargs["workcell"]).read_text()))(robot, origins)
        for _ in range(WARMUP_RENDERS):
            base.sim.render()
        frame = read_camera_tiles(base, ["left_back_cam"], [0])[0]
        imageio.imwrite(view_dir / "background.png", frame)
        (view_dir / "layout.json").write_text(json.dumps(bank[env_init[0]]["objects"], indent=1))
        print(f"[roll] figure view -> {view_dir}", flush=True)
        sys.stdout.flush()
        os._exit(0)
    if z is not None:
        env.set_z(z, np.arange(a.n_env))
    # The first action must see the pinned layout and the episode z.
    obs = env.observe()

    buf: list[list] = [[obs[i].copy()] for i in range(a.n_env)]
    # Per-step success streak, used by the renderer to cut each trajectory at success.
    sbuf: list[list] = [[0] for _ in range(a.n_env)]
    # Actions sent to the env (T rows; observations have T+1 rows).
    abuf: list[list] = [[] for _ in range(a.n_env)]
    # Full scene state before each action; env i ending at step t owns snaps[0:t+1].
    snaps: list[dict] = []
    meta = scene_state.names(base)
    alive = np.ones(a.n_env, dtype=bool)
    n_ep = n_ok = 0
    for _t in range(int(env.max_episode_steps) + 5):
        snaps.append(scene_state.snapshot(base))
        act = agent.sample_actions(0, prev_transition={"next_observation": obs}, training=False)
        nxt, _r, term, trunc, infos = env.step(act)
        done = np.asarray(term).astype(bool) | np.asarray(trunc).astype(bool)
        streak = base._succ_streak.detach().cpu().numpy()
        fin = infos["final_obs"]
        for i in np.flatnonzero(alive):
            abuf[i].append(np.asarray(act[i]).copy())
            if done[i]:
                buf[i].append(fin[i].copy())
                sbuf[i].append(int(streak[i]))
                ok = bool(streak[i] >= hold)
                data = _episode_arrays(buf[i], slices, a.seed, skill_z=(z[i] if z is not None else None), success=ok)
                data["succ_streak"] = np.asarray(sbuf[i], dtype=np.int32)
                data["success_hold_steps"] = np.asarray(hold, dtype=np.int32)
                data["init_id"] = np.asarray(env_init[i], dtype=np.int32)
                data["actions"] = np.stack(abuf[i]).astype(np.float32)
                data.update(scene_state.env_slice(snaps, i, 0, _t + 1))
                data.update(meta)
                np.savez_compressed(out_dir / f"ep_{n_ep:05d}.npz", **data)
                n_ep += 1
                n_ok += int(ok)
                alive[i] = False
            else:
                buf[i].append(nxt[i].copy())
                sbuf[i].append(int(streak[i]))
        obs = nxt
        if not alive.any():
            break
    print(f"[roll] done: {n_ep} episodes, success {n_ok}/{n_ep}", flush=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
