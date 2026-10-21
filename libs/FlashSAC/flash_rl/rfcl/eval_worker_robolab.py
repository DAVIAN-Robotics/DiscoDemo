"""Eval worker: evaluates the trainer's checkpoints in its own process and Isaac app.

Isaac allows one AppLauncher per process, so the worker builds its own environment (with the
measured exterior and wrist cameras), loads each requested checkpoint, and runs one deterministic
episode per env from a full reset. It reports the success rate (success reached at any step) and
the time to the first success, and writes a video of the first envs.

File IPC (see ``eval_request.py``): request ``request_{env_step}.json`` (only the newest pending
request is evaluated), result ``result_{env_step}.json``, shutdown ``stop``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

# Make flash_rl importable regardless of the spawning process's cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flash_rl.rfcl.eval_request import CONTROL_HZ, eval_result_path, write_json_atomic  # noqa: E402

# Exterior and wrist cameras, tiled left to right.
VIDEO_SENSORS = ("left_back_cam", "wrist_cam")
VIDEO_ENVS = 4
# Frames kept after the first success before a tile freezes.
VIDEO_SUCCESS_MARGIN = 5
# Playback speed, downscale factor (frames are stored downscaled) and libx264 quality of eval videos.
VIDEO_SPEED = 2.0
VIDEO_SCALE = 2
VIDEO_QUALITY = 5


def write_video(path: str, frames: np.ndarray) -> None:
    """Write ``[T, C, H, W]`` uint8 frames as H.264 + yuv420p + faststart (plays in browsers and wandb)."""
    import imageio

    imageio.mimwrite(
        path,
        frames.transpose(0, 2, 3, 1),
        fps=int(round(VIDEO_SPEED * CONTROL_HZ)),
        codec="libx264",
        quality=VIDEO_QUALITY,
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )


def run_clean_eval(
    env: Any, agent: Any, max_steps: int, n_video: int, on_reset: Callable[[Any], None] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reset all envs and roll out the deterministic policy for one episode per env.

    The env is built with ``success_terminates=False``, so episodes run to the time limit; envs
    that finish are frozen so auto-reset episodes do not count. Success is read from
    ``_succ_streak >= 1`` (the success predicate holds at that step).

    Parameters
    ----------
    env : RoboLabVectorEnv
        Eval env (with cameras when ``n_video > 0``).
    agent : Any
        ``sample_actions(step, prev_transition, training)`` as in the RL agent.
    max_steps : int
        Episode horizon.
    n_video : int
        Record the exterior and wrist cameras of the first ``n_video`` envs.
    on_reset : callable, optional
        Called with the unwrapped env right after the reset.

    Returns
    -------
    tuple
        ``success`` (n,) bool, ``first_success_step`` (n,) int (-1 if none) and video frames
        ``[n_video, T, C, H / VIDEO_SCALE, W / VIDEO_SCALE]`` uint8 (None when ``n_video == 0``).
    """
    from flash_rl.rfcl.video_eval_robolab import paint_success_border
    from real2sim.render_contract import read_camera_tiles, render_cameras

    obs, _ = env.reset(random_start_init=False)  # synchronized start for all envs
    base = env.unwrapped
    if on_reset is not None:
        on_reset(base)
    n = int(env.num_envs)
    success = np.zeros(n, dtype=bool)
    first_step = np.full(n, -1, dtype=np.int64)
    ep_done = np.zeros(n, dtype=bool)
    frames: list[np.ndarray] = []
    last: np.ndarray | None = None
    for t in range(max_steps):
        actions = np.array(agent.sample_actions(0, prev_transition={"next_observation": obs}, training=False))
        obs, _rew, terms, truncs, _ = env.step(actions)
        done = np.asarray(terms).astype(bool) | np.asarray(truncs).astype(bool)
        held = base._succ_streak.detach().cpu().numpy() >= 1
        new = (~ep_done) & held & ~success
        first_step[new] = t + 1
        success |= new
        if n_video > 0:
            # A tile stops on done or VIDEO_SUCCESS_MARGIN frames after its first success.
            vid = slice(0, n_video)
            tile_end = (ep_done | done)[vid] | (success[vid] & (t + 1 - first_step[vid] >= VIDEO_SUCCESS_MARGIN))
            if not (bool(tile_end.all()) and last is not None):
                render_cameras(base)
                fr = read_camera_tiles(base, VIDEO_SENSORS, list(range(n_video))).transpose(0, 3, 1, 2)
                k, c, h, w = fr.shape
                fr = fr.reshape(k, c, h // VIDEO_SCALE, VIDEO_SCALE, w // VIDEO_SCALE, VIDEO_SCALE).mean(axis=(3, 5))
                fr = np.rint(fr).astype(np.uint8)
                if last is not None:
                    fr[tile_end] = last[tile_end]
                frames.append(fr)
                last = fr
        ep_done |= done
        if bool(ep_done.all()):
            break
    video = None
    if frames:
        for fr in frames:
            paint_success_border(fr, success[:n_video], width=3)
        video = np.stack(frames, axis=1)
    return success, first_step, video


def main() -> None:
    """Build the eval env and agent, then serve checkpoint eval requests until ``stop``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-yaml", required=True, help="config.yaml of the training run")
    ap.add_argument("--ipc-dir", required=True)
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--parent-pid", type=int, required=True, help="exit when the trainer process is gone")
    args = ap.parse_args()

    import json

    from omegaconf import OmegaConf

    from flash_rl.agents import create_agent
    from flash_rl.envs.robolab import env_kwargs_from_cfg, make_robolab_rl_env
    from flash_rl.envs.skill_z import SkillZConcatWrapper
    from flash_rl.rfcl.eval_request import shrink_replay_buffer_for_inference
    from flash_rl.rfcl.video_eval_robolab import tile_grid

    cfg = OmegaConf.load(args.config_yaml)
    env = make_robolab_rl_env(
        int(args.num_envs),
        int(args.seed),
        cameras=True,
        # Episodes run to the time limit so every env is measured over the same horizon.
        success_terminates=False,
        **env_kwargs_from_cfg(cfg),
    )
    if int(cfg.skill.z_dim) > 0:
        env = SkillZConcatWrapper(env, int(cfg.skill.z_dim), seed=int(args.seed))
    _, env_info = env.reset()

    cfg.agent.use_compile = False  # compile warmup is not worth it for eval
    shrink_replay_buffer_for_inference(cfg)
    agent = create_agent(env.observation_space, env.action_space, env_info, cfg.agent)
    max_steps = int(env.max_episode_steps)
    n_video = min(VIDEO_ENVS, int(args.num_envs))
    print(f"[eval-worker] ready: num_envs={env.num_envs} max_ep={max_steps} ipc={args.ipc_dir}", flush=True)

    ipc = args.ipc_dir
    while not os.path.exists(os.path.join(ipc, "stop")):
        if os.getppid() != int(args.parent_pid):
            print("[eval-worker] trainer is gone; exiting", flush=True)
            break
        reqs = sorted(
            (f for f in os.listdir(ipc) if f.startswith("request_") and f.endswith(".json")),
            key=lambda f: int(f.split("_")[1].split(".")[0]),
        )
        if not reqs:
            time.sleep(2.0)
            continue
        with open(os.path.join(ipc, reqs[-1])) as f:
            req = json.load(f)
        for r in reqs:  # evaluate the newest request; drop stale ones
            os.remove(os.path.join(ipc, r))

        t0 = time.time()
        agent.load_actor(str(req["ckpt_dir"]))
        success, first_step, video = run_clean_eval(env, agent, max_steps, n_video)
        env_step = int(req["env_step"])
        video_path = None
        if video is not None:
            video_path = os.path.join(ipc, f"video_{env_step}.mp4")
            write_video(video_path, tile_grid(video))
        result = {
            "env_step": env_step,
            "success_rate": float(success.mean()),
            # Mean time (s) to the first success over successful episodes; None if none succeeded.
            "time_to_success_s": float(first_step[success].mean() / CONTROL_HZ) if success.any() else None,
            "n_episodes": int(success.size),
            "video": video_path,
        }
        write_json_atomic(eval_result_path(ipc, env_step), result)
        print(
            f"[eval-worker] env_step={env_step} success={result['success_rate']:.3f} "
            f"time={result['time_to_success_s']} ({time.time() - t0:.1f}s)",
            flush=True,
        )

    sys.stdout.flush()
    os._exit(0)  # skip Isaac teardown (can hang)


if __name__ == "__main__":
    # Hard-exit on any exception: a normal shutdown can hang in Isaac's atexit teardown.
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
