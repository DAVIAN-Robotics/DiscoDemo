"""Evaluate a pi0.5 checkpoint in Isaac Sim (the paper's SFT evaluation).

One episode in each of 50 parallel envs from the task's initial-state distribution, run to the task horizon.
Reported: success rate (the success predicate holds at any step) and the mean time to the first success.
The policy sees the exterior and wrist cameras through the same lens and photometric pipeline that rendered
the training data, and commands absolute joint targets.

Writes ``<out-dir>/result.json`` and ``<out-dir>/video.mp4`` (first 4 episodes, exterior | wrist). With
``--episode-videos`` it also writes one video per episode and ``episodes.json`` (per-episode success, time and
initial object poses) to ``<out-dir>/episodes/``.

Usage
-----
    python -m sft.sim_eval <checkpoint> --task pnp_banana --seed 1000 --out-dir <dir>

``<checkpoint>`` is a ``pretrained_model`` directory (``<output_dir>/checkpoints/last/pretrained_model``) or a
hub repo such as ``DAVIAN-Robotics/DiscoDemo-Stage3_SFT-...``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "libs/FlashSAC/configs"
#: Evaluation episodes (one per env).
NUM_EPISODES = 50
#: Dataset camera -> Isaac sensor (the dataset keeps the DROID camera names).
EXTERIOR_SENSOR = "left_back_cam"
WRIST_SENSOR = "wrist_cam"


def task_env_kwargs(task: str) -> dict[str, Any]:
    """Env arguments of ``task`` for pi0.5: the task's plant with absolute joint-position targets.

    Parameters
    ----------
    task : str
        ``pnp_banana`` | ``stack_cube`` | ``fmb_round`` | ``fmb_sqcircle``.

    Returns
    -------
    dict[str, Any]
        Keyword arguments for ``make_robolab_rl_env``.
    """
    from omegaconf import OmegaConf

    base = OmegaConf.load(CONFIGS / "fr3.yaml")
    t = OmegaConf.load(CONFIGS / "task" / f"{task}.yaml")
    return {
        "task": str(REPO / t.env.task),
        "env_version": str(t.env.env_version),
        "workcell": str(REPO / base.env.workcell),
        "init_pose": str(t.rfcl.init_pose),
        "success_hold_steps": int(base.success_hold_steps),
    }


class Pi05Agent:
    """Adapts ``Pi05Policy`` to the RL agent interface used by ``run_clean_eval``.

    Reads the cameras and the arm state from the env and maps the absolute joint targets to the env's
    normalized action in ``[-1, 1]^8``.
    """

    def __init__(self, base: Any, policy: Any):
        from robolab.robots.rl_obs import NormalizedJointPositionAction

        self.base = base
        self.policy = policy
        body = base.action_manager.get_term("body")
        assert isinstance(body, NormalizedJointPositionAction), type(body).__name__
        # The term maps a in [-1, 1] to the physical target a * scale + offset.
        self.scale = body._scale[0].detach().cpu().numpy()
        self.offset = body._offset[0].detach().cpu().numpy()

    def sample_actions(self, step: int, prev_transition: Any, training: bool) -> np.ndarray:
        """``[N, 8]`` normalized env action for the current scene."""
        from real2sim.render_contract import read_camera_tiles, render_cameras
        from robolab.robots.action_norm import ARM_J7_INDEX, PANDA_J7_MOUNT_OFFSET
        from robolab.robots.droid import arm_joint_pos_panda, gripper_pos_panda

        b = self.base
        render_cameras(b)
        idx = list(range(b.num_envs))
        exterior = read_camera_tiles(b, [EXTERIOR_SENSOR], idx)
        wrist = read_camera_tiles(b, [WRIST_SENSOR], idx)
        state = np.concatenate([arm_joint_pos_panda(b).cpu().numpy(), gripper_pos_panda(b).cpu().numpy()], axis=1)
        out = self.policy.act(exterior, wrist, state)  # [N, 8] command-frame joints + gripper {0, 1}
        q = out[:, :7].copy()
        q[:, ARM_J7_INDEX] += PANDA_J7_MOUNT_OFFSET  # command frame -> physical joint7
        arm = (q - self.offset) / self.scale
        return np.concatenate([arm, 2.0 * out[:, 7:8] - 1.0], axis=1).astype(np.float32)


def evaluate(checkpoint: str, task: str, seed: int, out_dir: Path, num_envs: int, episode_videos: bool) -> dict:
    """Run the evaluation and write ``result.json`` (and the videos) to ``out_dir``."""
    from flash_rl.envs.robolab import make_robolab_rl_env
    from flash_rl.rfcl.eval_request import CONTROL_HZ
    from flash_rl.rfcl.eval_worker_robolab import VIDEO_ENVS, run_clean_eval, write_video
    from flash_rl.rfcl.video_eval_robolab import tile_grid
    from sft.pi05_policy import Pi05Policy

    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_robolab_rl_env(
        num_envs, seed, cameras=True, success_terminates=False, to_numpy=True, **task_env_kwargs(task)
    )
    env.reset()
    base = env.unwrapped
    policy = Pi05Policy(checkpoint, device=str(base.device), prompt=str(base.cfg.instruction))
    agent = Pi05Agent(base, policy)
    policy.reset()

    init: dict[str, Any] = {}

    def record_init(b: Any) -> None:
        """Initial object poses ``[x, y, z, qw, qx, qy, qz]`` (env frame) of every episode."""
        o = b.scene.env_origins
        for name, obj in b.scene.rigid_objects.items():
            pose = obj.data.root_state_w[:, :7].clone()
            pose[:, :3] -= o
            init[name] = pose.cpu().numpy().round(5).tolist()

    n_video = num_envs if episode_videos else min(VIDEO_ENVS, num_envs)
    success, first_step, video = run_clean_eval(env, agent, int(env.max_episode_steps), n_video, on_reset=record_init)
    assert video is not None
    write_video(str(out_dir / "video.mp4"), tile_grid(video[:VIDEO_ENVS]))
    if episode_videos:
        ep_dir = out_dir / "episodes"
        ep_dir.mkdir(exist_ok=True)
        for k in range(num_envs):
            write_video(str(ep_dir / f"ep{k:02d}_{'success' if success[k] else 'failure'}.mp4"), video[k])
        episodes = [
            {
                "success": bool(success[k]),
                "time_to_success_s": float(first_step[k] / CONTROL_HZ) if success[k] else None,
                "init": {name: poses[k] for name, poses in init.items()},
            }
            for k in range(num_envs)
        ]
        (ep_dir / "episodes.json").write_text(json.dumps(episodes, indent=1))
    result = {
        "checkpoint": checkpoint,
        "task": task,
        "seed": seed,
        "n_episodes": int(num_envs),
        "success_rate": float(success.mean()),
        # Mean time (s) to the first success over successful episodes; None if none succeeded.
        "time_to_success_s": float(first_step[success].mean() / CONTROL_HZ) if success.any() else None,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(f"[sim_eval] {json.dumps(result)}", flush=True)
    return result


def main() -> None:
    """CLI entry point."""
    from real2sim.env_contract import TASKS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", help="pretrained_model directory or hub repo id")
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--seed", type=int, required=True, help="the paper evaluates with the training seed")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, default=NUM_EPISODES)
    ap.add_argument("--episode-videos", action="store_true", help="one video per episode + initial poses")
    a = ap.parse_args()
    evaluate(a.checkpoint, a.task, a.seed, a.out_dir, a.num_envs, a.episode_videos)


if __name__ == "__main__":
    # Exit with os._exit: Isaac Sim's atexit teardown can hang.
    code = 0
    try:
        main()
    except SystemExit as e:  # argparse --help / usage errors
        code = e.code if isinstance(e.code, int) else int(e.code is not None)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
