"""File IPC between the trainer and the eval worker (``eval_worker_robolab.py``).

The trainer writes ``request_{env_step}.json``; the worker answers with ``result_{env_step}.json``
and exits when ``stop`` appears. Isaac-free so it can be unit tested.
"""

from __future__ import annotations

import json
import os
from typing import Any

# Control rate of the FR3 tasks (Hz).
CONTROL_HZ = 20.0
# Rows the eval worker preallocates for the (unused) replay buffer of its agent.
INFERENCE_BUFFER_ROWS = 1024


def eval_request_path(ipc_dir: str, env_step: int) -> str:
    """Return ``{ipc_dir}/request_{env_step}.json``."""
    return os.path.join(ipc_dir, f"request_{int(env_step)}.json")


def eval_result_path(ipc_dir: str, env_step: int) -> str:
    """Return ``{ipc_dir}/result_{env_step}.json``."""
    return os.path.join(ipc_dir, f"result_{int(env_step)}.json")


def write_json_atomic(path: str, obj: dict[str, Any]) -> None:
    """Write ``obj`` through a temporary file and ``os.replace``."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def write_eval_request(ipc_dir: str, env_step: int, ckpt_dir: str) -> None:
    """Ask the worker to evaluate ``ckpt_dir`` (results are logged at ``env_step``)."""
    write_json_atomic(eval_request_path(ipc_dir, env_step), {"env_step": int(env_step), "ckpt_dir": str(ckpt_dir)})


def eval_harvest_deadline_s(max_episode_steps: int, base_s: float = 600.0, n_evals: float = 2.0) -> float:
    """Longest wait for the final result: ``base_s`` plus ``n_evals`` episode lengths of simulated time."""
    return base_s + n_evals * max_episode_steps / CONTROL_HZ


def shrink_replay_buffer_for_inference(cfg: Any) -> None:
    """Shrink the replay buffer of an inference-only agent config (in place)."""
    cfg.agent.buffer_max_length = INFERENCE_BUFFER_ROWS
    cfg.agent.buffer_min_length = INFERENCE_BUFFER_ROWS
