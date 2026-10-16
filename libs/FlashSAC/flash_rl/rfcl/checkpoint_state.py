"""Reverse-curriculum state saved with each checkpoint and restored on resume."""

from __future__ import annotations

from typing import Any


def serialize_reverse_curriculum_state(train_env: Any) -> dict[str, object]:
    """Frontiers, solved flags and frontier-window outcomes of every demonstration."""
    metadata = list(train_env.demo_metadata)
    return {
        "frontier": [int(item.start_step) for item in metadata],
        "reverse_solved": [bool(item.solved) for item in metadata],
        "frontier_outcomes": train_env.core.frontier_state(),
    }


def restore_reverse_curriculum_state(train_env: Any, state: dict[str, Any]) -> None:
    """Restore the state written by ``serialize_reverse_curriculum_state``."""
    metadata = list(train_env.demo_metadata)
    frontier = [int(v) for v in state["frontier"]]
    solved = [bool(v) for v in state["reverse_solved"]]
    if len(frontier) != len(metadata) or len(solved) != len(metadata):
        raise RuntimeError(f"resume state covers {len(frontier)} demos, expected {len(metadata)}")
    for item, start_step, is_solved in zip(metadata, frontier, solved, strict=True):
        if not 0 <= start_step < int(item.total_steps):
            raise RuntimeError(f"resume frontier {start_step} outside [0, {int(item.total_steps) - 1}]")
        if is_solved and start_step != 0:
            raise RuntimeError(f"resume marks a nonzero frontier solved: frontier={start_step}")
        item.start_step = start_step
        item.solved = is_solved
    train_env.core.set_frontier_state(state["frontier_outcomes"])
    # The start-state pool was built from the frontiers before the restore.
    train_env.core.mark_active_dirty()
