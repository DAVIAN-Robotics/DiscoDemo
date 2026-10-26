"""Per-episode skill-z invariant check (pure numpy, no Isaac dependency, so it is unit-testable)."""

from __future__ import annotations

import numpy as np


def resolve_episode_skill_z(z_rows: list, env_id: int) -> np.ndarray | None:
    """Check that all z rows of an episode are identical and return the representative z.

    z must be constant within an episode (the skill-z wrapper resamples only on done). A change means
    the episode boundary and the resample point are misaligned, so this raises instead of silently
    using the first value.

    Args:
        z_rows: z captured at every frame of the episode (empty if z is not used).
        env_id: env id reported in the error message.

    Returns:
        The episode z (float32), or None if ``z_rows`` is empty.
    """
    if not z_rows:
        return None
    z0 = np.asarray(z_rows[0], dtype=np.float32)
    for t, zt in enumerate(z_rows[1:], start=1):
        if not np.array_equal(np.asarray(zt, dtype=np.float32), z0):
            raise ValueError(
                f"skill_z changed within an episode (env {env_id}, frame {t}): {z0} -> {zt}. "
                "Episode boundary and z resampling are misaligned; check the collection loop."
            )
    return z0
