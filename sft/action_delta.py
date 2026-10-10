"""Chunk-relative delta action labels for pi0.5 SFT (as in openpi ``pi05_droid``).

pi0.5 DROID checkpoints predict normalized joint deltas from the current state; the gripper stays absolute.
Training labels are converted before the normalizer, and the action normalizer uses delta statistics::

    train:      delta[..., :7] = action[..., :7] - state[..., :7]
    inference:  action[..., :7] = delta[..., :7] + state[..., :7]
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

#: 7 delta joints followed by the absolute gripper.
DELTA_MASK: tuple[bool, ...] = (True,) * 7 + (False,)
#: Frames (in dataset order) used for the delta-action statistics.
DELTA_STATS_FRAMES = 300_000


def _n_delta_dims(mask: tuple[bool, ...]) -> int:
    """Number of delta dims; the delta dims must come first."""
    m = np.asarray(mask, dtype=bool)
    nd = int(m.sum())
    if not bool(m[:nd].all()):
        raise ValueError(f"delta mask must be prefix-True (delta joints first), got {mask}")
    return nd


def policy_action_window(policy_cfg: Any) -> tuple[int, int]:
    """Action label window ``(length, offset)`` of the policy (offset = first action relative to now)."""
    idx = list(policy_cfg.action_delta_indices)
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError(f"action_delta_indices is not contiguous: {idx}")
    return len(idx), idx[0]


# The registry name is stored in the preprocessor of every SFT checkpoint.
@ProcessorStepRegistry.register(name="action_delta_processor_step")
@dataclass
class ActionDeltaProcessorStep(ProcessorStep):
    """Absolute joint action chunk -> delta from the current state. Placed right before the normalizer.

    Inference transitions carry no action, so the step is a no-op there.
    """

    mask: tuple[bool, ...] = field(default_factory=lambda: DELTA_MASK)
    state_key: str = OBS_STATE

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Subtract the current state from the delta dims of the action chunk."""
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        state = transition[TransitionKey.OBSERVATION][self.state_key]  # (..., dim)
        # action: (..., chunk, dim).
        if state.ndim != action.ndim - 1 or state.shape[:-1] != action.shape[:-2]:
            raise ValueError(f"state{tuple(state.shape)} does not match action{tuple(action.shape)}")
        nd = _n_delta_dims(self.mask)
        action = action.clone()
        action[..., :nd] = action[..., :nd] - state[..., :nd].unsqueeze(-2)
        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Shapes are unchanged."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Serialized step arguments."""
        return {"mask": [bool(x) for x in self.mask], "state_key": self.state_key}


def compute_delta_action_stats(dataset: Any, horizon: int, offset: int) -> dict[str, np.ndarray]:
    """Action statistics after the same delta transform as ``ActionDeltaProcessorStep``.

    Uses the first ``DELTA_STATS_FRAMES`` frames in dataset order (action/state columns only, no video
    decoding): ``delta[t, j] = action[t + j] - state[t]`` for ``j`` in ``[offset, offset + horizon)`` within each
    episode. Every key of the dataset's action stats is recomputed (``q<NN>`` as quantiles), except ``count``.

    Parameters
    ----------
    dataset : LeRobotDataset
        Training dataset.
    horizon, offset : int
        Action window of the policy (``policy_action_window``).

    Returns
    -------
    dict[str, numpy.ndarray]
        Replacement for ``dataset.meta.stats["action"]``.
    """
    nd = _n_delta_dims(DELTA_MASK)
    hf = dataset.hf_dataset.with_format("numpy")
    n = min(len(hf), DELTA_STATS_FRAMES)
    cols = hf.select(range(n)) if n < len(hf) else hf
    act = np.asarray(cols["action"], dtype=np.float64)  # (n, action_dim)
    st = np.asarray(cols[OBS_STATE], dtype=np.float64)  # (n, state_dim)
    ep = np.asarray(cols["episode_index"]).reshape(-1)

    deltas: list[np.ndarray] = []
    for e in np.unique(ep):
        idx = np.nonzero(ep == e)[0]
        a, s, T = act[idx], st[idx], len(idx)
        for j in range(offset, offset + horizon):
            # Pairs (action[t + j], state[t]) inside the episode.
            lo, hi = max(j, 0), min(T + j, T)
            if hi <= lo:
                continue
            d = a[lo:hi].copy()
            d[:, :nd] -= s[lo - j : hi - j, :nd]
            deltas.append(d)
    D = np.concatenate(deltas, axis=0)  # (N, action_dim)

    orig = dataset.meta.stats["action"]
    reducers = {"mean": np.mean, "std": np.std, "min": np.min, "max": np.max}
    out: dict[str, np.ndarray] = {}
    for key, ref in orig.items():
        if key in reducers:
            v = reducers[key](D, axis=0)
        elif key.startswith("q") and key[1:].isdigit():
            v = np.quantile(D, int(key[1:]) / 100.0, axis=0)
        else:
            v = np.asarray(ref)
        out[key] = v.astype(np.asarray(ref).dtype)
    return out
