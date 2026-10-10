import numpy as np
import pytest
import torch

datasets = pytest.importorskip("datasets")
from lerobot.processor.core import TransitionKey  # noqa: E402

from sft.action_delta import DELTA_MASK, ActionDeltaProcessorStep, compute_delta_action_stats  # noqa: E402


def test_step_subtracts_state_from_joints_only():
    action = torch.arange(2 * 3 * 8, dtype=torch.float32).view(2, 3, 8)
    state = torch.ones(2, 8) * 10
    out = ActionDeltaProcessorStep()(
        {TransitionKey.ACTION: action, TransitionKey.OBSERVATION: {"observation.state": state}}
    )
    a = out[TransitionKey.ACTION]
    assert torch.equal(a[..., :7], action[..., :7] - 10)
    assert torch.equal(a[..., 7], action[..., 7])
    assert ActionDeltaProcessorStep(**ActionDeltaProcessorStep().get_config()).mask == list(DELTA_MASK)


def test_delta_stats():
    T = 6
    act = np.tile(np.arange(T, dtype=np.float32)[:, None], (1, 8))
    st = np.zeros((T, 8), dtype=np.float32)
    hf = datasets.Dataset.from_dict(
        {"action": act, "observation.state": st, "episode_index": np.zeros(T, dtype=np.int64)}
    )

    class Meta:
        stats = {"action": {"mean": np.zeros(8, np.float32), "max": np.zeros(8, np.float32), "count": np.array([T])}}

    class DS:
        hf_dataset = hf
        meta = Meta()

    s = compute_delta_action_stats(DS(), horizon=2, offset=0)
    # Pairs (a[t + j] - s[t]) for j in {0, 1} inside the episode: values 0..5 and 1..5.
    assert np.isclose(s["mean"][0], (sum(range(6)) + sum(range(1, 6))) / 11)
    assert s["max"][0] == 5.0
    assert s["count"].tolist() == [T]
