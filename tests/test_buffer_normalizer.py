import torch

from flash_rl.agents.utils.normalizer import Normalizer
from flash_rl.buffers import torch_buffer


def test_block_array_across_blocks(monkeypatch):
    monkeypatch.setattr(torch_buffer, "BLOCK_ROWS", 4)
    arr = torch_buffer._BlockArray(10, (2,), torch.float32, torch.device("cpu"))
    data = torch.arange(20, dtype=torch.float32).view(10, 2)
    arr.write(0, data[:3])
    arr.write(3, data[3:9])
    assert len(arr.blocks) == 3 and arr.blocks[-1].shape[0] == 2  # last block holds only rows 8-9
    idx = torch.tensor([0, 3, 4, 7, 8, 5])
    assert torch.equal(arr.gather(idx), data[idx])
    assert torch.equal(arr.head(9), data[:9])


def test_normalizer_bounds_and_clamp():
    n = Normalizer(2)
    n.set_obs_bounds(torch.tensor([0.0, -2.0]), torch.tensor([1.0, 2.0]))
    y = n.normalize_obs(torch.tensor([[0.5, 2.0], [3.0, -4.0]]))
    assert torch.allclose(y, torch.tensor([[0.0, 1.0], [1.0, -1.0]]), atol=1e-5)
