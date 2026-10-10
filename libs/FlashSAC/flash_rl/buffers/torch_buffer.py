"""Uniform replay buffer backed by PyTorch tensors."""

import os
from collections import deque
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from flash_rl.buffers.base_buffer import BaseBuffer, Batch
from flash_rl.types import NDArray

# Mapping from numpy dtypes to torch dtypes
_NP_TO_TORCH_DTYPE: dict[np.dtype[Any], torch.dtype] = {
    np.dtype(np.float64): torch.float32,  # enforce float32
    np.dtype(np.float32): torch.float32,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.bool_): torch.bool,
    np.dtype(np.uint8): torch.uint8,
}


def _numpy_dtype_to_torch(dtype: Any) -> torch.dtype:
    """Convert a numpy dtype to a torch dtype, enforcing float32 for float64."""
    dtype = np.dtype(dtype)
    if dtype in _NP_TO_TORCH_DTYPE:
        return _NP_TO_TORCH_DTYPE[dtype]
    return torch.float32


# Rows per storage block. Blocks are allocated on first write, so memory follows the number of
# stored transitions (at most one block of slack) instead of ``max_length``.
BLOCK_ROWS = 1 << 24


class _BlockArray:
    """Rows ``[0, max_length)`` of one field, stored in fixed-size blocks allocated on first write."""

    def __init__(self, max_length: int, row_shape: tuple[int, ...], dtype: torch.dtype, device: torch.device):
        self._max_length = int(max_length)
        self._row_shape = row_shape
        self._dtype = dtype
        self._device = device
        self._pin = device.type == "cpu" and torch.cuda.is_available()
        self.blocks: list[torch.Tensor] = []

    def _ensure(self, end_row: int) -> None:
        while len(self.blocks) * BLOCK_ROWS < end_row:
            rows = min(BLOCK_ROWS, self._max_length - len(self.blocks) * BLOCK_ROWS)
            self.blocks.append(
                torch.zeros((rows,) + self._row_shape, dtype=self._dtype, device=self._device, pin_memory=self._pin)
            )

    def write(self, start: int, values: torch.Tensor) -> None:
        """Write ``values`` to rows ``[start, start + len(values))`` (no wrap-around)."""
        end = start + int(values.shape[0])
        self._ensure(end)
        pos = start
        while pos < end:
            b, off = divmod(pos, BLOCK_ROWS)
            n = min(end - pos, self.blocks[b].shape[0] - off)
            self.blocks[b][off : off + n] = values[pos - start : pos - start + n].to(self._dtype)
            pos += n

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        """Rows at ``idx`` (any rows below the number written)."""
        if len(self.blocks) == 1:
            return self.blocks[0][idx]
        bid = idx // BLOCK_ROWS
        off = idx % BLOCK_ROWS
        out = self.blocks[0][off]
        for b in range(1, len(self.blocks)):
            blk = self.blocks[b]
            sel = (bid == b).view((-1,) + (1,) * len(self._row_shape))
            out = torch.where(sel, blk[off.clamp(max=blk.shape[0] - 1)], out)
        return out

    def head(self, n: int) -> torch.Tensor:
        """CPU copy of rows ``[0, n)``."""
        parts, left = [], n
        for blk in self.blocks:
            if left <= 0:
                break
            parts.append(blk[:left].to("cpu", copy=True))
            left -= blk.shape[0]
        if not parts:
            return torch.empty((0,) + self._row_shape, dtype=self._dtype)
        return torch.cat(parts) if len(parts) > 1 else parts[0]


class TorchUniformBuffer(BaseBuffer):
    """A uniform experience replay buffer using PyTorch tensors.

    Data is stored on the given device in blocks of ``BLOCK_ROWS`` rows that are allocated as the
    buffer fills, so memory grows with the stored transitions up to ``max_length``.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
        device_type: str,
        store_ep_uid: bool = False,
    ):
        # store_ep_uid: also store a per-transition episode uid (used by the skill success gate).
        self._store_ep_uid = bool(store_ep_uid)
        super().__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            sample_batch_size,
        )
        device_type = (
            device_type
            if device_type.startswith("cuda") and ":" in device_type
            else ("cuda:0" if device_type.startswith("cuda") else "cpu")
        )
        self._device = torch.device(device_type)
        self.reset()

    def __len__(self) -> int:
        """Number of stored transitions."""
        return self._num_in_buffer

    def reset(self) -> None:
        """Clear storage, the n-step deque and indices; float64 spaces are stored as float32."""
        m, dev = self._max_length, self._device
        assert self._observation_space.shape is not None and self._action_space.shape is not None
        obs_shape = (self._observation_space.shape[-1],)
        obs_dtype = _numpy_dtype_to_torch(self._observation_space.dtype or np.float32)
        act_dtype = _numpy_dtype_to_torch(self._action_space.dtype or np.float32)
        self._fields: dict[str, _BlockArray] = {
            "observation": _BlockArray(m, obs_shape, obs_dtype, dev),
            "next_observation": _BlockArray(m, obs_shape, obs_dtype, dev),
            "action": _BlockArray(m, (self._action_space.shape[-1],), act_dtype, dev),
            "reward": _BlockArray(m, (), torch.float32, dev),
            "terminated": _BlockArray(m, (), torch.float32, dev),
            "truncated": _BlockArray(m, (), torch.float32, dev),
        }
        if self._store_ep_uid:
            self._fields["ep_uid"] = _BlockArray(m, (), torch.int32, dev)
        self._n_step_transitions: deque[dict[str, Any]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0

    def _to_tensor(self, value: Any) -> torch.Tensor:
        """Convert a value to a tensor on the buffer device (cloned if already a tensor)."""
        if isinstance(value, torch.Tensor):
            return value.detach().to(self._device, copy=True)
        return torch.tensor(value, device=self._device)

    def _get_n_step_prev_transition(self) -> Batch:
        """Compute the n-step return, done status, and next observation from the queue.

        Mirrors NpyUniformBuffer._get_n_step_prev_transition exactly.
        """
        n_step_prev_transition = self._n_step_transitions[0]
        curr_transition = self._n_step_transitions[-1]

        # clone last transition
        n_step_reward = curr_transition["reward"].clone()
        n_step_terminated = curr_transition["terminated"].clone()
        n_step_truncated = curr_transition["truncated"].clone()
        n_step_next_observation = curr_transition["next_observation"].clone()

        for n_step_idx in reversed(range(self._n_step - 1)):
            transition = self._n_step_transitions[n_step_idx]
            reward = transition["reward"]  # (n,)
            terminated = transition["terminated"]  # (n,)
            truncated = transition["truncated"]  # (n,)
            next_observation = transition["next_observation"]  # (n, *obs_shape)

            # compute n-step return
            done = (terminated.bool() | truncated.bool()).float()
            n_step_reward = reward + self._gamma * n_step_reward * (1 - done)

            # assign next observation starting from done
            done_mask = done.bool()
            n_step_terminated[done_mask] = terminated[done_mask]
            n_step_truncated[done_mask] = truncated[done_mask]
            n_step_next_observation[done_mask] = next_observation[done_mask]

        n_step_prev_transition["reward"] = n_step_reward
        n_step_prev_transition["terminated"] = n_step_terminated
        n_step_prev_transition["truncated"] = n_step_truncated
        n_step_prev_transition["next_observation"] = n_step_next_observation

        return cast(Batch, n_step_prev_transition)

    def add(self, transition: Batch) -> None:
        """Push a transition to the n-step queue and write the n-step result once the queue is full."""
        self._n_step_transitions.append({key: self._to_tensor(value) for key, value in transition.items()})
        if len(self._n_step_transitions) < self._n_step:
            return
        n_step_prev_transition = cast(dict[str, torch.Tensor], self._get_n_step_prev_transition())
        add_batch_size = len(n_step_prev_transition["observation"])
        # Rows wrap around at max_length; split the write at the end.
        first = min(add_batch_size, self._max_length - self._current_idx)
        for key, arr in self._fields.items():
            # ep_uid is the queue head's; constant within an episode.
            values = n_step_prev_transition[key]
            arr.write(self._current_idx, values[:first])
            if first < add_batch_size:
                arr.write(0, values[first:])
        self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
        self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def can_sample(self) -> bool:
        """True once at least ``min_length`` transitions are stored."""
        return self._num_in_buffer >= self._min_length

    def sample(self, sample_idxs: NDArray | None = None) -> Batch:
        """Sample a batch at ``sample_idxs``, or uniformly at random if None."""
        if sample_idxs is None:
            idxs = torch.randint(0, self._num_in_buffer, (self._sample_batch_size,), device=self._device)
        else:
            idxs = torch.as_tensor(sample_idxs, device=self._device)
        return {key: arr.gather(idxs) for key, arr in self._fields.items()}

    def save(self, path: str) -> None:
        """Save the stored rows and indices to ``path`` (file size follows the number of stored rows)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dataset: dict[str, Any] = {key: arr.head(self._num_in_buffer) for key, arr in self._fields.items()}
        dataset["num_in_buffer"] = self._num_in_buffer
        dataset["current_idx"] = self._current_idx
        torch.save(dataset, path)

    def load(self, path: str) -> None:
        """Load rows and indices saved by ``save``.

        The in-flight n-step queue (at most ``n_step - 1`` transitions) is not saved.
        """
        dataset = torch.load(path, map_location=self._device)
        for key, arr in self._fields.items():
            arr.write(0, dataset[key])
        self._num_in_buffer = int(dataset["num_in_buffer"])
        self._current_idx = int(dataset["current_idx"])
        self._n_step_transitions.clear()

    def get_observations(self) -> torch.Tensor:
        """Return the stored observations (CPU copy)."""
        return self._fields["observation"].head(self._num_in_buffer)
