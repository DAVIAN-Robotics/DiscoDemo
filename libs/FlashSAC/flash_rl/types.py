"""Shared type aliases (numpy / torch arrays)."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = NDArray | torch.Tensor
