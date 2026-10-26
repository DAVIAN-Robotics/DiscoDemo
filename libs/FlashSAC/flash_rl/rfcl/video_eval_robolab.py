"""Video helpers for evaluation rollouts: success border and tile grid."""

from __future__ import annotations

import numpy as np


def paint_success_border(fr: np.ndarray, success: np.ndarray, width: int) -> None:
    """Paint a green border in place on tiles where ``success[i]`` is true (``fr [N, 3, H, W]`` uint8)."""
    green = np.array([0, 200, 0], dtype=np.uint8)[:, None, None]  # [C,1,1]
    for i in np.nonzero(np.asarray(success))[0]:
        f = fr[i]
        f[:, :width, :] = green
        f[:, -width:, :] = green
        f[:, :, :width] = green
        f[:, :, -width:] = green


def tile_grid(frames: np.ndarray) -> np.ndarray:
    """Tile ``[N, T, C, H, W]`` into one video ``[T, C, nrows*H, ncols*W]``.

    ``ncols = ceil(sqrt(N))``, ``nrows = ceil(N / ncols)``; N=3 or 4 gives a 2x2 grid.
    """
    n, t, c, h, w = frames.shape
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    grid = np.zeros((t, c, nrows * h, ncols * w), dtype=frames.dtype)
    for i in range(n):
        r, col = divmod(i, ncols)
        grid[:, :, r * h : (r + 1) * h, col * w : (col + 1) * w] = frames[i]
    return grid
