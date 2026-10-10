r"""Trajectory embedding and Vendi score shared by the diversity and coverage metrics.

Follows the trajectory representation of PGDG (arXiv:2605.21710) without its action channel:

* Per-step feature ``psi(s_t)`` = manipulated-object pose (position 3 + quaternion 4) and ``panda_hand`` pose
  (3 + 4); quaternions are sign-canonicalized to ``w >= 0``.
* Features are z-scored per task (statistics over all sources of the task).
* Each trajectory is linearly resampled to 64 steps and transformed per dimension with an orthonormal DCT-II;
  the 8 coefficients after DC are kept (112 dims).
* RBF kernel ``exp(-gamma * d^2)`` with ``gamma = 1 / median(d^2)`` over the task pool.
* Vendi score ``exp(-sum_i lambda_i log lambda_i)`` of ``K / m``, averaged over random subsets of size ``m``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.fft import dct

Arr = npt.NDArray[np.float64]
N_RESAMPLE = 64
K_DCT = 8


def canon_quat(q: Arr) -> Arr:
    """Flip ``[T, 4]`` wxyz quaternion signs so that ``w >= 0``."""
    out: Arr = q * np.where(q[:, :1] < 0, -1.0, 1.0)
    return out


def _resample(a: Arr) -> Arr:
    """Linearly resample the time axis of ``[T, D]`` to ``N_RESAMPLE`` points."""
    u = np.linspace(0.0, 1.0, len(a))
    g = np.linspace(0.0, 1.0, N_RESAMPLE)
    return np.stack([np.interp(g, u, a[:, j]) for j in range(a.shape[1])], axis=1)


def embed(x: Arr, mu: Arr, sd: Arr) -> Arr:
    """Normalize ``[T, D]``, resample, DCT-II and flatten the ``K_DCT`` coefficients after DC (``[K_DCT * D]``)."""
    c = dct(_resample((x - mu) / sd), axis=0, norm="ortho")
    out: Arr = c[1 : K_DCT + 1].reshape(-1)
    return out


def median_gamma(pool: Arr) -> float:
    """RBF width ``1 / median`` of the pairwise squared distances of ``[n, d]`` embeddings."""
    d2 = ((pool[:, None] - pool[None]) ** 2).sum(-1)
    return 1.0 / float(np.median(d2[np.triu_indices(len(pool), 1)]))


def vendi(e: Arr, gamma: float, m: int, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    """Vendi score (Friedman & Dieng, TMLR 2023) of ``[n, d]`` embeddings, averaged over ``reps`` subsets of size ``m``.

    Returns
    -------
    tuple of float
        (mean, std); ``(nan, nan)`` if ``n < m``.
    """
    if len(e) < m:
        return float("nan"), float("nan")
    lk = np.exp(-gamma * ((e[:, None] - e[None]) ** 2).sum(-1))
    v = []
    for _ in range(reps):
        i = rng.choice(len(e), m, replace=False)
        lam = np.clip(np.linalg.eigvalsh(lk[np.ix_(i, i)] / m), 0.0, None)
        lam = lam[lam > 1e-12]
        v.append(float(np.exp(-np.sum(lam * np.log(lam)))))
    return float(np.mean(v)), float(np.std(v))
