r"""Dataset coverage: Vendi score of trajectories sampled from a whole generated dataset (diverse initial states).

Uses the embedding and Vendi score of ``vendi.py`` on ``--n-traj`` randomly chosen trajectories per (task,
source). The hand pose is computed from the joint positions with forward kinematics (``_fk.py``). Each
trajectory is cut at its frame flagged successful (``success``).

Inputs are collected states: every ``traj_*`` group of every ``*.h5`` below the source directory (for a
generated dataset, ``<out>/shards`` of ``scripts/2_gendata/generate.sh``, which holds only the exported
trajectories), with ``states/objects/<object>/root_pose``, ``states/robot/joint_pos`` and ``success``.

Spec (JSON; relative paths are resolved against the spec file)::

    {"tasks": {"pnp_banana": {"object": "banana",
                              "sources": {"discodemo": {"pool": "gen/discodemo/shards"},
                                          "prfcl": {"pool": "gen/prfcl/shards"}}}}}

Trajectory sampling is seeded by the position of the task and source in the spec, so keep the order fixed.

Usage::

    python scripts/analysis/coverage_vendi.py --spec spec.json --out coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fk import ee_pose_many  # noqa: E402
from vendi import Arr, canon_quat, embed, median_gamma, vendi  # noqa: E402

#: Trajectories per source used to fit the kernel width.
GAMMA_POOL = 300


def _pool(d: Path) -> list[tuple[Path, int]]:
    """All ``(h5 path, traj index)`` pairs below ``d`` (recursive, sorted)."""
    out = []
    for f in sorted(d.rglob("*.h5")):
        with h5py.File(f, "r") as h:
            out += [(f, int(k.split("_")[1])) for k in h if k.startswith("traj_")]
    assert out, f"no trajectories below {d}"
    return out


def _trajs(items: list[tuple[Path, int]], obj: str) -> list[Arr]:
    """``[T, 14]`` features (object pose, hand pose) of the given trajectories."""
    poses: list[Arr] = []
    qs: list[Arr] = []
    by: dict[Path, list[int]] = {}
    for f, t in items:
        by.setdefault(f, []).append(t)
    for f, ts in by.items():
        with h5py.File(f, "r") as h:
            for t in ts:
                g = h[f"traj_{t}"]
                p = np.asarray(g[f"states/objects/{obj}/root_pose"], dtype=np.float64)
                end = max(2, int(np.argmax(np.asarray(g["success"]).astype(bool))) + 1)
                poses.append(np.concatenate([p[:end, :3], canon_quat(p[:end, 3:7])], 1))
                qs.append(np.asarray(g["states/robot/joint_pos"], dtype=np.float64)[:end])
    return [np.concatenate([p, ee], 1) for p, ee in zip(poses, ee_pose_many(qs), strict=True)]


def main() -> None:
    """Task x source coverage table."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True, help="input spec (JSON)")
    ap.add_argument("--out", type=Path, required=True, help="output JSON")
    ap.add_argument("--n-traj", type=int, default=1000, help="trajectories per source")
    ap.add_argument("--m", type=int, default=200, help="subset size")
    ap.add_argument("--reps", type=int, default=100, help="random subsets per Vendi estimate")
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text())
    base = a.spec.resolve().parent
    res: dict[str, Any] = {"n_traj": a.n_traj, "m": a.m, "reps": a.reps, "tasks": {}}
    for ti, (t, tspec) in enumerate(spec["tasks"].items()):
        rng = np.random.default_rng([1, ti])
        tr: dict[str, list[Arr]] = {}
        for si, (s, src) in enumerate(tspec["sources"].items()):
            pool = Path(src["pool"]) if Path(src["pool"]).is_absolute() else base / src["pool"]
            items = _pool(pool)
            pick = np.random.default_rng([0, ti, si]).permutation(len(items))[: a.n_traj]
            tr[s] = _trajs([items[i] for i in pick], str(tspec["object"]))
        allx = np.concatenate([x for v in tr.values() for x in v])
        mu, sd = allx.mean(0), allx.std(0) + 1e-8
        emb = {s: np.stack([embed(x, mu, sd) for x in v]) for s, v in tr.items()}
        gamma = median_gamma(
            np.concatenate([e[rng.choice(len(e), min(GAMMA_POOL, len(e)), replace=False)] for e in emb.values()])
        )
        row: dict[str, Any] = {}
        for s, e in emb.items():
            vs, vsd = vendi(e, gamma, a.m, a.reps, rng)
            row[s] = {"n": len(e), "vendi": vs, "vendi_sd": vsd}
            print(f"{t:14s} {s:12s} n={len(e):4d} vendi={vs:.2f} +- {vsd:.2f}", flush=True)
        res["tasks"][t] = row
    a.out.write_text(json.dumps(res, indent=1))
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
