r"""Fixed-scene diversity: Vendi score of successful rollouts from one initial scene, per data source.

Uses the embedding and Vendi score of ``vendi.py``. Inputs are rollout directories of ``ep_*.npz`` files written
by ``libs/real2sim/scripts/rollout_fixed_init.py`` (keys ``succ_streak``, ``state/obj/<object>/{pos,quat}``,
``state/art/robot/{body_pos,body_quat}`` and ``meta/art/robot/body_names``). A rollout is successful when
``succ_streak`` reaches the success hold; it is cut at its first success (the first step of that hold). The
first ``--limit`` successful rollouts (file order) of each source are used.

Spec (JSON; relative paths are resolved against the spec file, entries with ``*?[`` are globbed and sorted)::

    {"tasks": {"pnp_banana": {"object": "banana",
                              "sources": {"discodemo": {"dirs": ["rollouts/discodemo"]},
                                          "prfcl": {"dirs": ["rollouts/prfcl"]}}}}}

Usage::

    python scripts/analysis/diversity_vendi.py --spec spec.json --out diversity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vendi import Arr, canon_quat, embed, median_gamma, vendi  # noqa: E402


def select(dirs: list[Path], obj: str, hold: int, limit: int) -> list[Arr]:
    """Load at most ``limit`` successful rollouts as ``[T, 14]`` features (object pose, hand pose)."""
    out: list[Arr] = []
    for d in dirs:
        assert d.is_dir(), f"missing rollout directory {d}"
        for f in sorted(d.glob("ep_*.npz")):
            z = np.load(f)
            hit = np.flatnonzero(np.asarray(z["succ_streak"]) >= hold)
            if not len(hit):
                continue
            # First step of the success hold that completed.
            end = max(2, int(hit[0]) - (hold - 1) + 1)
            h = list(z["meta/art/robot/body_names"]).index("panda_hand")
            n = min(end, len(z["actions"]))
            x = np.concatenate(
                [
                    z[f"state/obj/{obj}/pos"][:n],
                    canon_quat(z[f"state/obj/{obj}/quat"][:n]),
                    z["state/art/robot/body_pos"][:n, h],
                    canon_quat(z["state/art/robot/body_quat"][:n, h]),
                ],
                axis=1,
            ).astype(np.float64)
            out.append(x)
            if len(out) >= limit:
                return out
    return out


def _resolve(entries: list[str], base: Path) -> list[Path]:
    """Resolve spec paths relative to ``base``; glob patterns are expanded in sorted order."""
    out: list[Path] = []
    for s in entries:
        p = Path(s) if Path(s).is_absolute() else base / s
        if any(ch in s for ch in "*?["):
            out += sorted(Path(p.anchor).glob(str(p.relative_to(p.anchor))))
        else:
            out.append(p)
    return out


def main() -> None:
    """Compute the per-task, per-source diversity table and write it as JSON."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True, help="input spec (JSON)")
    ap.add_argument("--out", type=Path, required=True, help="output JSON")
    ap.add_argument("--m", type=int, default=20, help="subset size")
    ap.add_argument("--reps", type=int, default=200, help="number of random subsets")
    ap.add_argument("--limit", type=int, default=24, help="successful rollouts per source")
    ap.add_argument("--success-hold-steps", type=int, default=20)
    a = ap.parse_args()

    spec = json.loads(a.spec.read_text())
    base = a.spec.resolve().parent
    rng = np.random.default_rng(0)
    res: dict[str, Any] = {"m": a.m, "reps": a.reps, "limit": a.limit, "tasks": {}}
    for task, tspec in spec["tasks"].items():
        obj = str(tspec["object"])
        trajs = {
            s: select(_resolve(list(v["dirs"]), base), obj, a.success_hold_steps, a.limit)
            for s, v in tspec["sources"].items()
        }
        allx = np.concatenate([x for v in trajs.values() for x in v])
        mu, sd = allx.mean(0), allx.std(0) + 1e-8
        emb = {s: np.stack([embed(x, mu, sd) for x in v]) for s, v in trajs.items()}
        gamma = median_gamma(np.concatenate(list(emb.values())))
        row: dict[str, Any] = {"object": obj, "gamma": gamma, "src": {}}
        for s, e in emb.items():
            vs, vsd = vendi(e, gamma, a.m, a.reps, rng)
            row["src"][s] = {"n": len(e), "vendi": vs, "vendi_sd": vsd}
            print(f"{task:14s} {s:12s} n={len(e):2d} vendi={vs:.2f} +- {vsd:.2f}")
        res["tasks"][task] = row
    a.out.write_text(json.dumps(res, indent=1))
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
