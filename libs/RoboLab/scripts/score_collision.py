"""Safety signals and the export filter for collected RL rollouts (CPU only, no simulator).

Two subcommands:

* ``score`` -- reads the ``episode_*.h5`` written by ``collect_rl_states.py`` and computes the safety
  signals of every trajectory (``collision_metrics.trajectory_signals``: contact peak / sustained contact,
  jam force, illegal contact IC and container displacement CD).
* ``select`` -- the export filter (``collision_metrics.export_filter_drops``), then ``--n-keep`` trajectories
  taken in order from one fixed random permutation of the pool (``--seed``).

usage::

    python libs/RoboLab/scripts/score_collision.py score --states-dir <states_dir> --out scores.json
    python libs/RoboLab/scripts/score_collision.py select --scores scores.json --n-keep 3000 --seed 0 --out keep.json

The ``select`` output is the ``--keep-json`` input of ``repartition_states_shards.py``.
"""

import argparse
import glob
import json
import os
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from flash_rl.rfcl import collision_metrics as cm


def score_trajectories(h5_paths: list[str]) -> list[dict[str, Any]]:
    """Safety signals of every collected trajectory.

    The contact pairs and the container come from the run config recorded in each h5
    (``policy_config_yaml``); insertion tasks (``safety_penalty.obj_press``) also need the jam signals.
    """
    import h5py

    rows: list[dict[str, Any]] = []
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            cfg = OmegaConf.load(str(f.attrs["policy_config_yaml"]))
            ic = cfg.safety_penalty.illegal_contact
            mask = cm.illegal_pair_mask(
                list(ic.robot_bodies), list(ic.scene_bodies), {(str(a), str(b)) for a, b in ic.allowed_pairs}
            )
            container = str(f.attrs["container"])
            jam = "obj_press" in cfg.safety_penalty
            # Lexicographic order (traj_0, traj_1, traj_10, ...): the selection permutation is applied over this
            # row order, so changing it changes which trajectories are exported.
            for gname in sorted(k for k in f if k.startswith("traj_")):
                g = f[gname]
                ex = g["extras"]
                assert ex["pair_force"].shape[1:] == mask.shape, f"{path}/{gname}: pair_force does not match the config"
                rows.append(
                    {
                        "traj_uid": str(g.attrs["traj_uid"]),
                        **cm.trajectory_signals(
                            ex["pair_force"][:],
                            mask,
                            g[f"states/objects/{container}/root_pose"][:, :2],
                            ex["press_axial"][:] if jam else None,
                            ex["press_lateral"][:] if jam else None,
                        ),
                    }
                )
    return rows


def select_kept(uids: list[str], removed: Any, *, n_keep: int, seed: int) -> list[str]:
    """First ``n_keep`` non-removed uids in one fixed permutation of the pool.

    Raises
    ------
    ValueError
        If the pool, or what survives the filter, is smaller than ``n_keep``.
    """
    rm = np.asarray(removed, dtype=bool)
    assert rm.shape == (len(uids),)
    perm_idx = np.random.default_rng(int(seed)).permutation(len(uids))
    kept = [uids[i] for i in perm_idx if not rm[i]]
    if len(kept) < n_keep:
        raise ValueError(f"{len(kept)} of {len(uids)} trajectories pass the filter < n_keep {n_keep}")
    return kept[:n_keep]


def main() -> int:
    """CLI: ``score`` / ``select``."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="per-trajectory safety signals")
    s.add_argument("--states-dir", required=True)
    s.add_argument("--out", required=True)
    sel = sub.add_parser("select", help="export filter, then n_keep trajectories from one permutation")
    sel.add_argument("--scores", required=True, help="output of `score`")
    sel.add_argument("--n-keep", type=int, default=3000)
    sel.add_argument("--seed", type=int, default=0)
    sel.add_argument("--out", required=True)
    args = p.parse_args()

    if args.cmd == "score":
        paths = sorted(glob.glob(os.path.join(args.states_dir, "episode_*.h5")))
        if not paths:
            raise FileNotFoundError(f"no episode_*.h5 in {args.states_dir}")
        rows = score_trajectories(paths)
        uids = [r["traj_uid"] for r in rows]
        assert len(set(uids)) == len(uids), "duplicate traj_uid"
        with open(args.out, "w") as f:
            json.dump({"states_dir": args.states_dir, "rows": rows}, f)
        print(f"[score] {len(rows)} trajectories -> {args.out}", flush=True)
        return 0

    with open(args.scores) as f:
        rows = json.load(f)["rows"]
    removed = np.array([cm.export_filter_drops(r) for r in rows], dtype=bool)
    kept = select_kept([r["traj_uid"] for r in rows], removed, n_keep=args.n_keep, seed=args.seed)
    with open(args.out, "w") as f:
        json.dump({"scores": args.scores, "pool": len(rows), "removed": int(removed.sum()), "kept": kept}, f, indent=1)
    print(f"[select] removed {int(removed.sum())}/{len(rows)}, kept {len(kept)} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
