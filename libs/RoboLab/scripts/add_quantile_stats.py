#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add q01/q99 to ``meta/stats.json`` of a LeRobot v3.0 dataset.

pi0.5 normalizes state/action with quantiles, so ``stats.json`` needs q01/q99. They are computed from the numeric
parquet columns only (no video decoding). Run once on the merged dataset, after aggregation.

usage::

    python libs/RoboLab/scripts/add_quantile_stats.py --root <lerobot_dataset> \
        [--keys action observation.state observation.velocity]
"""

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq


def _column_matrix(root: str, key: str) -> np.ndarray:
    """Stack column ``key`` of data/**/*.parquet into an [N, dim] float64 array."""
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files in {os.path.join(root, 'data')}")
    rows = []
    for f in files:
        col = pq.read_table(f, columns=[key]).column(key).to_pylist()
        rows.extend(col)
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 1:  # scalar column -> [N, 1]
        arr = arr[:, None]
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LeRobot v3.0 dataset root (with meta/stats.json)")
    ap.add_argument(
        "--keys",
        nargs="+",
        default=["action", "observation.state", "observation.velocity"],
        help="numeric feature keys to add q01/q99 for",
    )
    args = ap.parse_args()

    stats_path = os.path.join(args.root, "meta", "stats.json")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"meta/stats.json not found: {stats_path}")
    with open(stats_path) as f:
        stats = json.load(f)

    for key in args.keys:
        if key not in stats:
            print(f"[add_quantile] skip {key}: not in stats")
            continue
        mat = _column_matrix(args.root, key)
        q01 = np.quantile(mat, 0.01, axis=0)
        q99 = np.quantile(mat, 0.99, axis=0)
        # Same per-dim shape as min/max.
        ref = stats[key].get("min")
        ref_shape = np.asarray(ref).shape if ref is not None else q01.shape
        stats[key]["q01"] = q01.reshape(ref_shape).tolist()
        stats[key]["q99"] = q99.reshape(ref_shape).tolist()
        print(
            f"[add_quantile] {key}: N={mat.shape[0]} dim={mat.shape[1]} "
            f"q01[0]={q01.flat[0]:.4f} q99[0]={q99.flat[0]:.4f}"
        )

    tmp = stats_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(stats, f, indent=4)
    os.replace(tmp, stats_path)
    print(f"[add_quantile] wrote q01/q99 -> {stats_path}")


if __name__ == "__main__":
    main()
