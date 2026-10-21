#!/usr/bin/env python3
"""Redistribute collected trajectories into N shard directories so N render processes can run in parallel.

Each shard is ``<out-dir>/shard_<i>/episode_0.h5``. Trajectory groups are byte-copied (no recomputation) and
renumbered ``traj_0..traj_{k-1}`` per shard; file-level attrs (task, instruction, provenance) are copied to every
shard. Only the trajectories kept by the export filter (``--keep-json``, output of
``score_collision.py select``) are written.

usage::

    python libs/RoboLab/scripts/repartition_states_shards.py \
        --src-dir <states_dir> --out-dir <shards_dir> --n-shards 4 --keep-json keep.json
"""

import argparse
import glob
import os

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="collected states dir with episode_*.h5")
    ap.add_argument("--out-dir", required=True, help="parent dir for shard_<i>/episode_0.h5")
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--keep-json", required=True, help="score_collision.py select output")
    args = ap.parse_args()

    import json

    with open(args.keep_json) as fh:
        keep = set(json.load(fh)["kept"])
    found: set[str] = set()

    src_files = sorted(glob.glob(os.path.join(args.src_dir, "episode_*.h5")))
    if not src_files:
        raise FileNotFoundError(f"no episode_*.h5 in {args.src_dir}")

    # (src_file, traj_key) in file order
    global_list = []
    src_attrs = None
    for f in src_files:
        with h5py.File(f, "r") as hf:
            if src_attrs is None:
                src_attrs = dict(hf.attrs)
            if "task" not in hf.attrs or "instruction" not in hf.attrs:
                raise ValueError(
                    f"{f}: missing 'task'/'instruction' attr (incomplete collection). "
                    f"render_states_to_lerobot.py needs hf.attrs['task']."
                )
            tks = sorted([k for k in hf if k.startswith("traj_")], key=lambda x: int(x.split("_")[1]))
            for tk in tks:
                uid = str(hf[tk].attrs["traj_uid"])
                if uid not in keep:
                    continue
                found.add(uid)
                global_list.append((f, tk))
    if keep - found:
        missing = sorted(keep - found)
        raise ValueError(f"{len(missing)} selected uids are missing from {args.src_dir}: {missing[:8]}")

    n_total = len(global_list)
    # Interleaved split: every shard spans the whole collection order (contiguous blocks would not).
    chunks = [np.arange(si, n_total, args.n_shards) for si in range(args.n_shards)]
    print(f"[repart] src={args.src_dir} total_traj={n_total} -> {args.n_shards} shards", flush=True)

    written = 0
    src_handles = {f: h5py.File(f, "r") for f in src_files}
    try:
        for si, idxs in enumerate(chunks):
            shard_dir = os.path.join(args.out_dir, f"shard_{si}")
            os.makedirs(shard_dir, exist_ok=True)
            dst_path = os.path.join(shard_dir, "episode_0.h5")
            with h5py.File(dst_path, "w") as dst:
                for k, v in src_attrs.items():
                    dst.attrs[k] = v
                for local_i, gi in enumerate(idxs):
                    sf, tk = global_list[int(gi)]
                    src_handles[sf].copy(tk, dst, name=f"traj_{local_i}")
            written += len(idxs)
            print(f"[repart] shard_{si}: {len(idxs)} traj -> {dst_path}", flush=True)
    finally:
        for h in src_handles.values():
            h.close()

    if written != n_total:
        raise AssertionError(f"trajectory count mismatch: wrote {written} != total {n_total}")
    print(f"[repart] OK — {written} traj across {args.n_shards} shards (== source {n_total})", flush=True)


if __name__ == "__main__":
    main()
