"""Merge rendered LeRobot shards into one dataset and carry over ``meta/provenance.json``.

``lerobot.datasets.aggregate.aggregate_datasets`` only moves the LeRobot schema, so the provenance
file (which generator checkpoint produced the data) is copied from the first shard and the shard
names are appended. Run ``add_quantile_stats.py`` and ``normalize_videos_cfr.py`` on the result.

Examples
--------
``python libs/RoboLab/scripts/aggregate_shards.py --out data/pnp_banana_ours data/shards/r0 data/shards/r1``
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets


def main() -> None:
    """Aggregate shards, then copy provenance."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shards", nargs="+", type=Path, help="LeRobot shard directories")
    ap.add_argument("--out", type=Path, required=True, help="merged dataset directory (replaced if it exists)")
    a = ap.parse_args()
    for s in a.shards:
        if not (s / "meta/info.json").is_file():
            raise FileNotFoundError(f"not a LeRobot dataset: {s}")
    shutil.rmtree(a.out, ignore_errors=True)
    aggregate_datasets(repo_ids=[s.name for s in a.shards], aggr_repo_id=a.out.name, roots=a.shards, aggr_root=a.out)
    src = a.shards[0] / "meta/provenance.json"
    if src.is_file():
        prov = json.loads(src.read_text())
        prov["merged_from_shards"] = [s.name for s in a.shards]
        (a.out / "meta/provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"[aggregate] {len(a.shards)} shards -> {a.out}")


if __name__ == "__main__":
    main()
