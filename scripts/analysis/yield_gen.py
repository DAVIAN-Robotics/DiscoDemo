r"""Generation yield: the fraction of a generator's rollouts from the task's initial-state distribution that
succeed, before the export filter.

Reads the collection record ``collect_budget.json`` written by ``libs/RoboLab/scripts/collect_rl_states.py``
(``episodes_accepted`` / ``episodes_attempted``).

Spec (JSON; relative paths are resolved against the spec file)::

    {"tasks": {"pnp_banana": {"discodemo": "gen/discodemo/states/collect_budget.json",
                              "prfcl": "gen/prfcl/states/collect_budget.json"}}}

Usage::

    python scripts/analysis/yield_gen.py --spec spec.json --out yield.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def budget(p: Path) -> dict[str, Any]:
    """Accepted / attempted episodes from a collection record.

    Parameters
    ----------
    p : pathlib.Path
        JSON with ``episodes_accepted`` and ``episodes_attempted``.

    Returns
    -------
    dict
        ``{succ, tried, rate, src}``.
    """
    b = json.loads(p.read_text())
    a, n = int(b["episodes_accepted"]), int(b["episodes_attempted"])
    return {"succ": a, "tried": n, "rate": a / n, "src": str(p)}


def main() -> None:
    """Evaluate every cell of the spec and write the yields as JSON."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True, help="input spec (JSON)")
    ap.add_argument("--out", type=Path, required=True, help="output JSON")
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text())
    base = a.spec.resolve().parent
    out: dict[str, dict[str, Any]] = {}
    for t, cells in spec["tasks"].items():
        for s, path in cells.items():
            r = budget(Path(path) if Path(path).is_absolute() else base / path)
            out[f"{t}|{s}"] = r
            print(f"{t}|{s:12s} {r['rate']:.3f}  ({r['succ']}/{r['tried']})")
    a.out.write_text(json.dumps(out, indent=1))
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
