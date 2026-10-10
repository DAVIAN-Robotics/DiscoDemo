# SPDX-License-Identifier: Apache-2.0
r"""Fixed initial-state banks: file format and env assignment (Isaac-free).

A bank is a list of init records, each with object poses (env frame) and optionally the arm state::

    [{"objects": {"board": [x,y,z,qw,qx,qy,qz], ...},
      "arm": {"joint_pos": [...], "joint_vel": [...]}}, ...]

Without ``arm`` the arm keeps its reset state; banks dumped from a run carry the arm because a
pinned object would otherwise lose its grasp.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: One bank record, as loaded from json.
InitRecord = dict[str, Any]


def load_bank(path: Path) -> list[InitRecord]:
    """Load a bank json as a list of records.

    Parameters
    ----------
    path : pathlib.Path
        Bank json (see the module docstring).

    Returns
    -------
    list of dict
        The records.
    """
    raw = json.loads(Path(path).read_text())
    assert isinstance(raw, list) and all("objects" in r for r in raw), f"not a bank: {path}"
    return raw


def assign_envs(n_env: int, n_init: int) -> list[int]:
    """Assign ``env i -> i % n_init``.

    Parameters
    ----------
    n_env : int
        Number of parallel envs.
    n_init : int
        Number of inits in the bank.

    Returns
    -------
    list of int
        Init index per env.

    Raises
    ------
    ValueError
        If ``n_init <= 0`` or ``n_env`` is not a multiple of ``n_init`` (every init must
        get the same number of samples).
    """
    if n_init <= 0:
        raise ValueError(f"n_init must be positive: {n_init}")
    if n_env % n_init != 0:
        raise ValueError(f"n_env ({n_env}) is not a multiple of n_init ({n_init})")
    return [i % n_init for i in range(n_env)]


def object_names(bank: list[InitRecord]) -> list[str]:
    """Return the sorted object names shared by every init.

    Parameters
    ----------
    bank : list of InitRecord
        Result of ``load_bank``.

    Returns
    -------
    list of str
        Sorted object names.

    Raises
    ------
    ValueError
        If the inits have different object sets.
    """
    names = sorted(bank[0]["objects"])
    for i, rec in enumerate(bank[1:], start=1):
        if sorted(rec["objects"]) != names:
            raise ValueError(f"inits 0 and {i} have different objects: {names} vs {sorted(rec['objects'])}")
    return names
