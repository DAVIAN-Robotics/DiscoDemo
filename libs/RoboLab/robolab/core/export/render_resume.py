"""Resume bookkeeping for ``render_states_to_lerobot.py``: per-group part h5 files plus an atomic progress file.

The renderer draws trajectories in groups of ``num_envs``. After a group finishes, its demo states are written to a
part h5 and the group id is added to the progress file, so a restarted run can skip finished groups.

- Demo numbers are fixed by group boundaries (``group_plan``), so skipping groups does not renumber demos.
- The progress file stores a fingerprint of the render settings; resuming with different settings raises.
- Parts and the progress file are written to a temp name and moved with ``os.replace``.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROGRESS_NAME = "render_progress.json"
PARTS_DIR = "_render_parts"


def group_plan(n_trajs: int, n_envs: int) -> list[tuple[int, int, int]]:
    """Group boundaries ``(group id, first traj, end traj)``.

    Parameters
    ----------
    n_trajs : int
        Number of trajectories to render (after sorting).
    n_envs : int
        Envs rendered in parallel per group.

    Returns
    -------
    list[tuple[int, int, int]]
        In group order; demo number = first traj + slot.
    """
    if n_envs <= 0:
        raise ValueError(f"n_envs must be > 0, got {n_envs}")
    return [(gi, g0, min(g0 + n_envs, n_trajs)) for gi, g0 in enumerate(range(0, n_trajs, n_envs))]


def render_fingerprint(*, args: dict[str, Any], env: dict[str, str], uids: list[str], n_envs: int) -> str:
    """Hash of the render settings; resume is only allowed when it matches.

    Parameters
    ----------
    args : dict[str, Any]
        argparse arguments (without the resume flag itself).
    env : dict[str, str]
        Environment variables that change the render output.
    uids : list[str]
        Trajectory ids in render order (order determines demo numbers).
    n_envs : int
        Group size.

    Returns
    -------
    str
        sha256 hex digest.
    """
    blob = json.dumps({"args": args, "env": env, "uids": uids, "n_envs": n_envs}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def part_path(out_root: str | Path, gi: int) -> Path:
    """Part h5 path of group ``gi``."""
    return Path(out_root) / PARTS_DIR / f"group_{gi:05d}.h5"


def write_part(path: Path, demos: list[dict[str, Any]]) -> None:
    """Atomically write one group's demo states to a part h5.

    Parameters
    ----------
    path : Path
        ``part_path`` result.
    demos : list[dict[str, Any]]
        Each: ``num, T, actions[T,8], joint_position[T,8], joint_velocity[T,8], skill_z[T,z] | None``
        (same schema as ``data/demo_N`` in data.hdf5).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        g = f.create_group("data")
        for d in demos:
            demo = g.create_group(f"demo_{int(d['num'])}")
            demo.attrs["num_samples"] = int(d["T"])
            demo.attrs["success"] = True
            demo.create_dataset("actions", data=np.asarray(d["actions"], np.float32))
            st = demo.create_group("states").create_group("articulation").create_group("robot")
            st.create_dataset("joint_position", data=np.asarray(d["joint_position"], np.float32))
            st.create_dataset("joint_velocity", data=np.asarray(d["joint_velocity"], np.float32))
            if d.get("skill_z") is not None:
                demo.create_dataset("skill_z", data=np.asarray(d["skill_z"], np.float32))
    os.replace(tmp, path)


def merge_parts(data_grp: h5py.Group, parts: list[Path]) -> None:
    """Copy ``data/demo_N`` of all parts, in order, into ``data_grp``.

    Parameters
    ----------
    data_grp : h5py.Group
        ``data`` group of the output data.hdf5.
    parts : list[Path]
        Part paths in group order; all must exist.
    """
    for p in parts:
        if not p.is_file():
            raise FileNotFoundError(f"missing part {p} (marked done in the progress file)")
        with h5py.File(p, "r") as f:
            for name in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
                if name in data_grp:
                    raise ValueError(f"{name} appears in two parts (group boundaries differ)")
                f.copy(f["data"][name], data_grp, name=name)


class RenderProgress:
    """Progress file listing finished groups, written atomically.

    Parameters
    ----------
    out_root : Path
        Render output root (next to render_manifest.json).
    fingerprint : str
        ``render_fingerprint`` value.
    n_groups : int
        Total number of groups.
    done : set[int]
        Finished group ids.
    """

    def __init__(self, out_root: Path, fingerprint: str, n_groups: int, done: set[int]) -> None:
        self.out_root = out_root
        self.fingerprint = fingerprint
        self.n_groups = n_groups
        self.done = done

    @classmethod
    def open(cls, out_root: str | Path, *, fingerprint: str, n_groups: int, resume: bool) -> "RenderProgress":
        """Open the progress file: a fresh run clears old parts, a resumed run checks the settings match.

        Parameters
        ----------
        out_root : str | Path
            Render output root.
        fingerprint : str
            Render settings of this run.
        n_groups : int
            Number of groups of this run.
        resume : bool
            Resume from an earlier run.

        Returns
        -------
        RenderProgress
            ``done`` holds the groups to skip.

        Raises
        ------
        ValueError
            On resume, if the previous fingerprint or group count differs.
        """
        root = Path(out_root)
        root.mkdir(parents=True, exist_ok=True)
        pf = root / PROGRESS_NAME
        if resume and pf.is_file():
            prev = json.loads(pf.read_text())
            if prev["fingerprint"] != fingerprint:
                raise ValueError(
                    f"render fingerprint differs from the previous run ({prev['fingerprint'][:12]} != {fingerprint[:12]}); "
                    f"cannot resume with different settings. Rerun without --resume to start over: {root}"
                )
            if int(prev["n_groups"]) != n_groups:
                raise ValueError(f"n_groups differs from the previous run ({prev['n_groups']} != {n_groups}): {root}")
            done = {int(g) for g in prev["done"]}
            missing = [g for g in done if not part_path(root, g).is_file()]
            if missing:
                raise FileNotFoundError(f"groups marked done but missing their part: {sorted(missing)[:8]} ({root})")
            return cls(root, fingerprint, n_groups, done)
        if not resume:
            shutil.rmtree(root / PARTS_DIR, ignore_errors=True)
        p = cls(root, fingerprint, n_groups, set())
        p._write()
        return p

    def mark_done(self, gi: int) -> None:
        """Mark group ``gi`` done; call after its part is written."""
        if not part_path(self.out_root, gi).is_file():
            raise FileNotFoundError(f"group {gi} has no part file")
        self.done.add(gi)
        self._write()

    def _write(self) -> None:
        pf = self.out_root / PROGRESS_NAME
        tmp = pf.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"fingerprint": self.fingerprint, "n_groups": self.n_groups, "done": sorted(self.done)})
        )
        os.replace(tmp, pf)
