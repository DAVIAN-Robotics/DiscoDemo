# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot v3.0 dataset exporter for RoboLab.

RoboLab raw render (data.hdf5 + per-camera mp4) -> LeRobot v3.0 dataset.

LeRobot v3.0 layout::

    dataset/
    ├── meta/
    │   ├── info.json                          # features / fps / codebase_version=v3.0
    │   ├── stats.json                         # min/max/mean/std/count per feature
    │   ├── tasks.jsonl + tasks.parquet        # instructions (parquet indexed by task string)
    │   └── episodes/chunk-000/file-000.parquet  # episode metadata (indices + video segment timestamps)
    ├── data/chunk-000/file-000.parquet        # per-frame state/action/timestamp
    └── videos/<camera>/chunk-000/file-000.mp4 # one concatenated file per camera (seek via from/to_timestamp)

Notes:
- Each episode clip is re-encoded to avc1 (H.264, yuv420p) and then concatenated with ``-c copy``;
  avc1 is what LeRobot / torchcodec decode reliably. Concatenating without re-encoding breaks
  timestamps.
- Image stats are computed in a streaming fashion from one frame per sampled video (O(1) memory).
- The episodes parquet must contain ``meta/episodes/chunk_index`` / ``meta/episodes/file_index``;
  ``lerobot.datasets.aggregate.aggregate_datasets`` reads them.

Invalid input raises immediately. Optional features (ee_pose / joint_velocity / env_cfg.json) are
used only if present, but raise if present and malformed.

Usage::

    from robolab.core.export.lerobot_exporter import LeRobotExporter
    LeRobotExporter("output/exp", "output/exp_lerobot", robot_type="franka").export()
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

# pyarrow is imported lazily inside the parquet writers so this module stays importable without it.


class LeRobotExporter:
    """Converts RoboLab HDF5 render output to a LeRobot v3.0 dataset."""

    def __init__(
        self,
        robolab_output_dir: str,
        lerobot_output_dir: str | None = None,
        robot_type: str = "franka",
        fps: float = 20.0,
        repo_id: str | None = None,
        concatenate_videos: bool = True,
    ):
        """Initialize the exporter.

        Args:
            robolab_output_dir: RoboLab render output (data.hdf5 + mp4 per task subfolder).
            lerobot_output_dir: output path; defaults to robolab_output_dir/lerobot.
            robot_type: robot type for metadata ("franka" is stored as "unknown", unsupported by LeRobot).
            fps: dataset fps.
            repo_id: optional HF repo id.
            concatenate_videos: one mp4 per camera (default) or one mp4 per episode.
        """
        self.robolab_dir = Path(robolab_output_dir)
        self.lerobot_dir = Path(lerobot_output_dir) if lerobot_output_dir else self.robolab_dir / "lerobot"
        # LeRobot does not know "franka".
        self.robot_type = "unknown" if robot_type == "franka" else robot_type
        self.fps = fps
        self.repo_id = repo_id or f"robolab/{self.robolab_dir.name}"
        self.concatenate_videos = concatenate_videos

        # accumulation buffers
        self._tasks: list[dict] = []
        self._episodes_metadata: list[dict] = []
        self._all_data_rows: list[dict] = []
        self._current_index = 0

        # numeric feature stats (key -> list of per-frame vectors)
        self._stats: dict[str, list[np.ndarray]] = {}

        # video info
        self._video_features: dict[str, dict] = {}
        self._video_files: list[tuple[str, str, int]] = []  # (feature_name, src_path, episode_idx)
        # after concat: (feature_name, episode_idx) -> (file_index, from_ts, to_ts)
        self._video_timestamps: dict[tuple[str, int], tuple[int, float, float]] = {}
        # cache: task_dir -> {camera_name: {width, height}}
        self._camera_info_cache: dict[str, dict[str, dict]] = {}

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #
    def export(self) -> Path:
        """Convert the RoboLab output to LeRobot v3.0 and return the output path."""
        print(f"[LeRobotExporter] Exporting from: {self.robolab_dir}")
        print(f"[LeRobotExporter] Exporting to: {self.lerobot_dir}")

        self._create_directory_structure()

        hdf5_files = self._find_hdf5_files()
        if not hdf5_files:
            raise ValueError(f"No HDF5 files (data.hdf5 or run_*.hdf5) found in {self.robolab_dir}")
        print(f"[LeRobotExporter] Found {len(hdf5_files)} HDF5 files")

        for hdf5_path in hdf5_files:
            self._process_hdf5_file(hdf5_path)

        print("[LeRobotExporter] [1/6] writing data parquet...", flush=True)
        self._write_data_parquet()
        print("[LeRobotExporter] [2/6] building videos (avc1 re-encode+concat)...", flush=True)
        self._build_videos()
        print("[LeRobotExporter] [3/6] writing episodes metadata...", flush=True)
        self._write_episodes_metadata()
        print("[LeRobotExporter] [4/6] writing tasks...", flush=True)
        self._write_tasks()
        print("[LeRobotExporter] [5/6] computing stats...", flush=True)
        self._write_stats()
        print("[LeRobotExporter] [6/7] writing info.json...", flush=True)
        self._write_info()
        print("[LeRobotExporter] [7/7] writing provenance.json...", flush=True)
        self._write_provenance(hdf5_files)

        print(f"[LeRobotExporter] Export complete: {self.lerobot_dir}")
        return self.lerobot_dir

    def _write_provenance(self, hdf5_files: list[Path]) -> None:
        """Copy the source h5 attrs (e.g. the generating policy) to ``meta/provenance.json``.

        Kept outside ``info.json`` so the LeRobot schema stays untouched (aggregation ignores unknown
        files in meta/). Nothing is written if the attrs are empty.
        """
        attrs: dict = {}
        for p in hdf5_files:
            with h5py.File(p, "r") as hf:
                for k, v in hf.attrs.items():
                    attrs.setdefault(k, v.decode() if isinstance(v, bytes) else v)
        keep = {k: (v.item() if hasattr(v, "item") else v) for k, v in attrs.items()}
        if not keep:
            print("[LeRobotExporter] no provenance: source h5 attrs are empty", flush=True)
            return
        keep["source_hdf5"] = [str(p) for p in hdf5_files]
        (self.lerobot_dir / "meta" / "provenance.json").write_text(
            json.dumps(keep, indent=1, ensure_ascii=False, default=str)
        )

    def _create_directory_structure(self):
        for d in (
            self.lerobot_dir / "meta" / "episodes" / "chunk-000",
            self.lerobot_dir / "data" / "chunk-000",
            self.lerobot_dir / "videos",
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _find_hdf5_files(self) -> list[Path]:
        """Find HDF5 files: single ``data.hdf5`` and/or multi-env ``run_<i>.hdf5`` files."""
        hdf5_files: list[Path] = []

        def _collect_from_dir(d: Path):
            data_path = d / "data.hdf5"
            if data_path.exists():
                hdf5_files.append(data_path)
            run_files = sorted(d.glob("run_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
            hdf5_files.extend(run_files)

        for task_dir in self.robolab_dir.iterdir():
            if task_dir.is_dir():
                _collect_from_dir(task_dir)
        _collect_from_dir(self.robolab_dir)
        return hdf5_files

    # ------------------------------------------------------------------ #
    # HDF5 processing
    # ------------------------------------------------------------------ #
    def _process_hdf5_file(self, hdf5_path: Path):
        """Extract all demos of one HDF5 file into the accumulation buffers."""
        task_dir = hdf5_path.parent
        task_name = task_dir.name if task_dir != self.robolab_dir else "default_task"

        # run_idx for multi-env run files, None for data.hdf5
        stem = hdf5_path.stem
        run_idx = int(stem.split("_")[1]) if stem.startswith("run_") else None

        print(f"[LeRobotExporter] Processing: {hdf5_path}")
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                raise ValueError(f"HDF5 file has no 'data' group (broken render?): {hdf5_path}")
            data_group = f["data"]

            demo_names = sorted(
                (k for k in data_group.keys() if k.startswith("demo_")),
                key=lambda x: int(x.split("_")[1]),
            )
            n_demos = len(demo_names)
            print(f"  Found {n_demos} episodes", flush=True)

            t0 = time.time()
            for di, demo_name in enumerate(demo_names):
                if di % 500 == 0 or di == n_demos - 1:
                    el = time.time() - t0
                    done = di + 1
                    eta = el / done * (n_demos - done)
                    print(
                        f"  [process] {done}/{n_demos} demos ({100 * done // n_demos}%) "
                        f"elapsed={el:.0f}s ETA={eta:.0f}s",
                        flush=True,
                    )
                demo_group = data_group[demo_name]
                episode_idx = len(self._episodes_metadata)

                episode_data = self._extract_episode_data(demo_group)
                self._find_episode_videos(task_dir, episode_idx, demo_name, run_idx=run_idx)

                task_idx = self._get_or_create_task(task_name)
                instruction = self._tasks[task_idx]["task"]
                num_frames = len(episode_data)

                self._episodes_metadata.append({
                    "episode_index": episode_idx,
                    "meta/episodes/chunk_index": 0,  # single chunk (read by aggregate)
                    "meta/episodes/file_index": 0,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": self._current_index,
                    "dataset_to_index": self._current_index + num_frames,
                    "length": num_frames,
                    "task_index": task_idx,
                    "success": bool(demo_group.attrs.get("success", False)),
                    "tasks": [instruction],
                })

                for frame_idx, row in enumerate(episode_data):
                    row["episode_index"] = episode_idx
                    row["frame_index"] = frame_idx
                    row["index"] = self._current_index
                    row["task_index"] = task_idx
                    row["timestamp"] = frame_idx / self.fps
                    # next.done is True on the last frame of each episode
                    row["next.done"] = frame_idx == num_frames - 1
                    self._all_data_rows.append(row)
                    self._current_index += 1

                self._update_stats(episode_data)

    def _extract_episode_data(self, demo_group: h5py.Group) -> list[dict]:
        """Extract per-frame row dicts from a demo group.

        ``actions`` and robot ``joint_position`` are required; joint velocity, ee pose and skill_z
        are added when present.
        """
        num_samples = int(demo_group.attrs.get("num_samples", 0))
        if num_samples == 0 and "actions" in demo_group:
            num_samples = demo_group["actions"].shape[0]
        if num_samples == 0:
            raise ValueError(f"Demo group '{demo_group.name}' has 0 samples and no 'actions' to infer from")

        # required: actions
        if "actions" not in demo_group:
            raise ValueError(f"Demo group '{demo_group.name}' missing required 'actions'")
        actions = np.asarray(demo_group["actions"])

        # required: robot joint_position
        states = demo_group.get("states")
        if states is None or "articulation" not in states or "robot" not in states["articulation"]:
            raise ValueError(f"Demo group '{demo_group.name}' missing states/articulation/robot")
        robot_states = states["articulation"]["robot"]
        if "joint_position" not in robot_states:
            raise ValueError(f"Demo group '{demo_group.name}' missing robot joint_position")
        joint_positions = np.asarray(robot_states["joint_position"])

        # optional: joint_velocity
        joint_velocities = (
            np.asarray(robot_states["joint_velocity"]) if "joint_velocity" in robot_states else None
        )

        # optional: skill_z ([T, z_dim], only for z-conditioned policies). Omitted when absent rather
        # than zero-filled, so "no z" stays distinguishable from z = 0.
        skill_z = np.asarray(demo_group["skill_z"]) if "skill_z" in demo_group else None

        # optional: ee_pose (only if an ee_pose recorder was used)
        ee_position = ee_orientation = None
        if "ee_pose" in demo_group:
            ee_group = demo_group["ee_pose"]
            if "position" in ee_group:
                ee_position = np.asarray(ee_group["position"])
            if "orientation" in ee_group:
                ee_orientation = np.asarray(ee_group["orientation"])

        rows = []
        for i in range(num_samples):
            row = {
                "action": actions[i].tolist(),
                "observation.state": joint_positions[i].tolist(),
            }
            if joint_velocities is not None:
                row["observation.velocity"] = joint_velocities[i].tolist()
            if ee_position is not None:
                row["observation.ee_position"] = ee_position[i].tolist()
            if ee_orientation is not None:
                row["observation.ee_orientation"] = ee_orientation[i].tolist()
            if skill_z is not None:
                row["observation.skill_z"] = skill_z[i].tolist()
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ #
    # task / instruction
    # ------------------------------------------------------------------ #
    def _get_instruction_for_task(self, task_name: str) -> str:
        """Language instruction of a task: from episode results, else derived from the task name."""
        from robolab.core.logging.results import load_episode_results

        for result in load_episode_results(str(self.robolab_dir)):
            if result.get("task") == task_name and "instruction" in result:
                return result["instruction"]

        import re

        words = re.findall(r"[A-Z][a-z]*|[a-z]+", task_name.replace("Task", ""))
        return " ".join(words).lower().capitalize()

    def _get_or_create_task(self, task_name: str) -> int:
        instruction = self._get_instruction_for_task(task_name)
        for i, task in enumerate(self._tasks):
            if task["task"] == instruction:
                return i
        task_idx = len(self._tasks)
        self._tasks.append({"task_index": task_idx, "task": instruction})
        return task_idx

    # ------------------------------------------------------------------ #
    # video discovery
    # ------------------------------------------------------------------ #
    def _load_camera_info_from_env_cfg(self, task_dir: Path) -> dict[str, dict]:
        """Read camera width/height from env_cfg.json (empty dict if the file does not exist)."""
        env_cfg_path = task_dir / "env_cfg.json"
        if not env_cfg_path.exists():
            return {}
        with open(env_cfg_path) as f:
            cfg = json.load(f)
        cameras = {}
        for key, value in cfg.get("scene", {}).items():
            if isinstance(value, dict) and "camera" in value.get("class_type", "").lower():
                cameras[key] = {
                    "width": value.get("width", 640),
                    "height": value.get("height", 480),
                }
        return cameras

    def _camera_info(self, task_dir: Path) -> dict[str, dict]:
        key = str(task_dir)
        if key not in self._camera_info_cache:
            self._camera_info_cache[key] = self._load_camera_info_from_env_cfg(task_dir)
        return self._camera_info_cache[key]

    def _register_video_feature(self, feature_name: str, width: int, height: int, fps: float):
        """Register video feature metadata (once per feature)."""
        if feature_name in self._video_features:
            return
        self._video_features[feature_name] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "avc1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": float(fps),
                "video.channels": 3,
                "has_audio": False,
            },
        }

    def _find_episode_videos(self, task_dir: Path, episode_idx: int, demo_name: str, run_idx: int | None = None):
        """Find and register the per-camera mp4 files of an episode.

        Multi-env (``run_idx`` set): demo_num is the env id, pattern ``*_{run_idx}_env{env_id}__*.mp4``.
        Otherwise: pattern ``*_{demo_num}__*.mp4``. Falls back to a combined ``*_viewport*.mp4``.
        """
        demo_num = int(demo_name.split("_")[1])
        camera_info = self._camera_info(task_dir)

        if run_idx is not None:
            per_camera_pattern = f"*_{run_idx}_env{demo_num}__*.mp4"
        else:
            per_camera_pattern = f"*_{demo_num}__*.mp4"

        # per-camera videos ({instruction}_{ep}__{camera_name}.mp4)
        per_camera_found = False
        for video_file in task_dir.glob(per_camera_pattern):
            dunder_idx = video_file.stem.rfind("__")
            if dunder_idx == -1:
                continue
            cam_name = video_file.stem[dunder_idx + 2:]
            feature_name = f"observation.images.{cam_name}"
            self._video_files.append((feature_name, str(video_file), episode_idx))
            per_camera_found = True

            cam_meta = camera_info.get(cam_name)
            if cam_meta:
                width, height = cam_meta["width"], cam_meta["height"]
            else:
                width, height, _ = self._get_video_info(video_file)
            self._register_video_feature(feature_name, width, height, self.fps)

        if per_camera_found:
            return

        # fallback: combined viewport
        viewport_match = f"_{run_idx}_env{demo_num}_viewport" if run_idx is not None else None
        for video_file in task_dir.glob("*.mp4"):
            name = video_file.stem
            if "_viewport" not in name:
                continue
            if run_idx is not None:
                matched = viewport_match in name
            else:
                matched = f"_{demo_num}" in name or name.endswith(f"_{demo_num}")
            if matched:
                feature_name = "observation.images.front"
                self._video_files.append((feature_name, str(video_file), episode_idx))
                width, height, fps = self._get_video_info(video_file)
                self._register_video_feature(feature_name, width, height, fps if fps > 0 else self.fps)
                break

    def _get_video_info(self, video_path: Path) -> tuple[int, int, float]:
        """Return (width, height, fps) via OpenCV; raises if unreadable."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if width <= 0 or height <= 0:
            raise ValueError(f"Failed to read video dimensions from {video_path} (got {width}x{height})")
        return width, height, fps

    def _get_video_duration_seconds(self, video_path: Path) -> float:
        """Return the video duration in seconds via OpenCV; raises if unreadable."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not (fps and fps > 0 and n_frames >= 0):
            raise ValueError(f"Failed to read duration from {video_path} (frames={n_frames}, fps={fps})")
        return n_frames / fps

    def _reencode_to_avc1(self, src_path: Path, dst_path: Path, fps: float | None = None, max_frames: int | None = None):
        """Re-encode a video to H.264 (avc1) yuv420p; raises on failure.

        max_frames: maximum number of frames to encode (all if None); the episode length, so the clip
            never runs past its states.
        """
        fps = fps or self.fps
        # Keep ffmpeg off the terminal's stdin (-nostdin + DEVNULL): as a background job it would
        # otherwise be stopped by SIGTTIN/SIGTTOU and hang silently until the timeout.
        cmd = [
            "ffmpeg", "-nostdin", "-y",
            "-i", str(src_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-an",
        ]
        if max_frames is not None:
            cmd += ["-frames:v", str(int(max_frames))]
        cmd += [str(dst_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg re-encode failed for {src_path}: "
                f"{result.stderr[-500:] if result.stderr else result.stdout[-500:]}"
            )

    # ------------------------------------------------------------------ #
    # stats
    # ------------------------------------------------------------------ #
    def _update_stats(self, episode_data: list[dict]):
        """Accumulate per-frame values of numeric and index features."""
        index_like = ("episode_index", "frame_index", "index", "task_index", "timestamp")
        for row in episode_data:
            for key, value in row.items():
                if key == "next.done":
                    continue
                if isinstance(value, list):
                    self._stats.setdefault(key, []).append(np.array(value))
                elif key in index_like:
                    self._stats.setdefault(key, []).append(np.array([value]))

    def _compute_final_stats(self) -> dict:
        """Compute mean/std/min/max/count from the accumulated values."""

        def to_list(a):
            a = np.asarray(a)
            return a.tolist() if a.shape != () else [float(a)]

        stats = {}
        for key, values in self._stats.items():
            if not values:
                continue
            stacked = np.stack(values)
            std = np.nan_to_num(stacked.std(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
            stats[key] = {
                "mean": to_list(stacked.mean(axis=0)),
                "std": to_list(std),
                "min": to_list(stacked.min(axis=0)),
                "max": to_list(stacked.max(axis=0)),
                "count": [len(values)],
            }
        return stats

    def _compute_observation_images_stats(self) -> dict[str, dict]:
        """Per-channel min/max/mean/std/count for each ``observation.images.*`` camera (streaming).

        Uses the first frame of up to ``max_videos`` evenly sampled videos per camera with running
        sums, so memory is O(1); raises if a sampled frame cannot be read.
        """
        import cv2

        max_videos = 2000  # sampled videos per camera
        result = {}
        for camera_name in self._video_features:
            videos = [(path, ep_idx) for c, path, ep_idx in self._video_files if c == camera_name]
            if not videos:
                continue

            total_count = sum(
                self._episodes_metadata[ep_idx]["length"]
                for _, ep_idx in videos
                if 0 <= ep_idx < len(self._episodes_metadata)
            )

            stride = max(1, len(videos) // max_videos)
            sampled = videos[::stride][:max_videos]

            n_px = 0
            s = np.zeros(3, np.float64)
            ss = np.zeros(3, np.float64)
            mn = np.full(3, np.inf, np.float64)
            mx = np.full(3, -np.inf, np.float64)
            for src_path, _ in sampled:
                path = Path(src_path)
                if not path.exists():
                    raise FileNotFoundError(f"Sampled video for stats does not exist: {path}")
                cap = cv2.VideoCapture(str(path))
                ret, frame = cap.read()  # first frame only
                cap.release()
                if not ret:
                    raise RuntimeError(f"Failed to read first frame for stats from {path}")
                # BGR -> RGB, scale to [0, 1], flatten to (H*W, 3)
                px = (frame[:, :, ::-1].astype(np.float64) / 255.0).reshape(-1, 3)
                s += px.sum(axis=0)
                ss += (px * px).sum(axis=0)
                mn = np.minimum(mn, px.min(axis=0))
                mx = np.maximum(mx, px.max(axis=0))
                n_px += px.shape[0]

            if n_px == 0:
                raise RuntimeError(f"No pixels sampled for camera '{camera_name}' — cannot compute stats")

            mean = s / n_px
            var = np.maximum(ss / n_px - mean * mean, 0.0)
            std = np.nan_to_num(np.sqrt(var), nan=0.0, posinf=0.0, neginf=0.0)
            result[camera_name] = {
                "min": [[[float(mn[c])]] for c in range(3)],
                "max": [[[float(mx[c])]] for c in range(3)],
                "mean": [[[float(mean[c])]] for c in range(3)],
                "std": [[[float(std[c])]] for c in range(3)],
                "count": [total_count],
            }
        return result

    # ------------------------------------------------------------------ #
    # parquet / json writers
    # ------------------------------------------------------------------ #
    def _write_data_parquet(self):
        """Write all frames to a single data parquet file."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not self._all_data_rows:
            raise ValueError("No data rows to write — no episodes were extracted")

        # Schema and list lengths come from the first row; all rows must match.
        first_row = self._all_data_rows[0]
        index_keys = ("episode_index", "frame_index", "index", "task_index")
        schema_fields = []
        list_lengths: dict[str, int] = {}
        for key, value in first_row.items():
            if key in index_keys:
                schema_fields.append(pa.field(key, pa.int64()))
            elif key == "timestamp":
                schema_fields.append(pa.field(key, pa.float32()))
            elif key == "next.done":
                schema_fields.append(pa.field(key, pa.bool_()))
            elif isinstance(value, list):
                list_lengths[key] = len(value)
                schema_fields.append(pa.field(key, pa.list_(pa.float32())))
            else:
                raise TypeError(f"Unexpected scalar data field '{key}'={value!r} (type {type(value)})")
        schema = pa.schema(schema_fields)

        # Column-wise conversion; missing keys or length mismatches raise.
        columns: dict[str, list] = {field.name: [] for field in schema}
        for row in self._all_data_rows:
            for key in columns:
                if key not in row:
                    raise KeyError(f"Data row missing field '{key}': {row.keys()}")
                value = row[key]
                if key in list_lengths and len(value) != list_lengths[key]:
                    raise ValueError(
                        f"Inconsistent list length for '{key}': expected {list_lengths[key]}, got {len(value)}"
                    )
                columns[key].append(value)

        table = pa.table(columns, schema=schema)
        output_path = self.lerobot_dir / "data" / "chunk-000" / "file-000.parquet"
        pq.write_table(table, output_path)
        print(f"  Wrote {len(self._all_data_rows)} frames to {output_path}")

    def _write_episodes_metadata(self):
        """Write episode metadata parquet (index columns + per-camera video segment timestamps)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not self._episodes_metadata:
            raise ValueError("No episode metadata to write")

        # Fill video segment metadata; every episode must have a timestamp for every camera.
        for ep in self._episodes_metadata:
            ep_idx = ep["episode_index"]
            for camera_name in self._video_features:
                key = (camera_name, ep_idx)
                if key not in self._video_timestamps:
                    raise KeyError(
                        f"Missing video timestamp for episode {ep_idx} camera '{camera_name}' "
                        f"(video build did not register this segment)"
                    )
                file_index, from_ts, to_ts = self._video_timestamps[key]
                ep[f"videos/{camera_name}/chunk_index"] = 0
                ep[f"videos/{camera_name}/file_index"] = file_index
                ep[f"videos/{camera_name}/from_timestamp"] = from_ts
                ep[f"videos/{camera_name}/to_timestamp"] = to_ts

        # Includes meta/episodes/{chunk,file}_index and data/{chunk,file}_index (read by aggregate).
        schema_fields = [
            pa.field("episode_index", pa.int64()),
            pa.field("meta/episodes/chunk_index", pa.int64()),
            pa.field("meta/episodes/file_index", pa.int64()),
            pa.field("data/chunk_index", pa.int64()),
            pa.field("data/file_index", pa.int64()),
            pa.field("dataset_from_index", pa.int64()),
            pa.field("dataset_to_index", pa.int64()),
            pa.field("length", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("success", pa.bool_()),
            pa.field("tasks", pa.list_(pa.string())),
        ]
        for camera_name in self._video_features:
            schema_fields.extend([
                pa.field(f"videos/{camera_name}/chunk_index", pa.int64()),
                pa.field(f"videos/{camera_name}/file_index", pa.int64()),
                pa.field(f"videos/{camera_name}/from_timestamp", pa.float64()),
                pa.field(f"videos/{camera_name}/to_timestamp", pa.float64()),
            ])
        schema = pa.schema(schema_fields)

        columns: dict[str, list] = {field.name: [] for field in schema}
        for ep in self._episodes_metadata:
            for key in columns:
                if key not in ep:
                    raise KeyError(f"Episode metadata missing field '{key}': {ep.keys()}")
                value = ep[key]
                if key == "success":
                    value = bool(value)
                elif key == "tasks":
                    if not isinstance(value, list):
                        raise TypeError(f"'tasks' must be a list, got {type(value)}")
                elif "timestamp" in key:
                    value = float(value)
                else:
                    value = int(value)
                columns[key].append(value)

        table = pa.table(columns, schema=schema)
        output_path = self.lerobot_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        pq.write_table(table, output_path)
        print(f"  Wrote {len(self._episodes_metadata)} episode metadata entries")

    def _write_tasks(self):
        """Write tasks as JSONL and parquet."""
        jsonl_path = self.lerobot_dir / "meta" / "tasks.jsonl"
        with open(jsonl_path, "w") as f:
            for task in self._tasks:
                f.write(json.dumps(task) + "\n")

        # LeRobotDataset reads the task string via `meta.tasks.iloc[idx].name`, so the parquet must be
        # indexed by the task string (with a RangeIndex, .name would be an int).
        if not self._tasks:
            raise ValueError("No tasks to write")
        import pandas as pd

        df = pd.DataFrame({
            "task_index": [t["task_index"] for t in self._tasks],
            "task": [t["task"] for t in self._tasks],
        }).set_index("task")
        df.to_parquet(self.lerobot_dir / "meta" / "tasks.parquet")
        print(f"  Wrote {len(self._tasks)} tasks")

    def _write_stats(self):
        stats = self._compute_final_stats()
        stats.update(self._compute_observation_images_stats())
        output_path = self.lerobot_dir / "meta" / "stats.json"
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Wrote statistics for {len(stats)} features")

    def _write_info(self):
        if not self._all_data_rows:
            raise ValueError("No data rows — cannot write info.json")

        fps_int = int(round(self.fps))
        index_keys = ("episode_index", "frame_index", "index", "task_index")
        features = {}
        for key, value in self._all_data_rows[0].items():
            if key in index_keys:
                features[key] = {"dtype": "int64", "shape": [1], "names": None, "fps": fps_int}
            elif key == "timestamp":
                features[key] = {"dtype": "float32", "shape": [1], "names": None, "fps": fps_int}
            elif key == "next.done":
                features[key] = {"dtype": "bool", "shape": [1], "names": None, "fps": fps_int}
            elif isinstance(value, list):
                features[key] = {
                    "dtype": "float32",
                    "shape": [len(value)],
                    "names": self._get_feature_names(key, len(value)),
                    "fps": fps_int,
                }
        features.update(self._video_features)

        n_episodes = len(self._episodes_metadata)
        total_videos = (
            len({c for c, _, _ in self._video_files}) if self.concatenate_videos else len(self._video_files)
        )
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "total_episodes": n_episodes,
            "total_frames": len(self._all_data_rows),
            "total_tasks": len(self._tasks),
            "total_videos": total_videos,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": fps_int,
            "splits": {"train": f"0:{n_episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 500,
        }
        with open(self.lerobot_dir / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        print("  Wrote info.json")

    def _get_feature_names(self, key: str, length: int) -> dict | list | None:
        """Human-readable names per feature dimension."""
        if "joint" in key.lower() or key == "observation.state":
            return {"motors": [f"joint_{i}" for i in range(length)]}
        if key == "action":
            if length == 7:
                return {"motors": ["x", "y", "z", "rx", "ry", "rz", "gripper"]}
            if length == 8:
                return {"motors": ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper"]}
            return {"motors": [f"action_{i}" for i in range(length)]}
        if "position" in key.lower():
            if length == 3:
                return ["x", "y", "z"]
            if length == 7:
                return ["x", "y", "z", "qx", "qy", "qz", "qw"]
        if "orientation" in key.lower() and length == 4:
            return ["qx", "qy", "qz", "qw"]
        return None

    # ------------------------------------------------------------------ #
    # video build
    # ------------------------------------------------------------------ #
    def _build_videos(self):
        """Build one concatenated mp4 per camera (default) or one mp4 per episode."""
        if self.concatenate_videos:
            self._build_concatenated_videos_merged()
        else:
            self._build_videos_one_per_episode()

    def _build_videos_one_per_episode(self):
        """Re-encode one avc1 mp4 per episode (file-000.mp4, file-001.mp4, ...)."""
        for camera_name, src_path, episode_idx in self._video_files:
            path = Path(src_path)
            if not path.exists():
                raise FileNotFoundError(f"Source video does not exist: {path}")
            camera_dir = self.lerobot_dir / "videos" / camera_name / "chunk-000"
            camera_dir.mkdir(parents=True, exist_ok=True)
            dst_path = camera_dir / f"file-{episode_idx:03d}.mp4"
            ep_len = self._episodes_metadata[episode_idx]["length"]
            self._reencode_to_avc1(path, dst_path, max_frames=ep_len)
            duration = self._get_video_duration_seconds(dst_path)
            self._video_timestamps[(camera_name, episode_idx)] = (episode_idx, 0.0, duration)
        if self._video_files:
            print(f"  Wrote {len(self._video_files)} episode videos (one file per episode, avc1)")

    def _build_concatenated_videos_merged(self):
        """Build one avc1 mp4 per camera; episodes are addressed by from/to_timestamp. Needs ffmpeg.

        Each clip is re-encoded to avc1 first, then concatenated with ``-c copy``.
        """
        import shlex

        for camera_name in self._video_features:
            entries = sorted(
                ((p, e) for c, p, e in self._video_files if c == camera_name),
                key=lambda x: x[1],
            )
            if not entries:
                continue

            valid = [(Path(p), ep_idx) for p, ep_idx in entries]
            for path, _ in valid:
                if not path.exists():
                    raise FileNotFoundError(f"Source video does not exist: {path}")

            camera_dir = self.lerobot_dir / "videos" / camera_name / "chunk-000"
            camera_dir.mkdir(parents=True, exist_ok=True)
            out_path = camera_dir / "file-000.mp4"

            temp_dir = tempfile.mkdtemp()
            try:
                temp_paths = []
                cumulative_ts = 0.0
                n_clips = len(valid)
                t0 = time.time()
                print(f"  [video:{camera_name}] re-encoding {n_clips} clips to avc1...", flush=True)
                for vi, (path, ep_idx) in enumerate(valid):
                    if vi % 500 == 0 or vi == n_clips - 1:
                        el = time.time() - t0
                        done = vi + 1
                        eta = el / done * (n_clips - done)
                        print(
                            f"  [video:{camera_name}] {done}/{n_clips} ({100 * done // n_clips}%) "
                            f"elapsed={el:.0f}s ETA={eta:.0f}s",
                            flush=True,
                        )
                    temp_f = Path(temp_dir) / f"ep_{ep_idx:04d}.mp4"
                    ep_len = self._episodes_metadata[ep_idx]["length"]
                    self._reencode_to_avc1(path, temp_f, max_frames=ep_len)
                    duration = self._get_video_duration_seconds(temp_f)
                    temp_paths.append(temp_f)
                    self._video_timestamps[(camera_name, ep_idx)] = (
                        0,
                        cumulative_ts,
                        cumulative_ts + duration,
                    )
                    cumulative_ts += duration

                # concat avc1 segments with -c copy
                concat_path = Path(temp_dir) / "concat_list.txt"
                with open(concat_path, "w") as f:
                    for t in temp_paths:
                        f.write(f"file {shlex.quote(str(t.absolute()))}\n")
                # stdin closed for the same reason as in _reencode_to_avc1.
                cmd = [
                    "ffmpeg", "-nostdin", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_path),
                    "-c", "copy",
                    str(out_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, stdin=subprocess.DEVNULL)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg concat failed for {camera_name}: "
                        f"{result.stderr[-500:] if result.stderr else result.stdout[-500:]}"
                    )
                print(
                    f"  Concatenated {len(valid)} videos (avc1) -> "
                    f"{out_path.relative_to(self.lerobot_dir)}"
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)


def export_to_lerobot(
    robolab_output_dir: str,
    lerobot_output_dir: str | None = None,
    robot_type: str = "franka",
    fps: float = 20.0,
    concatenate_videos: bool = True,
) -> Path:
    """Convenience wrapper: convert a RoboLab output directory to LeRobot v3.0."""
    return LeRobotExporter(
        robolab_output_dir=robolab_output_dir,
        lerobot_output_dir=lerobot_output_dir,
        robot_type=robot_type,
        fps=fps,
        concatenate_videos=concatenate_videos,
    ).export()
