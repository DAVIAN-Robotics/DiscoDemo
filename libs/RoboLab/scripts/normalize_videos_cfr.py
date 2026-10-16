"""Re-encode every video of a LeRobot dataset to clean constant frame rate (CFR).

``aggregate_datasets`` concatenates videos with PyAV and can leave +-1 timebase PTS jitter in the merged mp4s,
which makes torchcodec miss frames at ``k / fps`` during training. Re-encoding with ``-fps_mode cfr`` puts every
frame back on the grid; frame counts are checked so episode timestamps stay valid. The fps is read from
``meta/info.json``; ``--fps`` is only a cross-check.

usage::

    python libs/RoboLab/scripts/normalize_videos_cfr.py <dataset_root> [--workers 4] [--crf 18]
"""

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path

import av
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _nframes(path: Path) -> int:
    c = av.open(str(path))
    n = c.streams.video[0].frames or sum(1 for _ in c.decode(video=0))
    c.close()
    return n


def resolve_fps(root: str | Path, fps: int | None) -> int:
    """Dataset fps from ``meta/info.json``.

    Parameters
    ----------
    root : str | Path
        LeRobot dataset root.
    fps : int | None
        Optional cross-check value.

    Returns
    -------
    int
        Dataset fps.

    Raises
    ------
    ValueError
        If ``fps`` differs from info.json (re-encoding would add or drop frames).
    """
    ds_fps = int(json.loads((Path(root) / "meta" / "info.json").read_text())["fps"])
    if fps is not None and int(fps) != ds_fps:
        raise ValueError(
            f"--fps {fps} != dataset fps {ds_fps} ({root}/meta/info.json); frames would be added or dropped"
        )
    return ds_fps


def normalize_one(path: Path, fps: int, crf: int) -> str:
    """Re-encode one mp4 in place to CFR; raise if the frame count changes."""
    before = _nframes(path)
    tmp = path.with_suffix(".cfr_tmp.mp4")
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-fps_mode",
        "cfr",
        "-crf",
        str(crf),
        "-an",
        str(tmp),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg CFR re-encode failed for {path}: {r.stderr[-400:]}")
    after = _nframes(tmp)
    if after != before:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"frame count changed for {path}: {before} -> {after}")
    tmp.replace(path)
    return f"{path.name}: {before} frames (CFR OK)"


def normalize_dataset(root: str, fps: int | None = None, workers: int = 4, crf: int = 18) -> None:
    """Re-encode all mp4s of a dataset to CFR at the dataset fps.

    Parameters
    ----------
    root : str
        LeRobot dataset root.
    fps : int | None
        Optional cross-check (``resolve_fps``).
    workers : int
        Parallel ffmpeg processes.
    crf : int
        x264 CRF.
    """
    fps = resolve_fps(root, fps)
    vids = sorted(Path(root).glob("videos/**/*.mp4"))
    assert vids, f"no videos: {root}/videos/**/*.mp4"
    print(f"[cfr] {len(vids)} videos -> clean CFR ({fps}fps, crf{crf}, {workers} workers)", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(normalize_one, v, fps, crf): v for v in vids}
        for done, f in enumerate(concurrent.futures.as_completed(futs), start=1):
            f.result()
            print(f"[cfr] {done}/{len(vids)} {futs[f].name}", flush=True)
    print("CFR_NORMALIZE_DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--fps", type=int, default=None, help="cross-check; must equal fps in meta/info.json")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--crf", type=int, default=18)
    a = p.parse_args()
    normalize_dataset(a.root, a.fps, a.workers, a.crf)
