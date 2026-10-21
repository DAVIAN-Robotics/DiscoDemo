"""Convert one raw render dir (robolab schema: data.hdf5 + per-camera mp4) to a LeRobot v3.0 dataset.

Run one process per render shard to keep the conversion memory peak isolated.

usage::

    python libs/RoboLab/scripts/convert_raw_to_lerobot.py <raw_dir> <lerobot_dir>
"""

import argparse
import importlib.util
import os
import sys
import tempfile
import traceback

_EXPORTER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "robolab", "core", "export", "lerobot_exporter.py"
)


def convert_one(raw_dir: str, lerobot_dir: str) -> str:
    """Convert ``raw_dir`` to a LeRobot dataset at ``lerobot_dir`` (20 fps) and return the output path."""
    import imageio_ffmpeg

    # Expose the ffmpeg bundled with imageio as `ffmpeg` on PATH for the exporter.
    ffd = tempfile.mkdtemp()
    os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), os.path.join(ffd, "ffmpeg"))
    os.environ["PATH"] = ffd + os.pathsep + os.environ.get("PATH", "")

    # Load the exporter as a standalone module so the robolab package (and Isaac) is not imported.
    spec = importlib.util.spec_from_file_location("lerobot_exporter_standalone", _EXPORTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.export_to_lerobot(
        robolab_output_dir=os.path.abspath(raw_dir), lerobot_output_dir=os.path.abspath(lerobot_dir)
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert a raw render dir to a LeRobot v3.0 dataset.")
    ap.add_argument("raw_dir")
    ap.add_argument("lerobot_dir")
    code = 0
    try:
        a = ap.parse_args()
        out = convert_one(a.raw_dir, a.lerobot_dir)
        print(f"CONVERT_OK {a.raw_dir} -> {out}", flush=True)
    except SystemExit as e:  # argparse --help / usage errors
        code = e.code if isinstance(e.code, int) else int(e.code is not None)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
