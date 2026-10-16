r"""Overlay FR3 end-effector trajectories on the figure view of a fixed scene (qualitative figures).

Each cell directory holds ``ep_*.npz`` rollouts from ``rollout_fixed_init.py`` (same fixed scene); the view
directory holds ``background.png`` and ``camera.json`` from ``rollout_fixed_init.py --export-view``. The
fingertip-centre (TCP) path of each successful rollout, up to its first success, is projected onto the
background, one panel per cell, plus a strip of all panels. Isaac is not needed here.

Per-task cuts used in the paper figures: on FMB-SqCircle each path is cut 40 frames after it first reaches
board height; on FMB-Round the final lateral drag along the board before the vertical insertion is removed.

Usage
-----
    python libs/real2sim/scripts/render_eef_2d.py --task fmb_round --view-dir <view> \
        --cells <rollouts>/discodemo:DiscoDemo:discodemo,<rollouts>/prfcl:P-RFCL:prfcl --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from real2sim.peg_board import HAND_TO_TCP_M

PALETTE = {"discodemo": "#d64a8b", "prfcl": "#1f77b4"}

#: Figure camera: looking from the front of the table (env frame), square 640 px, 44 deg field of view.
FIGURE_CAMERA = {"pos": (1.20, 0.00, 0.50), "tgt": (0.55, 0.00, 0.20), "side": 640, "fov_deg": 44.0}

#: Offset from the reported hand frame to the fingertip centre: the observed ``eef_orientation`` is a
#: remapped frame, so the offset is along its local -y.
EE_TO_TCP_LOCAL = (0.0, -HAND_TO_TCP_M, 0.0)
#: Trajectories per panel, line width (pt) and opacity, white halo around the lines (opacity, radius px).
LIMIT = 24
LINE_WIDTH = 1.5
LINE_ALPHA = 0.55
HALO_ALPHA = 0.65
HALO_DILATE = 6.0
#: FMB-Round drag removal: height band above the final height, and the highest median height (above the final
#: height) of a lateral move that is treated as a drag along the board.
DRAG_DZ = 0.015
DRAG_MAX_DZ = 0.05


def tcp_from_hand(ee_pos: np.ndarray, ee_quat: np.ndarray) -> np.ndarray:
    """Fingertip-centre path ``[T, 3]`` from the hand position ``[T, 3]`` and orientation (wxyz) ``[T, 4]``."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_quat(np.roll(np.asarray(ee_quat, dtype=np.float64), -1, axis=1))
    return np.asarray(ee_pos, dtype=np.float64) + rot.apply(np.asarray(EE_TO_TCP_LOCAL, dtype=np.float64))


def cam_params(pos, tgt, side: int, fov_deg: float, origin=(0.0, 0.0, 0.0)) -> dict:
    """Build projection parameters analytically from a camera pose.

    Parameters
    ----------
    pos : array-like
        Camera position (env-relative) [3].
    tgt : array-like
        Look-at point [3].
    side : int
        Square frame side (px).
    fov_deg : float
        Horizontal = vertical field of view (deg).
    origin : array-like, optional
        Env origin.

    Returns
    -------
    dict
        ``K``, ``R_ros``, ``pos_w`` and ``origin``.
    """
    p0 = np.asarray(pos, dtype=np.float64)
    fwd = np.asarray(tgt, dtype=np.float64) - p0
    fwd /= np.linalg.norm(fwd)
    xc = np.cross(np.array([0.0, 0.0, 1.0]), -fwd)
    xc /= np.linalg.norm(xc)
    yc = np.cross(-fwd, xc)
    # OpenGL (x right, y up, z back) -> ROS (x right, y down, z forward).
    R_ros = np.stack([xc, -yc, fwd], axis=1)
    f = side / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    K = np.array([[f, 0.0, side / 2.0], [0.0, f, side / 2.0], [0.0, 0.0, 1.0]])
    return {
        "K": K.tolist(),
        "R_ros": R_ros.tolist(),
        "pos_w": (p0 + np.asarray(origin, dtype=np.float64)).tolist(),
        "origin": list(map(float, origin)),
    }


def project(pts_w: np.ndarray, K: np.ndarray, R_ros: np.ndarray, pos_w: np.ndarray) -> np.ndarray:
    """Project world points [N,3] to pixels [N,2]; points behind the camera are NaN.

    Parameters
    ----------
    pts_w : numpy.ndarray
        World points [N,3].
    K : numpy.ndarray
        3x3 intrinsics.
    R_ros : numpy.ndarray
        Camera rotation (ROS convention), 3x3.
    pos_w : numpy.ndarray
        Camera world position [3].

    Returns
    -------
    numpy.ndarray
        Pixel coordinates [N,2].
    """
    xc = (pts_w - pos_w[None, :]) @ R_ros
    z = xc[:, 2]
    uvw = xc @ K.T
    px = np.full((pts_w.shape[0], 2), np.nan, dtype=np.float64)
    ok = z > 1e-6
    px[ok, 0] = uvw[ok, 0] / z[ok]
    px[ok, 1] = uvw[ok, 1] / z[ok]
    return px


#: Height band above the final height treated as "on the board" (m).
BOARD_BAND_M = 0.09


#: ``--cut-reach-cells``: board-height margin (m) and frames kept after reaching it.
REACH_DZ = 0.02
REACH_PAD = 40


def cut_after_reach(trajs: list[np.ndarray]) -> list[np.ndarray]:
    """Cut each trajectory ``REACH_PAD`` frames after it first reaches board height.

    Parameters
    ----------
    trajs : list of numpy.ndarray
        ``[T,3]`` TCP trajectories (env-relative).

    Returns
    -------
    list of numpy.ndarray
        Cut trajectories. Board height = median of per-trajectory minimum z + ``REACH_DZ``.
    """
    assert trajs, "no trajectories to cut"
    zc = float(np.median([t[:, 2].min() for t in trajs])) + REACH_DZ
    out = []
    for t in trajs:
        hit = np.flatnonzero(t[:, 2] <= zc)
        out.append(t[: int(hit[0]) + REACH_PAD + 1] if len(hit) else t)
    return out


def remove_final_drag(t: np.ndarray) -> np.ndarray:
    """Drop the final lateral drag along the board that is followed only by a vertical plunge.

    Motion is measured over a ``w``-step window because slow drags are not visible step by step.
    """
    drag_dz = DRAG_DZ
    # Drop the tail that stays within drag_dz of the final height.
    hi = np.flatnonzero(t[:, 2] > t[-1, 2] + drag_dz)
    if len(hi):
        t = t[: int(hi[-1]) + 2]
    # Then drop a final lateral drag along the board surface that is followed only
    # by a vertical plunge. Motion is measured over a w-step window because slow
    # drags are not visible step by step.
    w = 8
    if len(t) > w + 2:
        dxy = np.linalg.norm(t[w:, :2] - t[:-w, :2], axis=1)
        dz = np.abs(t[w:, 2] - t[:-w, 2])
        lat = (dxy > 0.003) & (dz < 0.5 * dxy)
        seg = np.linalg.norm(np.diff(t, axis=0), axis=1)
        after = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]  # path length remaining after point k
        i = len(lat) - 1
        while i >= 0 and not lat[i]:
            i -= 1
        k = min(i + w, len(t) - 1)
        # What follows must be a short, nearly vertical plunge.
        plunge = after[k] <= BOARD_BAND_M and np.linalg.norm(t[-1, :2] - t[k, :2]) <= 0.015
        if i >= 0 and plunge:
            # Walk back through the drag; stop only at a clear descent longer than w.
            desc = (dz >= 0.003) & (dz >= 0.5 * dxy)
            j, start, gap = i, i, 0
            while j >= 0:
                if lat[j]:
                    start, gap = j, 0
                elif desc[j]:
                    gap += 1
                    if gap > w:
                        break
                j -= 1
            # Only a lateral move near board height counts as a drag (sideways alignment higher above the board
            # before a slow vertical insertion does not).
            drag_h = float(np.median(t[start : i + w + 1, 2]) - t[-1, 2])
            lateral = np.linalg.norm(t[i + w, :2] - t[start, :2]) > 0.01 or start < i - w
            if lateral and drag_h <= DRAG_MAX_DZ:
                t = t[: max(2, start + 1)]
    return t


def load_cell(d: Path, task: str) -> list[np.ndarray]:
    """TCP paths ``[T, 3]`` (env frame) of the first ``LIMIT`` successful rollouts of one cell, cut at success."""
    out = []
    for f in sorted(d.glob("ep_*.npz")):
        z = np.load(f)
        hold = int(z["success_hold_steps"])
        hit = np.flatnonzero(np.asarray(z["succ_streak"]) >= hold)
        if not len(hit):
            continue
        # The first step of the success hold that completed.
        end = max(2, int(hit[0]) - (hold - 1) + 1)
        t = tcp_from_hand(z["ee_pos"], z["ee_quat"])[:end].copy()
        if task == "fmb_round" and len(t) > 2:
            t = remove_final_drag(t)
        out.append(t)
        if len(out) >= LIMIT:
            break
    if task == "fmb_sqcircle":
        out = cut_after_reach(out)
    return out


def _halo(bg: np.ndarray, pxs: list[np.ndarray], lw: float, alpha: float, dilate: float) -> np.ndarray:
    """Lighten the background only around the drawn lines.

    Parameters
    ----------
    bg : numpy.ndarray
        Background (H, W, 3), uint8 or float in [0, 1].
    pxs : list of numpy.ndarray
        Projected pixels per trajectory (N, 2); NaN breaks the line.
    lw : float
        Line width (pt), also used for the mask.
    alpha : float
        White opacity (0-1).
    dilate : float
        Dilation radius (background pixels); edges are blurred with half of it.

    Returns
    -------
    numpy.ndarray
        Background with the halo, float in [0, 1].
    """
    import cv2
    import matplotlib.pyplot as plt  # main() selects the Agg backend first

    h, w = bg.shape[:2]
    fig = plt.figure(figsize=(w / 200, h / 200), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    for px in pxs:
        ax.plot(px[:, 0], px[:, 1], color="black", lw=lw, solid_capstyle="round")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., 0]
    plt.close(fig)
    assert buf.shape == (h, w), f"mask canvas {buf.shape} != background {(h, w)}"
    m = (buf < 128).astype(np.float32)
    r = max(int(round(dilate)), 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(dilate / 2.0, 0.5))
    f = bg[..., :3].astype(np.float64)
    f = f / 255.0 if bg.dtype == np.uint8 else f
    a_ = (alpha * np.clip(m, 0.0, 1.0))[..., None]
    return np.clip(f * (1.0 - a_) + a_, 0.0, 1.0)


def main() -> None:
    """Draw one overlay panel per cell and a strip of all panels."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("pnp_banana", "stack_cube", "fmb_round", "fmb_sqcircle"))
    ap.add_argument("--view-dir", required=True, help="background.png, camera.json and layout.json of the scene")
    ap.add_argument(
        "--cells",
        required=True,
        help="comma-separated 'rollout_dir:Title:discodemo|prfcl' (the last field picks the colour)",
    )
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    view = Path(a.view_dir)
    cam = json.loads((view / "camera.json").read_text())
    K, R_ros, pos_w, origin = (np.asarray(cam[k], dtype=np.float64) for k in ("K", "R_ros", "pos_w", "origin"))
    bg = np.asarray(imageio.imread(view / "background.png"))
    h, w = bg.shape[:2]
    # The insertion rows are shown enlarged in the paper, so their panels are drawn at twice the resolution.
    out_scale = 2.0 if a.task.startswith("fmb") else 1.0

    # [sanity] The pinned object positions must land on the objects in the background.
    for nm, p7 in json.loads((view / "layout.json").read_text()).items():
        px = project(np.asarray([p7[:3]], dtype=np.float64) + origin[None, :], K, R_ros, pos_w)[0]
        print(f"[sanity] {nm} env-rel={[round(v, 3) for v in p7[:3]]} -> px=({px[0]:.1f}, {px[1]:.1f})", flush=True)

    panels = []
    for spec in a.cells.split(","):
        cell_dir, title, color = spec.split(":")
        trajs = load_cell(Path(cell_dir), a.task)
        pxs = [project(t + origin[None, :], K, R_ros, pos_w) for t in trajs]
        pxs = [px for px in pxs if np.isfinite(px).all(axis=1).sum() >= 2]
        fig = plt.figure(figsize=(w / 200, h / 200), dpi=200)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(_halo(bg, pxs, LINE_WIDTH, HALO_ALPHA, HALO_DILATE))
        for px in pxs:
            ax.plot(px[:, 0], px[:, 1], color=PALETTE[color], lw=LINE_WIDTH, alpha=LINE_ALPHA, solid_capstyle="round")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        f = out_dir / f"panel_{Path(cell_dir).name}.png"
        fig.savefig(f, dpi=200 * out_scale)
        plt.close(fig)
        print(f"[png] {Path(cell_dir).name}: {len(pxs)} traj -> {f}", flush=True)
        panels.append((f, title))

    # Reserve room for the titles and save with a tight bbox so they are not clipped.
    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.28), dpi=220)
    for ax, (f, title) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.imshow(imageio.imread(f))
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.01, wspace=0.03)
    fig.savefig(out_dir / "strip.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[png] strip -> {out_dir / 'strip.png'}", flush=True)


if __name__ == "__main__":
    main()
