"""Matplotlib figures for the web UI, rendered to PNG bytes. Headless Agg."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import geometry as geo
from . import metrics as metrics_mod


def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_trajectory(sol, gt=None, display_transform=None):
    """Top-down view of solved camera centers (+ GT if available)."""
    frames = sorted(sol.cameras)
    C = np.array([geo.camera_center(*sol.cameras[f]) for f in frames])
    rot_only = np.array([bool(sol.frame_info.get(f, {}).get("rotation_only"))
                         for f in frames])
    fig, ax = plt.subplots(figsize=(6, 5.4))
    if gt is not None:
        gtc = metrics_mod.gt_cameras(gt)
        Cg = np.array([gtc[f][2] for f in frames if f in gtc])
        # align solution to GT for a comparable overlay
        s, R_al, t_al = metrics_mod._alignment_to_gt(sol, gt)
        Ca = C @ (s * R_al).T + t_al
        ax.plot(Cg[:, 0], Cg[:, 1], "-", color="0.4", lw=1.5, label="ground truth")
        ax.plot(Ca[:, 0], Ca[:, 1], "-", color="tab:blue", lw=1.2, label="solved (aligned)")
        ax.scatter(Ca[rot_only, 0], Ca[rot_only, 1], c="tab:red", s=14,
                   label="rotation-only", zorder=3)
    else:
        ax.plot(C[:, 0], C[:, 1], "-o", ms=2, color="tab:blue", label="solved (raw, up-to-scale)")
        ax.scatter(C[rot_only, 0], C[rot_only, 1], c="tab:red", s=14, label="rotation-only")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("camera trajectory (top view)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def plot_camera_errors(metrics):
    cam = metrics["cameras"]
    pf = cam["per_frame"]
    f = [r["frame"] for r in pf]
    pos = [r["pos_err_m"] for r in pf]
    rot = [r["rot_err_deg"] for r in pf]
    seg = [r["segment"] for r in pf]
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    for ax, vals, name in [(axes[0], pos, "position error (m)"),
                           (axes[1], rot, "rotation error (deg)")]:
        ax.plot(f, vals, ".-", ms=3, lw=0.8, color="tab:blue")
        _shade_segments(ax, f, seg)
        ax.set_ylabel(name, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"pos RMSE {cam['pos_rmse_m']:.3f} m   rot RMSE {cam['rot_rmse_deg']:.3f} deg",
                      fontsize=9)
    axes[1].set_xlabel("frame")
    return _fig_to_png(fig)


def _shade_segments(ax, frames, segs):
    """Shade pure-rotation segments."""
    in_rot = [s == "rotate" for s in segs]
    start = None
    ymax = ax.get_ylim()[1]
    for i, r in enumerate(in_rot):
        if r and start is None:
            start = frames[i]
        if start is not None and (not r or i == len(in_rot) - 1):
            end = frames[i]
            ax.axvspan(start, end, color="tab:red", alpha=0.12)
            start = None
    ax.set_ylim(0, ymax)


def plot_parallax(sol, gt=None):
    frames = sorted(sol.cameras)
    par = [sol.frame_info.get(f, {}).get("parallax_deg") for f in frames]
    rot = [bool(sol.frame_info.get(f, {}).get("rotation_only")) for f in frames]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    xs = [f for f, p in zip(frames, par) if p is not None]
    ys = [p for p in par if p is not None]
    ax.plot(xs, ys, ".-", ms=3, lw=0.8, color="tab:blue", label="parallax (deg)")
    ax.axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax.text(len(frames) * 0.01, 1.05, "degeneracy threshold", fontsize=7, color="0.4")
    rx = [f for f, r in zip(frames, rot) if r]
    if rx:
        ax.scatter(rx, [0.05] * len(rx), c="tab:red", s=16, marker="|",
                   label="flagged rotation-only")
    ax.set_xlabel("frame")
    ax.set_ylabel("parallax (deg)", fontsize=8)
    ax.set_title("motion degeneracy: parallax per frame (rotation segments have ~0)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def plot_reprojection(metrics):
    reproj = metrics["reprojection"]
    pf = reproj["per_frame"]
    f = [r["frame"] for r in pf]
    mean = [r["mean_px"] for r in pf]
    p95 = [r["p95_px"] for r in pf]
    n = [r["n_obs"] for r in pf]
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.4), sharex=True)
    axes[0].plot(f, mean, ".-", ms=3, lw=0.8, label="mean", color="tab:blue")
    axes[0].plot(f, p95, ".-", ms=3, lw=0.8, label="p95", color="tab:orange")
    axes[0].set_ylabel("reprojection (px)", fontsize=8)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"reprojection error: mean {reproj['mean_px']:.3f}px, "
                      f"{reproj['n_outliers_rejected']} outliers rejected", fontsize=9)
    axes[1].bar(f, n, color="tab:green", width=1.0)
    axes[1].set_ylabel("observations", fontsize=8)
    axes[1].set_xlabel("frame")
    axes[1].grid(alpha=0.3)
    return _fig_to_png(fig)


def plot_track_quality(tracks, injected_flags=None):
    lens = np.array([len(tr["frames"]) for tr in tracks])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9))
    axes[0].hist(lens, bins=30, color="tab:blue")
    axes[0].set_xlabel("track length (frames)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{len(tracks)} tracks, mean {lens.mean():.1f} frames", fontsize=9)
    # observations per frame
    n_frames = int(max(np.concatenate([tr["frames"] for tr in tracks]))) + 1
    per = np.zeros(n_frames, int)
    for tr in tracks:
        for fr in tr["frames"]:
            per[int(fr)] += 1
    axes[1].plot(per, lw=1.0, color="tab:green")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("active tracks")
    axes[1].set_title("track density (no synchronized collapse)", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def plot_marker_swimming(metrics):
    ms = metrics.get("marker_swimming")
    if not ms or not ms.get("per_frame"):
        return None
    f = [r["frame"] for r in ms["per_frame"]]
    mean = [r["mean_px"] for r in ms["per_frame"]]
    mx = [r["max_px"] for r in ms["per_frame"]]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(f, mean, ".-", ms=3, lw=0.8, label="mean", color="tab:blue")
    ax.plot(f, mx, ".-", ms=3, lw=0.8, label="max", color="tab:red")
    ax.set_xlabel("frame")
    ax.set_ylabel("marker swim (px)", fontsize=8)
    ax.set_title(f"virtual marker swimming vs background: mean {ms['mean_px']:.2f}px, "
                 f"p95 {ms['p95_px']:.2f}px", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _fig_to_png(fig)


def plot_ground_contact_compare(res_delayed, res_inherited):
    """Bar comparison of the two triangulation strategies' ground-contact."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))
    labels = ["delayed", "inherited"]
    deferred = [res_delayed["n_deferred_no_estimate"], res_inherited["n_deferred_no_estimate"]]
    estimated = [res_delayed["n_estimated"], res_inherited["n_estimated"]]
    x = np.arange(2)
    axes[0].bar(x, estimated, label="estimated", color="tab:blue")
    axes[0].bar(x, deferred, bottom=estimated, label="deferred (no estimate)",
                color="0.7")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("rotation-born ground tracks")
    axes[0].set_title("coverage: deferred = honest 'no estimate'", fontsize=9)
    axes[0].legend(fontsize=7)
    med = [res_delayed["median_abs_z_err_m"], res_inherited["median_abs_z_err_m"]]
    p90 = [res_delayed["p90_abs_z_err_m"], res_inherited["p90_abs_z_err_m"]]
    axes[1].bar(x - 0.18, med, width=0.36, label="median |z err|", color="tab:green")
    axes[1].bar(x + 0.18, p90, width=0.36, label="p90 |z err|", color="tab:orange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("ground-contact |z error| (m)")
    axes[1].set_title("ground-contact error of estimated points", fontsize=9)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    return _fig_to_png(fig)
