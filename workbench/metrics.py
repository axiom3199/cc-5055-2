"""Evaluation against synthetic ground truth: camera trajectory errors,
reprojection statistics, ground-contact error of map points, and virtual
marker 'swimming'. Only used when ground truth exists (test footage)."""
import numpy as np

from . import geometry as geo


def gt_cameras(gt):
    """Return dict frame -> (R_wc, t_wc, C) from gt.json structure."""
    out = {}
    for fr in gt["frames"]:
        R = np.array(fr["R_wc"]).reshape(3, 3)
        t = np.array(fr["t_wc"])
        out[fr["idx"]] = (R, t, geo.camera_center(R, t))
    return out


def match_track_to_gt(tracks, gt, max_dist_px=3.0):
    """Associate each track with a GT scene point (nearest projection in the
    birth frame) or, for texture-corner tracks on the ground, with the
    ray-ground intersection. Returns dict track_id -> (gt_pid or None, xyz)."""
    gt_pts = np.asarray(gt["points"])
    K = np.asarray(gt["K"])
    gt_cams = gt_cameras(gt)
    out = {}
    for tr in tracks:
        born = int(tr["frames"][0])
        R, t, _ = gt_cams[born]
        Xc = geo.world_to_cam(R, t, gt_pts)
        uv = geo.project_points(Xc, K)
        d2 = (uv[:, 0] - tr["xy"][0][0]) ** 2 + (uv[:, 1] - tr["xy"][0][1]) ** 2
        d2[Xc[:, 2] < 0.5] = np.inf
        gid = int(np.argmin(d2))
        if d2[gid] <= max_dist_px ** 2:
            out[tr["id"]] = (gid, gt_pts[gid].copy())
            continue
        # texture corner: intersect the birth-frame ray with the ground z=0
        c2w = np.asarray(gt["frames"][born]["cam_to_world"]).reshape(4, 4)
        ray = c2w[:3, :3] @ (np.linalg.inv(K) @ np.array([tr["xy"][0][0], tr["xy"][0][1], 1.0]))
        Cw = c2w[:3, 3]
        if abs(ray[2]) < 1e-9:
            continue
        tt = -Cw[2] / ray[2]
        if tt <= 0:
            continue
        out[tr["id"]] = (None, Cw + tt * ray)
    return out


def _alignment_to_gt(solution, gt):
    """Rotation-informed similarity alignment of the solution to GT."""
    gt_cams = gt_cameras(gt)
    frames = sorted(solution.cameras.keys())
    frames = [f for f in frames if f in gt_cams]
    C_sol = np.array([geo.camera_center(*solution.cameras[f]) for f in frames])
    C_gt = np.array([gt_cams[f][2] for f in frames])
    G = np.array([solution.cameras[f][0].T @ gt_cams[f][0] for f in frames])
    R_al = geo.project_to_so3(G.mean(axis=0)).T
    A, b = [], []
    for i in range(len(frames)):
        Rc = R_al @ C_sol[i]
        for k in range(3):
            row = np.zeros(4)
            row[0] = Rc[k]
            row[1 + k] = 1.0
            A.append(row)
            b.append(C_gt[i][k])
    x, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return float(x[0]), R_al, x[1:4]


def evaluate_cameras(solution, gt):
    """Align the solved trajectory to GT and report per-frame position (m) and
    rotation (deg) errors.

    Alignment: a planar/shallow camera arc makes the rotation of a
    position-only Umeyama fit weakly observable, so we determine the alignment
    rotation from the orientation residuals (SO(3) average of R_sol^T R_gt)
    and then fit scale+translation on the camera centers. This is the honest
    way to compare trajectories; both alignment variants are reported.
    """
    gt_cams = gt_cameras(gt)
    frames = sorted(solution.cameras.keys())
    frames = [f for f in frames if f in gt_cams]
    C_sol = np.array([geo.camera_center(*solution.cameras[f]) for f in frames])
    C_gt = np.array([gt_cams[f][2] for f in frames])

    # rotation from orientation residuals
    G = np.array([solution.cameras[f][0].T @ gt_cams[f][0] for f in frames])
    R_al = geo.project_to_so3(G.mean(axis=0)).T
    # scale + translation from centers given R_al
    A, b = [], []
    for i in range(len(frames)):
        Rc = R_al @ C_sol[i]
        for k in range(3):
            row = np.zeros(4)
            row[0] = Rc[k]
            row[1 + k] = 1.0
            A.append(row)
            b.append(C_gt[i][k])
    x, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    s, t_al = float(x[0]), x[1:4]

    # reference: position-only Umeyama (weakly observable rotation)
    s_u, R_u, t_u = geo.umeyama(C_sol, C_gt, with_scale=True)

    rows = []
    for i, f in enumerate(frames):
        C_al = s * (R_al @ C_sol[i]) + t_al
        pos_err = float(np.linalg.norm(C_al - C_gt[i]))
        R_sol = solution.cameras[f][0]
        R_gt = gt_cams[f][0]
        rot_err = geo.rotation_angle_deg(R_sol @ R_al.T, R_gt)
        rows.append({"frame": f, "pos_err_m": pos_err, "rot_err_deg": rot_err,
                     "segment": gt["frames"][f]["segment"]})
    pos = np.array([r["pos_err_m"] for r in rows])
    rot = np.array([r["rot_err_deg"] for r in rows])
    return {
        "scale_recovered": s,
        "alignment": {"s": s, "R": R_al.tolist(), "t": t_al.tolist()},
        "alignment_umeyama_centers": {"s": s_u, "R": R_u.tolist(), "t": t_u.tolist()},
        "per_frame": rows,
        "pos_rmse_m": float(np.sqrt(np.mean(pos ** 2))),
        "pos_max_m": float(pos.max()),
        "rot_rmse_deg": float(np.sqrt(np.mean(rot ** 2))),
        "rot_max_deg": float(rot.max()),
    }


def reprojection_stats(solution):
    per_frame = {}
    for (pid, f, x, y), r, outl in zip(solution.observations, solution.obs_residual,
                                       solution.obs_outlier):
        per_frame.setdefault(f, []).append(r)
    rows = []
    for f in sorted(per_frame):
        arr = np.asarray(per_frame[f])
        rows.append({"frame": f, "n_obs": len(arr), "mean_px": float(arr.mean()),
                     "p95_px": float(np.percentile(arr, 95))})
    allr = np.asarray(solution.obs_residual) if solution.obs_residual else np.zeros(1)
    return {"per_frame": rows,
            "mean_px": float(allr.mean()),
            "median_px": float(np.median(allr)),
            "p95_px": float(np.percentile(allr, 95)),
            "n_outliers_rejected": int(sol_outlier_count(solution))}


def sol_outlier_count(solution):
    return sum(1 for o in solution.obs_outlier if o)


def ground_contact_error(solution, gt, rot_segments=None):
    """For map points born during pure-rotation segments whose GT point lies on
    the ground: |height error| after GT similarity alignment. Also reports
    coverage (deferred tracks have no point - the honest outcome of 'delayed').
    """
    gt_ground = np.asarray(gt["ground_mask"], bool)
    rot_frames = {f["idx"] for f in gt["frames"] if f["segment"] == "rotate"}
    track_gt = match_track_to_gt(solution._tracks.values(), gt)
    s, R_al, t_al = _alignment_to_gt(solution, gt)

    rows = []
    n_deferred = 0
    for tr in solution._tracks.values():
        born = int(tr["frames"][0])
        if born not in rot_frames or tr["id"] not in track_gt:
            continue
        gid, gxyz = track_gt[tr["id"]]
        on_ground = bool(gt_ground[gid]) if gid is not None else True
        if not on_ground:
            continue
        tid = tr["id"]
        if tid not in solution.points:
            n_deferred += 1
            continue
        X_al = geo.apply_similarity(s, R_al, t_al, solution.points[tid])
        z_err = float(X_al[2] - gxyz[2])
        rows.append({"track": int(tid), "born_frame": born,
                     "inherited": bool(solution.point_meta[tid]["inherited"]),
                     "z_err_m": z_err, "abs_z_err_m": abs(z_err)})
    arr = np.array([r["abs_z_err_m"] for r in rows]) if rows else np.zeros(1)
    return {"n_ground_tracks_born_in_rotation": len(rows) + n_deferred,
            "n_deferred_no_estimate": n_deferred,
            "n_estimated": len(rows),
            "median_abs_z_err_m": float(np.median(arr)),
            "mean_abs_z_err_m": float(arr.mean()),
            "p90_abs_z_err_m": float(np.percentile(arr, 90)),
            "max_abs_z_err_m": float(arr.max()),
            "rows": rows}


def marker_swimming(solution, gt, marker_vertices_display, display_transform):
    """How much does the virtual marker swim against the background? Project
    the marker corners with the solved cameras and with the GT cameras. The
    marker lives in display coordinates; the solved cameras see it via the
    display transform, the GT cameras see it via the solution->GT alignment.
    Per-frame pixel deviation is the visible swim."""
    gt_cams = gt_cameras(gt)
    K = np.asarray(gt["K"])
    V = np.atleast_2d(marker_vertices_display)
    M = np.asarray(display_transform["matrix"], float)
    Minv = np.linalg.inv(M)
    V_raw = V @ Minv[:3, :3].T + Minv[:3, 3]           # display -> raw solution
    s, R_al, t_al = _alignment_to_gt(solution, gt)      # raw -> GT world
    V_gt = geo.apply_similarity(s, R_al, t_al, V_raw)
    rows = []
    for f in sorted(solution.cameras.keys()):
        if f not in gt_cams:
            continue
        R_s, t_s = solution.cameras[f]
        R_g, t_g, _ = gt_cams[f]
        uv_s = geo.project_world(R_s, t_s, V_raw, K)
        uv_g = geo.project_world(R_g, t_g, V_gt, K)
        dev = np.linalg.norm(uv_s - uv_g, axis=1)
        rows.append({"frame": int(f), "mean_px": float(dev.mean()),
                     "max_px": float(dev.max())})
    if not rows:
        return {"per_frame": [], "mean_px": None, "p95_px": None}
    allm = np.array([r["mean_px"] for r in rows])
    return {"per_frame": rows, "mean_px": float(allm.mean()),
            "p95_px": float(np.percentile(allm, 95))}
