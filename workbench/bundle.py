"""Bundle adjustment: joint refinement of camera poses and 3D points by
minimizing reprojection error with a robust loss. No trajectory smoothing is
applied anywhere - cameras are only constrained by their observations."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from . import geometry as geo


def _rodrigues_batch(rvecs):
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    theta_safe = np.where(theta < 1e-12, 1.0, theta)
    k = rvecs / theta_safe
    Kx = np.zeros((len(rvecs), 3, 3))
    Kx[:, 0, 1] = -k[:, 2]; Kx[:, 0, 2] = k[:, 1]
    Kx[:, 1, 0] = k[:, 2];  Kx[:, 1, 2] = -k[:, 0]
    Kx[:, 2, 0] = -k[:, 1]; Kx[:, 2, 1] = k[:, 0]
    eye = np.eye(3)[None]
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    small = (theta[:, 0] < 1e-12)
    Rs = eye + s * Kx + (1.0 - c) * (Kx @ Kx)
    Rs[small] = np.eye(3)
    return Rs


def bundle_adjust(cameras, points, observations, K, max_nfev=100, f_scale=2.0,
                  loss="soft_l1", verbose=0):
    """cameras: dict frame_idx -> (R, t) world-to-cam.
    points: dict pid -> (3,) world point.
    observations: list of (pid, frame_idx, x, y).
    Gauge: the lowest-indexed frame is held fixed entirely; the second lowest
    keeps its translation fixed (rotation free) to anchor the arbitrary scale.

    Loss: soft_l1 (smooth sqrt-huber). A plain huber loss stalls when the init
    is not yet below f_scale - it downweights nearly every residual and the
    trust-region gradient collapses (optimality stays huge). soft_l1 keeps a
    usable gradient everywhere while still damping gross outliers; the explicit
    >4px rejection pass then removes the remaining ones.
    Returns (cameras_out, points_out, info).
    """
    frames = sorted(cameras.keys())
    if len(frames) < 2 or not points:
        return cameras, points, {"skipped": True}
    f_fixed, f_scale_fixed = frames[0], frames[1]
    # Gauge (7 dof): frame 0 fully fixed (6); frame 1 translation constrained to
    # a sphere of its initial radius (1 dof: scale) while its DIRECTION stays
    # free. Freezing the whole translation vector would over-constrain the
    # problem and bake the two-view baseline direction error into the solution.
    scale_anchor_d = float(np.linalg.norm(cameras[f_scale_fixed][1]))

    cam_index = {}   # frame -> (param_offset, dim)
    offset = 0
    for f in frames:
        if f == f_fixed:
            cam_index[f] = (None, 0)
        elif f == f_scale_fixed:
            cam_index[f] = (offset, 6)   # rvec(3) + translation direction(3)
            offset += 6
        else:
            cam_index[f] = (offset, 6)
            offset += 6
    pids = sorted(points.keys())
    pt_offset = {pid: offset + 3 * i for i, pid in enumerate(pids)}
    n_params = offset + 3 * len(pids)

    x0 = np.zeros(n_params)
    for f in frames:
        off, dim = cam_index[f]
        if dim == 0:
            continue
        R, t = cameras[f]
        x0[off:off + 3] = geo.rvec_from_matrix(R)
        if dim == 6:
            x0[off + 3:off + 6] = t
    for pid, i in pt_offset.items():
        x0[i:i + 3] = points[pid]

    obs = [(p, f, x, y) for (p, f, x, y) in observations
           if p in pt_offset and f in cam_index]
    n_obs = len(obs)
    if n_obs == 0:
        return cameras, points, {"skipped": True}

    obs_pid = np.array([o[0] for o in obs])
    obs_frame = np.array([o[1] for o in obs])
    obs_uv = np.array([[o[2], o[3]] for o in obs])
    frame_slot = {f: i for i, f in enumerate(frames)}
    obs_slot = np.array([frame_slot[f] for f in obs_frame])
    pt_slot = {pid: i for i, pid in enumerate(pids)}
    obs_ptslot = np.array([pt_slot[p] for p in obs_pid])

    fixed_rvec = {}
    for f in frames:
        off, dim = cam_index[f]
        if dim == 0:
            fixed_rvec[f] = geo.rvec_from_matrix(cameras[f][0])

    def residuals(x):
        rvecs = np.zeros((len(frames), 3))
        ts = np.zeros((len(frames), 3))
        for i, f in enumerate(frames):
            off, dim = cam_index[f]
            if dim == 0:
                rvecs[i] = fixed_rvec[f]
                ts[i] = cameras[f][1]
            elif f == f_scale_fixed:
                rvecs[i] = x[off:off + 3]
                d = x[off + 3:off + 6]
                nrm = np.linalg.norm(d)
                ts[i] = d * (scale_anchor_d / nrm) if nrm > 1e-12 else cameras[f][1]
            else:
                rvecs[i] = x[off:off + 3]
                ts[i] = x[off + 3:off + 6]
        pts_arr = x[offset:].reshape(-1, 3)
        X = pts_arr[obs_ptslot]
        Rs = _rodrigues_batch(rvecs[obs_slot])
        tsel = ts[obs_slot]
        Xc = np.einsum("nij,nj->ni", Rs, X) + tsel
        z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
        res = np.empty(2 * n_obs)
        res[0::2] = u - obs_uv[:, 0]
        res[1::2] = v - obs_uv[:, 1]
        return res

    S = lil_matrix((2 * n_obs, n_params), dtype=np.uint8)
    for k in range(n_obs):
        off, dim = cam_index[obs_frame[k]]
        if dim > 0:
            S[2 * k:2 * k + 2, off:off + dim] = 1
        pi = pt_offset[obs_pid[k]]
        S[2 * k:2 * k + 2, pi:pi + 3] = 1
    S = S.tocsr()

    cost_init = float(np.sum(residuals(x0) ** 2))
    sol = least_squares(residuals, x0, jac_sparsity=S, method="trf",
                        loss=loss, f_scale=f_scale, x_scale="jac",
                        max_nfev=max_nfev, verbose=verbose)

    cams_out = {}
    for f in frames:
        off, dim = cam_index[f]
        if dim == 0:
            cams_out[f] = cameras[f]
        elif f == f_scale_fixed:
            d = sol.x[off + 3:off + 6]
            nrm = np.linalg.norm(d)
            t = d * (scale_anchor_d / nrm) if nrm > 1e-12 else cameras[f][1]
            cams_out[f] = (geo.rodrigues(sol.x[off:off + 3]), t)
        else:
            cams_out[f] = (geo.rodrigues(sol.x[off:off + 3]), sol.x[off + 3:off + 6])
    pts_out = {pid: sol.x[i:i + 3].copy() for pid, i in pt_offset.items()}

    res = residuals(sol.x).reshape(-1, 2)
    info = {"cost_init": cost_init,
            "cost_final": float(2 * sol.cost),
            "nfev": int(sol.nfev), "optimality": float(sol.optimality),
            "n_obs": n_obs, "n_params": n_params,
            "rms_px": float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))),
            "residuals": res,
            "obs_pid": obs_pid, "obs_frame": obs_frame}
    return cams_out, pts_out, info
