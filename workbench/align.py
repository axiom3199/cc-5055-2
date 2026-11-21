"""User-supplied constraints -> display coordinate frame.

The raw SfM solution is up-to-scale (similarity). A known distance between two
points fixes scale; a set of ground points fixes the ground plane (normal -> +Z,
plane -> z=0). Yaw is fixed deterministically by mapping the first->last camera
ground-projected direction to +X. The transform is stored separately from the
raw solution.
"""
import numpy as np

from . import geometry as geo


def fit_ground_plane(points_xyz):
    """Robust plane fit: PCA initialization, then iteratively reweighted
    least squares to reject outlier points. Returns (centroid, normal)."""
    X = np.asarray(points_xyz, float)
    w = np.ones(len(X))
    n = np.array([0.0, 0.0, 1.0])
    c = X.mean(axis=0)
    for _ in range(8):
        cw = (X * w[:, None]).sum(axis=0) / w.sum()
        Xw = (X - cw) * np.sqrt(w)[:, None]
        _, _, Vt = np.linalg.svd(Xw, full_matrices=False)
        n = Vt[-1]
        n = n / np.linalg.norm(n)
        d = (X - cw) @ n
        s = 1.4826 * np.median(np.abs(d)) + 1e-9
        w = 1.0 / (1.0 + (d / (2.0 * s)) ** 2)
        c = cw
    return c, n


def compute_display_transform(solution_points, camera_centers, dist_pair, dist_meters,
                              ground_ids):
    """solution_points: dict pid -> (3,) raw coordinates.
    camera_centers: (m,3) raw camera centers (for yaw convention).
    dist_pair: (pid_a, pid_b); dist_meters: known real distance.
    ground_ids: list of pids lying on the ground plane.
    Returns dict with matrix (4x4), scale, and diagnostics.
    """
    pa = np.asarray(solution_points[dist_pair[0]], float)
    pb = np.asarray(solution_points[dist_pair[1]], float)
    raw_d = float(np.linalg.norm(pa - pb))
    if raw_d < 1e-12:
        raise ValueError("distance anchor points coincide in the solution")
    scale = float(dist_meters) / raw_d

    gnd = np.array([solution_points[i] for i in ground_ids], float)
    centroid, normal = fit_ground_plane(gnd)
    # orient normal so that cameras are on the positive side
    cam_mean = np.asarray(camera_centers, float).mean(axis=0)
    if np.dot(normal, cam_mean - centroid) < 0:
        normal = -normal

    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z_axis)
    c = np.dot(normal, z_axis)
    if np.linalg.norm(v) < 1e-9:
        R_tilt = np.eye(3) if c > 0 else geo.rodrigues(np.array([np.pi, 0, 0]))
    else:
        Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_tilt = np.eye(3) + Kx + Kx @ Kx * (1 - c) / (np.linalg.norm(v) ** 2)

    # yaw: first->last camera direction (ground-projected) -> +X
    C0 = R_tilt @ camera_centers[0]
    C1 = R_tilt @ camera_centers[-1]
    d = C1 - C0
    d[2] = 0.0
    if np.linalg.norm(d) > 1e-9:
        yaw = -np.arctan2(d[1], d[0])
    else:
        yaw = 0.0
    R_yaw = geo.rodrigues(np.array([0.0, 0.0, yaw]))
    R_align = R_yaw @ R_tilt

    # translation: ground centroid -> origin (plane becomes z=0). The transform
    # is X_disp = scale * R_align @ X + t_align, so the scale must be included.
    t_align = -scale * (R_align @ centroid)
    M = geo.similarity_matrix(scale, R_align, t_align)
    return {
        "matrix": M.tolist(),
        "scale": scale,
        "rotation": R_align.tolist(),
        "translation": t_align.tolist(),
        "raw_distance": raw_d,
        "known_distance_m": float(dist_meters),
        "ground_plane_rms_m": float(np.sqrt(np.mean(
            (np.asarray([geo.apply_similarity(scale, R_align, t_align, p)[2]
                         for p in gnd])) ** 2))),
        "inputs": {"distance_pair": [int(dist_pair[0]), int(dist_pair[1])],
                   "distance_m": float(dist_meters),
                   "ground_ids": [int(i) for i in ground_ids]},
    }


def apply_transform(transform, X):
    M = np.asarray(transform["matrix"], float)
    X = np.asarray(X, float)
    single = X.ndim == 1
    X2 = np.atleast_2d(X)
    out = X2 @ M[:3, :3].T + M[:3, 3]
    return out[0] if single else out
