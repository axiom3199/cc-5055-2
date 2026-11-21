"""Core geometry: pinhole camera model, rotations, triangulation, similarity fit."""
import numpy as np
import cv2


def rodrigues(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def rvec_from_matrix(R):
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.reshape(3)


def project_to_so3(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_angle_deg(R1, R2):
    R = R1.T @ R2
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def camera_center(R, t):
    """World-to-camera (R, t) -> camera center in world coords."""
    return -R.T @ t


def world_to_cam(R, t, X):
    return X @ R.T + t


def project_points(X_cam, K):
    z = X_cam[:, 2]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * X_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * X_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def project_world(R, t, Xw, K):
    return project_points(world_to_cam(R, t, np.atleast_2d(Xw)), K)


def triangulate_multiview(poses, points2d, K):
    """DLT triangulation from m>=2 views. poses: list of (R,t), points2d: (m,2) pixels."""
    A = []
    for (R, t), (u, v) in zip(poses, points2d):
        P = K @ np.hstack([R, t.reshape(3, 1)])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return X[:3]
    return X[:3] / X[3]


def parallax_angle_deg(C1, C2, X):
    """Angle at point X between rays from camera centers C1 and C2."""
    v1 = C1 - X
    v2 = C2 - X
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def umeyama(src, dst, with_scale=True):
    """Similarity transform mapping src -> dst: dst ~= s * R @ src + t.

    src, dst: (n,3) arrays of corresponding points.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / src.shape[0]
        s = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def apply_similarity(s, R, t, X):
    return s * (np.asarray(X) @ R.T) + t


def similarity_matrix(s, R, t):
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M
