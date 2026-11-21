"""Degenerate / pure-rotation motion classification between two views.

A pure rotation about the camera center induces x2 ~ K R K^-1 x1 exactly, so a
homography fits AND K^-1 H K is (near-)orthogonal. A planar scene also admits a
homography, but K^-1 H K is then not a rotation. Parallax angle from essential-
matrix triangulation is the second, independent cue.
"""
import cv2
import numpy as np

from . import geometry as geo


def classify_motion(pts1, pts2, K, ransac_thresh_px=1.0):
    """pts1, pts2: (n,2) pixel correspondences. Returns a dict with:
    n_matches, ess_inliers, hom_inliers, parallax_deg (median, essential inliers),
    rot_orth_error (||R^T R - I||_F for R = K^-1 H K), is_rotation_only, is_degenerate,
    R_rel (rotation from cam1 to cam2 if rotation-only, else None).
    """
    pts1 = np.asarray(pts1, np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, np.float64).reshape(-1, 2)
    out = {"n_matches": len(pts1), "ess_inliers": 0, "hom_inliers": 0,
           "parallax_deg": 0.0, "rot_orth_error": np.inf,
           "is_rotation_only": False, "is_degenerate": True, "R_rel": None}
    if len(pts1) < 8:
        return out

    # Essential-matrix estimation is itself degenerate under pure rotation (the
    # polynomial solver can fail outright); treat any failure as "no essential
    # geometry" and fall back to the homography cue below.
    E = mask_e = None
    try:
        E, mask_e = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999,
                                         ransac_thresh_px)
    except cv2.error:
        E, mask_e = None, None
    try:
        H, mask_h = cv2.findHomography(pts1, pts2, cv2.RANSAC, 1.5)
    except cv2.error:
        H, mask_h = None, None
    n_e = int(mask_e.sum()) if mask_e is not None else 0
    n_h = int(mask_h.sum()) if mask_h is not None else 0
    out["ess_inliers"] = n_e
    out["hom_inliers"] = n_h

    parallax = 0.0
    if E is not None and E.size and n_e >= 8:
        E = E.reshape(-1, 3, 3)[0] if E.size > 9 else E
        try:
            _, R, t, mask_rp = cv2.recoverPose(E, pts1, pts2, K)
        except cv2.error:
            R = None
        inl = mask_e.reshape(-1) > 0
        if R is not None and inl.sum() >= 8:
            P2 = (R, t.reshape(3))
            p1n = pts1[inl]
            p2n = pts2[inl]
            C1 = np.zeros(3)
            C2 = geo.camera_center(R, t.reshape(3))
            Xs = _triangulate_pair((np.eye(3), np.zeros(3)), P2, K, p1n, p2n)
            angles = [geo.parallax_angle_deg(C1, C2, X) for X in Xs]
            parallax = float(np.median(angles)) if angles else 0.0
    out["parallax_deg"] = parallax

    R_rel = None
    orth_err = np.inf
    if H is not None:
        Kinv = np.linalg.inv(K)
        Rcand = Kinv @ H @ K
        R_rel = geo.project_to_so3(Rcand)
        orth_err = float(np.linalg.norm(Rcand.T @ Rcand - np.eye(3), "fro"))
    out["rot_orth_error"] = orth_err

    rotation_consistent = orth_err < 0.08 and n_h >= 0.6 * max(n_e, 1)
    low_parallax = parallax < 0.3
    out["is_rotation_only"] = bool(rotation_consistent or low_parallax)
    out["is_degenerate"] = bool(parallax < 1.0)
    if rotation_consistent:
        out["R_rel"] = R_rel
    return out


def _triangulate_pair(P1, P2, K, pts1, pts2):
    R1, t1 = P1
    R2, t2 = P2
    out = []
    for a, b in zip(pts1, pts2):
        out.append(geo.triangulate_multiview([(R1, t1), (R2, t2)], np.array([a, b]), K))
    return np.asarray(out)
