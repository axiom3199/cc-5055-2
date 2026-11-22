"""Incremental structure-from-motion solver.

Pipeline: two-view initialization -> sequential pose registration (PnP, with a
rotation-only homography fallback for degenerate segments) -> parallax-gated
triangulation ('delayed') or immediate depth inheritance ('inherited') ->
global bundle adjustment with robust loss and outlier rejection.

The raw solution is up-to-scale: no metric claims are made until an external
scale constraint is applied (see align.py).

Entry points:
  solve_graph(view, ...)  the camera observation interface: reads an immutable
                          ObservationView snapshot (obsgraph.py); the solver
                          never touches the live observation graph.
  solve(tracks, ...)      compatibility adapter for the old entry; wraps raw
                          track dicts in a view and calls solve_graph.
"""
import numpy as np
import cv2

from . import geometry as geo
from .degeneracy import classify_motion
from .bundle import bundle_adjust
from .obsgraph import ObservationView, ResidualStore, ObsId

MIN_PARALLAX_INIT = 5.0     # degrees, for the seed pair (near-planar scenes need real baseline)
MIN_INIT_INLIER_FRAC = 0.5  # cheirality-consistent fraction of matches required
MIN_PARALLAX_TRI = 1.5      # degrees, for accepting a triangulated point
REPROJ_TRI = 2.5            # px, acceptance for a new triangulation
OUTLIER_REJECT = 4.0        # px, post-BA observation rejection


class Solution:
    def __init__(self):
        self.cameras = {}        # frame -> (R, t) world-to-cam
        self.points = {}         # track_id -> (3,)
        self.point_meta = {}     # track_id -> dict(inherited, born_frame, n_obs)
        self.frame_info = {}     # frame -> dict(method, parallax_deg, rotation_only, ...)
        self.observations = []   # (pid, frame, x, y)
        self.obs_residual = []   # parallel to observations, filled after BA
        self.obs_outlier = []
        self.residual_store = None   # ResidualStore keyed by ObsId (evaluation layer)
        self.init_info = {}
        self.ba_info = {}
        self.logs = []
        self.strategy = "delayed"

    def log(self, msg):
        self.logs.append(msg)

    def build_residual_store(self):
        """Key the post-BA residuals by unified observation identity. Strict:
        every kept observation must carry a residual and an outlier flag."""
        if not (len(self.observations) == len(self.obs_residual)
                == len(self.obs_outlier)):
            raise RuntimeError(
                f"{len(self.observations)} observations but "
                f"{len(self.obs_residual)} residuals and "
                f"{len(self.obs_outlier)} outlier flags: the evaluation "
                f"layer would be silently incomplete")
        store = ResidualStore()
        for (pid, f, _, _), r, o in zip(self.observations, self.obs_residual,
                                        self.obs_outlier):
            store.set(ObsId(pid, f), r, o)
        self.residual_store = store
        return store


def _median_parallax(R, t, p1, p2, K):
    """Median parallax of correspondences under a candidate relative pose."""
    if len(p1) == 0:
        return 0.0
    C1 = np.zeros(3)
    C2 = geo.camera_center(R, t)
    P1 = (np.eye(3), np.zeros(3))
    P2 = (R, t)
    angles = []
    for a, b in zip(p1, p2):
        X = geo.triangulate_multiview([P1, P2], np.array([a, b]), K)
        angles.append(geo.parallax_angle_deg(C1, C2, X))
    return float(np.median(angles))


def _frame_obs_index(tracks, n_frames):
    per_frame = [[] for _ in range(n_frames)]
    for tr in tracks:
        for j, f in enumerate(tr["frames"]):
            per_frame[int(f)].append((tr["id"], j))
    return per_frame


def _shared_matches(tracks, fa, fb):
    d = {}
    for tr in tracks:
        pos = {int(f): i for i, f in enumerate(tr["frames"])}
        if fa in pos and fb in pos:
            d[tr["id"]] = (tr["xy"][pos[fa]], tr["xy"][pos[fb]])
    return d


def solve(tracks, n_frames, K, strategy="delayed", ba_max_nfev=100, log=print):
    """Compatibility adapter for the pre-graph entry point. Wraps the raw
    track dicts in an ObservationView and delegates to solve_graph; results
    are identical to calling solve_graph on a graph built from the same
    tracks."""
    view = ObservationView.from_track_dicts(tracks, n_frames)
    return solve_graph(view, K=K, strategy=strategy, ba_max_nfev=ba_max_nfev,
                       log=log)


def solve_graph(view, K, strategy="delayed", ba_max_nfev=100, log=print):
    """Camera observation interface: solve from an immutable ObservationView
    snapshot. The live observation graph is never passed in, so the solver
    cannot modify the UI-visible point set; results reach the graph only
    through an explicit commit (obsgraph.ObservationGraph.commit_solution)."""
    tracks = view.tracks()
    n_frames = view.n_frames
    sol = Solution()
    sol.strategy = strategy
    per_frame = _frame_obs_index(tracks, n_frames)
    track_by_id = {tr["id"]: tr for tr in tracks}

    # ---------- two-view initialization ----------
    # A seed pair needs real baseline: with a near-planar scene and small
    # parallax the essential matrix is ill-conditioned and recoverPose returns
    # a plausible-looking but wrong translation. Require both enough parallax
    # and a high cheirality-consistent inlier fraction.
    init = None
    f0 = 0
    for fj in range(4, min(45, n_frames)):
        matches = _shared_matches(tracks, f0, fj)
        if len(matches) < 60:
            continue
        p1 = np.array([v[0] for v in matches.values()])
        p2 = np.array([v[1] for v in matches.values()])
        E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.size > 9:
            E = E.reshape(-1, 3, 3)[0] if E is not None and E.size > 9 else E
        n_in, R, t, mask_rp = cv2.recoverPose(E, p1, p2, K, mask=mask)
        frac = n_in / max(len(matches), 1)
        # median parallax of cheirality-consistent points under the recovered pose
        inl_ids = [k for k, m in enumerate(mask_rp.reshape(-1)) if m > 0]
        par = _median_parallax(R, t.reshape(3), p1[inl_ids], p2[inl_ids], K)
        log(f"[init] pair (0,{fj}): matches={len(matches)} rp_inliers={n_in} "
            f"({frac:.0%}) parallax={par:.2f}deg")
        if frac >= MIN_INIT_INLIER_FRAC and par >= MIN_PARALLAX_INIT:
            init = (fj, matches, E, mask_rp, R, t.reshape(3))
            break
    if init is None:
        raise RuntimeError(
            "no initialization pair with sufficient parallax and consistent "
            "geometry; the footage may be rotation-dominated or too short - "
            "refusing to emit a metric-looking solution without scale evidence")
    fj, matches, E, mask_rp, R, t = init
    inl = mask_rp.reshape(-1) > 0
    sol.cameras[f0] = (np.eye(3), np.zeros(3))
    sol.cameras[fj] = (R, t)
    sol.frame_info[f0] = {"method": "init", "rotation_only": False, "parallax_deg": None}
    sol.frame_info[fj] = {"method": "init", "rotation_only": False, "parallax_deg": None}
    sol.init_info = {"pair": [f0, fj], "inliers": int(inl.sum())}
    log(f"[init] selected pair (0,{fj}) with {int(inl.sum())} cheirality-consistent "
        f"inliers; baseline fixed to unit length (arbitrary scale)")

    # triangulate seed points
    ids = list(matches.keys())
    p1 = np.array([v[0] for v in matches.values()])
    p2 = np.array([v[1] for v in matches.values()])
    for k, tid in enumerate(ids):
        if not inl[k]:
            continue
        X = geo.triangulate_multiview([sol.cameras[f0], sol.cameras[fj]],
                                      np.array([p1[k], p2[k]]), K)
        if _accept_point(X, {f0: p1[k], fj: p2[k]}, sol.cameras, K):
            sol.points[tid] = X
            sol.point_meta[tid] = {"inherited": False, "born_frame": f0}
    log(f"[init] triangulated {len(sol.points)} seed points")

    # ---------- sequential registration ----------
    solved = [fj]
    order = list(range(fj + 1, n_frames)) + list(range(1, fj))
    for f in order:
        ok = _register_frame(sol, f, per_frame, track_by_id, K, log)
        if ok:
            solved.append(f)
    log(f"[solve] registered {len(sol.cameras)}/{n_frames} frames, "
        f"{len(sol.points)} map points before BA")

    # ---------- bundle adjustment + outlier rejection ----------
    sol._tracks = track_by_id
    _run_ba(sol, K, ba_max_nfev, log)
    _enrich_frame_geometry(sol, K)
    return sol


def _enrich_frame_geometry(sol, K):
    """Recompute per-frame parallax and the rotation-only flag from the SOLVED
    geometry (camera centers + refined map points), not from the essential
    matrix. The essential matrix is degenerate under pure rotation, so the
    init-time classification produces meaningless parallax spikes there; the
    solved baseline is the honest signal. A frame is rotation-only when its
    baseline to the previous solved frame subtends ~no angle at the scene."""
    frames = sorted(sol.cameras)
    centers = {f: geo.camera_center(*sol.cameras[f]) for f in frames}
    # map points observed per frame
    obs_by_frame = {}
    for (pid, f, x, y) in sol.observations:
        if pid in sol.points:
            obs_by_frame.setdefault(f, []).append(pid)
    for i in range(1, len(frames)):
        f, g = frames[i - 1], frames[i]
        shared = set(obs_by_frame.get(f, ())) & set(obs_by_frame.get(g, ()))
        if len(shared) < 5:
            continue
        angs = [geo.parallax_angle_deg(centers[f], centers[g], sol.points[pid])
                for pid in shared]
        par = float(np.median(angs))
        info = sol.frame_info.setdefault(g, {})
        info["parallax_deg"] = par
        info["rotation_only"] = par < 0.3


def _accept_point(X, obs_by_frame, cameras, K, thresh=REPROJ_TRI):
    errs = []
    for f, uv in obs_by_frame.items():
        R, t = cameras[f]
        Xc = R @ X + t
        if Xc[2] <= 0.05:
            return False
        p = geo.project_points(Xc[None], K)[0]
        errs.append(np.linalg.norm(p - uv))
    return max(errs) < thresh


def _register_frame(sol, f, per_frame, track_by_id, K, log):
    # 2D-3D correspondences from existing map points. Inherited-depth points
    # are excluded: their 3D position is a guess and must not drive camera
    # registration (they only exist for the strategy comparison / display).
    obj, img, src_tid = [], [], []
    for tid, j in per_frame[f]:
        if tid in sol.points and not sol.point_meta[tid].get("inherited", False):
            obj.append(sol.points[tid])
            img.append(track_by_id[tid]["xy"][j])
            src_tid.append(tid)
    method = None
    if len(obj) >= 8:
        obj = np.asarray(obj)
        img = np.asarray(img)
        okp, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, K, None, iterationsCount=300, reprojectionError=3.0,
            confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
        n_inl = 0 if inl is None else len(inl)
        if okp and n_inl >= 10:
            rvec, tvec = cv2.solvePnP(obj[inl[:, 0]], img[inl[:, 0]], K, None,
                                      rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)[1:]
            sol.cameras[f] = (geo.rodrigues(rvec.reshape(3)), tvec.reshape(3))
            method = "pnp"
            sol.frame_info[f] = {"method": "pnp", "pnp_inliers": int(n_inl)}
    if method is None:
        # rotation-only fallback: estimate orientation from homography vs a
        # nearby solved frame; position stays at the reference camera center.
        ref = _nearest_solved(sol, f)
        if ref is None:
            sol.frame_info[f] = {"method": "failed"}
            log(f"[frame {f}] FAILED: no solved reference")
            return False
        matches = _shared_frame_matches(sol, f, ref, per_frame, track_by_id)
        if len(matches) < 20:
            sol.frame_info[f] = {"method": "failed"}
            log(f"[frame {f}] FAILED: only {len(matches)} matches to frame {ref}")
            return False
        p_ref = np.array([m[0] for m in matches])
        p_f = np.array([m[1] for m in matches])
        info = classify_motion(p_ref, p_f, K)
        if info["R_rel"] is None:
            sol.frame_info[f] = {"method": "failed"}
            log(f"[frame {f}] FAILED: no rotation-consistent homography "
                f"(orth_err={info['rot_orth_error']:.3f})")
            return False
        R_ref, t_ref = sol.cameras[ref]
        C_ref = geo.camera_center(R_ref, t_ref)
        R_f = info["R_rel"] @ R_ref
        t_f = -R_f @ C_ref
        sol.cameras[f] = (R_f, t_f)
        sol.frame_info[f] = {"method": "rotation_only", "ref": ref,
                             "parallax_deg": info["parallax_deg"]}
        log(f"[frame {f}] ROTATION-ONLY vs frame {ref}: orientation from homography, "
            f"position pinned (parallax={info['parallax_deg']:.2f}deg)")

    # Degeneracy classification over a wide-enough baseline: consecutive-frame
    # parallax during translation is small and would be misread as rotation, so
    # compare against a solved frame several steps back. Pure rotation shows no
    # parallax no matter the baseline; translation reveals itself.
    back = [s for s in sol.cameras if s <= f - 4]
    prev = max(back) if back else max([s for s in sol.cameras if s < f], default=None)
    if prev is not None:
        matches = _shared_frame_matches(sol, f, prev, per_frame, track_by_id)
        if len(matches) >= 15:
            p_prev = np.array([m[0] for m in matches])
            p_f = np.array([m[1] for m in matches])
            info = classify_motion(p_prev, p_f, K)
            sol.frame_info[f]["parallax_deg"] = info["parallax_deg"]
            sol.frame_info[f]["rotation_only"] = info["is_rotation_only"]
    sol.frame_info[f].setdefault("rotation_only", False)

    _triangulate_frame(sol, f, per_frame, track_by_id, K, log)
    return True


def _nearest_solved(sol, f):
    cands = [s for s in sol.cameras if s != f]
    if not cands:
        return None
    return min(cands, key=lambda s: abs(s - f))


def _shared_frame_matches(sol, fa, fb, per_frame, track_by_id):
    """Observations (xy in fa, xy in fb) of tracks seen in both frames."""
    obs_a = {tid: j for tid, j in per_frame[fa]}
    out = []
    for tid, jb in per_frame[fb]:
        if tid in obs_a:
            out.append((track_by_id[tid]["xy"][obs_a[tid]], track_by_id[tid]["xy"][jb]))
    return out


def _triangulate_frame(sol, f, per_frame, track_by_id, K, log):
    n_new = 0
    for tid, j in per_frame[f]:
        # A track holding only an inherited-depth guess is still a candidate:
        # upgrade it to a real triangulation as soon as parallax exists.
        if tid in sol.points and not sol.point_meta[tid].get("inherited"):
            continue
        tr = track_by_id[tid]
        obs_frames = [int(fr) for fr in tr["frames"]]
        solved_obs = [(fr, i) for i, fr in enumerate(obs_frames) if fr in sol.cameras]
        if len(solved_obs) < 2:
            continue
        # Pick the observation pair with the largest camera-baseline (cheap
        # center-distance test), then triangulate only that pair. Triangulating
        # all O(n^2) pairs for every candidate every frame is the bottleneck
        # for the inherited strategy, which re-checks many points per frame.
        centers = {fr: geo.camera_center(*sol.cameras[fr]) for fr, _ in solved_obs}
        best_pair, best_d = None, 1e-9
        for a in range(len(solved_obs)):
            fa, _ = solved_obs[a]
            Ca = centers[fa]
            for b in range(a + 1, len(solved_obs)):
                fb, _ = solved_obs[b]
                d = np.linalg.norm(Ca - centers[fb])
                if d > best_d:
                    best_d, best_pair = d, (a, b)
        best = None
        if best_pair is not None:
            a, b = best_pair
            fa, ia = solved_obs[a]
            fb, ib = solved_obs[b]
            Ra, ta = sol.cameras[fa]
            Rb, tb = sol.cameras[fb]
            X = geo.triangulate_multiview([(Ra, ta), (Rb, tb)],
                                          np.array([tr["xy"][ia], tr["xy"][ib]]), K)
            ang = geo.parallax_angle_deg(centers[fa], centers[fb], X)
            best = (ang, X)
        if best is not None and best[0] >= MIN_PARALLAX_TRI:
            # multi-view refinement with all solved observations
            poses = [sol.cameras[fr] for fr, _ in solved_obs]
            uvs = np.array([tr["xy"][i] for _, i in solved_obs])
            X = geo.triangulate_multiview(poses, uvs, K)
            obs_by_frame = {fr: tr["xy"][i] for fr, i in solved_obs}
            if _accept_point(X, obs_by_frame, sol.cameras, K):
                was_inherited = tid in sol.points
                sol.points[tid] = X
                sol.point_meta[tid] = {"inherited": False,
                                       "born_frame": int(obs_frames[0]),
                                       "parallax_deg": best[0]}
                if not was_inherited:
                    n_new += 1
        elif sol.strategy == "inherited" and tid not in sol.points and len(solved_obs) >= 3:
            _inherit_depth(sol, tid, tr, solved_obs, per_frame, track_by_id, K)
            if tid in sol.points:
                n_new += 1
    if n_new:
        log(f"[frame {f}] triangulated {n_new} new points "
            f"(strategy={sol.strategy})")


def _inherit_depth(sol, tid, tr, solved_obs, per_frame, track_by_id, K):
    """Create a point immediately by inheriting the median depth of nearby map
    points in the birth frame. Geometrically wrong in general; used only for the
    inherited-depth strategy comparison."""
    fb, ib = solved_obs[0]
    R, t = sol.cameras[fb]
    C = geo.camera_center(R, t)
    xy_new = tr["xy"][ib]
    neigh = []
    for tid2, j2 in per_frame[fb]:
        if tid2 in sol.points:
            Xc = R @ sol.points[tid2] + t
            if Xc[2] > 0.1:
                d_img = np.linalg.norm(track_by_id[tid2]["xy"][j2] - xy_new)
                neigh.append((d_img, Xc[2]))
    if len(neigh) < 5:
        return
    neigh.sort()
    depth = float(np.median([d for _, d in neigh[:8]]))
    ray_cam = np.linalg.inv(K) @ np.array([xy_new[0], xy_new[1], 1.0])
    X = C + R.T @ (ray_cam * depth)
    sol.points[tid] = X
    sol.point_meta[tid] = {"inherited": True, "born_frame": int(tr["frames"][0]),
                           "inherited_depth": depth}


def _run_ba(sol, K, max_nfev, log):
    def gather():
        # Inherited-depth points are excluded from BA: their 3D position is an
        # unconstrained guess (no parallax), so letting them into the joint
        # optimization would corrupt the very cameras we want to compare. They
        # remain in sol.points as display-only estimates for the strategy
        # comparison, keeping both strategies on an identical camera trajectory.
        obs = []
        for tid, X in sol.points.items():
            if sol.point_meta[tid].get("inherited"):
                continue
            tr = sol._tracks[tid]
            for j, fr in enumerate(tr["frames"]):
                fr = int(fr)
                if fr in sol.cameras:
                    obs.append((tid, fr, float(tr["xy"][j][0]), float(tr["xy"][j][1])))
        return obs

    sol.observations = gather()
    cams, pts, info = bundle_adjust(sol.cameras, sol.points, sol.observations, K,
                                    max_nfev=max_nfev)
    sol.cameras, sol.points = cams, pts
    sol.ba_info = {k: v for k, v in info.items() if k != "residuals"}
    res = info.get("residuals")
    if res is None:
        # BA skipped (no points survived): observations are empty as well, so
        # the evaluation layer is consistently empty rather than missing.
        sol.build_residual_store()
        return
    rnorm = np.linalg.norm(res, axis=1)
    keep = rnorm < OUTLIER_REJECT
    n_rej = int((~keep).sum())
    log(f"[BA] pass 1: rms={info['rms_px']:.3f}px, rejecting {n_rej} observations "
        f"> {OUTLIER_REJECT}px")
    if n_rej > 0:
        obs_kept = [o for o, k in zip(sol.observations, keep) if k]
        # the survivors are inliers by construction; the rejection count lives
        # in ba_info["rejected_outliers"], keeping one flag per observation
        sol.obs_outlier = [False] * len(obs_kept)
        cams, pts, info2 = bundle_adjust(sol.cameras, sol.points, obs_kept, K,
                                         max_nfev=max_nfev)
        sol.cameras, sol.points = cams, pts
        sol.observations = obs_kept
        res2 = info2.get("residuals")
        sol.obs_residual = list(np.linalg.norm(res2, axis=1)) if res2 is not None else []
        sol.ba_info.update({"rms_px_final": info2.get("rms_px"),
                            "rejected_outliers": n_rej,
                            "cost_final2": info2.get("cost_final")})
        log(f"[BA] pass 2: rms={info2.get('rms_px', float('nan')):.3f}px, "
            f"nfev={info2.get('nfev')}")
    else:
        sol.obs_outlier = [False] * len(sol.observations)
        sol.obs_residual = list(rnorm)

    # Prune under-constrained points: after rejection, a point seen in fewer
    # than 3 views has no redundancy and drifts freely, distorting the shared
    # cameras. Drop it (and its observations) and do a final clean pass.
    cnt = {}
    for (pid, f, x, y) in sol.observations:
        cnt[pid] = cnt.get(pid, 0) + 1
    weak = {pid for pid, c in cnt.items() if c < 3}
    if weak:
        obs3 = [o for o in sol.observations if o[0] not in weak]
        keep_flag = [o[0] not in weak for o in sol.observations]
        for pid in weak:
            sol.points.pop(pid, None)
            sol.point_meta.pop(pid, None)
        cams, pts, info3 = bundle_adjust(sol.cameras, sol.points, obs3, K,
                                         max_nfev=max_nfev)
        sol.cameras, sol.points = cams, pts
        sol.observations = obs3
        sol.obs_outlier = [f for f, k in zip(sol.obs_outlier, keep_flag) if k]
        res3 = info3.get("residuals")
        sol.obs_residual = list(np.linalg.norm(res3, axis=1)) if res3 is not None else []
        sol.ba_info.update({"rms_px_pass3": info3.get("rms_px"),
                            "pruned_weak_points": len(weak)})
        log(f"[BA] pass 3: pruned {len(weak)} under-constrained points, "
            f"rms={info3.get('rms_px', float('nan')):.3f}px")
    sol.build_residual_store()
