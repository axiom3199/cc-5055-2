"""Run orchestration and artifact IO. A run directory holds:
  video/frame_XXXX.png   synthetic input footage
  gt.json                ground truth (synthetic only)
  tracks.npz             raw KLT tracks (+ injected outlier flags)
  solution_raw.json      up-to-scale solve result (NO metric claims)
  observations.npz       observation residuals / outlier flags after BA
  display_transform.json user-constraint similarity transform (separate file)
  metrics.json           evaluation (uses GT; test footage only)
  render/frame_XXXX.png  per-frame overlay renders
"""
import json
import os
import time

import numpy as np

from . import synth, tracking, sfm, align, metrics, marker as marker_mod
from . import geometry as geo
from .obsgraph import ObservationGraph
from .anchors import LensProject, LensKind


def create_run(root, name, cfg_dict):
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(cfg_dict, fh, indent=1)
    set_status(run_dir, "created")
    return run_dir


def set_status(run_dir, status):
    with open(os.path.join(run_dir, "status.json"), "w") as fh:
        json.dump({"status": status, "time": time.time()}, fh)


def get_status(run_dir):
    try:
        with open(os.path.join(run_dir, "status.json")) as fh:
            return json.load(fh)
    except OSError:
        return {"status": "unknown"}


def load_config(run_dir):
    with open(os.path.join(run_dir, "config.json")) as fh:
        return json.load(fh)


def load_gt(run_dir):
    with open(os.path.join(run_dir, "gt.json")) as fh:
        return json.load(fh)


def load_intrinsics(run_dir):
    """Camera intrinsics K (3x3) and image size [w, h]. Synthetic runs read them
    from gt.json; real footage needs an intrinsics.json the user supplies
    ({"K": [[fx,0,cx],[0,fy,cy],[0,0,1]], "image": [w,h]})."""
    if os.path.exists(os.path.join(run_dir, "gt.json")):
        gt = load_gt(run_dir)
        return np.asarray(gt["K"]), list(gt["image"])
    p = os.path.join(run_dir, "intrinsics.json")
    if os.path.exists(p):
        with open(p) as fh:
            d = json.load(fh)
        return np.asarray(d["K"], float), list(d["image"])
    raise RuntimeError(
        "no camera intrinsics: synthetic runs have gt.json; for real footage "
        "add intrinsics.json with K and image size")


def step_generate(run_dir):
    cfg = load_config(run_dir)
    sc = synth.SynthConfig(
        profile=cfg.get("profile", "survey"),
        n_frames=int(cfg.get("n_frames", 100)),
        seed=int(cfg.get("seed", 7)))
    synth.generate(run_dir, sc)
    set_status(run_dir, "generated")
    return sc


def step_track(run_dir, max_tracks=600):
    cfg = load_config(run_dir)
    tracks, n_frames = tracking.build_tracks(os.path.join(run_dir, "video"),
                                             max_tracks=max_tracks)
    ratio = float(cfg.get("outlier_ratio", 0.0))
    injected = []
    if ratio > 0:
        injected = tracking.inject_outliers(tracks, ratio, seed=cfg.get("seed", 7))
    ot, of, oxy = tracking.tracks_to_arrays(tracks)
    outlier_flags = np.zeros(len(ot), bool)
    idx = {}
    for k in range(len(ot)):
        idx.setdefault(int(ot[k]), []).append(k)
    for tid, obs_i in injected:
        # obs_i counts within the track's own observation list
        if tid in idx and obs_i < len(idx[tid]):
            outlier_flags[idx[tid][obs_i]] = True
    np.savez_compressed(os.path.join(run_dir, "tracks.npz"),
                        obs_track=ot, obs_frame=of, obs_xy=oxy,
                        injected_outlier=outlier_flags, n_frames=n_frames)
    set_status(run_dir, "tracked")
    return tracks, n_frames, len(injected)


def load_tracks(run_dir):
    z = np.load(os.path.join(run_dir, "tracks.npz"))
    ot, of, oxy = z["obs_track"], z["obs_frame"], z["obs_xy"]
    tracks = {}
    for k in range(len(ot)):
        tid = int(ot[k])
        tr = tracks.setdefault(tid, {"id": tid, "frames": [], "xy": []})
        tr["frames"].append(int(of[k]))
        tr["xy"].append(oxy[k].astype(np.float32))
    out = []
    for tid in sorted(tracks):
        tr = tracks[tid]
        tr["frames"] = np.asarray(tr["frames"], np.int32)
        tr["xy"] = np.asarray(tr["xy"], np.float32)
        out.append(tr)
    return out, int(z["n_frames"]), z["injected_outlier"]


def load_or_build_graph(run_dir):
    """The run's observation graph: the persisted one if present (it carries
    the user's observation deletions and anchors), otherwise built fresh
    from the raw tracks."""
    if os.path.exists(os.path.join(run_dir, "obsgraph.npz")):
        return ObservationGraph.load(run_dir)
    tracks, n_frames, _ = load_tracks(run_dir)
    return ObservationGraph.from_tracks(tracks, n_frames)


def step_solve(run_dir, strategy=None, ba_max_nfev=100, log=print):
    cfg = load_config(run_dir)
    strategy = strategy or cfg.get("strategy", "delayed")
    K, image_size = load_intrinsics(run_dir)
    graph = load_or_build_graph(run_dir)
    logs = []
    def collect(msg):
        logs.append(msg)
        log(msg)
    # the solver reads an immutable view; its result reaches the graph only
    # through the explicit commit below - it never mutates the UI point set
    sol = sfm.solve_graph(graph.view(), K=K, strategy=strategy,
                          ba_max_nfev=ba_max_nfev, log=collect)
    graph.commit_solution(sol)
    graph.save(run_dir)
    save_solution(run_dir, sol, K, image_size, logs)
    set_status(run_dir, f"solved_{strategy}")
    return sol


def save_solution(run_dir, sol, K, image_size, logs):
    frames = []
    for f in sorted(sol.cameras):
        R, t = sol.cameras[f]
        info = sol.frame_info.get(f, {})
        frames.append({"idx": int(f), "R": R.reshape(-1).tolist(), "t": t.tolist(),
                       "method": info.get("method", "?"),
                       "rotation_only": bool(info.get("rotation_only", False)),
                       "parallax_deg": info.get("parallax_deg")})
    points = []
    for pid, X in sol.points.items():
        meta = sol.point_meta.get(pid, {})
        points.append({"id": int(pid), "X": list(map(float, X)),
                       "inherited": bool(meta.get("inherited", False)),
                       "born_frame": int(meta.get("born_frame", -1))})
    data = {
        "units": "arbitrary (up-to-scale similarity; metric values require display_transform.json)",
        "K": np.asarray(K).tolist(),
        "image": list(image_size),
        "strategy": sol.strategy,
        "init": sol.init_info,
        "ba": {k: v for k, v in sol.ba_info.items() if not isinstance(v, np.ndarray)},
        "frames": frames,
        "points": points,
        "logs": logs,
    }
    with open(os.path.join(run_dir, f"solution_raw.json"), "w") as fh:
        json.dump(data, fh)
    obs = np.asarray(sol.observations, np.float64) if sol.observations else np.zeros((0, 4))
    np.savez_compressed(os.path.join(run_dir, "observations.npz"),
                        obs=obs,
                        residual=np.asarray(sol.obs_residual, np.float64),
                        outlier=np.asarray(sol.obs_outlier, bool))


def load_solution(run_dir):
    with open(os.path.join(run_dir, "solution_raw.json")) as fh:
        data = json.load(fh)
    sol = sfm.Solution()
    sol.strategy = data["strategy"]
    sol.init_info = data.get("init", {})
    sol.ba_info = data.get("ba", {})
    sol.logs = data.get("logs", [])
    for fr in data["frames"]:
        f = int(fr["idx"])
        sol.cameras[f] = (np.asarray(fr["R"]).reshape(3, 3), np.asarray(fr["t"]))
        sol.frame_info[f] = {"method": fr["method"],
                             "rotation_only": fr["rotation_only"],
                             "parallax_deg": fr.get("parallax_deg")}
    for p in data["points"]:
        pid = int(p["id"])
        sol.points[pid] = np.asarray(p["X"])
        sol.point_meta[pid] = {"inherited": p["inherited"],
                               "born_frame": p["born_frame"]}
    z = np.load(os.path.join(run_dir, "observations.npz"))
    obs = z["obs"]
    sol.observations = [(int(p), int(f), float(x), float(y)) for p, f, x, y in obs]
    sol.obs_residual = list(z["residual"])
    sol.obs_outlier = list(z["outlier"])
    sol.build_residual_store()   # strict: raises if a residual is missing
    if os.path.exists(os.path.join(run_dir, "obsgraph.npz")):
        # the solution's point ids belong to the graph's tracks (fragments
        # after a deletion), so join against the graph, not the raw tracks
        graph = ObservationGraph.load(run_dir)
        sol._tracks = {t.id: {"id": t.id, "frames": t.frames, "xy": t.xy}
                       for t in graph.tracks.snapshot()}
    else:
        tracks, n_frames, injected = load_tracks(run_dir)
        sol._tracks = {tr["id"]: tr for tr in tracks}
    return sol, np.asarray(data["K"]), data


def simulate_constraints(run_dir, n_ground=25):
    """Simulate the user-supplied constraints for synthetic footage, where GT
    lets us identify ground points and a true distance. In real use the user
    clicks these; here we derive them from the GT association so the demo and
    tests can run unattended. Returns (dist_pair, dist_meters, ground_ids)."""
    sol, K, _ = load_solution(run_dir)
    gt = load_gt(run_dir)
    track_gt = metrics.match_track_to_gt(list(sol._tracks.values()), gt)
    ground_mask = np.asarray(gt["ground_mask"], bool)
    gt_pts = np.asarray(gt["points"])
    # solved map points whose track matches a GT ground point
    cand = []   # (track_id, gt_xyz)
    for tid, X in sol.points.items():
        if tid in track_gt and not sol.point_meta[tid].get("inherited"):
            gid, gxyz = track_gt[tid]
            if gid is not None and ground_mask[gid]:
                cand.append((tid, gt_pts[gid]))
    if len(cand) < 3:
        raise RuntimeError("not enough ground-associated map points to simulate constraints")
    ground_ids = [tid for tid, _ in cand[:n_ground]]
    # distance pair: two ground points far apart for a well-conditioned scale
    best = None
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            d = float(np.linalg.norm(cand[i][1] - cand[j][1]))
            if best is None or d > best[0]:
                best = (d, cand[i][0], cand[j][0])
    dist_meters, pa, pb = best
    return (pa, pb), dist_meters, ground_ids


def step_align(run_dir, dist_pair, dist_meters, ground_ids):
    sol, K, _ = load_solution(run_dir)
    frames = sorted(sol.cameras)
    centers = np.array([geo.camera_center(*sol.cameras[f]) for f in frames])
    tf = align.compute_display_transform(sol.points, centers, dist_pair,
                                         dist_meters, ground_ids)
    tf["note"] = ("similarity transform from raw solution coordinates to display "
                  "coordinates; derived from user-supplied known distance and "
                  "ground constraint. Raw solution remains up-to-scale.")
    with open(os.path.join(run_dir, "display_transform.json"), "w") as fh:
        json.dump(tf, fh, indent=1)
    # register the constraints as anchors in the observation graph, so both
    # lens projects (monocular / stereo) share the same ground anchor and the
    # scale anchor's lifetime is managed by undoable transactions
    graph = load_or_build_graph(run_dir)
    edit = graph.begin_edit()
    edit.stage_anchors(dist_pair=dist_pair, dist_meters=dist_meters,
                       ground_ids=ground_ids)
    edit.commit()
    graph.save(run_dir)
    return tf


def step_export_landing_points(run_dir, kind=LensKind.MONOCULAR,
                               baseline_m=None, baseline_pair=None):
    """Metric ground-contact ('landing') points for a lens project over this
    run. Raises MetricSuspendedError when the project has no length reference
    (e.g. a monocular project whose only distance anchor was deleted); the
    overlay preview is unaffected by the suspension."""
    graph = load_or_build_graph(run_dir)
    sol, K, _ = load_solution(run_dir)
    project = LensProject(kind, graph, baseline_m=baseline_m,
                          baseline_pair=baseline_pair)
    out = project.export_landing_points(sol)
    out["lens_kind"] = kind
    with open(os.path.join(run_dir, "landing_points.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def load_display_transform(run_dir):
    p = os.path.join(run_dir, "display_transform.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def step_evaluate(run_dir):
    sol, K, _ = load_solution(run_dir)
    gt = load_gt(run_dir)
    cam = metrics.evaluate_cameras(sol, gt)
    reproj = metrics.reprojection_stats(sol)
    gc = metrics.ground_contact_error(solution=sol, gt=gt)
    out = {"cameras": cam, "reprojection": reproj, "ground_contact": gc}
    tf = load_display_transform(run_dir)
    mk = load_marker(run_dir)
    if tf is not None and mk is not None:
        m = marker_mod.make_cube_marker(**mk["args"])
        out["marker_swimming"] = metrics.marker_swimming(sol, gt, m["vertices"], tf)
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(out, fh)
    return out


def save_marker(run_dir, marker_args):
    with open(os.path.join(run_dir, "marker.json"), "w") as fh:
        json.dump({"args": marker_args}, fh)


def load_marker(run_dir):
    p = os.path.join(run_dir, "marker.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def step_render(run_dir, marker_args=None):
    sol, K, _ = load_solution(run_dir)
    tf = load_display_transform(run_dir)
    mk = None
    if marker_args is not None:
        save_marker(run_dir, marker_args)
        mk = marker_mod.make_cube_marker(**marker_args)
    else:
        saved = load_marker(run_dir)
        if saved:
            mk = marker_mod.make_cube_marker(**saved["args"])
    paths = marker_mod.render_sequence(run_dir, sol, K, mk, tf)
    set_status(run_dir, "rendered")
    return paths
