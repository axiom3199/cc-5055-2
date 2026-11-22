"""Web workbench for camera-motion solving on live-action-style footage.

Pages:
  /                     dashboard: list runs, create a run
  /runs/<name>          run overview: status, metrics, plots, scale warning
  /runs/<name>/viewer   per-frame overlay viewer (map points, marker, HUD)
  /runs/<name>/tracks   track quality
  /compare              delayed vs inherited triangulation comparison

Long pipeline steps run in background threads; the overview polls status.
"""
import json
import os
import threading
import traceback

import numpy as np
from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_file, url_for)

from . import pipeline, plots
from . import marker as marker_mod
from .obsgraph import StaleEditError, ReidentifyRequired
from .anchors import LensKind, MetricSuspendedError

RUNS_ROOT = os.environ.get("WORKBENCH_RUNS", os.path.join(os.getcwd(), "runs"))
os.makedirs(RUNS_ROOT, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

TASKS = {}   # run_name -> {"step": str, "log": [..], "error": str|None, "done": bool}
TASK_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# background task runner
# --------------------------------------------------------------------------
def _run_task(name, fn):
    def runner():
        with TASK_LOCK:
            TASKS[name] = {"step": "running", "log": [], "error": None, "done": False}
        try:
            fn(lambda msg: TASKS[name]["log"].append(str(msg)))
            with TASK_LOCK:
                TASKS[name]["done"] = True
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            with TASK_LOCK:
                TASKS[name]["error"] = f"{exc}\n{traceback.format_exc(limit=3)}"
                TASKS[name]["done"] = True
    threading.Thread(target=runner, daemon=True).start()


def _task(name):
    with TASK_LOCK:
        return dict(TASKS.get(name, {"step": "idle", "log": [], "error": None, "done": True}))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _run_dir(name):
    d = os.path.join(RUNS_ROOT, name)
    if not os.path.isdir(d):
        abort(404)
    return d


def _list_runs():
    out = []
    for n in sorted(os.listdir(RUNS_ROOT)):
        d = os.path.join(RUNS_ROOT, n)
        if not os.path.isdir(d):
            continue
        cfg = {}
        try:
            cfg = pipeline.load_config(d)
        except OSError:
            pass
        st = pipeline.get_status(d)
        out.append({"name": n, "config": cfg, "status": st.get("status", "?")})
    return out


def _load_metrics(run_dir):
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", runs=_list_runs())


@app.route("/runs/new", methods=["POST"])
def create_run():
    name = request.form.get("name", "").strip()
    if not name or any(c in name for c in "/.."):
        abort(400, "invalid run name")
    cfg = {
        "profile": request.form.get("profile", "survey"),
        "n_frames": int(request.form.get("n_frames", 100)),
        "seed": int(request.form.get("seed", 7)),
        "outlier_ratio": float(request.form.get("outlier_ratio", 0.0)),
        "strategy": request.form.get("strategy", "delayed"),
    }
    run_dir = pipeline.create_run(RUNS_ROOT, name, cfg)
    return redirect(url_for("run_overview", name=name))


@app.route("/runs/<name>/run_all", methods=["POST"])
def run_all(name):
    run_dir = _run_dir(name)
    cfg = pipeline.load_config(run_dir)

    def job(log):
        if not os.path.exists(os.path.join(run_dir, "gt.json")):
            log("generating footage...")
            pipeline.step_generate(run_dir)
        log("tracking...")
        pipeline.step_track(run_dir)
        log("solving...")
        pipeline.step_solve(run_dir, strategy=cfg.get("strategy", "delayed"), log=log)
        log("evaluating...")
        pipeline.step_evaluate(run_dir)
        log("done")

    _run_task(name, job)
    return redirect(url_for("run_overview", name=name))


@app.route("/runs/<name>/task")
def task_status(name):
    t = _task(name)
    st = pipeline.get_status(_run_dir(name))
    return jsonify({"task": t, "status": st})


@app.route("/runs/<name>/solve", methods=["POST"])
def solve_only(name):
    """Camera-solving control: runs the solver kernel on the current
    observation graph (including any committed observation deletions).
    Kernel failures feed back into the solve task status polled by the page."""
    run_dir = _run_dir(name)
    cfg = pipeline.load_config(run_dir)

    def job(log):
        pipeline.step_solve(run_dir, strategy=cfg.get("strategy", "delayed"),
                            log=log)
        pipeline.step_evaluate(run_dir)
    _run_task(name, job)
    return redirect(url_for("run_overview", name=name))


# --------------------------------------------------------------------------
# observation edits (undoable transactions over the observation graph)
# --------------------------------------------------------------------------
@app.route("/runs/<name>/edits")
def list_edits(name):
    run_dir = _run_dir(name)
    graph = pipeline.load_or_build_graph(run_dir)
    return jsonify({"version": graph.version,
                    "anchors": graph.anchors.to_dict(),
                    "edits": [{"id": e["id"], "base_version": e["base_version"],
                               "version": e["version"], "ops": e["ops"],
                               "undone": bool(e.get("undone")),
                               "undo_of": e.get("undo_of")}
                              for e in graph.edits()]})


@app.route("/runs/<name>/edits/delete_track", methods=["POST"])
def delete_track(name):
    """Explicit observation-deletion interface. The delete is an undoable
    transaction; it never goes through the solver or the UI point set."""
    run_dir = _run_dir(name)
    graph = pipeline.load_or_build_graph(run_dir)
    tid = int(request.form["track_id"])
    edit = graph.begin_edit()
    edit.delete_track(tid)
    try:
        eid = edit.commit()
    except StaleEditError as exc:
        abort(409, str(exc))
    graph.save(run_dir)
    return jsonify({"edit": eid, "version": graph.version})


@app.route("/runs/<name>/edits/delete_scale_anchor", methods=["POST"])
def delete_scale_anchor(name):
    run_dir = _run_dir(name)
    graph = pipeline.load_or_build_graph(run_dir)
    aid = int(request.form["anchor_id"])
    edit = graph.begin_edit()
    edit.delete_scale_anchor(aid)
    try:
        eid = edit.commit()
    except StaleEditError as exc:
        abort(409, str(exc))
    graph.save(run_dir)
    return jsonify({"edit": eid, "version": graph.version})


@app.route("/runs/<name>/edits/<int:eid>/undo", methods=["POST"])
def undo_edit(name, eid):
    run_dir = _run_dir(name)
    graph = pipeline.load_or_build_graph(run_dir)
    try:
        graph.undo(eid)
    except ReidentifyRequired as exc:
        # the track was split and its members no longer identify the original
        # constraint - the user must re-identify; nothing was bound by name
        return jsonify({"error": "reidentify_required", "detail": str(exc)}), 409
    except (ValueError, IndexError) as exc:
        abort(404, str(exc))
    graph.save(run_dir)
    return jsonify({"undone": eid, "version": graph.version})


# --------------------------------------------------------------------------
# metric landing-point export (suspends without a length reference)
# --------------------------------------------------------------------------
@app.route("/runs/<name>/export/landing")
def export_landing(name):
    run_dir = _run_dir(name)
    kind = request.args.get("lens", LensKind.MONOCULAR)
    baseline_m = request.args.get("baseline_m", type=float)
    pair = request.args.get("baseline_pair", "")
    baseline_pair = tuple(int(x) for x in pair.split(",") if x.strip()) or None
    if baseline_pair is not None and len(baseline_pair) != 2:
        abort(400, "baseline_pair must be two frame indices")
    try:
        out = pipeline.step_export_landing_points(
            run_dir, kind=kind, baseline_m=baseline_m, baseline_pair=baseline_pair)
    except MetricSuspendedError as exc:
        # preview keeps working; only the unit-bearing export is suspended
        return jsonify({"error": "metric_suspended", "detail": str(exc)}), 409
    return jsonify(out)


# --------------------------------------------------------------------------
# run overview
# --------------------------------------------------------------------------
@app.route("/runs/<name>")
def run_overview(name):
    run_dir = _run_dir(name)
    cfg = pipeline.load_config(run_dir)
    st = pipeline.get_status(run_dir)
    metrics = _load_metrics(run_dir)
    tf = pipeline.load_display_transform(run_dir)
    marker = pipeline.load_marker(run_dir)
    sol_data = None
    p = os.path.join(run_dir, "solution_raw.json")
    if os.path.exists(p):
        with open(p) as fh:
            sol_data = json.load(fh)
    has_gt = os.path.exists(os.path.join(run_dir, "gt.json"))
    return render_template("run.html", name=name, cfg=cfg, status=st,
                           metrics=metrics, transform=tf, marker=marker,
                           sol=sol_data, has_gt=has_gt, task=_task(name))


@app.route("/runs/<name>/align", methods=["POST"])
def align(name):
    run_dir = _run_dir(name)
    mode = request.form.get("mode", "auto")
    if mode == "auto":
        dist_pair, dist_m, ground_ids = pipeline.simulate_constraints(run_dir)
    else:
        pa = int(request.form["dist_a"]); pb = int(request.form["dist_b"])
        dist_m = float(request.form["dist_meters"])
        ground_ids = [int(x) for x in request.form["ground_ids"].split(",") if x.strip()]
        dist_pair = (pa, pb)
    pipeline.step_align(run_dir, dist_pair, dist_m, ground_ids)
    pipeline.step_evaluate(run_dir)   # refresh marker swimming etc.
    return redirect(url_for("run_overview", name=name))


@app.route("/runs/<name>/marker", methods=["POST"])
def marker(name):
    run_dir = _run_dir(name)
    args = {"center_xy": [float(request.form["cx"]), float(request.form["cy"])],
            "size": float(request.form.get("size", 0.5))}
    pipeline.save_marker(run_dir, args)

    def job(log):
        pipeline.step_render(run_dir)
        pipeline.step_evaluate(run_dir)
    _run_task(name, job)
    return redirect(url_for("run_overview", name=name))


@app.route("/runs/<name>/render", methods=["POST"])
def render(name):
    run_dir = _run_dir(name)

    def job(log):
        pipeline.step_render(run_dir)
    _run_task(name, job)
    return redirect(url_for("run_overview", name=name))


# --------------------------------------------------------------------------
# plots (generated on demand)
# --------------------------------------------------------------------------
@app.route("/runs/<name>/plot/<kind>.png")
def plot(name, kind):
    run_dir = _run_dir(name)
    sol, K, _ = pipeline.load_solution(run_dir)
    gt = pipeline.load_gt(run_dir) if os.path.exists(os.path.join(run_dir, "gt.json")) else None
    metrics = _load_metrics(run_dir)
    if kind == "trajectory":
        png = plots.plot_trajectory(sol, gt)
    elif kind == "camera_errors" and metrics:
        png = plots.plot_camera_errors(metrics)
    elif kind == "parallax":
        png = plots.plot_parallax(sol, gt)
    elif kind == "reprojection" and metrics:
        png = plots.plot_reprojection(metrics)
    elif kind == "marker_swimming" and metrics:
        png = plots.plot_marker_swimming(metrics)
        if png is None:
            abort(404)
    else:
        abort(404)
    return send_file(__import__("io").BytesIO(png), mimetype="image/png")


# --------------------------------------------------------------------------
# per-frame viewer
# --------------------------------------------------------------------------
@app.route("/runs/<name>/viewer")
def viewer(name):
    run_dir = _run_dir(name)
    n = _n_frames(run_dir)
    return render_template("viewer.html", name=name, n_frames=n)


def _n_frames(run_dir):
    vd = os.path.join(run_dir, "video")
    return len([f for f in os.listdir(vd) if f.endswith(".png")])


@app.route("/runs/<name>/frame/<int:idx>")
def frame_image(name, idx):
    run_dir = _run_dir(name)
    # prefer a pre-rendered overlay; otherwise render on the fly
    rp = os.path.join(run_dir, "render", f"frame_{idx:04d}.png")
    if os.path.exists(rp):
        return send_file(rp)
    sol, K, _ = pipeline.load_solution(run_dir)
    tf = pipeline.load_display_transform(run_dir)
    mk = pipeline.load_marker(run_dir)
    mk_obj = marker_mod.make_cube_marker(**mk["args"]) if mk else None
    from PIL import Image
    base_path = os.path.join(run_dir, "video", f"frame_{idx:04d}.png")
    if not os.path.exists(base_path):
        abort(404)
    base = np.asarray(Image.open(base_path).convert("L"))
    over = marker_mod.render_frame_overlay(base, sol, idx, K, mk_obj, tf)
    import io
    buf = io.BytesIO()
    Image.fromarray(over).save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/runs/<name>/frame_info/<int:idx>")
def frame_info(name, idx):
    run_dir = _run_dir(name)
    sol, K, _ = pipeline.load_solution(run_dir)
    info = sol.frame_info.get(idx, {})
    return jsonify({"frame": idx, "solved": idx in sol.cameras,
                    "method": info.get("method"),
                    "rotation_only": bool(info.get("rotation_only")),
                    "parallax_deg": info.get("parallax_deg")})


# --------------------------------------------------------------------------
# track quality
# --------------------------------------------------------------------------
@app.route("/runs/<name>/tracks")
def tracks_page(name):
    run_dir = _run_dir(name)
    tracks, n_frames, injected = pipeline.load_tracks(run_dir)
    n_injected = int(np.asarray(injected).sum()) if injected is not None else 0
    return render_template("tracks.html", name=name, n_tracks=len(tracks),
                           n_frames=n_frames, n_injected=n_injected)


@app.route("/runs/<name>/plot/tracks.png")
def tracks_plot(name):
    run_dir = _run_dir(name)
    tracks, n_frames, _ = pipeline.load_tracks(run_dir)
    png = plots.plot_track_quality(tracks)
    return send_file(__import__("io").BytesIO(png), mimetype="image/png")


# --------------------------------------------------------------------------
# strategy comparison
# --------------------------------------------------------------------------
@app.route("/compare")
def compare():
    res = {}
    for strat in ("delayed", "inherited"):
        d = os.path.join(RUNS_ROOT, f"cmp_{strat}")
        if os.path.isdir(d):
            m = _load_metrics(d)
            if m:
                res[strat] = m
    png = None
    if "delayed" in res and "inherited" in res:
        png = plots.plot_ground_contact_compare(
            res["delayed"]["ground_contact"], res["inherited"]["ground_contact"])
    return render_template("compare.html", res=res, has_plot=png is not None)


@app.route("/compare/plot.png")
def compare_plot():
    res = {}
    for strat in ("delayed", "inherited"):
        d = os.path.join(RUNS_ROOT, f"cmp_{strat}")
        m = _load_metrics(d) if os.path.isdir(d) else None
        if m:
            res[strat] = m
    if "delayed" not in res or "inherited" not in res:
        abort(404)
    png = plots.plot_ground_contact_compare(
        res["delayed"]["ground_contact"], res["inherited"]["ground_contact"])
    return send_file(__import__("io").BytesIO(png), mimetype="image/png")


@app.route("/compare/run", methods=["POST"])
def compare_run():
    """Solve the same footage with both strategies for a clean comparison."""
    base = request.form.get("base", "cmp_src")
    src = os.path.join(RUNS_ROOT, base)
    if not os.path.isdir(src):
        abort(404, "base run not found")

    def job(log):
        for strat in ("delayed", "inherited"):
            dst = os.path.join(RUNS_ROOT, f"cmp_{strat}")
            os.makedirs(dst, exist_ok=True)
            for fn in ("config.json", "gt.json", "tracks.npz"):
                s = os.path.join(src, fn)
                if os.path.exists(s):
                    import shutil
                    shutil.copy(s, os.path.join(dst, fn))
            if not os.path.exists(os.path.join(dst, "video")):
                os.symlink(os.path.join(src, "video"), os.path.join(dst, "video"))
            log(f"solving {strat}...")
            pipeline.step_solve(dst, strategy=strat, log=log)
            pipeline.step_evaluate(dst)
        log("done")
    _run_task("cmp", job)
    return redirect(url_for("compare"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
