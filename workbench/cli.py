"""Command-line interface for the camera-motion workbench.

  python -m workbench.cli generate RUN [--profile survey] [--frames 100] [--seed 7]
  python -m workbench.cli track RUN [--outlier-ratio 0.05]
  python -m workbench.cli solve RUN [--strategy delayed] [--ba-nfev 300]
  python -m workbench.cli align RUN --auto            # or --dist A B METERS --ground IDS
  python -m workbench.cli evaluate RUN
  python -m workbench.cli marker RUN --cx 1.0 --cy 0.5 --size 0.5
  python -m workbench.cli render RUN
  python -m workbench.cli export-landing RUN [--lens monocular|stereo] [--baseline-m 0.12 --baseline-pair 0,1]
  python -m workbench.cli pipeline RUN                # generate..evaluate in one go
  python -m workbench.cli compare RUN                 # solve both strategies, report
  python -m workbench.cli serve                       # web UI

RUN is a directory under WORKBENCH_RUNS (default ./runs) or an absolute path.
"""
import argparse
import json
import os
import sys

from . import pipeline

RUNS_ROOT = os.environ.get("WORKBENCH_RUNS", os.path.join(os.getcwd(), "runs"))


def _run_dir(arg):
    return arg if os.path.isabs(arg) else os.path.join(RUNS_ROOT, arg)


def _print_metrics(m):
    cam = m["cameras"]
    print(f"  cameras : pos RMSE {cam['pos_rmse_m']:.3f} m (max {cam['pos_max_m']:.3f}), "
          f"rot RMSE {cam['rot_rmse_deg']:.3f} deg (max {cam['rot_max_deg']:.3f})")
    r = m["reprojection"]
    print(f"  reproj  : mean {r['mean_px']:.3f} px, p95 {r['p95_px']:.3f} px, "
          f"{r['n_outliers_rejected']} outliers rejected")
    gc = m["ground_contact"]
    line = (f"  ground  : {gc['n_ground_tracks_born_in_rotation']} rotation-born, "
            f"{gc['n_deferred_no_estimate']} deferred, {gc['n_estimated']} estimated")
    if gc["n_estimated"]:
        line += (f", median |z| {gc['median_abs_z_err_m']:.3f} m, "
                 f"p90 {gc['p90_abs_z_err_m']:.3f} m")
    print(line)
    if m.get("marker_swimming") and m["marker_swimming"].get("mean_px") is not None:
        ms = m["marker_swimming"]
        print(f"  marker  : swim mean {ms['mean_px']:.2f} px, p95 {ms['p95_px']:.2f} px")


def cmd_generate(a):
    run_dir = _run_dir(a.run)
    pipeline.create_run(RUNS_ROOT, a.run, {
        "profile": a.profile, "n_frames": a.frames, "seed": a.seed,
        "outlier_ratio": a.outlier_ratio, "strategy": a.strategy})
    pipeline.step_generate(run_dir)
    print(f"[{a.run}] generated {a.frames} frames, profile={a.profile}")


def cmd_track(a):
    run_dir = _run_dir(a.run)
    tracks, n_frames, n_inj = pipeline.step_track(run_dir)
    print(f"[{a.run}] tracked {len(tracks)} tracks over {n_frames} frames"
          + (f", injected {n_inj} wrong matches" if n_inj else ""))


def cmd_solve(a):
    run_dir = _run_dir(a.run)
    sol = pipeline.step_solve(run_dir, strategy=a.strategy, ba_max_nfev=a.ba_nfev)
    print(f"[{a.run}] solved {len(sol.cameras)} cameras, {len(sol.points)} points "
          f"(strategy={sol.strategy})")


def cmd_align(a):
    run_dir = _run_dir(a.run)
    if a.auto:
        dist_pair, dist_m, ground_ids = pipeline.simulate_constraints(run_dir)
        print(f"[{a.run}] simulated constraints: pair={dist_pair} dist={dist_m:.3f} m, "
              f"{len(ground_ids)} ground points")
    else:
        dist_pair = (a.dist[0], a.dist[1])
        dist_m = a.dist[2]
        ground_ids = [int(x) for x in a.ground.split(",") if x.strip()]
    tf = pipeline.step_align(run_dir, dist_pair, dist_m, ground_ids)
    print(f"[{a.run}] display transform: scale={tf['scale']:.4f}, "
          f"ground RMS={tf['ground_plane_rms_m']:.3f} m (saved separately from raw solution)")


def cmd_evaluate(a):
    m = pipeline.step_evaluate(_run_dir(a.run))
    print(f"[{a.run}] metrics:")
    _print_metrics(m)


def cmd_marker(a):
    run_dir = _run_dir(a.run)
    pipeline.save_marker(run_dir, {"center_xy": [a.cx, a.cy], "size": a.size})
    print(f"[{a.run}] marker placed at ({a.cx},{a.cy}) size {a.size}")


def cmd_render(a):
    paths = pipeline.step_render(_run_dir(a.run))
    print(f"[{a.run}] rendered {len(paths)} frames")


def cmd_export_landing(a):
    from .anchors import MetricSuspendedError
    pair = tuple(int(x) for x in a.baseline_pair.split(",")) if a.baseline_pair else None
    try:
        out = pipeline.step_export_landing_points(
            _run_dir(a.run), kind=a.lens, baseline_m=a.baseline_m,
            baseline_pair=pair)
    except MetricSuspendedError as exc:
        print(f"[{a.run}] metric export SUSPENDED: {exc}")
        sys.exit(2)
    print(f"[{a.run}] {len(out['points'])} landing points, units={out['units']}, "
          f"scale from {out['scale_evidence']} "
          f"(ground RMS {out['ground_plane_rms_m']:.3f} m)")


def cmd_pipeline(a):
    run_dir = _run_dir(a.run)
    if not os.path.exists(os.path.join(run_dir, "config.json")):
        cmd_generate(a)
    if not os.path.exists(os.path.join(run_dir, "gt.json")):
        pipeline.step_generate(run_dir)
    cmd_track(a)
    cmd_solve(a)
    try:
        cmd_align(argparse.Namespace(run=a.run, auto=True))
    except Exception as exc:  # noqa: BLE001
        print(f"[{a.run}] align skipped: {exc}")
    cmd_evaluate(a)


def cmd_compare(a):
    """Solve the same tracked footage with both strategies and report."""
    import shutil
    src = _run_dir(a.run)
    for strat in ("delayed", "inherited"):
        dst = _run_dir(f"cmp_{strat}")
        os.makedirs(dst, exist_ok=True)
        for fn in ("config.json", "gt.json", "tracks.npz"):
            s = os.path.join(src, fn)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(dst, fn))
        vd = os.path.join(dst, "video")
        if not os.path.exists(vd):
            os.symlink(os.path.join(src, "video"), vd)
        pipeline.step_solve(dst, strategy=strat, log=lambda m: None)
        m = pipeline.step_evaluate(dst)
        print(f"[{strat}]")
        _print_metrics(m)


def cmd_serve(a):
    from .app import app
    app.run(host=a.host, port=a.port, debug=False)


def main(argv=None):
    p = argparse.ArgumentParser(prog="workbench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def base(name, **kw):
        sp = sub.add_parser(name, **kw)
        sp.add_argument("run")
        return sp

    g = base("generate")
    g.add_argument("--profile", default="survey")
    g.add_argument("--frames", type=int, default=100)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--outlier-ratio", type=float, default=0.0)
    g.add_argument("--strategy", default="delayed")
    g.set_defaults(fn=cmd_generate)

    t = base("track")
    t.add_argument("--outlier-ratio", type=float, default=None)
    t.set_defaults(fn=cmd_track)

    s = base("solve")
    s.add_argument("--strategy", default=None)
    s.add_argument("--ba-nfev", type=int, default=300)
    s.set_defaults(fn=cmd_solve)

    al = base("align")
    al.add_argument("--auto", action="store_true")
    al.add_argument("--dist", nargs=3, type=float, metavar=("A", "B", "METERS"))
    al.add_argument("--ground", default="")
    al.set_defaults(fn=cmd_align)

    base("evaluate").set_defaults(fn=cmd_evaluate)

    mk = base("marker")
    mk.add_argument("--cx", type=float, default=1.0)
    mk.add_argument("--cy", type=float, default=0.5)
    mk.add_argument("--size", type=float, default=0.5)
    mk.set_defaults(fn=cmd_marker)

    base("render").set_defaults(fn=cmd_render)

    el = base("export-landing")
    el.add_argument("--lens", default="monocular",
                    choices=["monocular", "stereo"])
    el.add_argument("--baseline-m", type=float, default=None)
    el.add_argument("--baseline-pair", default="")
    el.set_defaults(fn=cmd_export_landing)

    pl = base("pipeline")
    pl.add_argument("--profile", default="survey")
    pl.add_argument("--frames", type=int, default=100)
    pl.add_argument("--seed", type=int, default=7)
    pl.add_argument("--outlier-ratio", type=float, default=0.0)
    pl.add_argument("--strategy", default="delayed")
    pl.add_argument("--ba-nfev", type=int, default=300)
    pl.set_defaults(fn=cmd_pipeline)

    base("compare").set_defaults(fn=cmd_compare)

    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=5000)
    sv.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    # strategy/outlier defaults fall back to the run config when not given
    if getattr(a, "strategy", None) is None and a.cmd == "solve":
        a.strategy = None
    a.fn(a)


if __name__ == "__main__":
    main()
