"""Refactor equivalence evidence: the layered observation graph must not
change camera solving's external behavior.

Equivalence is bounded by reprojection error and parameter degrees of
freedom (BA n_params), over before/after diffs built from point deletion
and long-track selection. Everything is generated in a clean directory by
the fixtures - no hand-edited data.
"""
import numpy as np
import pytest

from workbench import pipeline, sfm
from workbench.obsgraph import ObservationGraph, ObsId, UnknownObservationError

BA_NFEV = 60


def _assert_same_solution(a, b, atol=1e-6):
    """Camera-overlay equivalence: identical cameras and points, equal
    reprojection statistics and parameter degrees of freedom."""
    assert set(a.cameras) == set(b.cameras)
    for f in a.cameras:
        Ra, ta = a.cameras[f]
        Rb, tb = b.cameras[f]
        assert np.allclose(Ra, Rb, atol=atol), f"camera {f} rotation diverged"
        assert np.allclose(ta, tb, atol=atol), f"camera {f} translation diverged"
    assert set(a.points) == set(b.points)
    for pid in a.points:
        assert np.allclose(a.points[pid], b.points[pid], atol=atol), \
            f"point {pid} diverged"
    # reprojection evidence
    assert a.observations == b.observations
    assert np.allclose(a.obs_residual, b.obs_residual, atol=atol)
    assert a.obs_outlier == b.obs_outlier
    # parameter degrees of freedom evidence
    assert a.ba_info["n_params"] == b.ba_info["n_params"]
    assert a.ba_info["n_obs"] == b.ba_info["n_obs"]


def test_new_interface_matches_legacy_entry(small_run):
    """The graph-interface solve (already stored in small_run) and the old
    solve(tracks, ...) entry on the same footage: identical solution."""
    sol_new, K, _ = pipeline.load_solution(small_run)   # solved via solve_graph
    tracks, n_frames, _ = pipeline.load_tracks(small_run)
    sol_old = sfm.solve(tracks, n_frames, K, strategy="delayed",
                        ba_max_nfev=BA_NFEV, log=lambda m: None)
    _assert_same_solution(sol_old, sol_new)
    # execution evidence, old vs new
    assert sol_old.ba_info["n_params"] == sol_new.ba_info["n_params"] > 0
    assert np.isfinite(sol_new.ba_info["rms_px"])


def test_point_deletion_diff(small_run):
    """Before/after diff built by deleting observations: a transaction on the
    graph vs the pre-refactor way (mutating track dicts, splitting the track
    by hand with the same fresh ids). Both diffs agree, and solving the
    diffed problem through either entry gives identical reprojection error
    and parameter degrees of freedom."""
    tracks, n_frames, _ = pipeline.load_tracks(small_run)
    K, _ = pipeline.load_intrinsics(small_run)
    track_by_id = {t["id"]: t for t in tracks}

    # pick a long track and delete interior observations -> splits in two
    tid = max((t["id"] for t in tracks if len(t["frames"]) >= 20),
              key=lambda i: len(track_by_id[i]["frames"]))
    victim = track_by_id[tid]
    cut = [int(f) for f in victim["frames"][5:8]]      # three interior frames

    # new world: undoable transaction on the observation graph
    graph = ObservationGraph.from_tracks(tracks, n_frames)
    edit = graph.begin_edit()
    edit.delete_observations([ObsId(tid, f) for f in cut])
    edit.commit()
    view_tracks = graph.view().tracks()

    # old world: hand-mutated track dicts replicating the documented split
    # semantics (fragments get fresh ids in order, old id retired)
    max_id = max(t["id"] for t in tracks)
    manual = []
    for t in tracks:
        if t["id"] != tid:
            manual.append(t)
            continue
        keep = [i for i, f in enumerate(t["frames"]) if int(f) not in cut]
        runs, cur = [], []
        for i in keep:
            if cur and int(t["frames"][i]) != int(t["frames"][cur[-1]]) + 1:
                runs.append(cur)
                cur = []
            cur.append(i)
        if cur:
            runs.append(cur)
        for k, run in enumerate(runs):
            manual.append({"id": max_id + 1 + k,
                           "frames": t["frames"][run],
                           "xy": t["xy"][run]})

    # the two diffs agree exactly (ids, frames, pixels); the solver also sees
    # them in the same (id-sorted) order
    manual = sorted(manual, key=lambda t: t["id"])
    assert len(view_tracks) == len(manual)
    for a, b in zip(view_tracks, manual):
        assert a["id"] == b["id"]
        assert np.array_equal(a["frames"], b["frames"])
        assert np.allclose(a["xy"], b["xy"])

    sol_new = sfm.solve_graph(graph.view(), K=K, strategy="delayed",
                              ba_max_nfev=BA_NFEV, log=lambda m: None)
    sol_old = sfm.solve(manual, n_frames, K, strategy="delayed",
                        ba_max_nfev=BA_NFEV, log=lambda m: None)
    _assert_same_solution(sol_old, sol_new)


def test_long_track_diff(small_run):
    """Before/after diff built by selecting long tracks only: deleting every
    short track through the graph transaction interface equals handing the
    old entry a pre-filtered track list."""
    tracks, n_frames, _ = pipeline.load_tracks(small_run)
    K, _ = pipeline.load_intrinsics(small_run)
    min_len = 10
    short_ids = [t["id"] for t in tracks if len(t["frames"]) < min_len]
    long_tracks = [t for t in tracks if len(t["frames"]) >= min_len]
    assert 0 < len(short_ids) < len(tracks)

    graph = ObservationGraph.from_tracks(tracks, n_frames)
    edit = graph.begin_edit()
    for tid in short_ids:
        edit.delete_track(tid)
    edit.commit()

    sol_new = sfm.solve_graph(graph.view(), K=K, strategy="delayed",
                              ba_max_nfev=BA_NFEV, log=lambda m: None)
    sol_old = sfm.solve(long_tracks, n_frames, K, strategy="delayed",
                        ba_max_nfev=BA_NFEV, log=lambda m: None)
    _assert_same_solution(sol_old, sol_new)


def test_resolve_after_deletion_stays_consistent(small_run, tmp_path):
    """After an observation deletion, re-solving through the pipeline reads
    the edited graph; the reloaded solution joins cleanly against the graph
    tracks (fragment ids) and evaluates without KeyErrors."""
    import shutil
    run_copy = tmp_path / "edited"
    shutil.copytree(small_run, run_copy)
    graph = pipeline.load_or_build_graph(str(run_copy))
    tid = graph.tracks.ids()[7]
    edit = graph.begin_edit()
    edit.delete_track(tid)
    edit.commit()
    graph.save(str(run_copy))

    pipeline.step_solve(str(run_copy), strategy="delayed", ba_max_nfev=BA_NFEV,
                        log=lambda m: None)
    sol, K, _ = pipeline.load_solution(str(run_copy))
    assert tid not in sol.points          # the deleted track is gone for good
    for pid in sol.points:
        assert pid in sol._tracks         # every point joins a graph track
    m = pipeline.step_evaluate(str(run_copy))
    assert np.isfinite(m["reprojection"]["mean_px"])


def test_interface_does_not_mask_errors_with_defaults(small_run, tmp_path):
    """The camera observation interface must surface corrupt or incomplete
    data, never paper over it with defaults."""
    import json
    import os
    import shutil

    # a solution whose residual layer is short must fail loudly on load
    run_copy = tmp_path / "tampered"
    shutil.copytree(small_run, run_copy)
    z = np.load(run_copy / "observations.npz")
    np.savez_compressed(run_copy / "observations.npz",
                        obs=z["obs"],
                        residual=z["residual"][:-7],   # drop residuals
                        outlier=z["outlier"])
    with pytest.raises(RuntimeError, match="residuals"):
        pipeline.load_solution(str(run_copy))

    # missing observation identity is an error, not a default
    sol, K, _ = pipeline.load_solution(small_run)
    with pytest.raises(UnknownObservationError):
        sol.residual_store.get(ObsId(10 ** 9, 10 ** 9))

    # solve_graph takes its frame count from the view, not from a default
    tracks, n_frames, _ = pipeline.load_tracks(small_run)
    view = ObservationGraph.from_tracks(tracks, n_frames).view()
    assert view.n_frames == n_frames
    with pytest.raises(TypeError):
        sfm.solve_graph(view)          # K is required, no default
