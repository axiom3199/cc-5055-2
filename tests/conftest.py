import json
import os
import shutil

import numpy as np
import pytest

from workbench import pipeline


def _make_run(root, name, profile="survey", n_frames=50, seed=7, outlier_ratio=0.0):
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump({"profile": profile, "n_frames": n_frames, "seed": seed,
                   "outlier_ratio": outlier_ratio, "strategy": "delayed"}, fh)
    return run_dir


def _clone_tracked(src, dst):
    """Copy a tracked run (config/gt/tracks + video symlink) into a new dir so
    each strategy solves into its own directory."""
    os.makedirs(dst, exist_ok=True)
    for fn in ("config.json", "gt.json", "tracks.npz"):
        shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))
    vd = os.path.join(dst, "video")
    if not os.path.exists(vd):
        os.symlink(os.path.join(src, "video"), vd)
    return dst


@pytest.fixture(scope="session")
def survey_tracked(tmp_path_factory):
    """A small survey-profile run, generated + tracked once (60 frames)."""
    root = tmp_path_factory.mktemp("runs")
    run_dir = _make_run(root, "survey_src", profile="survey", n_frames=50)
    pipeline.step_generate(run_dir)
    pipeline.step_track(run_dir)
    return run_dir


@pytest.fixture(scope="session")
def survey_run(survey_tracked, tmp_path_factory):
    """Delayed-strategy solve in its own directory."""
    dst = _clone_tracked(survey_tracked, os.path.join(
        os.path.dirname(survey_tracked), "survey_delayed"))
    pipeline.step_solve(dst, strategy="delayed", ba_max_nfev=120, log=lambda m: None)
    return dst


@pytest.fixture(scope="session")
def survey_solution(survey_run):
    sol, K, _ = pipeline.load_solution(survey_run)
    return sol


@pytest.fixture(scope="session")
def survey_metrics(survey_run):
    return pipeline.step_evaluate(survey_run)


@pytest.fixture(scope="session")
def inherited_run(survey_tracked, tmp_path_factory):
    dst = _clone_tracked(survey_tracked, os.path.join(
        os.path.dirname(survey_tracked), "survey_inherited"))
    pipeline.step_solve(dst, strategy="inherited", ba_max_nfev=120, log=lambda m: None)
    return dst


@pytest.fixture(scope="session")
def inherited_solution(inherited_run):
    sol, K, _ = pipeline.load_solution(inherited_run)
    return sol


@pytest.fixture(scope="session")
def robust_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs_rob")
    run_dir = _make_run(root, "robust", profile="survey", n_frames=50,
                        outlier_ratio=0.05)
    pipeline.step_generate(run_dir)
    pipeline.step_track(run_dir)
    pipeline.step_solve(run_dir, strategy="delayed", ba_max_nfev=120,
                        log=lambda m: None)
    return run_dir


@pytest.fixture(scope="session")
def robust_solution(robust_run):
    sol, K, _ = pipeline.load_solution(robust_run)
    return sol


@pytest.fixture(scope="session")
def small_run(tmp_path_factory):
    """A 24-frame survey run with a reduced track count, solved once through
    the current (graph) pipeline. Shared by the refactor-equivalence and UI
    tests so each does not pay for its own full solve."""
    root = tmp_path_factory.mktemp("runs_small")
    run_dir = _make_run(root, "small", profile="survey", n_frames=24)
    pipeline.step_generate(run_dir)
    pipeline.step_track(run_dir, max_tracks=350)
    pipeline.step_solve(run_dir, strategy="delayed", ba_max_nfev=60,
                        log=lambda m: None)
    return run_dir
