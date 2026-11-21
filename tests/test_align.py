import numpy as np

from workbench import pipeline, align
from workbench import geometry as geo


def test_ground_plane_fit():
    rng = np.random.default_rng(0)
    # plane z = 2 with noise and outliers
    X = rng.uniform(-5, 5, (200, 3))
    X[:, 2] = 2.0 + rng.normal(0, 0.01, 200)
    X[::20] += rng.normal(0, 3, 3)          # outliers
    c, n = align.fit_ground_plane(X)
    assert abs(abs(n[2]) - 1.0) < 0.05       # normal ~ +Z
    assert abs(c[2] - 2.0) < 0.1


def test_display_transform_scale_and_ground(survey_run, survey_solution):
    # simulate user constraints from GT and build the transform
    dist_pair, dist_m, ground_ids = pipeline.simulate_constraints(survey_run)
    tf = pipeline.step_align(survey_run, dist_pair, dist_m, ground_ids)
    # the known distance must be reproduced in display coordinates
    sol, K, _ = pipeline.load_solution(survey_run)
    pa = align.apply_transform(tf, sol.points[dist_pair[0]])
    pb = align.apply_transform(tf, sol.points[dist_pair[1]])
    assert np.isclose(np.linalg.norm(pa - pb), dist_m, rtol=1e-3)
    # ground points must lie near z=0
    zs = [align.apply_transform(tf, sol.points[g])[2] for g in ground_ids]
    assert np.std(zs) < 0.5
    # transform stored separately from the raw solution
    import os
    assert os.path.exists(os.path.join(survey_run, "display_transform.json"))
    assert os.path.exists(os.path.join(survey_run, "solution_raw.json"))


def test_transform_is_similarity(survey_run, survey_solution):
    dist_pair, dist_m, ground_ids = pipeline.simulate_constraints(survey_run)
    tf = pipeline.step_align(survey_run, dist_pair, dist_m, ground_ids)
    M = np.asarray(tf["matrix"])
    R = M[:3, :3] / tf["scale"]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)   # rotation * uniform scale
