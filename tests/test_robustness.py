import numpy as np

from workbench import pipeline


def test_outliers_injected(robust_run):
    _, _, injected = pipeline.load_tracks(robust_run)
    assert int(np.asarray(injected).sum()) > 100


def test_robust_solution_stays_accurate(robust_run, robust_solution):
    m = pipeline.step_evaluate(robust_run)
    cam = m["cameras"]
    # with 5% wrong matches the solver must not collapse
    assert len(robust_solution.cameras) == 50
    assert cam["pos_rmse_m"] < 0.8
    assert cam["rot_rmse_deg"] < 2.0
    # and the robust BA must reject a substantial number of outliers
    assert m["reprojection"]["n_outliers_rejected"] > 500
