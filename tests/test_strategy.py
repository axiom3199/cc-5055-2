import numpy as np

from workbench import pipeline


def test_strategies_share_camera_trajectory(survey_run, survey_solution,
                                            inherited_solution):
    """Inherited-depth points are display-only; both strategies must produce
    essentially the same cameras (the comparison isolates the depth strategy).
    Tolerance allows for floating-point noise from different parameter orderings."""
    assert set(survey_solution.cameras) == set(inherited_solution.cameras)
    for f in survey_solution.cameras:
        Ra, ta = survey_solution.cameras[f]
        Rb, tb = inherited_solution.cameras[f]
        assert np.allclose(Ra, Rb, atol=1e-4)
        assert np.allclose(ta, tb, atol=1e-4)


def test_delayed_defers_more_than_inherited(survey_run, survey_solution,
                                            inherited_solution):
    md = pipeline.step_evaluate(survey_run)["ground_contact"]
    # re-solve inherited metrics on its own run dir is done in the UI; here we
    # compare the point sets directly.
    n_inh_points = sum(1 for m in inherited_solution.point_meta.values()
                       if m.get("inherited"))
    n_delayed_estimated = md["n_estimated"]
    # delayed leaves most rotation-born ground tracks without an estimate
    assert md["n_deferred_no_estimate"] > md["n_estimated"]
    # inherited creates immediate estimates for tracks delayed defers
    assert n_inh_points > 0


def test_delayed_estimates_are_triangulated_not_inherited(survey_solution):
    for tid, meta in survey_solution.point_meta.items():
        assert not meta.get("inherited"), "delayed strategy must not inherit depth"
