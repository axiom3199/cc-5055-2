import numpy as np

from workbench import pipeline


def test_full_pipeline_solves_all_frames(survey_solution, survey_metrics):
    sol = survey_solution
    assert len(sol.cameras) == 50
    assert len(sol.points) > 300
    # init actually happened with sufficient parallax
    assert sol.init_info["inliers"] > 100


def test_camera_accuracy(survey_metrics):
    cam = survey_metrics["cameras"]
    # honest thresholds for noisy KLT on weak-geometry monocular footage
    assert cam["pos_rmse_m"] < 0.7
    assert cam["rot_rmse_deg"] < 1.5
    reproj = survey_metrics["reprojection"]
    assert reproj["mean_px"] < 1.5


def test_rotation_segments_detected(survey_solution):
    sol = survey_solution
    rot = sorted(f for f in sol.cameras if sol.frame_info[f].get("rotation_only"))
    # survey profile (50 frames): rotation segments are frames 20-29 and 40-49
    assert len(rot) > 8, "rotation segments should be flagged"
    in_a = [f for f in rot if 18 <= f <= 31]
    in_b = [f for f in rot if 38 <= f <= 49]
    assert len(in_a) >= 5 and len(in_b) >= 5, \
        f"expected rotation flags in both segments, got {rot}"
    # translation frames (e.g. 3..18) must NOT be flagged
    early = [f for f in rot if 3 <= f <= 18]
    assert len(early) == 0


def test_raw_solution_has_no_metric_claim(survey_run):
    import json, os
    with open(os.path.join(survey_run, "solution_raw.json")) as fh:
        data = json.load(fh)
    assert "arbitrary" in data["units"] or "up-to-scale" in data["units"]
