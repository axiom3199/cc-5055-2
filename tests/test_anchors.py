"""Lens projects and scale anchors: one ground anchor serves both a
monocular and a stereo project; the stereo project's known baseline keeps
metric export alive after the distance anchor is deleted, while the
monocular project must suspend unit-bearing export (preview unaffected)."""
import numpy as np
import pytest

from workbench.anchors import (AnchorSet, LensProject, LensKind,
                               MetricSuspendedError, UnknownAnchorError)
from workbench.obsgraph import ObservationGraph
from workbench.sfm import Solution


def _solution():
    """Raw up-to-scale solution: raw = metric / 2 (true scale 2.0)."""
    sol = Solution()
    metric_pts = {0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0),
                  3: (1.0, 1.0, 0.0), 4: (0.5, 0.5, 1.0)}
    sol.points = {pid: np.asarray(p, float) / 2.0 for pid, p in metric_pts.items()}
    # metric camera centers, baseline 0.12 m between frames 0 and 1
    for f, C in ((0, (0.3, 0.4, 1.5)), (1, (0.42, 0.4, 1.5)), (2, (0.54, 0.4, 1.5))):
        Craw = np.asarray(C, float) / 2.0
        sol.cameras[f] = (np.eye(3), -Craw)
    return sol


def _graph_with_anchors():
    g = ObservationGraph(3)
    g.anchors.upsert_distance_anchor((0, 1), 1.0)   # the only length reference
    g.anchors.add_ground([0, 1, 2, 3])              # the shared ground anchor
    return g


def test_same_ground_anchor_serves_both_lens_projects():
    g = _graph_with_anchors()
    sol = _solution()
    mono = LensProject(LensKind.MONOCULAR, g)
    stereo = LensProject(LensKind.STEREO, g, baseline_m=0.12, baseline_pair=(0, 1))
    tm = mono.metric_transform(sol)
    ts = stereo.metric_transform(sol)
    # one ground anchor -> one ground plane -> identical display orientation
    assert np.allclose(tm["rotation"], ts["rotation"])
    assert np.allclose(tm["scale"], ts["scale"])
    assert tm["scale_evidence"] == "distance_anchor"


def test_monocular_export_suspends_after_deleting_only_length_reference():
    g = _graph_with_anchors()
    sol = _solution()
    mono = LensProject(LensKind.MONOCULAR, g)
    out = mono.export_landing_points(sol)
    assert out["units"] == "m"
    assert {p["id"] for p in out["points"]} == {0, 1, 2, 3}   # elevated point excluded
    # metric distance between anchor points is reproduced
    p0, p1 = (np.array(p["xyz_m"]) for p in out["points"] if p["id"] in (0, 1))
    assert np.isclose(np.linalg.norm(p1 - p0), 1.0, atol=1e-6)

    aid = next(iter(g.anchors.distance_anchors))
    edit = g.begin_edit()
    edit.delete_scale_anchor(aid)
    edit.commit()
    assert not mono.metric_available(sol)
    with pytest.raises(MetricSuspendedError):
        mono.export_landing_points(sol)


def test_stereo_keeps_metric_landing_points_after_anchor_deletion():
    g = _graph_with_anchors()
    sol = _solution()
    stereo = LensProject(LensKind.STEREO, g, baseline_m=0.12, baseline_pair=(0, 1))
    aid = next(iter(g.anchors.distance_anchors))
    edit = g.begin_edit()
    edit.delete_scale_anchor(aid)
    edit.commit()
    # no distance anchor left, but the known baseline still fixes scale
    assert stereo.metric_available(sol)
    out = stereo.export_landing_points(sol)
    assert out["scale_evidence"] == "stereo_baseline"
    p0, p1 = (np.array(p["xyz_m"]) for p in out["points"] if p["id"] in (0, 1))
    assert np.isclose(np.linalg.norm(p1 - p0), 1.0, atol=1e-6)


def test_anchor_deletion_is_undoable_and_restores_metric():
    g = _graph_with_anchors()
    sol = _solution()
    mono = LensProject(LensKind.MONOCULAR, g)
    aid = next(iter(g.anchors.distance_anchors))
    edit = g.begin_edit()
    edit.delete_scale_anchor(aid)
    eid = edit.commit()
    assert not mono.metric_available(sol)
    g.undo(eid)
    assert mono.metric_available(sol)
    out = mono.export_landing_points(sol)
    assert out["scale_evidence"] == "distance_anchor"


def test_anchor_set_strict_lookup_and_persistence():
    a = AnchorSet()
    with pytest.raises(UnknownAnchorError):
        a.get_distance_anchor(7)
    aid = a.upsert_distance_anchor((3, 4), 2.0)
    a.add_ground([1, 2, 3])
    b = AnchorSet.from_dict(a.to_dict())
    assert b.get_distance_anchor(aid)["meters"] == 2.0
    assert b.ground_ids == {1, 2, 3}
    # upserting the same point pair replaces instead of duplicating
    a.upsert_distance_anchor((3, 4), 3.0)
    assert len(a.distance_anchors) == 1
    assert a.get_distance_anchor(aid)["meters"] == 3.0


def test_stereo_project_requires_a_known_baseline():
    with pytest.raises(ValueError):
        LensProject(LensKind.STEREO, ObservationGraph(3))
    with pytest.raises(ValueError):
        LensProject(LensKind.STEREO, ObservationGraph(3),
                    baseline_m=-1.0, baseline_pair=(0, 1))
