"""Observation graph: layered storage, unified observation identity, undoable
edit transactions, and edit-log validation on reload."""
import numpy as np
import pytest

from workbench.obsgraph import (ObservationGraph, ObsId, TrackRecord,
                                StaleEditError, ReidentifyRequired,
                                UnknownObservationError, CorruptEditLogError)


def _graph():
    # track 1: frames 0..9; track 2: frames 5..14; track 3: frames 20..29
    g = ObservationGraph(30)
    for tid, f0 in ((1, 0), (2, 5), (3, 20)):
        frames = np.arange(f0, f0 + 10, dtype=np.int32)
        xy = np.stack([frames * 2.0, frames * 3.0], axis=1).astype(np.float32)
        g.tracks.add(TrackRecord(tid, frames, xy))
    return g


def test_unified_observation_identity_spans_layers():
    g = _graph()
    # one identity addresses both the track layer (pixel) and, after a
    # solution commit, the residual layer (evaluation)
    g.residuals.set(ObsId(1, 4), 0.31, outlier=False)
    rec = g.tracks.get(1)
    j = list(rec.frames).index(4)
    assert np.allclose(rec.xy[j], [8.0, 12.0])
    assert g.residuals.get(ObsId(1, 4)) == (0.31, False)


def test_strict_lookup_no_default_masking():
    g = _graph()
    with pytest.raises(UnknownObservationError):
        g.tracks.get(999)
    with pytest.raises(UnknownObservationError):
        g.residuals.get(ObsId(1, 4))
    with pytest.raises(UnknownObservationError):
        g.cameras.get(0)
    with pytest.raises(UnknownObservationError):
        g.residuals.stats()


def _split_graph():
    """A graph whose track 1 was split by deleting observations 1@4, 1@5."""
    g = _graph()
    edit = g.begin_edit()
    edit.delete_observations([ObsId(1, 4), ObsId(1, 5)])
    eid = edit.commit()
    return g, eid


def test_delete_observations_splits_track_and_retires_id():
    g, _ = _split_graph()
    assert 1 not in g.tracks          # original id retired, never reused
    frags = [g.tracks.get(i) for i in g.tracks.ids() if g.tracks.get(i).origin == 1]
    assert len(frags) == 2
    assert sorted(len(f.frames) for f in frags) == [4, 4]   # 0-3 and 6-9
    members = set()
    for f in frags:
        members |= f.members()
    assert members == {ObsId(1, f) for f in [0, 1, 2, 3, 6, 7, 8, 9]}


def test_undo_restores_split_track_by_observation_members():
    g, eid = _split_graph()
    g.undo(eid)
    rec = g.tracks.get(1)             # original id and members recovered
    assert list(rec.frames) == list(range(10))
    assert np.allclose(rec.xy[4], [8.0, 12.0])


def test_undo_requires_reidentification_when_members_changed():
    g, eid = _split_graph()
    # a second delete eats into one fragment: members no longer union cleanly
    frag_id = [i for i in g.tracks.ids() if g.tracks.get(i).origin == 1][0]
    f0 = int(g.tracks.get(frag_id).frames[0])
    edit2 = g.begin_edit()
    edit2.delete_observations([ObsId(frag_id, f0)])
    edit2.commit()
    with pytest.raises(ReidentifyRequired):
        g.undo(eid)


def test_nested_split_undo_restores_only_the_edited_lineage():
    g, eid_a = _split_graph()          # track 1 -> fragments F1, F2
    frags = [i for i in g.tracks.ids() if g.tracks.get(i).origin == 1]
    f1, f2 = sorted(frags)
    f1_frames = list(g.tracks.get(f1).frames)
    # edit B: delete one observation from fragment F1 (splits F1 again)
    edit_b = g.begin_edit()
    edit_b.delete_observations([ObsId(f1, f1_frames[2])])
    eid_b = edit_b.commit()
    # undoing B restores F1 exactly and leaves the sibling F2 untouched
    g.undo(eid_b)
    assert list(g.tracks.get(f1).frames) == f1_frames
    assert list(g.tracks.get(f2).frames) == list(range(6, 10))
    # undoing A now restores the original track 1
    g.undo(eid_a)
    assert list(g.tracks.get(1).frames) == list(range(10))


def test_out_of_order_undo_of_nested_split_requires_reidentification():
    g, eid_a = _split_graph()
    frag_id = [i for i in g.tracks.ids() if g.tracks.get(i).origin == 1][0]
    f0 = int(g.tracks.get(frag_id).frames[0])
    edit_b = g.begin_edit()
    edit_b.delete_observations([ObsId(frag_id, f0)])
    edit_b.commit()
    # undoing the OUTER split while the inner delete is still committed must
    # refuse: the live members do not identify the original constraint
    with pytest.raises(ReidentifyRequired):
        g.undo(eid_a)


def test_stale_commit_rejected_when_optimization_interleaves():
    g = _graph()
    g.anchors.upsert_distance_anchor((10, 11), 2.5)
    aid = next(iter(g.anchors.distance_anchors))
    # background optimization opens its commit at version V...
    bg = g.begin_edit()
    bg.stage_solution(_fake_solution())
    # ...and the user deletes the scale anchor at the same base version
    user = g.begin_edit()
    user.delete_scale_anchor(aid)
    user.commit()                      # lands first
    with pytest.raises(StaleEditError):
        bg.commit()                    # the optimization commit is rejected
    assert g.anchors.distance_anchors == {}
    # the other order is symmetric
    g2 = _graph()
    g2.anchors.upsert_distance_anchor((10, 11), 2.5)
    aid2 = next(iter(g2.anchors.distance_anchors))
    user2 = g2.begin_edit()
    user2.delete_scale_anchor(aid2)
    bg2 = g2.begin_edit()
    bg2.stage_solution(_fake_solution())
    bg2.commit()
    with pytest.raises(StaleEditError):
        user2.commit()
    assert len(g2.anchors.distance_anchors) == 1


def test_reopened_project_contains_only_legal_commits(tmp_path):
    g = _graph()
    g.anchors.upsert_distance_anchor((10, 11), 2.5)
    aid = next(iter(g.anchors.distance_anchors))
    bg = g.begin_edit()
    bg.stage_solution(_fake_solution())
    user = g.begin_edit()
    user.delete_scale_anchor(aid)
    user.commit()
    with pytest.raises(StaleEditError):
        bg.commit()
    run_dir = str(tmp_path / "run")
    import os
    os.makedirs(run_dir)
    g.save(run_dir)
    g2 = ObservationGraph.load(run_dir)
    legal = [e for e in g2.edits() if not e.get("undo_of")]
    assert len(legal) == 1
    assert legal[0]["ops"][0]["op"] == "delete_scale_anchor"
    assert g2.version == g.version


def test_corrupt_log_with_racing_commits_is_rejected(tmp_path):
    import os, json
    g = _graph()
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    e1 = g.begin_edit(); e1.delete_track(1); e1.commit()
    g.save(run_dir)
    # forge a second commit on the same base version (a crashed racer)
    with open(os.path.join(run_dir, "edits.jsonl")) as fh:
        lines = fh.readlines()
    rec = json.loads(lines[0])
    rec["id"] = 1
    rec["ops"] = [{"op": "delete_observations", "track_id": 2, "frames": [5]}]
    with open(os.path.join(run_dir, "edits.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    with pytest.raises(CorruptEditLogError):
        ObservationGraph.load(run_dir)


def test_graph_roundtrip_preserves_tracks_and_version(tmp_path):
    import os
    g = _graph()
    e = g.begin_edit(); e.delete_observations([ObsId(2, 7)]); e.commit()
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    g.save(run_dir)
    g2 = ObservationGraph.load(run_dir)
    assert g2.version == g.version
    assert g2.tracks.ids() == g.tracks.ids()
    for tid in g.tracks.ids():
        a, b = g.tracks.get(tid), g2.tracks.get(tid)
        assert a.origin == b.origin
        assert np.array_equal(a.frames, b.frames)
        assert np.allclose(a.xy, b.xy)


def test_solver_view_is_a_snapshot_the_solver_cannot_mutate():
    g = _graph()
    view = g.view()
    tracks = view.tracks()
    tracks[0]["xy"][0] = [-999.0, -999.0]     # solver-side mutation
    rec = g.tracks.get(1)
    assert not np.allclose(rec.xy[0], [-999.0, -999.0])
    assert view.version == g.version


def _fake_solution():
    from workbench.sfm import Solution
    sol = Solution()
    sol.cameras = {0: (np.eye(3), np.zeros(3)),
                   1: (np.eye(3), np.array([-1.0, 0, 0]))}
    sol.observations = [(1, 0, 0.0, 0.0), (1, 1, 2.0, 3.0)]
    sol.obs_residual = [0.1, 0.2]
    sol.obs_outlier = [False, True]
    return sol


def test_commit_solution_writes_camera_and_residual_layers():
    g = _graph()
    g.commit_solution(_fake_solution())
    assert g.cameras.frames() == [0, 1]
    assert g.residuals.get(ObsId(1, 1)) == (0.2, True)
    assert g.residuals.stats()["n_outliers"] == 1


def test_staged_anchor_registration_is_undoable():
    g = _graph()
    edit = g.begin_edit()
    edit.stage_anchors(dist_pair=(10, 11), dist_meters=2.5, ground_ids=[10, 11])
    eid = edit.commit()
    assert len(g.anchors.distance_anchors) == 1
    assert g.anchors.ground_ids == {10, 11}
    g.undo(eid)
    assert g.anchors.distance_anchors == {}
    assert g.anchors.ground_ids == set()
    # re-registering with a different distance, then undoing the SECOND
    # registration, must restore the first value (not the empty state)
    e1 = g.begin_edit(); e1.stage_anchors(dist_pair=(10, 11), dist_meters=2.5)
    e1.commit()
    e2 = g.begin_edit(); e2.stage_anchors(dist_pair=(10, 11), dist_meters=9.0)
    e2id = e2.commit()
    g.undo(e2id)
    anc = next(iter(g.anchors.distance_anchors.values()))
    assert anc["meters"] == 2.5
