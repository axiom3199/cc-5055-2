"""UI-level camera solving: the lens-tracking window keeps its solve
controls, the kernel drives them and reports failures back, and after
specifying the ground the viewer still shows the virtual marker and
reprojection residuals - now served by the refactored modules."""
import io
import json
import os
import shutil
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WORKBENCH_SKIP_UI"), reason="UI tests disabled")


@pytest.fixture(scope="module")
def client(small_run, tmp_path_factory):
    runs_root = tmp_path_factory.mktemp("ui_runs")
    os.environ["WORKBENCH_RUNS"] = str(runs_root)
    from workbench import app as app_mod
    app_mod.app.config["TESTING"] = True
    # a solved run, copied so the tests can mutate it freely
    shutil.copytree(small_run, os.path.join(str(runs_root), "ui_small"))
    return app_mod.app.test_client()


def _poll_task(client, name, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/runs/{name}/task").get_json()
        if j["task"]["done"]:
            return j["task"]
        time.sleep(0.5)
    raise TimeoutError(f"task for {name} did not finish")


def test_solve_control_success(client):
    """The solve control in the run page drives the solver kernel to
    completion; the task status reports success."""
    r = client.post("/runs/ui_small/solve", follow_redirects=False)
    assert r.status_code == 302
    task = _poll_task(client, "ui_small")
    assert task["error"] is None
    j = client.get("/runs/ui_small/task").get_json()
    assert j["status"]["status"].startswith("solved")


def test_solve_failure_feeds_back_to_the_solve_task(client):
    """A kernel refusal (footage too short to initialize) must surface in the
    solve task the page polls - not vanish into a log file."""
    client.post("/runs/new", data={"name": "ui_fail", "n_frames": 4,
                                   "profile": "survey", "seed": 7,
                                   "outlier_ratio": 0.0, "strategy": "delayed"})
    client.post("/runs/ui_fail/run_all")
    task = _poll_task(client, "ui_fail")
    assert task["error"] is not None
    assert "no initialization pair" in task["error"]


def test_ground_then_viewer_shows_marker_and_residuals(client):
    """Specify the ground, place a virtual object: the per-frame viewer still
    composites the solved geometry (map points keyed through the residual
    layer) and the marker."""
    r = client.post("/runs/ui_small/align", data={"mode": "auto"})
    assert r.status_code == 302
    client.post("/runs/ui_small/marker",
                data={"cx": 1.0, "cy": 0.5, "size": 0.5})
    img = client.get("/runs/ui_small/frame/0")
    assert img.status_code == 200
    assert img.mimetype == "image/png"
    from PIL import Image
    arr = np.asarray(Image.open(io.BytesIO(img.data)).convert("RGB")).astype(int)
    # map points from the residual layer (green) and the marker (orange)
    green = (np.abs(arr[..., 0] - 60) < 40) & (arr[..., 1] > 200) & (arr[..., 2] < 170)
    orange = (arr[..., 0] > 200) & (np.abs(arr[..., 1] - 80) < 60) & (arr[..., 2] < 90)
    assert green.sum() > 50, "no map-point overlay pixels found"
    assert orange.sum() > 10, "no virtual-marker pixels found"
    info = client.get("/runs/ui_small/frame_info/0").get_json()
    assert info["solved"] and info["method"]


def test_landing_export_suspend_and_baseline_fallback(client):
    """Monocular: deleting the only length reference suspends the metric
    export (preview unaffected). Stereo: the known baseline keeps it alive.
    Undoing the deletion restores the monocular export."""
    r = client.get("/runs/ui_small/export/landing?lens=monocular")
    assert r.status_code == 200
    out = r.get_json()
    assert out["units"] == "m" and out["points"]

    edits = client.get("/runs/ui_small/edits").get_json()
    aid = edits["anchors"]["distance_anchors"][0]["id"]
    r = client.post("/runs/ui_small/edits/delete_scale_anchor",
                    data={"anchor_id": aid})
    assert r.status_code == 200
    eid = r.get_json()["edit"]

    r = client.get("/runs/ui_small/export/landing?lens=monocular")
    assert r.status_code == 409
    assert r.get_json()["error"] == "metric_suspended"
    # the preview is unaffected by the suspension
    assert client.get("/runs/ui_small/frame/1").status_code == 200
    # a stereo project with a known baseline still exports metric points
    r = client.get("/runs/ui_small/export/landing"
                   "?lens=stereo&baseline_m=0.12&baseline_pair=0,1")
    assert r.status_code == 200
    assert r.get_json()["scale_evidence"] == "stereo_baseline"

    r = client.post(f"/runs/ui_small/edits/{eid}/undo")
    assert r.status_code == 200
    r = client.get("/runs/ui_small/export/landing?lens=monocular")
    assert r.status_code == 200


def test_observation_delete_and_undo_via_ui(client):
    """Observation deletion goes through the explicit transaction interface;
    undo restores the track by observation members."""
    from workbench.obsgraph import ObservationGraph
    run_dir = os.path.join(os.environ["WORKBENCH_RUNS"], "ui_small")
    g0 = ObservationGraph.load(run_dir)
    tid = g0.tracks.ids()[5]
    n_frames_of_track = len(g0.tracks.get(tid).frames)
    v0 = g0.version

    r = client.post("/runs/ui_small/edits/delete_track",
                    data={"track_id": tid})
    assert r.status_code == 200
    eid = r.get_json()["edit"]
    g1 = ObservationGraph.load(run_dir)
    assert g1.version == v0 + 1
    assert tid not in g1.tracks

    r = client.post(f"/runs/ui_small/edits/{eid}/undo")
    assert r.status_code == 200
    g2 = ObservationGraph.load(run_dir)
    assert len(g2.tracks.get(tid).frames) == n_frames_of_track


def test_index_lists_runs(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"ui_small" in r.data
