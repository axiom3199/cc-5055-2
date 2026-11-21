import numpy as np

from workbench import geometry as geo
from workbench.bundle import bundle_adjust

K = np.array([[700.0, 0, 320], [0, 700, 240], [0, 0, 1]])


def _scene(npts=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-4, 4, (npts, 3))
    X[:, 2] += 10.0
    return X


def test_bundle_adjust_reduces_reprojection():
    rng = np.random.default_rng(1)
    X = _scene()
    # three cameras with real baselines
    cams = {0: (np.eye(3), np.zeros(3))}
    for i, C in enumerate([np.array([1.0, 0, 0]), np.array([2.0, 0.3, 0])], start=1):
        cams[i] = (np.eye(3), -C.copy())
    # noisy observations
    obs = []
    for pid, x in enumerate(X):
        for f, (R, t) in cams.items():
            uv = geo.project_points((R @ x + t)[None], K)[0]
            uv += rng.normal(0, 0.5, 2)
            obs.append((pid, f, float(uv[0]), float(uv[1])))
    points = {i: x.copy() for i, x in enumerate(X)}
    # perturb cameras and points
    cams_p = {f: (R, t + rng.normal(0, 0.05, 3)) for f, (R, t) in cams.items()}
    cams_p[0] = cams[0]
    pts_p = {i: x + rng.normal(0, 0.1, 3) for i, x in points.items()}
    cams_o, pts_o, info = bundle_adjust(cams_p, pts_p, obs, K, max_nfev=200)
    # converges from the perturbed start down to the noise floor (~0.5px)
    assert info["cost_final"] < 0.05 * info["cost_init"]
    assert info["rms_px"] < 1.0
    assert info["optimality"] < 100.0      # actually converged, not stalled
    # gauge: frame 0 must be held fixed
    assert np.allclose(cams_o[0][1], cams[0][1])


def test_bundle_adjust_gauge_scale_is_fixed():
    X = _scene(40)
    cams = {0: (np.eye(3), np.zeros(3)),
            1: (np.eye(3), np.array([-1.0, 0, 0]))}
    obs = []
    for pid, x in enumerate(X):
        for f, (R, t) in cams.items():
            uv = geo.project_points((R @ x + t)[None], K)[0]
            obs.append((pid, f, float(uv[0]), float(uv[1])))
    points = {i: x.copy() for i, x in enumerate(X)}
    cams_o, _, _ = bundle_adjust(cams, points, obs, K, max_nfev=50)
    # the second camera's baseline length must be preserved (scale anchor)
    assert np.isclose(np.linalg.norm(cams_o[1][1]), 1.0, rtol=1e-6)
