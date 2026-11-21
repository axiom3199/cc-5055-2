import numpy as np

from workbench import geometry as geo
from workbench.degeneracy import classify_motion

K = np.array([[700.0, 0, 320], [0, 700, 240], [0, 0, 1]])


def _project(R, t, X):
    return geo.project_points((R @ X.T).T + t, K)


def test_classify_pure_rotation():
    rng = np.random.default_rng(2)
    X = rng.uniform(-5, 5, (200, 3))
    X[:, 2] += 12.0
    R2 = geo.rodrigues(np.array([0.0, 0.05, 0.0]))   # pure yaw, no translation
    p1 = _project(np.eye(3), np.zeros(3), X)
    p2 = _project(R2, np.zeros(3), X)
    info = classify_motion(p1, p2, K)
    assert info["is_rotation_only"]
    assert info["parallax_deg"] < 0.3
    assert info["R_rel"] is not None
    # recovered relative rotation should match the true yaw
    assert geo.rotation_angle_deg(info["R_rel"], R2) < 1.0


def test_classify_translation():
    rng = np.random.default_rng(3)
    X = rng.uniform(-5, 5, (200, 3))
    X[:, 2] += 12.0
    t2 = np.array([0.8, 0.0, 0.0])                    # real baseline
    R2 = geo.rodrigues(np.array([0.0, 0.02, 0.0]))
    p1 = _project(np.eye(3), np.zeros(3), X)
    p2 = _project(R2, t2, X)
    info = classify_motion(p1, p2, K)
    assert not info["is_rotation_only"]
    assert info["parallax_deg"] > 1.0


def test_classify_near_degenerate_small_baseline():
    rng = np.random.default_rng(4)
    X = rng.uniform(-5, 5, (200, 3))
    X[:, 2] += 12.0
    t2 = np.array([0.01, 0.0, 0.0])                   # tiny baseline
    p1 = _project(np.eye(3), np.zeros(3), X)
    p2 = _project(np.eye(3), t2, X)
    info = classify_motion(p1, p2, K)
    assert info["is_degenerate"]
