import numpy as np

from workbench import geometry as geo


def test_rodrigues_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        rvec = rng.normal(0, 0.5, 3)
        R = geo.rodrigues(rvec)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(geo.rvec_from_matrix(R), rvec, atol=1e-8)


def test_camera_center_and_projection():
    R = geo.rodrigues(np.array([0.1, 0.2, 0.3]))
    C = np.array([1.0, 2.0, 3.0])
    t = -R @ C
    assert np.allclose(geo.camera_center(R, t), C)
    # a point at the camera center projects to nothing meaningful; a point
    # straight ahead projects to the principal point
    K = np.array([[700.0, 0, 320], [0, 700, 240], [0, 0, 1]])
    X = C + R.T @ np.array([0, 0, 5.0])   # 5 units in front of the camera
    uv = geo.project_world(R, t, X[None], K)[0]
    assert np.allclose(uv, [320, 240], atol=1e-6)


def test_triangulate_multiview_recovers_point():
    K = np.array([[700.0, 0, 320], [0, 700, 240], [0, 0, 1]])
    X = np.array([0.5, -0.3, 8.0])
    cams = []
    obs = []
    for C in (np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.5, 0.5, 0])):
        R = np.eye(3)
        t = -R @ C
        cams.append((R, t))
        obs.append(geo.project_points((R @ X + t)[None], K)[0])
    Xh = geo.triangulate_multiview(cams, np.array(obs), K)
    assert np.allclose(Xh, X, atol=1e-6)


def test_parallax_angle():
    C1 = np.array([0.0, 0, 0])
    C2 = np.array([1.0, 0, 0])
    X = np.array([0.0, 0, 10.0])
    ang = geo.parallax_angle_deg(C1, C2, X)
    assert 5.0 < ang < 6.0           # ~atan(1/10) in deg
    # pure rotation: same center -> zero parallax
    assert geo.parallax_angle_deg(C1, C1, X) == 0.0


def test_umeyama_exact_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(0, 1, (50, 3))
    s_true = 3.7
    R_true = geo.rodrigues(np.array([0.3, -0.2, 0.5]))
    t_true = np.array([1.0, -2.0, 0.5])
    dst = s_true * (src @ R_true.T) + t_true
    s, R, t = geo.umeyama(src, dst, with_scale=True)
    assert np.isclose(s, s_true, rtol=1e-6)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-6)
