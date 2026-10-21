import numpy as np
from vendi import canon_quat, median_gamma, vendi


def test_canon_quat():
    q = np.array([[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, 0.5]])
    assert (canon_quat(q)[:, 0] >= 0).all()
    assert np.allclose(np.abs(canon_quat(q)), np.abs(q))


def test_vendi_extremes():
    rng = np.random.default_rng(0)
    same = np.ones((10, 3))
    assert np.isclose(vendi(same, 1.0, 5, 3, rng)[0], 1.0)
    far = np.eye(10) * 100.0
    assert np.isclose(vendi(far, 1.0, 5, 3, rng)[0], 5.0)
    assert np.isnan(vendi(far, 1.0, 20, 3, rng)[0])
    assert median_gamma(far) > 0
