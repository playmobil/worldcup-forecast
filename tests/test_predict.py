import numpy as np

from wcforecast.predict import calibrate, poisson_1x2


def test_equal_lambdas_are_symmetric():
    h, d, a = poisson_1x2(1.3, 1.3)
    assert abs(h - a) < 1e-9
    assert abs(h + d + a - 1.0) < 1e-9


def test_stronger_team_more_likely_to_win():
    h_even, _, _ = poisson_1x2(1.3, 1.3)
    h_fav, _, a_fav = poisson_1x2(2.0, 0.8)
    assert h_fav > a_fav
    assert h_fav > h_even


def test_vectorised_matches_scalar():
    H, D, A = poisson_1x2(np.array([1.3, 2.0]), np.array([1.3, 0.8]))
    assert H.shape == (2,)
    assert abs(H[0] - poisson_1x2(1.3, 1.3)[0]) < 1e-9
    assert abs(H[1] - poisson_1x2(2.0, 0.8)[0]) < 1e-9


def test_dixon_coles_raises_draw_probability():
    _, d0, _ = poisson_1x2(1.3, 1.3, dixon_coles_rho=0.0)
    _, d1, _ = poisson_1x2(1.3, 1.3, dixon_coles_rho=-0.1)
    assert d1 > d0


def test_calibrate_normalises_and_boosts_draw():
    p = calibrate([0.7, 0.2, 0.1])
    assert abs(p.sum() - 1.0) < 1e-9
    boosted = calibrate([0.7, 0.2, 0.1], temperature=1.0, draw_boost=0.05)
    assert boosted[1] > 0.2
