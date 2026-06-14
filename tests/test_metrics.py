import numpy as np

from wcforecast import validate as V

UNIFORM = np.full((1, 3), 1 / 3)


def test_log_loss_uniform():
    assert abs(V.log_loss(UNIFORM, [0])[0] - 1.09861) < 1e-3


def test_brier_uniform():
    assert abs(V.brier(UNIFORM, [0])[0] - 0.66667) < 1e-3


def test_rps_ordered():
    # outcome = home win; cum (1/3, 2/3, 1) vs (1,1,1)
    assert abs(V.rps(UNIFORM, [0])[0] - 0.27778) < 1e-3
    # outcome = draw is "closer" on the ordinal axis
    assert abs(V.rps(UNIFORM, [1])[0] - 0.11111) < 1e-3


def test_perfect_forecast_is_zero():
    P = np.array([[1.0, 0.0, 0.0]])
    assert V.log_loss(P, [0])[0] < 1e-6
    assert V.brier(P, [0])[0] < 1e-12
    assert V.rps(P, [0])[0] < 1e-12


def test_confident_wrong_is_penalised():
    P = np.array([[0.98, 0.01, 0.01]])
    assert abs(V.log_loss(P, [2])[0] - 4.60517) < 1e-3
