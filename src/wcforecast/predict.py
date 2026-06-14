"""Turning goal-rate estimates into 1X2 probabilities, plus the calibration layer.

These are pure functions (no model/PyMC dependency) so they are trivial to test.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

# Validated calibration constants (temperature scaling + neutral-venue draw boost).
# See docs/FINDINGS.md: temperature helps out-of-sample; isotonic over-fits.
CALIB_TEMPERATURE = 1.08
CALIB_DRAW_BOOST = 0.03


def poisson_1x2(lam_home, lam_away, max_goals: int = 15, dixon_coles_rho=0.0):
    """Win/Draw/Loss probabilities by enumerating the Poisson score grid.

    ``lam_home``/``lam_away`` may be scalars or 1-D arrays (vectorised over posterior
    draws); the return shape matches. ``dixon_coles_rho`` applies the low-score
    correction (0 = independent Poisson). Each result is renormalised to sum to 1.
    """
    lh = np.atleast_1d(np.asarray(lam_home, dtype=float))
    la = np.atleast_1d(np.asarray(lam_away, dtype=float))
    scalar = np.ndim(lam_home) == 0 and np.ndim(lam_away) == 0
    k = np.arange(max_goals + 1)
    pmf_h = poisson.pmf(k[None, :], lh[:, None])          # (S, K) home goals
    pmf_a = poisson.pmf(k[None, :], la[:, None])          # (S, K) away goals
    joint = pmf_h[:, :, None] * pmf_a[:, None, :]         # (S, x=home, y=away)

    rho = np.atleast_1d(np.asarray(dixon_coles_rho, dtype=float))
    if np.any(rho != 0.0):
        joint[:, 0, 0] *= np.maximum(1 - lh * la * rho, 1e-9)
        joint[:, 0, 1] *= np.maximum(1 + lh * rho, 1e-9)
        joint[:, 1, 0] *= np.maximum(1 + la * rho, 1e-9)
        joint[:, 1, 1] *= np.maximum(1 - rho, 1e-9)

    x = k[:, None]
    y = k[None, :]
    home = (joint * (x > y)).sum(axis=(1, 2))
    draw = (joint * (x == y)).sum(axis=(1, 2))
    away = (joint * (x < y)).sum(axis=(1, 2))
    total = home + draw + away
    home, draw, away = home / total, draw / total, away / total
    if scalar:
        return float(home[0]), float(draw[0]), float(away[0])
    return home, draw, away


def calibrate(probs, temperature: float = CALIB_TEMPERATURE,
              draw_boost: float = CALIB_DRAW_BOOST) -> np.ndarray:
    """Temperature scaling + draw boost. Accepts a single ``(3,)`` vector or an
    ``(n, 3)`` batch; returns the calibrated array of the same shape."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    p = p ** (1.0 / temperature)
    p = p / p.sum(axis=-1, keepdims=True)
    p = p.copy()
    p[..., 1] += draw_boost
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=-1, keepdims=True)
