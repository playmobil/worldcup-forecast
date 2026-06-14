"""Hierarchical Bayesian Poisson model with partial pooling (PyMC).

Each team has latent attack/defence offsets around a structural prior; data-rich
teams are pulled by their results, data-poor teams shrink back to the prior. PyMC is
imported lazily so the rest of the package works without it.
"""
from __future__ import annotations

import os
import pickle
import platform
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .predict import poisson_1x2
from .teams import INDEX, N_TEAMS, TEAM_NAMES


def _ensure_arm64_compiler() -> None:
    """Force ``-arch arm64`` on Apple Silicon (PyTensor sometimes emits x86_64
    objects that then fail to ``dlopen`` in an arm64 interpreter)."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        flags = os.environ.get("PYTENSOR_FLAGS", "")
        if "arch arm64" not in flags:
            os.environ["PYTENSOR_FLAGS"] = (flags + ",gcc__cxxflags=-arch arm64").lstrip(",")


def _dixon_coles_logtau(hs, as_, lam_h, lam_a, rho):
    """log of the Dixon-Coles low-score correction (PyTensor, for the weighted path)."""
    import pytensor.tensor as pt

    m00 = ((hs == 0) & (as_ == 0)).astype(float)
    m01 = ((hs == 0) & (as_ == 1)).astype(float)
    m10 = ((hs == 1) & (as_ == 0)).astype(float)
    m11 = ((hs == 1) & (as_ == 1)).astype(float)
    tau = (1.0 + m00 * (-lam_h * lam_a * rho) + m01 * (lam_h * rho)
           + m10 * (lam_a * rho) + m11 * (-rho))
    return pt.log(pt.maximum(tau, 1e-9))


@dataclass
class PoissonModel:
    """A fitted model: the posterior plus convenience prediction methods."""

    idata: object                 # arviz.InferenceData
    team_names: list[str]

    def _posterior(self, thin: int = 1):
        post = self.idata.posterior
        n = len(self.team_names)
        atk = post["atk"].values.reshape(-1, n)[::thin]
        deff = post["def"].values.reshape(-1, n)[::thin]
        mu = post["mu"].values.reshape(-1)[::thin]
        hadv = post["home_adv"].values.reshape(-1)[::thin]
        rho = post["rho"].values.reshape(-1)[::thin] if "rho" in post else 0.0
        return atk, deff, mu, hadv, rho

    def match_probs(self, home: str, away: str, home_advantage: float = 0.0,
                    max_goals: int = 15, thin: int = 1) -> tuple[float, float, float]:
        """Posterior-predictive (P_home, P_draw, P_away).

        ``home_advantage`` is 1.0 only when ``home`` truly plays at home (a non-neutral
        venue); 0.0 at neutral venues — matching how the model was trained.
        """
        assert home_advantage in (0.0, 1.0, 0, 1), "home_advantage must be 0 or 1"
        atk, deff, mu, hadv, rho = self._posterior(thin)
        h, a = INDEX[home], INDEX[away]
        lam_h = np.exp(mu + atk[:, h] - deff[:, a] + hadv * float(home_advantage))
        lam_a = np.exp(mu + atk[:, a] - deff[:, h])
        ph, pd, pa = poisson_1x2(lam_h, lam_a, max_goals=max_goals, dixon_coles_rho=rho)
        return float(np.mean(ph)), float(np.mean(pd)), float(np.mean(pa))

    def strengths(self) -> dict[str, float]:
        """Expected goal difference vs an average team — a readable strength summary
        (do NOT use ``atk - def``: when ka≈kd it cancels)."""
        atk, deff, mu, *_ = self._posterior()
        a, d, m = atk.mean(0), deff.mean(0), float(mu.mean())
        gd = np.exp(m + a) - np.exp(m - d)
        return {t: float(gd[i]) for i, t in enumerate(self.team_names)}

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"idata": self.idata, "team_names": self.team_names}, f)

    @classmethod
    def load(cls, path) -> "PoissonModel":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["idata"], d["team_names"])


def fit(matches: pd.DataFrame, structural, weights=None, dixon_coles: bool = False,
        draws: int = 1000, tune: int = 1000, chains: int = 2,
        seed: int = 42, target_accept: float = 0.92, progressbar: bool = False) -> PoissonModel:
    """Fit the hierarchical Poisson model.

    Parameters
    ----------
    matches : DataFrame from :func:`wcforecast.data.load_matches` (has hi/ai/home_not_neutral).
    structural : structural prior vector (order = ``TEAM_NAMES``), drives the team priors.
    weights : optional per-match likelihood weights (e.g. recency weighting).
    dixon_coles : add the low-score correction (extra parameter ``rho``).
    """
    _ensure_arm64_compiler()
    import pymc as pm

    s = np.asarray(structural, dtype=float)
    hi = matches["hi"].to_numpy()
    ai = matches["ai"].to_numpy()
    hs = matches["home_score"].to_numpy().astype(int)
    as_ = matches["away_score"].to_numpy().astype(int)
    hn = matches["home_not_neutral"].to_numpy()
    w = None if weights is None else np.asarray(weights, dtype=float)

    with pm.Model():
        ka = pm.Normal("ka", 0.4, 0.4)
        kd = pm.Normal("kd", 0.4, 0.4)
        mu = pm.Normal("mu", 0.1, 0.3)
        home_adv = pm.HalfNormal("home_adv", 0.3)
        sig_a = pm.HalfNormal("sig_a", 0.3)
        sig_d = pm.HalfNormal("sig_d", 0.3)
        # Non-centred parameterisation (removes the funnel; far better mixing).
        atk_z = pm.Normal("atk_z", 0, 1, shape=N_TEAMS)
        def_z = pm.Normal("def_z", 0, 1, shape=N_TEAMS)
        atk = pm.Deterministic("atk", ka * s + atk_z * sig_a)
        deff = pm.Deterministic("def", kd * s + def_z * sig_d)

        lam_h = pm.math.exp(mu + atk[hi] - deff[ai] + home_adv * hn)
        lam_a = pm.math.exp(mu + atk[ai] - deff[hi])
        rho = pm.Normal("rho", 0.0, 0.1) if dixon_coles else None

        if w is None and not dixon_coles:
            pm.Poisson("hg", lam_h, observed=hs)
            pm.Poisson("ag", lam_a, observed=as_)
        else:
            lp = pm.logp(pm.Poisson.dist(lam_h), hs) + pm.logp(pm.Poisson.dist(lam_a), as_)
            if dixon_coles:
                lp = lp + _dixon_coles_logtau(hs, as_, lam_h, lam_a, rho)
            if w is not None:
                lp = w * lp
            pm.Potential("loglik", lp.sum())

        idata = pm.sample(draws, tune=tune, chains=chains, cores=1,
                          target_accept=target_accept, random_seed=seed,
                          progressbar=progressbar)
    return PoissonModel(idata, TEAM_NAMES)
