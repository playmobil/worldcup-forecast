"""worldcup-forecast — a structural + Bayesian forecaster for the FIFA World Cup.

Quick start
-----------
>>> from wcforecast import data, ratings, model, simulate
>>> results = data.load_results()
>>> s = ratings.structural_index("2026-06-11", results,
...                              fifa=data.load_fifa_snapshot(),
...                              squad=data.load_squad_values())
>>> matches = data.load_matches(start="2006-01-01", cutoff="2026-06-11")
>>> m = model.fit(matches, s, weights=ratings.recency_weights(matches, "2026-06-11"),
...               dixon_coles=True)
>>> simulate.champion_probabilities(m, s, n_sims=20000).head()
"""
from __future__ import annotations

from . import data, markets, model, predict, ratings, simulate, validate
from .model import PoissonModel, fit
from .predict import calibrate, poisson_1x2
from .ratings import elo_ratings, recency_weights, structural_index
from .simulate import champion_probabilities, monte_carlo
from .teams import GROUPS, TEAM_NAMES, TEAMS

__version__ = "0.1.0"
__all__ = [
    "data", "ratings", "model", "predict", "simulate", "validate", "markets",
    "fit", "PoissonModel", "poisson_1x2", "calibrate",
    "structural_index", "elo_ratings", "recency_weights",
    "champion_probabilities", "monte_carlo",
    "TEAMS", "GROUPS", "TEAM_NAMES", "__version__",
]
