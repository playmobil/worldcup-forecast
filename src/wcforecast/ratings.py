"""Team-strength estimation: Elo, the structural (Klement) prior, and weighting.

All quantities are computed *as of a cutoff date* (only matches strictly before the
cutoff are used) so they can drive leakage-free forecasts and walk-forward backtests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .teams import TEAMS, TEAM_NAMES

_CONTINENTAL = {"UEFA Euro", "Copa América", "Copa America", "African Cup of Nations",
                "AFC Asian Cup", "Gold Cup", "CONCACAF Championship"}


def tournament_weight(name: str) -> float:
    """Importance weight for a competition (drives Elo K and recency weighting)."""
    n = str(name)
    if n == "FIFA World Cup":
        return 1.0
    if "World Cup" in n and "qualification" not in n:
        return 0.9
    if n in _CONTINENTAL:
        return 0.85
    if "Nations League" in n:
        return 0.7
    if "qualification" in n:
        return 0.6
    if n == "Friendly":
        return 0.3
    return 0.5


def elo_ratings(results: pd.DataFrame, cutoff: str,
                k: float = 40.0, home_advantage: float = 65.0,
                init: float = 1500.0) -> dict[str, float]:
    """World-Football-Elo ratings as of ``cutoff`` (K=40 validated optimal).

    Iterates every international match before the cutoff (all teams, so ratings are
    well-calibrated) and returns ratings for the 48 World Cup teams.
    """
    df = results[results["date"] < cutoff].sort_values("date")
    elo: dict[str, float] = {}
    for r in df.itertuples(index=False):
        try:
            hs, as_ = int(r.home_score), int(r.away_score)
        except (ValueError, TypeError):
            continue
        h, a = r.home_team, r.away_team
        rh, ra = elo.get(h, init), elo.get(a, init)
        ha = 0.0 if r.neutral else home_advantage
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / 400.0))
        score_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        gd = abs(hs - as_)
        gmult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
        delta = k * tournament_weight(r.tournament) * gmult * (score_h - exp_h)
        elo[h], elo[a] = rh + delta, ra - delta
    return {t: elo.get(t, init) for t in TEAM_NAMES}


def recency_weights(matches: pd.DataFrame, cutoff: str,
                    half_life_days: float = 1095.0) -> np.ndarray:
    """Per-match likelihood weight = tournament importance × exp(-age / half-life)."""
    cut = pd.to_datetime(cutoff)
    days = (cut - pd.to_datetime(matches["date"])).dt.days.clip(lower=0).to_numpy()
    tw = matches["tournament"].map(tournament_weight).fillna(0.5).to_numpy()
    return tw * np.exp(-days / half_life_days)


def _football_culture(results: pd.DataFrame, cutoff: str, elo: dict[str, float]) -> np.ndarray:
    """Culture proxy in [0,1]: blend of World-Cup appearances and long-run Elo."""
    wc = results[(results["tournament"] == "FIFA World Cup") & (results["date"] < cutoff)].copy()
    wc["year"] = wc["date"].str[:4]
    appear = np.array([len(set(wc.loc[(wc.home_team == t) | (wc.away_team == t), "year"]))
                       for t in TEAM_NAMES], dtype=float)
    appear = appear / appear.max() if appear.max() > 0 else appear
    e = np.array([elo[t] for t in TEAM_NAMES], dtype=float)
    e = (e - e.min()) / (e.max() - e.min()) if e.max() > e.min() else np.zeros_like(e)
    return 0.5 * appear + 0.5 * e


def _z(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def structural_index(cutoff: str, results: pd.DataFrame,
                     fifa: dict[str, float] | None = None,
                     squad: dict[str, float] | None = None,
                     elo: dict[str, float] | None = None) -> np.ndarray:
    """Standardised structural prior s_i for every team (order = ``TEAM_NAMES``).

    Current-strength **anchor**: the standardised mean of whichever of FIFA points
    and log squad value are supplied (the leakage-free 2026 inputs); falls back to
    leakage-safe Elo when neither is given (e.g. historical backtests). On top of the
    anchor sit the Klement slow variables (GDP, population, temperature, host) plus a
    football-culture term and a population×culture "talent pool" interaction.
    """
    if elo is None:
        elo = elo_ratings(results, cutoff)
    culture = _football_culture(results, cutoff, elo)

    signals = []
    if fifa and all(t in fifa for t in TEAM_NAMES):
        signals.append(_z([fifa[t] for t in TEAM_NAMES]))
    if squad and all(t in squad for t in TEAM_NAMES):
        signals.append(_z(np.log10([squad[t] for t in TEAM_NAMES])))
    anchor = np.mean(signals, axis=0) if signals else _z([elo[t] for t in TEAM_NAMES])

    gdp = np.array([TEAMS[t].gdp_per_capita for t in TEAM_NAMES], dtype=float)
    pop = np.array([TEAMS[t].population_m for t in TEAM_NAMES], dtype=float)
    temp = np.array([TEAMS[t].avg_temp_c for t in TEAM_NAMES], dtype=float)
    host = np.array([1.0 if TEAMS[t].host else 0.0 for t in TEAM_NAMES])
    lg = np.log10(gdp)
    talent = np.log10(pop) * culture

    s = (anchor
         + 0.15 * (lg - 4.3) - 0.05 * np.maximum(0, lg - 4.8)   # GDP, inverted-U
         + 0.10 * (np.log10(pop) - 1.3)                          # talent pool size
         - 0.20 * ((temp - 13) / 12) ** 2                        # climate inverted-U
         + 0.35 * host                                           # host boost
         + 0.25 * culture                                        # football culture
         + 0.15 * (talent - talent.mean()))                      # population × culture
    return _z(s)
