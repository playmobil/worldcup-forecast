"""Data loading: match results, FIFA rankings, squad values, World Bank.

Design rules:
  * **Leakage gate** — :func:`load_matches` takes an exclusive ``cutoff`` so a model
    is only ever trained on matches strictly before the date it predicts.
  * **Reproducibility** — large public datasets are auto-downloaded into ``data/raw``;
    small frozen 2026 snapshots live in ``data/snapshots`` (committed).
  * Market data lives in :mod:`wcforecast.markets`, never here (it is benchmark-only).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .teams import TEAM_NAMES, INDEX

DATA_DIR = Path(os.environ.get("WCFORECAST_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
SNAPSHOTS = DATA_DIR / "snapshots"
RAW = DATA_DIR / "raw"
CACHE = DATA_DIR / "cache"

MARTJ42_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FIFA_HISTORY_URL = ("https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/"
                    "master/ranking_fifa_historical.csv")
WORLD_BANK_API = "https://api.worldbank.org/v2"

# External datasets spell some nations differently from our canonical team names.
_FIFA_SNAPSHOT_NAMES = {"Korea Republic": "South Korea", "USA": "United States",
                        "Czechia": "Czech Republic", "Côte d'Ivoire": "Ivory Coast",
                        "Congo DR": "DR Congo", "Cabo Verde": "Cape Verde",
                        "Türkiye": "Turkey", "IR Iran": "Iran"}
_FIFA_HISTORY_NAMES = {"USA": "United States", "Korea Republic": "South Korea",
                       "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran",
                       "Congo DR": "DR Congo", "Cape Verde Islands": "Cape Verde",
                       "Czechia": "Czech Republic", "Türkiye": "Turkey"}
_SQUAD_NAMES = {"USA": "United States", "Korea, South": "South Korea",
                "Czechia": "Czech Republic", "Cote d'Ivoire": "Ivory Coast",
                "Côte d'Ivoire": "Ivory Coast", "Turkiye": "Turkey",
                "Democratic Republic of the Congo": "DR Congo",
                "Bosnia-Herzegovina": "Bosnia and Herzegovina", "Curacao": "Curaçao"}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


# --------------------------------------------------------------------------- #
# Match results (martj42)                                                      #
# --------------------------------------------------------------------------- #
def load_results(refresh: bool = False) -> pd.DataFrame:
    """All played international matches (1872–present). Auto-downloads once."""
    path = RAW / "results.csv"
    if refresh or not path.exists():
        _download(MARTJ42_URL, path)
    return pd.read_csv(path).dropna(subset=["home_score", "away_score"])


def load_matches(start: str = "2006-01-01", cutoff: str | None = None,
                 teams: list[str] | None = None) -> pd.DataFrame:
    """Training set: played matches in ``[start, cutoff)`` with both teams in ``teams``.

    ``cutoff`` is the leakage gate (strict ``<``). Returns the frame with helper
    columns ``hi``/``ai`` (team indices) and ``home_not_neutral`` (1.0 for a true
    home side, 0.0 at neutral venues).
    """
    teams = teams or TEAM_NAMES
    tset = set(teams)
    df = load_results()
    df = df[df["date"] >= start]
    if cutoff is not None:
        df = df[df["date"] < cutoff]
    df = df[df["home_team"].isin(tset) & df["away_team"].isin(tset)].copy()
    idx = {t: i for i, t in enumerate(teams)}
    df["hi"] = df["home_team"].map(idx)
    df["ai"] = df["away_team"].map(idx)
    df["home_not_neutral"] = (~df["neutral"]).astype(float)
    return df


# --------------------------------------------------------------------------- #
# FIFA ranking (frozen 2026 snapshot + downloadable monthly history)          #
# --------------------------------------------------------------------------- #
def load_fifa_snapshot(path: Path | None = None) -> dict[str, float]:
    """Frozen pre-2026-tournament FIFA points, keyed by canonical team name."""
    path = path or SNAPSHOTS / "fifa_ranking_2026-06-11.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["team"] = df["team"].map(lambda t: _FIFA_SNAPSHOT_NAMES.get(str(t).strip(), str(t).strip()))
    return {r.team: float(r.points) for r in df.itertuples() if r.team in INDEX}


def load_fifa_history(refresh: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Monthly FIFA points 1992–2024 as ``{team: (sorted_dates, points)}`` for
    leakage-safe lookups via :func:`fifa_at`. Spliced with the 2026 snapshot."""
    path = RAW / "fifa_ranking_history.csv"
    if refresh or not path.exists():
        _download(FIFA_HISTORY_URL, path)
    f = pd.read_csv(path)
    f["m"] = f["team"].map(lambda t: _FIFA_HISTORY_NAMES.get(str(t).strip(), str(t).strip()))
    f = f[["m", "date", "total_points"]].rename(columns={"total_points": "p"})
    snap = load_fifa_snapshot()
    f = pd.concat([f, pd.DataFrame({"m": list(snap), "date": "2026-06-11", "p": list(snap.values())})])
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t, g in f[f["m"].isin(INDEX)].sort_values("date").groupby("m"):
        out[t] = (g["date"].to_numpy(), g["p"].to_numpy(dtype=float))
    return out


def fifa_at(history: dict, team: str, date: str) -> float:
    """Most recent FIFA points for ``team`` on or before ``date`` (NaN if unknown)."""
    if team not in history:
        return float("nan")
    dates, pts = history[team]
    i = int(np.searchsorted(dates, date, side="right")) - 1
    return float(pts[i]) if i >= 0 else float("nan")


# --------------------------------------------------------------------------- #
# Squad market values (frozen 2026 snapshot; no history)                      #
# --------------------------------------------------------------------------- #
def load_squad_values(path: Path | None = None) -> dict[str, float]:
    """National-team squad market value in EUR, keyed by canonical team name."""
    path = path or SNAPSHOTS / "squad_values_2026.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["m"] = df["country"].map(lambda t: _SQUAD_NAMES.get(str(t).strip(), str(t).strip()))
    return {r.m: float(r.market_value_eur) for r in df.itertuples()
            if r.m in INDEX and str(r.market_value_eur).replace(".", "").isdigit()}


# --------------------------------------------------------------------------- #
# World Bank (optional refinement of GDP / population)                        #
# --------------------------------------------------------------------------- #
def load_world_bank(year: int, indicator: str = "NY.GDP.PCAP.CD",
                    iso3: dict[str, str] | None = None) -> dict[str, float]:
    """Fetch a World Bank indicator (e.g. GDP per capita) for the given year.

    Optional; the structural index falls back to the values baked into
    :mod:`wcforecast.teams` when this is not used.
    """
    if not iso3:
        raise ValueError("provide an iso3 mapping {team: ISO3}")
    codes = ";".join(sorted(set(iso3.values())))
    r = requests.get(f"{WORLD_BANK_API}/country/{codes}/indicator/{indicator}",
                     params={"format": "json", "date": f"{year - 4}:{year}", "per_page": "20000"},
                     timeout=30)
    r.raise_for_status()
    rows = r.json()[1] or []
    best: dict[str, tuple[int, float]] = {}
    for rec in rows:
        v, c, d = rec.get("value"), rec.get("countryiso3code"), rec.get("date")
        if v is None:
            continue
        if c not in best or int(d) > best[c][0]:
            best[c] = (int(d), float(v))
    inv = {code: team for team, code in iso3.items()}
    return {inv[c]: val for c, (_, val) in best.items() if c in inv}
