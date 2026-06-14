"""Out-of-sample validation: leakage-safe features, proper scoring, significance.

This is the project's measuring instrument. Every candidate improvement is judged on a
**locked test window** (default 2024-01 → 2026-06) with paired-bootstrap significance —
never on a handful of World Cup matches. See ``docs/FINDINGS.md`` for results.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .data import fifa_at, load_fifa_history, load_results
from .ratings import tournament_weight
from .teams import CONFEDERATION, INDEX

DEV_END = "2024-01-01"
TEST_END = "2026-06-11"
FEATURE_START = "2010-01-01"


# --------------------------- metrics --------------------------- #
def _onehot(y):
    e = np.zeros((len(y), 3))
    e[np.arange(len(y)), np.asarray(y)] = 1.0
    return e


def log_loss(probs, y):
    P = np.asarray(probs, dtype=float)
    return -np.log(np.clip(P[np.arange(len(y)), np.asarray(y)], 1e-12, 1.0))


def rps(probs, y):
    cp = np.cumsum(np.asarray(probs, dtype=float), axis=1)
    ce = np.cumsum(_onehot(y), axis=1)
    return 0.5 * ((cp[:, 0] - ce[:, 0]) ** 2 + (cp[:, 1] - ce[:, 1]) ** 2)


def brier(probs, y):
    return ((np.asarray(probs, dtype=float) - _onehot(y)) ** 2).sum(axis=1)


def summary(probs, y) -> dict:
    return {"log_loss": float(log_loss(probs, y).mean()),
            "rps": float(rps(probs, y).mean()),
            "brier": float(brier(probs, y).mean()),
            "n": int(len(y))}


def paired_bootstrap(probs_a, probs_b, y, n: int = 10000, seed: int = 0):
    """95% CI of mean(log_loss(a) − log_loss(b)). Negative ⇒ ``a`` is better."""
    d = log_loss(probs_a, y) - log_loss(probs_b, y)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def reliability(probs, y, bins: int = 6):
    """Reliability table: (lo, hi, mean predicted prob, empirical frequency, count)."""
    pr = np.asarray(probs, dtype=float).ravel()
    hit = _onehot(y).ravel()
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (pr >= lo) & (pr < hi if i < bins - 1 else pr <= hi)
        if m.sum() >= 10:
            out.append((lo, hi, float(pr[m].mean()), float(hit[m].mean()), int(m.sum())))
    return out


# --------------------------- features --------------------------- #
def build_features(start: str = FEATURE_START, k: float = 40.0, hfa: float = 65.0) -> pd.DataFrame:
    """Leakage-safe per-match features for every both-in-48 match since ``start``:
    incremental Elo, FIFA (historical lookup), recent form, venue and confederation."""
    results = load_results().sort_values("date")
    hist = load_fifa_history()
    elo: dict[str, float] = {}
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    rows = []
    for r in results.itertuples(index=False):
        try:
            hs, as_ = int(r.home_score), int(r.away_score)
        except (ValueError, TypeError):
            continue
        h, a = r.home_team, r.away_team
        rh, ra = elo.get(h, 1500.0), elo.get(a, 1500.0)
        if h in INDEX and a in INDEX and r.date >= start:
            rows.append({
                "date": r.date, "home": h, "away": a,
                "hnn": 0.0 if r.neutral else 1.0,
                "y": 0 if hs > as_ else (1 if hs == as_ else 2),
                "elo_diff": rh - ra, "elo_sum": rh + ra - 3000.0,
                "fifa_h": fifa_at(hist, h, r.date), "fifa_a": fifa_at(hist, a, r.date),
                "form_diff": (np.mean(form[h]) if form[h] else 0.0)
                             - (np.mean(form[a]) if form[a] else 0.0),
                "cross_confed": 0.0 if CONFEDERATION.get(h) == CONFEDERATION.get(a) else 1.0,
            })
        ha = 0.0 if r.neutral else hfa
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / 400.0))
        sh = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        gd = abs(hs - as_)
        gmult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
        delta = k * tournament_weight(r.tournament) * gmult * (sh - exp_h)
        elo[h], elo[a] = rh + delta, ra - delta
        form[h].append(hs - as_)
        form[a].append(as_ - hs)
    df = pd.DataFrame(rows)
    lowf = np.nanpercentile(np.r_[df["fifa_h"], df["fifa_a"]], 5)
    df["fifa_h"] = df["fifa_h"].fillna(lowf)
    df["fifa_a"] = df["fifa_a"].fillna(lowf)
    df["fifa_diff"] = df["fifa_h"] - df["fifa_a"]
    df["fifa_sum"] = df["fifa_h"] + df["fifa_a"] - 3000.0
    return df


def _logit(dev, test, cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(dev[cols].to_numpy(), dev["y"].to_numpy())
    P = np.zeros((len(test), 3))
    P[:, m.named_steps["logisticregression"].classes_] = m.predict_proba(test[cols].to_numpy())
    return P


def scorecard(dev_end: str = DEV_END, test_end: str = TEST_END) -> pd.DataFrame:
    """Fit simple strength baselines on the dev window and score them on the locked
    test window, with paired-bootstrap significance vs the Elo baseline."""
    feats = build_features()
    dev = feats[feats["date"] < dev_end]
    test = feats[(feats["date"] >= dev_end) & (feats["date"] < test_end)]
    y = test["y"].to_numpy()

    preds = {
        "uniform": np.full((len(test), 3), 1 / 3),
        "elo": _logit(dev, test, ["hnn", "elo_diff", "elo_sum"]),
        "fifa": _logit(dev, test, ["hnn", "fifa_diff", "fifa_sum"]),
        "elo+fifa": _logit(dev, test, ["hnn", "elo_diff", "elo_sum", "fifa_diff", "fifa_sum"]),
    }
    rows = []
    for name, P in preds.items():
        s = summary(P, y)
        if name == "elo":
            ci = ""
        else:
            md, lo, hi = paired_bootstrap(P, preds["elo"], y)
            sig = "sig" if hi < 0 else ("worse" if lo > 0 else "n.s.")
            ci = f"{md:+.4f} [{lo:+.3f}, {hi:+.3f}] {sig}"
        rows.append({"model": name, **s, "vs_elo": ci})
    return pd.DataFrame(rows)
