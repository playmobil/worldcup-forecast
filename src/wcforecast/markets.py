"""Market odds — **benchmark only**.

This module fetches prediction-market (Polymarket) and traditional-bookmaker
(The Odds API) prices. Per the design's independence principle, market data is used
ONLY to score/benchmark the forecast and is NEVER an input to the model.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

import requests

from .data import CACHE
from .teams import INDEX

# --------------------------------------------------------------------------- #
# Polymarket (prediction market, no API key)                                  #
# --------------------------------------------------------------------------- #
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
WORLD_CUP_TAG_ID = 102232
GAME_SLUG_RE = re.compile(r"^fifwc-.+-\d{4}-\d{2}-\d{2}$")
_PM_NAMES = {"Korea Republic": "South Korea", "Czechia": "Czech Republic",
             "Bosnia-Herzegovina": "Bosnia and Herzegovina", "USA": "United States",
             "Türkiye": "Turkey", "Cabo Verde": "Cape Verde", "Curacao": "Curaçao",
             "Côte d'Ivoire": "Ivory Coast", "Congo DR": "DR Congo"}


def _pm_name(n):
    return _PM_NAMES.get((n or "").strip(), (n or "").strip())


def devig(p_home, p_draw, p_away):
    """Normalise three implied probabilities to sum 1 (linear de-vig); None if absurd."""
    s = p_home + p_draw + p_away
    return None if not (0.5 < s < 2.0) else (p_home / s, p_draw / s, p_away / s)


def _pm_split_title(title):
    parts = re.split(r"\s+vs\.?\s+", title or "", maxsplit=1)
    return (_pm_name(parts[0]), _pm_name(parts[1])) if len(parts) == 2 else (None, None)


def polymarket_finished_games(end_date_min="2026-06-01", end_date_max=None, closed="true"):
    """List finished (or started-but-decided) World Cup match events."""
    if end_date_max is None:
        end_date_max = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    games, offset = [], 0
    while True:
        r = requests.get(f"{GAMMA_API}/events", params={
            "tag_id": WORLD_CUP_TAG_ID, "closed": closed,
            "end_date_min": end_date_min, "end_date_max": end_date_max,
            "limit": 100, "offset": offset}, timeout=15)
        r.raise_for_status()
        batch = r.json()
        games += [{"slug": e["slug"], "title": e["title"], "kickoff": e["endDate"]}
                  for e in batch if GAME_SLUG_RE.match(e["slug"])]
        if len(batch) < 100:
            break
        offset += 100
    return sorted(games, key=lambda g: g["kickoff"])


def polymarket_outcome(slug):
    """Realised result of a settled event: dict(outcome=home/draw/away, winner, ...) or None."""
    e = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=15).json()
    if not e:
        return None
    event = e[0]
    ml = [m for m in event["markets"] if m.get("sportsMarketType") == "moneyline"]
    home, away = _pm_split_title(event.get("title", ""))
    hits = []
    for m in ml:
        outs = json.loads(m["outcomes"])
        prices = [float(p) for p in json.loads(m["outcomePrices"])]
        if prices[outs.index("Yes")] > 0.99:
            hits.append(m["groupItemTitle"])
    if len(hits) != 1:
        return None
    if hits[0].startswith("Draw"):
        return {"outcome": "draw", "winner": None, "home_team": home, "away_team": away}
    winner = _pm_name(hits[0])
    return {"outcome": "home" if winner == home else "away", "winner": winner,
            "home_team": home, "away_team": away}


def _clob_price_before(token_id, kickoff_ts, fidelity=3600):
    try:
        r = requests.get(f"{CLOB_API}/prices-history", params={
            "market": token_id, "interval": "max", "fidelity": fidelity}, timeout=15)
        r.raise_for_status()
        pts = r.json().get("history", [])
    except Exception:
        return None
    elig = [p for p in pts if float(p.get("t", 1e18)) < kickoff_ts]
    return float(max(elig, key=lambda p: float(p["t"]))["p"]) if elig else None


def polymarket_prematch_1x2(slug, fidelity=3600):
    """Pre-kickoff implied 1X2 (de-vigged), relative to the title's home/away. Or None."""
    e = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=15).json()
    if not e:
        return None
    event = e[0]
    ml = [m for m in event["markets"] if m.get("sportsMarketType") == "moneyline"]
    if len(ml) != 3:
        return None
    home, away = _pm_split_title(event.get("title", ""))
    kts = datetime.fromisoformat(event["endDate"].replace("Z", "+00:00")).timestamp()
    px = {}
    for m in ml:
        outs = json.loads(m["outcomes"])
        tok = json.loads(m["clobTokenIds"])
        p = _clob_price_before(tok[outs.index("Yes")], kts, fidelity)
        name = m["groupItemTitle"]
        px["draw" if name.startswith("Draw") else _pm_name(name)] = p
    if None in (px.get(home), px.get("draw"), px.get(away)):
        return None
    dv = devig(px[home], px["draw"], px[away])
    return None if dv is None else {"home": dv[0], "draw": dv[1], "away": dv[2],
                                    "home_team": home, "away_team": away}


# --------------------------------------------------------------------------- #
# Traditional bookmakers (The Odds API, needs ODDS_API_KEY) — cached          #
# --------------------------------------------------------------------------- #
ODDS_API = "https://api.the-odds-api.com/v4"
_ODDS_CACHE = CACHE / "bookmaker_odds.json"
_ODDS_TTL = 6 * 3600
_BK_NAMES = {"USA": "United States", "Korea Republic": "South Korea", "South Korea": "South Korea",
             "Czechia": "Czech Republic", "Turkiye": "Turkey", "Türkiye": "Turkey",
             "Cape Verde Islands": "Cape Verde", "Cote d'Ivoire": "Ivory Coast",
             "Côte d'Ivoire": "Ivory Coast", "DR Congo": "DR Congo", "Curacao": "Curaçao",
             "IR Iran": "Iran"}


def _odds_key(key=None):
    key = key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("Set ODDS_API_KEY (free at https://the-odds-api.com) to fetch "
                         "bookmaker odds, or skip with --no-bookmaker.")
    return key


def fetch_bookmaker_odds(key=None, regions="eu,uk", use_cache=True, max_age=_ODDS_TTL):
    """Raw World Cup odds from The Odds API, cached to avoid burning the free quota."""
    sig = f"{regions}|h2h"
    if use_cache and _ODDS_CACHE.exists():
        try:
            c = json.loads(_ODDS_CACHE.read_text())
            if c.get("sig") == sig and (time.time() - c.get("ts", 0)) < max_age:
                return c["data"]
        except Exception:
            pass
    r = requests.get(f"{ODDS_API}/sports/soccer_fifa_world_cup/odds", params={
        "apiKey": _odds_key(key), "regions": regions, "markets": "h2h",
        "oddsFormat": "decimal"}, timeout=20)
    r.raise_for_status()
    data = r.json()
    CACHE.mkdir(parents=True, exist_ok=True)
    _ODDS_CACHE.write_text(json.dumps({"ts": time.time(), "sig": sig, "data": data}))
    return data


def bookmaker_consensus_1x2(event):
    """Average implied 1X2 across all bookmakers in an event, de-vigged. Or None."""
    home, away = event["home_team"], event["away_team"]
    acc = {"home": [], "draw": [], "away": []}
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            o = {x["name"]: x["price"] for x in mk["outcomes"]}
            if home in o and away in o and "Draw" in o:
                acc["home"].append(1.0 / o[home])
                acc["draw"].append(1.0 / o["Draw"])
                acc["away"].append(1.0 / o[away])
    if not acc["home"]:
        return None
    ph, pd_, pa = (sum(acc[k]) / len(acc[k]) for k in ("home", "draw", "away"))
    dv = devig(ph, pd_, pa)
    return None if dv is None else {
        "home_team": _BK_NAMES.get(home, home), "away_team": _BK_NAMES.get(away, away),
        "kickoff": event.get("commence_time"), "n_books": len(acc["home"]),
        "overround": ph + pd_ + pa, "p1x2": dv}


def bookmaker_games(key=None, regions="eu,uk"):
    """Per-match bookmaker consensus 1X2 for all upcoming World Cup matches."""
    out = [bookmaker_consensus_1x2(e) for e in fetch_bookmaker_odds(key, regions=regions)]
    out = [g for g in out if g and g["home_team"] in INDEX and g["away_team"] in INDEX]
    return sorted(out, key=lambda g: g["kickoff"] or "")
