# Data

The library separates **committed frozen snapshots** (small, reproducible inputs) from
**derived/large data** (auto-downloaded or computed, git-ignored).

## Committed: `data/snapshots/` (the leakage-safe 2026 inputs)

| File | Source | Notes |
|---|---|---|
| `fifa_ranking_2026-06-11.csv` | FIFA Men's World Ranking | Frozen the day the 2026 World Cup began. Pre-tournament → leakage-free for 2026 forecasting. |
| `squad_values_2026.csv` | Transfermarkt national-team squad market values | Single 2026 snapshot. **No history**, so it is used only for the 2026 forward forecast, never in historical backtests. |

These are the "current strength" anchors for the 2026 forecast (see `docs/DESIGN.md`).

## Auto-downloaded / derived (git-ignored)

| Path | Source | When |
|---|---|---|
| `data/raw/results.csv` | [martj42/international_results](https://github.com/martj42/international_results) — every international match since 1872 | downloaded on first run |
| `data/raw/fifa_ranking_history.csv` | [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking) — monthly FIFA points 1992–2024 | downloaded on first run |
| World Bank GDP / population | [World Bank Indicators API](https://data.worldbank.org) (no key) | fetched + cached on demand |
| `data/cache/*.pkl` | fitted PyMC posteriors, etc. | computed on demand |

## Markets (benchmark only — never fed to the model)

Bookmaker odds via [The Odds API](https://the-odds-api.com) require a free key in `ODDS_API_KEY`.
Polymarket (prediction market) needs no key. Both are used **only** to benchmark the forecast,
per the design's independence principle (`docs/DESIGN.md`).
