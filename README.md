# worldcup-forecast

A structural + Bayesian forecaster for the FIFA World Cup — and, just as importantly, an
**honest, rigorously-validated** account of *what actually improves* national-team match
prediction.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-passing-brightgreen)

Most World Cup models publish a champion list and a flattering accuracy number. This one
ships a **walk-forward validation harness** and a **findings ledger** that states — with
significance tests — which ideas help and which don't, including the uncomfortable ones.
The headline result is itself honest: after an exhaustive search, the model sits at the
*structural efficiency frontier*, and the betting market is very hard to beat.

---

## What it does

- **Team strength** from a frozen pre-tournament FIFA snapshot + Transfermarkt squad value
  + a leakage-safe Elo, wrapped in Klement-style structural priors (GDP, population,
  climate, host advantage, football-culture).
- **Match model**: a hierarchical Bayesian Poisson with partial pooling (PyMC) — data-rich
  teams are pulled by results; data-poor teams shrink to the structural prior.
- **Tournament**: Monte-Carlo simulation of the official 2026 bracket (12 groups, 8 best
  third-placed teams, 73–104 knockout map) for champion/stage probabilities.
- **Validation**: out-of-sample walk-forward scoring (log-loss / RPS / Brier) with
  paired-bootstrap significance — never judged on a handful of matches.
- **Market benchmark**: Polymarket and traditional bookmaker odds for comparison only.
  Per the design's independence principle, **market data is never an input to the model**.

## Quickstart

```bash
git clone <repo-url> && cd worldcup-forecast
pip install -e .                 # or: make install

wcforecast forecast              # 2026 champion probabilities (dual-track)
wcforecast predict Brazil Morocco
wcforecast validate              # out-of-sample scorecard with significance tests
ODDS_API_KEY=xxxx wcforecast odds   # live bookmaker consensus (optional, free key)
```

Python ≥ 3.10. First run auto-downloads the public match/ranking datasets; the small
frozen 2026 inputs are bundled in `data/snapshots/`.

## Example output

```
2026 World Cup — champion probability (Monte Carlo)

  team                      accuracy %  independent %
  France                          13.8           13.3
  Spain                           12.9           11.4
  England                         11.4           11.3
  Argentina                        8.6            9.9
  United States                    6.7            7.7
  ...
```

Two tracks are reported side by side:
- **accuracy** — the full model (FIFA + squad-value anchor + match data);
- **independent** — the structural ("Klement") prior alone, market-free and interpretable.

Their differences are *features to explain*, not errors to hide.

## Honest findings — the distinctive part

Every candidate was tested out-of-sample on a **locked 305-match window (2024–2026)** with
paired-bootstrap significance. Full ledger: [`docs/FINDINGS.md`](docs/FINDINGS.md).

| Idea | Verdict |
|---|---|
| Leakage-safe Elo (K=40) + structural prior | ✅ solid baseline |
| Temperature calibration + neutral-draw boost | ✅ small but significant |
| Model averaging (Bayesian + Elo-logit) | ✅ lowers variance |
| Squad value as a strength anchor | ✅ helps 2026 forecast (aligns ~30% closer to the sharp market on mismatches) |
| FIFA vs Elo anchor | ➖ tie (FIFA's value is data quality, not extra signal) |
| Extra features (form, rest, confederation) | ➖ no significant gain |
| Head-to-head records | ❌ ≈ zero once strength is controlled |
| Dixon-Coles low-score correction | ❌ ρ ≈ 0 on this data |
| Gradient-boosting (LightGBM) ensemble & isotonic calibration | ❌ over-fit sparse data, significantly worse |

**Bottom line:** simple/low-variance moves help; flexible/complex ones over-fit national-team
data. The model is at the structural frontier — single-match World Cup outcomes are roughly
half luck, and the market is hard to beat. That conclusion is the product.

## How it works

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full methodology and the design trade-offs
(why partial pooling, why the market is benchmark-only, how leakage is prevented).

## Project layout

```
worldcup-forecast/
├── src/wcforecast/
│   ├── teams.py        # the 48 teams, official draw, confederations
│   ├── data.py         # leakage-safe loaders (martj42, FIFA, squad, World Bank)
│   ├── ratings.py      # Elo + structural (Klement) prior index
│   ├── model.py        # hierarchical Bayesian Poisson (PyMC)
│   ├── predict.py      # score-grid 1X2 + calibration
│   ├── simulate.py     # official bracket + Monte-Carlo champion odds
│   ├── validate.py     # walk-forward OOS harness + metrics + significance
│   ├── markets.py      # Polymarket + bookmaker odds (benchmark only)
│   └── cli.py          # `wcforecast` command-line interface
├── tests/              # fast, offline unit tests
├── docs/               # DESIGN.md (methodology) + FINDINGS.md (validated ledger)
└── data/               # frozen 2026 snapshots (committed) + caches (git-ignored)
```

## Library use

```python
from wcforecast import data, ratings, model, simulate, predict

results = data.load_results()
s = ratings.structural_index("2026-06-11", results,
                             fifa=data.load_fifa_snapshot(),
                             squad=data.load_squad_values())
matches = data.load_matches(start="2006-01-01", cutoff="2026-06-11")
m = model.fit(matches, s, weights=ratings.recency_weights(matches, "2026-06-11"),
              dixon_coles=True)

simulate.champion_probabilities(m, s, n_sims=20000).head()
predict.calibrate(m.match_probs("Brazil", "Morocco", home_advantage=0.0))
```

## Design principle: independence from the market

The model never ingests betting odds (as a feature, calibration target, or ensemble member).
Its purpose is an *independent, interpretable structural signal*; markets are used only to
benchmark it. Beating the market is **not** the goal — and the validation confirms it would
be very hard anyway.

## Limitations

- Single-match prediction has a low ceiling (much of the outcome is luck); see FINDINGS.
- Squad values and the FIFA snapshot are single 2026 snapshots — used for the forward
  forecast, not historical backtests (no history available).
- Free bookmaker-odds tier covers live odds only, so the market benchmark for *historical*
  validation is limited to outcomes, not historical prices.

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

Inspired by Joachim Klement's structural World Cup model. Data: martj42/international_results,
Dato-Futbol/fifa-ranking, FIFA, Transfermarkt, World Bank, Polymarket, The Odds API.
