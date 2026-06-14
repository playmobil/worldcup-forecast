# worldcup-forecast

[简体中文](README.md) | **English**

A structural + Bayesian forecaster for the FIFA World Cup — and, just as importantly, an
**honest, rigorously-validated** account of *what actually improves* national-team match
prediction.

*This project is inspired by Joachim Klement's structural World Cup model.*

![Joachim Klement — correctly predicted the World Cup champion three times running: 2014 (Germany), 2018 (France), 2022 (Argentina)](joachim-klement.jpg)

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-passing-brightgreen)

Most World Cup models publish a champion list and a flattering accuracy number. This one
ships a **walk-forward validation harness** and a **findings ledger** that states — with
significance tests — which ideas help and which don't, including the uncomfortable ones.
The headline result is itself honest: after an exhaustive search, the model sits at the
*structural efficiency frontier*, and the betting market is very hard to beat.

---

## 2026 World Cup match results (live)

An illustrative bet log following the dual-track forecast: $10,000 per match, split across
home/draw/away in proportion to the model's predicted probabilities (min unit $100), settled
at the actual result's odds.

| Match | Model W/D/L % | Stake W/D/L | Actual | Odds | P/L |
|---|---|---|---|---|---|
| Mexico–South Africa | 63/24/12 | 6400/2400/1200 | home | 1.44 | −784 |
| South Korea–Czech Republic | 39/32/29 | 3900/3200/2900 | home | 2.67 | +413 |
| Canada–Bosnia | 54/28/18 | 5400/2800/1800 | draw | 3.64 | +192 |
| United States–Paraguay | 61/25/14 | 6100/2500/1400 | home | 2.15 | +3,115 |
| Qatar–Switzerland | 13/26/61 | 1300/2600/6100 | draw | 8.00 | +10,800 |
| Brazil–Morocco | 42/31/28 | 4100/3100/2800 | draw | 3.92 | +2,152 |
| Haiti–Scotland | 19/30/51 | 1900/3000/5100 | away | 1.55 | −2,095 |
| Australia–Turkey | 26/31/43 | 2600/3100/4300 | home | 5.71 | +4,846 |
| **Total (8 matches, $80,000 staked)** | | | | | **+18,639** |

> ⚠️ Honest note: only 8 matches, and the return is dominated by a single longshot
> (Qatar–Switzerland draw @8.00, +$10,800, ~60% of the total); against the ~5.9% bookmaker vig
> the strategy is negative-EV overall. This is variance, **not** evidence of beating the market
> — which is exactly the point this project keeps making. See [`docs/FINDINGS.md`](docs/FINDINGS.md).

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

## Built on Klement's structural approach

Joachim Klement's premise is that national-team strength is mostly *structural* — not star
power, but slow-moving country variables, the tournament's own path, and a dose of luck. This
project keeps exactly those inputs as a **leakage-safe structural prior** (the `independent`
track), then lets match results refine it through partial pooling. Each factor his model looks
at maps to a concrete term in `ratings.structural_index` / `simulate.py`:

| Klement factor | How this project implements it |
|---|---|
| FIFA ranking | Standardized FIFA points as the current-strength **anchor**, frozen pre-tournament (blended with Transfermarkt squad value for 2026). |
| Population | A `log10(population)` "talent-pool size" term plus a population × culture interaction — a big population helps only where football is actually followed. |
| GDP per capita | `log10(GDP)` as an **inverted-U** (development helps, then flattens); the coefficient is reasoned, not fit. |
| Football's role in society | A leakage-safe **culture** proxy: half past-World-Cup appearances, half long-run Elo. |
| Climate & match environment | A mean-temperature **inverted-U** (penalty for leaving a temperate optimum) plus a host-nation boost. |
| Group & knockout path | Monte-Carlo over the **official 2026 bracket** — the real draw: 12 groups → top two + 8 best third-placed teams (eligibility-respecting matching) → R32…final. |
| A degree of randomness | Poisson goal noise each match, ≈⅓-strength extra time and penalty coin-flips in knockouts, and a fresh posterior draw per simulated tournament. |

**Where it goes further.** That structural prior is the *shrinkage target* of a hierarchical
Bayesian Poisson — data-rich teams move toward their results, data-poor teams stay near the
prior — and a leakage-safe Elo plus squad value are added as extra anchors. Coefficients are
set from prior reasoning, never fit on the ~36 rows of World-Cup-only history (which yields
nonsense, e.g. a negative GDP coefficient). And every addition is checked out-of-sample with
significance tests — see Honest findings above.

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
