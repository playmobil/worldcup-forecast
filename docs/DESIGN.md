# Design & methodology

A concise specification of how `worldcup-forecast` works and why.

## 1. Goal and philosophy

The model produces an **independent, interpretable structural signal** for World Cup
outcomes. It deliberately does **not** chase maximum accuracy:

- If you only want accuracy, use the betting market — it is near-efficient and very hard
  to beat. Our own validation (`FINDINGS.md`) confirms this.
- The value here is a transparent, reproducible view that is *orthogonal* to the market,
  plus a rigorous account of which modelling choices actually pay off.

Consequently, **market data (Polymarket, bookmaker odds) is never an input** — not as a
feature, a calibration target, or an ensemble member. It is used only to benchmark.

## 2. Data layer (`data.py`)

| Source | Use | Leakage control |
|---|---|---|
| martj42/international_results | training matches (1872–present) | `load_matches(cutoff=...)` keeps only matches strictly before the predicted date |
| FIFA ranking — 2026 snapshot (bundled) | current-strength anchor for 2026 | frozen the day the tournament began |
| FIFA ranking — monthly history (downloaded) | leakage-safe FIFA at any past date | `fifa_at()` takes the latest entry ≤ date |
| Transfermarkt squad values — 2026 snapshot | strength anchor for 2026 | single snapshot → forward forecast only |
| World Bank GDP / population | Klement slow variables | slow-moving; optional refinement |

## 3. Strength estimation (`ratings.py`)

**Elo** (`elo_ratings`) — World-Football-Elo style: K scaled by competition importance,
margin-of-victory multiplier, home-field bonus on non-neutral venues. Computed *to a
cutoff* from all international matches (K=40 chosen by out-of-sample tuning).

**Structural index** (`structural_index`) — a standardized prior strength per team:

```
anchor  = mean of standardized { FIFA points, log squad value }   (2026)   or  z(Elo)  (history)
s_raw   = anchor
        + 0.15·(log10 gdp − 4.3) − 0.05·max(0, log10 gdp − 4.8)    # GDP, inverted-U
        + 0.10·(log10 pop − 1.3)                                   # talent-pool size
        − 0.20·((temp − 13)/12)²                                   # climate inverted-U
        + 0.35·host                                                # host advantage
        + 0.25·culture + 0.15·(log10 pop × culture)                # football culture
s       = standardize(s_raw)
```

Coefficients are set from prior reasoning, **not** fit on the small World-Cup-only history
(flat regressions on ~36 rows produce nonsense — e.g. a negative GDP coefficient). Culture
is a leakage-safe blend of past World-Cup appearances and long-run Elo.

## 4. Match model (`model.py`, `predict.py`)

Hierarchical Bayesian Poisson with **partial pooling**:

```
atk[i] = ka·s[i] + z_a[i]·σ_a        z_a[i] ~ Normal(0, 1)     # non-centred
def[i] = kd·s[i] + z_d[i]·σ_d        z_d[i] ~ Normal(0, 1)
log λ_home = μ + atk[home] − def[away] + home_adv·(non-neutral)
log λ_away = μ + atk[away] − def[home]
home_goals ~ Poisson(λ_home),  away_goals ~ Poisson(λ_away)
```

- **Non-centred** parameterisation removes the hierarchical funnel (0 divergences, r̂≈1.0).
- **Partial pooling**: data-rich teams move to their results; data-poor teams shrink to the
  structural prior — the antidote to over-fitting sparse national-team data.
- **Recency weighting** (optional): each match's log-likelihood is weighted by
  `tournament_importance × exp(-age/half_life)` via a `pm.Potential`.
- **Dixon-Coles** (optional): low-score correlation parameter ρ (found ≈ 0 here).
- 1X2 probabilities come from enumerating the Poisson score grid (`poisson_1x2`),
  averaged over posterior draws (posterior predictive).
- **Calibration** (`calibrate`): temperature scaling + a small neutral-venue draw boost,
  fit out-of-sample. (Isotonic was tried and over-fit — see FINDINGS.)

## 5. Tournament simulation (`simulate.py`)

The official 2026 bracket: 12 groups → top two + eight best third-placed teams (assigned to
slots by an eligibility-respecting bipartite matching) → R32…final. Each simulated
tournament draws a fresh posterior sample, so champion/stage probabilities marginalise both
parameter uncertainty and match randomness. Extra time ≈ ⅓-strength goals; penalties ≈ a
near-coin-flip slightly favouring the stronger side.

## 6. Validation (`validate.py`)

- **Walk-forward / locked test.** Features are leakage-safe (Elo, FIFA, form computed to
  each match date). Models are tuned on a development window (`< 2024-01`) and scored once
  on a **locked test window** (2024-01 → 2026-06, ~305 matches). The handful of played 2026
  matches are *not* an optimisation target.
- **Proper scoring.** Log-loss (primary; sample-efficient), RPS (ordinal), Brier; plus
  reliability and sharpness.
- **Significance.** Paired bootstrap on per-match log-loss differences (95% CI). A change is
  adopted only if it is a robust out-of-sample win.

## 7. Anti-patterns (deliberately avoided)

- No "current form" feature pile-up (player value beyond a squad aggregate, injuries, xG):
  catastrophic 48-nation coverage and over-fitting.
- No flat regression on the tiny World-Cup-only history.
- No market data fed into the model.
- No single-champion claim — always a probability distribution.
- No tuning to a few matches — see the over-fitting demonstration in `FINDINGS.md`.
