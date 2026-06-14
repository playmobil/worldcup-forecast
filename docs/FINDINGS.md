# Findings — an honest ledger

Every candidate improvement was evaluated **out-of-sample** on a locked test window
(2024-01 → 2026-06, ~305 international matches, both teams in the 48) with paired-bootstrap
significance. Strengths/features are computed to each match date (leakage-safe). Reproduce
with `wcforecast validate`.

## Out-of-sample scorecard (305 matches, log-loss; lower is better)

| Predictor | log-loss | vs. Elo baseline (95% CI) |
|---|---|---|
| uniform (1/3, 1/3, 1/3) | 1.099 | +0.051 |
| **Elo (baseline)** | **1.048** | — |
| FIFA | 1.051 | +0.003 [−0.019, +0.025] — tie |
| Elo + FIFA | 1.043 | −0.005 [−0.012, +0.003] — n.s. |
| Bayesian Poisson (pooled, DC, recency) | 1.047 | −0.003 [−0.022, +0.016] — n.s. |
| Bayesian + temperature calibration | 1.041 | calibration alone: −0.006 [−0.011, −0.000] ✅ |
| model averaging + calibration | **1.038** | −0.012 [−0.024, +0.001] — best, borderline |
| LightGBM ensemble | 1.096 | +0.046 [+0.009, +0.085] ❌ worse |
| isotonic calibration | 1.234 | +0.185 ❌ worse |

## ✅ Kept (robust, validated)

- **Elo K=40** (vs default 20) — best on the development window.
- **Temperature calibration + neutral-venue draw boost** — the only *individually
  significant* improvement (the model is mildly over-confident and under-predicts draws,
  especially at neutral venues: predicted ≈0.26 vs actual ≈0.33).
- **Model averaging** (Bayesian + Elo-logit) — lowers variance.

Together: ≈ −1% log-loss over a simple Elo logit (borderline significant).

## ✅ Squad value (2026 forecast)

Transfermarkt squad value is *more independent* of Elo/FIFA (corr ≈ 0.76/0.80, vs FIFA–Elo
0.94) and harsh on minnows. Validated against the **sharp bookmaker market** (not 6 matches):
adding it as a strength anchor cuts the model-vs-consensus gap by **~30% on lopsided games**
(e.g. Germany–Curaçao 0.79→0.83 toward the market's 0.92). It is used in the 2026 forward
forecast. Caveat: a single snapshot with no history → cannot be validated on the 305-match
backtest, and it does not help the tiny played-2026 sample (which is noise + favourite-draws).

## ❌ Rejected (with evidence)

- **FIFA vs Elo anchor** — statistically tied; FIFA's value is data quality, not signal.
- **Extra features** (recent form, rest days, confederation flags) — no significant gain.
- **Head-to-head records** — once team strength is controlled, ΔR² ≈ 0.001 (t ≈ 1.6). The
  "bogey team" effect is mostly the strength gap plus small-sample noise.
- **Dixon-Coles** — ρ ≈ 0 on this data (tried twice).
- **Gradient-boosting (LightGBM) ensemble** and **isotonic calibration** — both
  *significantly worse*: flexible learners over-fit sparse national-team data. (The
  literature's larger gains come from club football with bookmaker-odds features.)
- **Bayesian structure vs a simple Elo logit** — tied. The hierarchy is principled and
  needed for the tournament simulation, but it does not beat a good rating + logit on 1X2.

## Why you must not optimise for a handful of matches

Tuning the calibration to make the (then) 6 played 2026 matches look best drops their
log-loss to **0.776** (impressive) — but the *same* parameters move the 305-match
out-of-sample log-loss from 1.039 **to 1.105** (worse than doing nothing). Six matches are
noise; "make the recent games look best" is a deceptive objective. This is why every
decision above is made on the locked test window with significance, not on World-Cup
outcomes.

## Bottom line

Simple, low-variance moves give small real gains; every flexible/complex move over-fits.
The model is at the **structural efficiency frontier**: single-match World Cup prediction
has a low ceiling (≈ half luck) and the market is hard to beat. Reaching that conclusion
*rigorously* — and being able to say which ideas are dead ends — is the point of the project.
