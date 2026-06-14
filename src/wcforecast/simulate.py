"""Tournament simulation: official 2026 bracket + Monte-Carlo champion/stage odds.

The knockout map (matches 73–104, eight best third-placed teams) follows the official
2026 format. Parameter uncertainty is propagated by drawing a fresh posterior sample
for each simulated tournament.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .teams import GROUPS, INDEX, N_TEAMS, TEAM_NAMES, TEAMS


def _T(*groups):
    return ("T", frozenset(groups))


# R32 slots: ('W', g)=group winner, ('R', g)=runner-up, ('T', {...})=eligible 3rd.
R32 = {
    73: (("R", "A"), ("R", "B")),          74: (("W", "E"), _T("A", "B", "C", "D", "F")),
    75: (("W", "F"), ("R", "C")),          76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), _T("C", "D", "F", "G", "H")), 78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), _T("C", "E", "F", "H", "I")), 80: (("W", "L"), _T("E", "H", "I", "J", "K")),
    81: (("W", "D"), _T("B", "E", "F", "I", "J")), 82: (("W", "G"), _T("A", "E", "H", "I", "J")),
    83: (("R", "K"), ("R", "L")),          84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), _T("E", "F", "G", "I", "J")), 86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), _T("D", "E", "I", "J", "L")), 88: (("R", "D"), ("R", "G")),
}
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL = {104: (101, 102)}
THIRD_SLOTS = [74, 77, 79, 80, 81, 82, 85, 87]
_HOST = np.array([1.0 if TEAMS[t].host else 0.0 for t in TEAM_NAMES])


def _match_thirds_to_slots(qual_thirds: dict[str, str], rng) -> dict[int, str]:
    """Assign the eight qualifying third-placed teams to slots respecting eligibility
    (a bipartite matching — structurally equivalent to FIFA's 495-row table)."""
    slot_elig = {m: R32[m][1][1] for m in THIRD_SLOTS}
    teams = list(qual_thirds.items())
    order = list(range(len(teams)))
    rng.shuffle(order)
    slots = list(THIRD_SLOTS)
    rng.shuffle(slots)
    assign, used = {}, set()

    def backtrack(si: int) -> bool:
        if si == len(slots):
            return True
        m = slots[si]
        for ti in order:
            team, g = teams[ti]
            if ti not in used and g in slot_elig[m]:
                used.add(ti)
                assign[m] = team
                if backtrack(si + 1):
                    return True
                used.discard(ti)
                del assign[m]
        return False

    backtrack(0)
    return assign


def _play(i, j, atk, deff, mu, hadv, knockout, rng):
    li = np.exp(mu + atk[i] - deff[j] + hadv * _HOST[i])
    lj = np.exp(mu + atk[j] - deff[i] + hadv * _HOST[j])
    gi, gj = rng.poisson(li), rng.poisson(lj)
    if not knockout:
        return gi, gj, None
    if gi != gj:
        return gi, gj, (i if gi > gj else j)
    gi2, gj2 = rng.poisson(li / 3), rng.poisson(lj / 3)        # extra time ≈ 1/3 strength
    if gi2 != gj2:
        return gi, gj, (i if gi2 > gj2 else j)
    p = 1.0 / (1.0 + np.exp(-0.4 * ((atk[i] - deff[i]) - (atk[j] - deff[j]))))  # penalties
    return gi, gj, (i if rng.random() < p else j)


def simulate_once(atk, deff, mu, hadv, tiebreak, rng):
    """Play one full tournament; return (champion_index, {team_index: deepest_stage}).

    Stage codes: 32, 16, 8, 4, 2 (final), 1 (champion); smaller = deeper.
    """
    g_first, g_second, thirds = {}, {}, []
    for g, members in GROUPS.items():
        idxs = [INDEX[t.name] for t in members]
        st = {k: [0, 0, 0] for k in idxs}  # points, goal-diff, goals-for
        for a in range(4):
            for b in range(a + 1, 4):
                gi, gj, _ = _play(idxs[a], idxs[b], atk, deff, mu, hadv, False, rng)
                st[idxs[a]][2] += gi
                st[idxs[a]][1] += gi - gj
                st[idxs[b]][2] += gj
                st[idxs[b]][1] += gj - gi
                if gi > gj:
                    st[idxs[a]][0] += 3
                elif gj > gi:
                    st[idxs[b]][0] += 3
                else:
                    st[idxs[a]][0] += 1
                    st[idxs[b]][0] += 1
        ranked = sorted(idxs, key=lambda k: (st[k][0], st[k][1], st[k][2], tiebreak[k], rng.random()),
                        reverse=True)
        g_first[g], g_second[g] = ranked[0], ranked[1]
        thirds.append((g, ranked[2], st[ranked[2]]))

    thirds.sort(key=lambda x: (x[2][0], x[2][1], x[2][2], tiebreak[x[1]], rng.random()), reverse=True)
    qual = {tm: g for g, tm, _ in thirds[:8]}
    slot_team = _match_thirds_to_slots(qual, rng)

    def resolve(slot, match_no):
        kind = slot[0]
        if kind == "W":
            return g_first[slot[1]]
        if kind == "R":
            return g_second[slot[1]]
        return slot_team[match_no]

    win, reached = {}, {}

    def mark(t, stage):
        reached[t] = min(reached.get(t, 99), stage)

    for m, (sa, sb) in R32.items():
        a, b = resolve(sa, m), resolve(sb, m)
        mark(a, 32)
        mark(b, 32)
        _, _, w = _play(a, b, atk, deff, mu, hadv, True, rng)
        win[m] = w
        mark(w, 16)
    for rnd, stage in ((R16, 8), (QF, 4), (SF, 2)):
        for m, (x, y) in rnd.items():
            _, _, w = _play(win[x], win[y], atk, deff, mu, hadv, True, rng)
            win[m] = w
            mark(w, stage)
    (mf, (x, y)), = FINAL.items()
    _, _, champ = _play(win[x], win[y], atk, deff, mu, hadv, True, rng)
    mark(champ, 1)
    return champ, reached


def monte_carlo(atk, deff, mu, hadv, tiebreak, n_sims: int = 10000, seed: int = 7) -> pd.DataFrame:
    """Run ``n_sims`` tournaments. ``atk``/``deff`` may be posterior draws ``(S, N)``
    (a fresh draw per tournament → propagates uncertainty) or a point ``(N,)``."""
    rng = np.random.default_rng(seed)
    atk = np.atleast_2d(atk)
    deff = np.atleast_2d(deff)
    mu = np.atleast_1d(mu)
    hadv = np.atleast_1d(hadv)
    s = atk.shape[0]
    champ = np.zeros(N_TEAMS)
    final = np.zeros(N_TEAMS)
    semi = np.zeros(N_TEAMS)
    quarter = np.zeros(N_TEAMS)
    for _ in range(n_sims):
        k = rng.integers(s) if s > 1 else 0
        c, reached = simulate_once(atk[k], deff[k], mu[k % len(mu)], hadv[k % len(hadv)],
                                   tiebreak, rng)
        champ[c] += 1
        for t, stage in reached.items():
            if stage <= 2:
                final[t] += 1
            if stage <= 4:
                semi[t] += 1
            if stage <= 8:
                quarter[t] += 1
    df = pd.DataFrame({
        "team": TEAM_NAMES,
        "champion": 100 * champ / n_sims,
        "final": 100 * final / n_sims,
        "semifinal": 100 * semi / n_sims,
        "quarterfinal": 100 * quarter / n_sims,
    }).sort_values("champion", ascending=False).reset_index(drop=True)
    return df


def champion_probabilities(model, structural, n_sims: int = 10000, seed: int = 7) -> pd.DataFrame:
    """Champion/stage probabilities from a fitted :class:`PoissonModel` (posterior MC)."""
    atk, deff, mu, hadv, _ = model._posterior()
    return monte_carlo(atk, deff, mu, hadv, np.asarray(structural, dtype=float), n_sims, seed)
