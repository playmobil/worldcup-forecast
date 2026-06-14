"""The 48 teams of the 2026 World Cup, the official draw, and helpers.

Per-team structural variables (Klement-style slow variables) are *approximate* and
serve as a fallback prior; the live forecast prefers the frozen FIFA snapshot and
Transfermarkt squad values loaded in :mod:`wcforecast.data`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    """Structural attributes of a national team."""

    name: str
    fifa: int                 # approximate frozen FIFA points (fallback prior)
    gdp_per_capita: float     # USD
    population_m: float       # millions
    avg_temp_c: float         # mean annual temperature
    host: bool                # 2026 co-host (USA / Mexico / Canada)
    confederation: str        # UEFA / CONMEBOL / CAF / AFC / CONCACAF / OFC


def _t(name, fifa, gdp, pop, temp, host, conf):
    return Team(name, fifa, gdp, pop, temp, host, conf)


# Group -> ordered list of teams (2026 official draw).
GROUPS: dict[str, list[Team]] = {
    "A": [_t("Mexico", 1650, 14000, 130, 21, True, "CONCACAF"),
          _t("South Africa", 1440, 6500, 60, 17, False, "CAF"),
          _t("South Korea", 1580, 36000, 52, 13, False, "AFC"),
          _t("Czech Republic", 1500, 32000, 10.5, 9, False, "UEFA")],
    "B": [_t("Canada", 1500, 55000, 40, 5, True, "CONCACAF"),
          _t("Bosnia and Herzegovina", 1350, 8000, 3.2, 11, False, "UEFA"),
          _t("Qatar", 1430, 80000, 2.7, 27, False, "AFC"),
          _t("Switzerland", 1640, 100000, 8.8, 9, False, "UEFA")],
    "C": [_t("Brazil", 1780, 11000, 216, 25, False, "CONMEBOL"),
          _t("Morocco", 1690, 4000, 37, 17, False, "CAF"),
          _t("Haiti", 1250, 1700, 11.5, 25, False, "CONCACAF"),
          _t("Scotland", 1490, 50000, 5.5, 8, False, "UEFA")],
    "D": [_t("United States", 1660, 86000, 335, 12, True, "CONCACAF"),
          _t("Paraguay", 1450, 6000, 6.9, 23, False, "CONMEBOL"),
          _t("Australia", 1500, 65000, 26, 18, False, "AFC"),
          _t("Turkey", 1560, 13000, 85, 14, False, "UEFA")],
    "E": [_t("Germany", 1700, 54000, 84, 9, False, "UEFA"),
          _t("Curaçao", 1320, 20000, 0.15, 28, False, "CONCACAF"),
          _t("Ivory Coast", 1490, 2700, 28, 26, False, "CAF"),
          _t("Ecuador", 1560, 6500, 18, 21, False, "CONMEBOL")],
    "F": [_t("Netherlands", 1750, 62000, 18, 10, False, "UEFA"),
          _t("Japan", 1650, 33000, 125, 15, False, "AFC"),
          _t("Sweden", 1530, 56000, 10.5, 7, False, "UEFA"),
          _t("Tunisia", 1500, 4000, 12, 19, False, "CAF")],
    "G": [_t("Belgium", 1740, 55000, 12, 10, False, "UEFA"),
          _t("Egypt", 1510, 4000, 112, 22, False, "CAF"),
          _t("Iran", 1630, 4500, 89, 17, False, "AFC"),
          _t("New Zealand", 1300, 48000, 5.2, 11, False, "OFC")],
    "H": [_t("Spain", 1875, 33000, 48, 15, False, "UEFA"),
          _t("Cape Verde", 1390, 4200, 0.6, 24, False, "CAF"),
          _t("Saudi Arabia", 1420, 33000, 37, 25, False, "AFC"),
          _t("Uruguay", 1680, 22000, 3.4, 17, False, "CONMEBOL")],
    "I": [_t("France", 1870, 46000, 68, 11, False, "UEFA"),
          _t("Senegal", 1640, 1800, 18, 28, False, "CAF"),
          _t("Iraq", 1430, 5500, 45, 22, False, "AFC"),
          _t("Norway", 1500, 88000, 5.5, 5, False, "UEFA")],
    "J": [_t("Argentina", 1890, 13000, 46, 15, False, "CONMEBOL"),
          _t("Algeria", 1500, 5500, 45, 17, False, "CAF"),
          _t("Austria", 1560, 57000, 9, 7, False, "UEFA"),
          _t("Jordan", 1390, 4500, 11, 19, False, "AFC")],
    "K": [_t("Portugal", 1770, 28000, 10, 16, False, "UEFA"),
          _t("DR Congo", 1420, 700, 102, 25, False, "CAF"),
          _t("Uzbekistan", 1430, 3000, 35, 13, False, "AFC"),
          _t("Colombia", 1700, 7000, 52, 24, False, "CONMEBOL")],
    "L": [_t("England", 1820, 50000, 68, 10, False, "UEFA"),
          _t("Croatia", 1700, 22000, 3.9, 12, False, "UEFA"),
          _t("Ghana", 1450, 2400, 34, 27, False, "CAF"),
          _t("Panama", 1430, 19000, 4.4, 27, False, "CONCACAF")],
}

TEAMS: dict[str, Team] = {t.name: t for g in GROUPS.values() for t in g}
TEAM_NAMES: list[str] = list(TEAMS)
INDEX: dict[str, int] = {name: i for i, name in enumerate(TEAM_NAMES)}
N_TEAMS = len(TEAM_NAMES)
HOSTS: frozenset[str] = frozenset(name for name, t in TEAMS.items() if t.host)
CONFEDERATION: dict[str, str] = {name: t.confederation for name, t in TEAMS.items()}

assert N_TEAMS == 48, f"expected 48 teams, got {N_TEAMS}"


def group_of(team: str) -> str:
    """Return the group letter a team belongs to."""
    for g, members in GROUPS.items():
        if any(t.name == team for t in members):
            return g
    raise KeyError(team)
