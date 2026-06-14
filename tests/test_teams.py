from wcforecast import data
from wcforecast.teams import CONFEDERATION, GROUPS, HOSTS, TEAM_NAMES, TEAMS


def test_forty_eight_teams():
    assert len(TEAMS) == 48
    assert len(TEAM_NAMES) == len(set(TEAM_NAMES)) == 48


def test_twelve_groups_of_four():
    assert len(GROUPS) == 12
    assert all(len(g) == 4 for g in GROUPS.values())


def test_three_hosts():
    assert HOSTS == {"United States", "Mexico", "Canada"}


def test_six_confederations():
    assert set(CONFEDERATION.values()) == {"UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"}


def test_frozen_snapshots_cover_all_teams():
    fifa = data.load_fifa_snapshot()
    squad = data.load_squad_values()
    assert all(t in fifa for t in TEAM_NAMES)
    assert all(t in squad for t in TEAM_NAMES)
