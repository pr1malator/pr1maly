"""
Tests for src/processor.py
These tests exercise the metric calculation logic without requiring a real
.dem file or demoparser2 installation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.processor import (
    _calculate_kast_rounds,
    _collect_all_steam_ids,
    _compute_hltv_rating,
    _count_multikill_rounds,
    _count_total_rounds,
    _count_valid_assists,
    _calculate_damage,
    _detect_player_team,
    _filter_attacker,
    _filter_victim,
    _filter_assister,
    calculate_all_players_stats,
    calculate_match_stats,
)

STEAM_ID = "76561198012345678"


# ---------------------------------------------------------------------------
# Helpers to build minimal DataFrames
# ---------------------------------------------------------------------------


def _make_death_df(events: list[dict]) -> pd.DataFrame:
    """Build a player_death-style DataFrame."""
    return pd.DataFrame(events)


def _make_hurt_df(events: list[dict]) -> pd.DataFrame:
    """Build a player_hurt-style DataFrame."""
    return pd.DataFrame(events)


def _make_round_end_df(n_rounds: int, winner: str = "CT") -> pd.DataFrame:
    return pd.DataFrame({
        "round": list(range(1, n_rounds + 1)),
        "winner": [winner] * n_rounds,
    })


# ---------------------------------------------------------------------------
# _count_total_rounds
# ---------------------------------------------------------------------------


def test_count_total_rounds_uses_max_round():
    df = pd.DataFrame({"round": [1, 2, 3, 4, 5]})
    assert _count_total_rounds(df) == 5


def test_count_total_rounds_empty_df():
    assert _count_total_rounds(pd.DataFrame()) == 0


def test_count_total_rounds_no_round_col():
    df = pd.DataFrame({"winner": [2, 3, 2]})
    assert _count_total_rounds(df) == 3


# ---------------------------------------------------------------------------
# DataFrame filter helpers
# ---------------------------------------------------------------------------


def test_filter_attacker_returns_correct_rows():
    df = _make_death_df(
        [
            {"attacker_steamid": STEAM_ID, "user_steamid": "111"},
            {"attacker_steamid": "999", "user_steamid": STEAM_ID},
        ]
    )
    result = _filter_attacker(df, STEAM_ID)
    assert len(result) == 1
    assert result.iloc[0]["user_steamid"] == "111"


def test_filter_victim_returns_correct_rows():
    df = _make_death_df(
        [
            {"attacker_steamid": STEAM_ID, "user_steamid": "111"},
            {"attacker_steamid": "999", "user_steamid": STEAM_ID},
        ]
    )
    result = _filter_victim(df, STEAM_ID)
    assert len(result) == 1
    assert result.iloc[0]["attacker_steamid"] == "999"


def test_filter_assister_returns_correct_rows():
    df = _make_death_df(
        [
            {"attacker_steamid": "999", "assister_steamid": STEAM_ID, "user_steamid": "111"},
            {"attacker_steamid": "888", "assister_steamid": "777", "user_steamid": "222"},
        ]
    )
    result = _filter_assister(df, STEAM_ID)
    assert len(result) == 1


def test_filter_attacker_empty_df():
    assert _filter_attacker(pd.DataFrame(), STEAM_ID).empty


# ---------------------------------------------------------------------------
# _compute_hltv_rating
# ---------------------------------------------------------------------------


def test_hltv_rating_average_player():
    # Known-good approximation for an average player
    rating = _compute_hltv_rating(kast=75.0, kpr=0.68, dpr=0.68, impact=0.96, adr=75.0)
    assert 0.9 < rating < 1.1, f"Expected ~1.0, got {rating}"


def test_hltv_rating_high_performer():
    rating = _compute_hltv_rating(kast=85.0, kpr=1.1, dpr=0.5, impact=1.8, adr=95.0)
    assert rating > 1.2, f"Expected rating > 1.2, got {rating}"


def test_hltv_rating_low_performer():
    rating = _compute_hltv_rating(kast=50.0, kpr=0.4, dpr=0.9, impact=0.3, adr=50.0)
    assert rating < 0.9, f"Expected rating < 0.9, got {rating}"


# ---------------------------------------------------------------------------
# _calculate_kast_rounds
# ---------------------------------------------------------------------------


def test_kast_all_rounds_contribute():
    rounds = [
        {"kills": 1, "assists": 0, "survived": 0, "traded": 0},
        {"kills": 0, "assists": 1, "survived": 0, "traded": 0},
        {"kills": 0, "assists": 0, "survived": 1, "traded": 0},
        {"kills": 0, "assists": 0, "survived": 0, "traded": 1},
    ]
    assert _calculate_kast_rounds(rounds) == 4


def test_kast_empty_round_not_counted():
    rounds = [
        {"kills": 0, "assists": 0, "survived": 0, "traded": 0},
        {"kills": 1, "assists": 0, "survived": 0, "traded": 0},
    ]
    assert _calculate_kast_rounds(rounds) == 1


def test_kast_empty_list():
    assert _calculate_kast_rounds([]) == 0


# ---------------------------------------------------------------------------
# calculate_match_stats (integration-level with synthetic data)
# ---------------------------------------------------------------------------


def _build_parsed_data(n_rounds: int = 5):
    """Build a minimal parsed_data dict simulating n_rounds of activity."""
    deaths = []
    hurts = []
    for r in range(1, n_rounds + 1):
        # Player gets 1 kill per round, 0 deaths
        deaths.append(
            {
                "round": r,
                "attacker_steamid": STEAM_ID,
                "attacker_name": "TestPlayer",
                "attacker_team_num": 3,
                "user_steamid": "enemy_sid",
                "user_name": "Enemy",
                "user_team_num": 2,
                "assister_steamid": None,
            }
        )
        hurts.append(
            {
                "round": r,
                "attacker_steamid": STEAM_ID,
                "dmg_health": 90,
            }
        )

    return {
        "player_death": pd.DataFrame(deaths),
        "player_hurt": pd.DataFrame(hurts),
        "round_end": _make_round_end_df(n_rounds),
        "header": {"map_name": "de_dust2"},
    }


def test_calculate_match_stats_basic():
    parsed = _build_parsed_data(n_rounds=10)
    stats = calculate_match_stats(parsed, STEAM_ID)

    assert stats["map_name"] == "de_dust2"
    assert stats["total_rounds"] == 10
    assert stats["kills"] == 10
    assert stats["deaths"] == 0
    assert stats["kpr"] == pytest.approx(1.0)
    assert stats["dpr"] == pytest.approx(0.0)
    assert stats["adr"] == pytest.approx(90.0)
    assert stats["kast"] == pytest.approx(100.0)
    assert stats["hltv_rating"] > 0
    # K/D with 0 deaths returns kills as float
    assert stats["kd_ratio"] == pytest.approx(10.0)
    # 1 kill per round => no multi-kills
    assert stats["rounds_2k"] == 0
    assert stats["rounds_3k"] == 0
    assert stats["rounds_4k"] == 0
    assert stats["rounds_5k"] == 0


def test_calculate_match_stats_player_name_detected():
    parsed = _build_parsed_data(n_rounds=5)
    stats = calculate_match_stats(parsed, STEAM_ID)
    assert stats["player_name"] == "TestPlayer"


def test_calculate_match_stats_round_stats_length():
    parsed = _build_parsed_data(n_rounds=7)
    stats = calculate_match_stats(parsed, STEAM_ID)
    assert len(stats["round_stats"]) == 7


def test_calculate_match_stats_empty_data():
    parsed = {
        "player_death": pd.DataFrame(),
        "player_hurt": pd.DataFrame(),
        "round_end": pd.DataFrame(),
        "header": {"map_name": "de_nuke"},
    }
    stats = calculate_match_stats(parsed, STEAM_ID)
    assert stats["kills"] == 0
    assert stats["deaths"] == 0
    assert stats["total_rounds"] == 1  # avoids division by zero
    assert stats["kd_ratio"] == pytest.approx(0.0)
    assert stats["match_result"] == "unknown"


# ---------------------------------------------------------------------------
# _count_multikill_rounds
# ---------------------------------------------------------------------------


def test_multikill_counts_basic():
    rounds = [
        {"kills": 0}, {"kills": 1}, {"kills": 2},
        {"kills": 3}, {"kills": 4}, {"kills": 5},
    ]
    result = _count_multikill_rounds(rounds)
    assert result == {2: 1, 3: 1, 4: 1, 5: 1}


def test_multikill_counts_multiple_2ks():
    rounds = [{"kills": 2}, {"kills": 2}, {"kills": 2}]
    result = _count_multikill_rounds(rounds)
    assert result[2] == 3
    assert result[3] == 0


def test_multikill_six_kills_counts_as_5k():
    rounds = [{"kills": 6}]
    result = _count_multikill_rounds(rounds)
    assert result[5] == 1


def test_multikill_empty():
    assert _count_multikill_rounds([]) == {2: 0, 3: 0, 4: 0, 5: 0}


# ---------------------------------------------------------------------------
# _detect_player_team
# ---------------------------------------------------------------------------


def test_detect_player_team_from_attacker():
    df = _make_death_df([
        {"attacker_steamid": STEAM_ID, "attacker_team_num": 3,
         "user_steamid": "enemy", "user_team_num": 2},
    ])
    assert _detect_player_team(df, STEAM_ID) == "CT"


def test_detect_player_team_from_victim():
    df = _make_death_df([
        {"attacker_steamid": "enemy", "attacker_team_num": 2,
         "user_steamid": STEAM_ID, "user_team_num": 3},
    ])
    assert _detect_player_team(df, STEAM_ID) == "CT"


def test_detect_player_team_empty():
    assert _detect_player_team(pd.DataFrame(), STEAM_ID) is None


def test_detect_player_team_halftime_swap():
    """Team detection should use earliest round, not mode, to handle halftime."""
    # Player is CT (team_num=3) in rounds 1-12, T (team_num=2) in rounds 13-24.
    # With more events in the second half, mode would wrongly pick T.
    events = []
    for r in range(1, 13):
        events.append({
            "round": r, "attacker_steamid": STEAM_ID,
            "attacker_team_num": 3, "user_steamid": "enemy", "user_team_num": 2,
        })
    for r in range(13, 25):
        events.append({
            "round": r, "attacker_steamid": STEAM_ID,
            "attacker_team_num": 2, "user_steamid": "enemy", "user_team_num": 3,
        })
        # Extra event in second half so mode would pick T
        events.append({
            "round": r, "attacker_steamid": STEAM_ID,
            "attacker_team_num": 2, "user_steamid": "enemy2", "user_team_num": 3,
        })
    df = _make_death_df(events)
    # Should return CT (first half), not T (mode)
    assert _detect_player_team(df, STEAM_ID) == "CT"


def test_collect_all_steam_ids_includes_assisters():
    """Players who only appear as assisters should be collected."""
    df = _make_death_df([
        {"attacker_steamid": "a1", "user_steamid": "v1",
         "assister_steamid": "assist_only"},
    ])
    ids = _collect_all_steam_ids(df)
    assert "assist_only" in ids


# ---------------------------------------------------------------------------
# _collect_all_steam_ids
# ---------------------------------------------------------------------------


def test_collect_all_steam_ids():
    df = _make_death_df([
        {"attacker_steamid": STEAM_ID, "user_steamid": "enemy1"},
        {"attacker_steamid": "enemy1", "user_steamid": STEAM_ID},
        {"attacker_steamid": "enemy2", "user_steamid": "enemy1"},
    ])
    ids = _collect_all_steam_ids(df)
    assert set(ids) == {STEAM_ID, "enemy1", "enemy2"}


def test_collect_all_steam_ids_empty():
    assert _collect_all_steam_ids(pd.DataFrame()) == []


def test_collect_all_steam_ids_filters_junk():
    df = _make_death_df([
        {"attacker_steamid": STEAM_ID, "user_steamid": "0"},
        {"attacker_steamid": None, "user_steamid": STEAM_ID},
    ])
    ids = _collect_all_steam_ids(df)
    assert STEAM_ID in ids
    assert "0" not in ids


# ---------------------------------------------------------------------------
# calculate_all_players_stats
# ---------------------------------------------------------------------------


def test_all_players_stats_returns_both_players():
    parsed = _build_parsed_data(n_rounds=5)
    players = calculate_all_players_stats(parsed, STEAM_ID, 5)
    assert len(players) == 2  # STEAM_ID + enemy_sid
    names = {p["name"] for p in players}
    assert "TestPlayer" in names


def test_all_players_stats_user_flagged():
    parsed = _build_parsed_data(n_rounds=5)
    players = calculate_all_players_stats(parsed, STEAM_ID, 5)
    user_entries = [p for p in players if p["is_user"]]
    assert len(user_entries) == 1
    assert user_entries[0]["steam_id"] == STEAM_ID


def test_all_players_stats_empty():
    parsed = {
        "player_death": pd.DataFrame(),
        "player_hurt": pd.DataFrame(),
        "round_end": pd.DataFrame(),
        "header": {},
    }
    assert calculate_all_players_stats(parsed, STEAM_ID, 0) == []


def test_calculate_match_stats_includes_all_players():
    parsed = _build_parsed_data(n_rounds=5)
    stats = calculate_match_stats(parsed, STEAM_ID)
    assert "all_players" in stats
    assert len(stats["all_players"]) == 2


# ---------------------------------------------------------------------------
# _calculate_damage (HP-tracking)
# ---------------------------------------------------------------------------


def test_damage_hp_tracking_caps_overkill():
    """dmg_health exceeding victim's remaining HP should be capped."""
    hurt_df = _make_hurt_df([
        # Round 1: player hits victim for 80, then 50 (victim at 20 HP for
        # second hit).  Actual damage = 80 + 20 = 100, NOT 80 + 50 = 130.
        {"round": 1, "tick": 100, "attacker_steamid": STEAM_ID,
         "attacker_team_num": 3, "user_steamid": "v1", "user_team_num": 2,
         "dmg_health": 80, "health": 20},
        {"round": 1, "tick": 110, "attacker_steamid": STEAM_ID,
         "attacker_team_num": 3, "user_steamid": "v1", "user_team_num": 2,
         "dmg_health": 50, "health": 0},
    ])
    assert _calculate_damage(hurt_df, STEAM_ID) == 100


def test_damage_hp_tracking_shared_victim():
    """When two attackers share a victim, damage is split by actual HP lost."""
    hurt_df = _make_hurt_df([
        # Round 1: other player hits victim for 60 (victim at 40 HP),
        # then our player hits for 80 (victim at 0 HP, actual = 40).
        {"round": 1, "tick": 100, "attacker_steamid": "other",
         "attacker_team_num": 3, "user_steamid": "v1", "user_team_num": 2,
         "dmg_health": 60, "health": 40},
        {"round": 1, "tick": 110, "attacker_steamid": STEAM_ID,
         "attacker_team_num": 3, "user_steamid": "v1", "user_team_num": 2,
         "dmg_health": 80, "health": 0},
    ])
    assert _calculate_damage(hurt_df, STEAM_ID) == 40


def test_damage_hp_tracking_multiple_rounds():
    """Victim HP resets to 100 each round."""
    hurt_df = _make_hurt_df([
        {"round": 1, "tick": 100, "attacker_steamid": STEAM_ID,
         "attacker_team_num": 3, "user_steamid": "v1", "user_team_num": 2,
         "dmg_health": 30, "health": 70},
        {"round": 2, "tick": 200, "attacker_steamid": STEAM_ID,
         "attacker_team_num": 3, "user_steamid": "v1", "user_team_num": 2,
         "dmg_health": 50, "health": 50},
    ])
    assert _calculate_damage(hurt_df, STEAM_ID) == 80


def test_damage_excludes_team_damage():
    """Friendly fire should not count toward ADR."""
    hurt_df = _make_hurt_df([
        {"round": 1, "tick": 100, "attacker_steamid": STEAM_ID,
         "attacker_team_num": 3, "user_steamid": "teammate", "user_team_num": 3,
         "dmg_health": 50, "health": 50},
        {"round": 1, "tick": 110, "attacker_steamid": STEAM_ID,
         "attacker_team_num": 3, "user_steamid": "enemy", "user_team_num": 2,
         "dmg_health": 60, "health": 40},
    ])
    assert _calculate_damage(hurt_df, STEAM_ID) == 60


# ---------------------------------------------------------------------------
# _count_valid_assists
# ---------------------------------------------------------------------------


def test_valid_assists_requires_same_round_damage():
    """Assists without damage in the kill's round should be excluded."""
    death_df = _make_death_df([
        {"round": 2, "attacker_steamid": "killer", "assister_steamid": STEAM_ID,
         "user_steamid": "v1"},
    ])
    # pr1me damaged v1 in round 1 but NOT round 2
    hurt_df = _make_hurt_df([
        {"round": 1, "attacker_steamid": STEAM_ID, "user_steamid": "v1",
         "dmg_health": 50},
    ])
    assists_df = _filter_assister(death_df, STEAM_ID)
    assert _count_valid_assists(assists_df, hurt_df, STEAM_ID) == 0


def test_valid_assists_counts_same_round_damage():
    """Assists with damage in the kill's round should be counted."""
    death_df = _make_death_df([
        {"round": 2, "attacker_steamid": "killer", "assister_steamid": STEAM_ID,
         "user_steamid": "v1"},
    ])
    hurt_df = _make_hurt_df([
        {"round": 2, "attacker_steamid": STEAM_ID, "user_steamid": "v1",
         "dmg_health": 50},
    ])
    assists_df = _filter_assister(death_df, STEAM_ID)
    assert _count_valid_assists(assists_df, hurt_df, STEAM_ID) == 1


def test_valid_assists_empty():
    """No assists should return 0."""
    assert _count_valid_assists(pd.DataFrame(), pd.DataFrame(), STEAM_ID) == 0


# ---------------------------------------------------------------------------
# Trade detection in _build_round_stats
# ---------------------------------------------------------------------------


def test_trade_detection_within_5_seconds():
    """Player is traded if killer dies within 320 ticks."""
    deaths = [
        # Player dies at tick 1000
        {"round": 1, "tick": 1000, "attacker_steamid": "killer",
         "attacker_team_num": 2, "user_steamid": STEAM_ID,
         "user_team_num": 3, "assister_steamid": None},
        # Killer dies at tick 1200 (within 320 ticks)
        {"round": 1, "tick": 1200, "attacker_steamid": "teammate",
         "attacker_team_num": 3, "user_steamid": "killer",
         "user_team_num": 2, "assister_steamid": None},
    ]
    parsed = {
        "player_death": pd.DataFrame(deaths),
        "player_hurt": pd.DataFrame(),
        "round_end": _make_round_end_df(1),
        "header": {"map_name": "de_dust2"},
    }
    stats = calculate_match_stats(parsed, STEAM_ID)
    assert stats["round_stats"][0]["traded"] == 1


def test_no_trade_outside_5_seconds():
    """Player is NOT traded if killer dies after 320 ticks."""
    deaths = [
        {"round": 1, "tick": 1000, "attacker_steamid": "killer",
         "attacker_team_num": 2, "user_steamid": STEAM_ID,
         "user_team_num": 3, "assister_steamid": None},
        # Killer dies at tick 1500 (beyond 320 ticks)
        {"round": 1, "tick": 1500, "attacker_steamid": "teammate",
         "attacker_team_num": 3, "user_steamid": "killer",
         "user_team_num": 2, "assister_steamid": None},
    ]
    parsed = {
        "player_death": pd.DataFrame(deaths),
        "player_hurt": pd.DataFrame(),
        "round_end": _make_round_end_df(1),
        "header": {"map_name": "de_dust2"},
    }
    stats = calculate_match_stats(parsed, STEAM_ID)
    assert stats["round_stats"][0]["traded"] == 0
