"""
Tests for src/processor.py
These tests exercise the metric calculation logic without requiring a real
.dem file or demoparser2 installation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.domain.calibration import hltv_rating
from src.domain.metrics.aim import (
    _AIM_RATING_WEIGHTS,
    _CONFIDENCE_K,
    _median,
    _peek_bucket,
)
from src.domain.metrics.utility import (
    _UTILITY_RATING_WEIGHTS,
    _genuine_purchases,
)
from src.processor import (
    _analyze_movement,
    _analyze_reaction_time,
    _calculate_aim_stats,
    _calculate_damage,
    _calculate_kast_rounds,
    _collect_all_steam_ids,
    _count_multikill_rounds,
    _count_total_rounds,
    _count_valid_assists,
    _detect_player_team,
    _filter_assister,
    _filter_attacker,
    _filter_victim,
    _first_shot_tick,
    _get_round_damage_taken,
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
    rating = hltv_rating(kast=75.0, kpr=0.68, dpr=0.68, impact=0.96, adr=75.0)
    assert 0.9 < rating < 1.1, f"Expected ~1.0, got {rating}"


def test_hltv_rating_high_performer():
    rating = hltv_rating(kast=85.0, kpr=1.1, dpr=0.5, impact=1.8, adr=95.0)
    assert rating > 1.2, f"Expected rating > 1.2, got {rating}"


def test_hltv_rating_low_performer():
    rating = hltv_rating(kast=50.0, kpr=0.4, dpr=0.9, impact=0.3, adr=50.0)
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


# ---------------------------------------------------------------------------
# Movement / counter-strafe classification
# ---------------------------------------------------------------------------

ATTACKER_ID = 76561198012345678
VICTIM_ID = 76561198087654321


def _make_velocity_df(speeds_by_tick: dict[int, float], steamid: int = ATTACKER_ID):
    """Build a velocities-style DataFrame from {tick: speed} (moving along +X)."""
    return pd.DataFrame([
        {
            "steamid": steamid, "tick": t,
            "velocity_X": s, "velocity_Y": 0.0,
            "X": 0.0, "Y": 0.0, "Z": 0.0,
            "yaw": 0.0, "pitch": 0.0,
        }
        for t, s in sorted(speeds_by_tick.items())
    ])


def test_movement_standing_when_never_moved():
    """Zero speed throughout the window is a genuine static hold."""
    df = _make_velocity_df({t: 0.0 for t in range(68, 101)})
    result = _analyze_movement(df, ATTACKER_ID, 100)
    assert result["movement_quality"] == "standing"
    assert result["pre_speed"] == 0.0


def test_movement_running_when_fast_at_shot():
    df = _make_velocity_df({t: 150.0 for t in range(68, 101)})
    result = _analyze_movement(df, ATTACKER_ID, 100)
    assert result["movement_quality"] == "running"


def test_movement_counterstrafe_not_reported_as_standing():
    """A clean counter-strafe reads ~0 u/s at the shot, exactly like standing.

    This is the regression the classifier exists for: speed at the shot alone
    cannot tell the two apart, so the peak speed beforehand and how fast it
    collapsed have to decide it.
    """
    speeds = {t: 250.0 for t in range(68, 96)}
    # Active counter-strafe: ~250 u/s cancelled inside a handful of ticks.
    speeds.update({96: 200.0, 97: 120.0, 98: 60.0, 99: 20.0, 100: 0.0})
    result = _analyze_movement(_make_velocity_df(speeds), ATTACKER_ID, 100)

    assert result["movement_quality"] == "counter-strafed"
    assert result["movement_quality"] != "standing"
    assert result["shot_speed"] == 0.0        # indistinguishable from standing
    assert result["pre_speed"] == 250.0       # ...except for this
    assert result["stop_ticks"] == 3


def test_movement_coasting_to_a_halt_is_not_a_counterstrafe():
    """Releasing the keys and letting friction bleed speed off is 'stopped'.

    sv_friction 5.2 decays speed ~8%/tick at 64-tick, so a full stop takes
    roughly four times as long as counter-strafing it.
    """
    speeds = {}
    speed = 250.0
    for t in range(68, 101):
        speeds[t] = speed
        speed *= 0.91875
    result = _analyze_movement(_make_velocity_df(speeds), ATTACKER_ID, 100)

    assert result["movement_quality"] == "stopped"
    assert result["stop_ticks"] > 7


# ---------------------------------------------------------------------------
# Reaction time
# ---------------------------------------------------------------------------


def _make_aim_df(attacker_yaw_by_tick: dict[int, float]):
    """Attacker at the origin looking at yaw; victim parked 500 units along +X.

    A yaw of 0 therefore points exactly at the victim.
    """
    rows = []
    for t, yaw in sorted(attacker_yaw_by_tick.items()):
        rows.append({
            "steamid": ATTACKER_ID, "tick": t,
            "X": 0.0, "Y": 0.0, "Z": 0.0, "yaw": yaw, "pitch": 0.0,
            "velocity_X": 0.0, "velocity_Y": 0.0,
        })
        rows.append({
            "steamid": VICTIM_ID, "tick": t,
            "X": 500.0, "Y": 0.0, "Z": 0.0, "yaw": 0.0, "pitch": 0.0,
            "velocity_X": 0.0, "velocity_Y": 0.0,
        })
    return pd.DataFrame(rows)


def test_reaction_time_measures_a_genuine_flick():
    """Crosshair swings onto the target 5 ticks before the shot."""
    yaws = {t: 90.0 for t in range(36, 95)}   # looking well away
    yaws.update({t: 0.0 for t in range(95, 101)})  # snapped on target
    result = _analyze_reaction_time(_make_aim_df(yaws), ATTACKER_ID, VICTIM_ID, 100)

    assert result is not None
    assert result["reaction_ticks"] == 5
    assert result["reaction_ms"] == 78
    assert result["category"] == "lightning"


def test_reaction_time_none_when_pre_aimed_for_whole_window():
    """Never off target means there is no acquisition to measure."""
    yaws = {t: 0.0 for t in range(36, 101)}
    assert _analyze_reaction_time(_make_aim_df(yaws), ATTACKER_ID, VICTIM_ID, 100) is None


def test_reaction_time_none_when_sample_is_truncated():
    """A short sample must not be reported as a slow reaction.

    Only 21 ticks of history exist here.  Reporting the oldest sampled tick as
    the acquisition would invent a 312 ms 'slow' reaction for what is actually
    a perfectly pre-aimed kill — and the invented number would grow with the
    length of the engagement rather than with anything the player did.
    """
    yaws = {t: 0.0 for t in range(80, 101)}
    assert _analyze_reaction_time(_make_aim_df(yaws), ATTACKER_ID, VICTIM_ID, 100) is None


# ---------------------------------------------------------------------------
# First-shot anchoring
# ---------------------------------------------------------------------------


def test_first_shot_tick_finds_earliest_shot_in_window():
    fires = pd.DataFrame([
        {"round": 5, "user_steamid": STEAM_ID, "tick": 940},
        {"round": 5, "user_steamid": STEAM_ID, "tick": 990},
        {"round": 5, "user_steamid": STEAM_ID, "tick": 1000},
    ])
    assert _first_shot_tick(fires, STEAM_ID, 5, 1000) == 940


def test_first_shot_tick_ignores_other_rounds_and_stale_shots():
    fires = pd.DataFrame([
        {"round": 4, "user_steamid": STEAM_ID, "tick": 950},   # wrong round
        {"round": 5, "user_steamid": STEAM_ID, "tick": 800},   # before window
        {"round": 5, "user_steamid": "someone_else", "tick": 940},
        {"round": 5, "user_steamid": STEAM_ID, "tick": 985},
    ])
    assert _first_shot_tick(fires, STEAM_ID, 5, 1000) == 985


def test_first_shot_tick_falls_back_to_hit_tick():
    assert _first_shot_tick(None, STEAM_ID, 5, 1000) == 1000
    assert _first_shot_tick(pd.DataFrame(), STEAM_ID, 5, 1000) == 1000


def test_movement_walking_never_needed_a_stop():
    """Peak speed under the accuracy threshold means no stop was ever required.

    The standing/stopped boundary is the accuracy threshold itself: a player
    who only ever walked could shoot straight the whole time, so this is not a
    'stopped' encounter even though the speed did drop.
    """
    speeds = {t: 70.0 for t in range(68, 96)}
    speeds.update({t: 0.0 for t in range(96, 101)})
    result = _analyze_movement(_make_velocity_df(speeds), ATTACKER_ID, 100)

    assert result["movement_quality"] == "standing"
    assert result["stop_ticks"] is None
    assert result["pre_speed"] == 70.0


def test_movement_ignores_round_restart_teleport_samples():
    """Respawn teleports surface as five-figure velocities on a single tick.

    One left in the window would become pre_speed and mask what the player
    actually did, so implausible samples are dropped, not clamped.
    """
    speeds = {t: 0.0 for t in range(68, 101)}
    speeds[80] = 94758.5  # engine reporting a teleport as movement
    result = _analyze_movement(_make_velocity_df(speeds), ATTACKER_ID, 100)

    assert result is not None
    assert result["movement_quality"] == "standing"
    assert result["pre_speed"] == 0.0
    assert result["stop_ticks"] is None


def test_movement_none_when_the_shot_tick_itself_is_an_artifact():
    """If the sample at the shot is garbage, the real speed is unknown."""
    speeds = {t: 0.0 for t in range(68, 100)}
    speeds[100] = 94758.5
    assert _analyze_movement(_make_velocity_df(speeds), ATTACKER_ID, 100) is None


# ---------------------------------------------------------------------------
# Aim aggregation: estimators, sample size and rating weights
# ---------------------------------------------------------------------------


def _round_with(kills):
    return {"round": 1, "kills_detail": kills, "damage_encounters": []}


def _kill(*, preaim=None, movement=None, ttk=None, reaction=None):
    k = {"weapon": "AK-47"}
    if preaim is not None:
        k["preaim"] = {"crosshair_error": preaim, "preaim_quality": "good"}
    if movement is not None:
        k["movement"] = {
            "shot_speed": movement, "pre_speed": movement,
            "stop_ticks": None, "window_span": 32,
            "movement_quality": "standing", "movement_direction": "still",
        }
    if ttk is not None:
        k["ttd"] = {"ttk_seconds": ttk, "shots_fired": 4, "hits": 3}
    if reaction is not None:
        k["reaction"] = {"reaction_ticks": 10, "reaction_ms": reaction, "category": "average"}
    return k


def test_median_ignores_the_tail_a_mean_would_chase():
    assert _median([1.0, 2.0, 3.0, 4.0, 100.0]) == 3.0
    assert _median([1.0, 3.0]) == 2.0
    assert _median([]) is None


def test_aim_metrics_report_median_and_sample_size():
    """One wild engagement should not become the headline number."""
    kills = [_kill(preaim=e) for e in (4.0, 5.0, 6.0, 7.0, 90.0)]
    aim = _calculate_aim_stats([_round_with(kills)])

    pa = aim["preaim"]
    assert pa["median"] == 6.0
    assert pa["avg"] == 22.4      # what a mean would have reported
    assert pa["n"] == 5
    assert pa["confidence"] == "low"


def test_confidence_label_tracks_sample_size():
    for n, expected in ((3, "low"), (10, "medium"), (25, "high")):
        aim = _calculate_aim_stats([_round_with([_kill(preaim=5.0)] * n)])
        assert aim["preaim"]["confidence"] == expected, n


def test_long_engagements_excluded_from_time_to_kill():
    """Fights past a second stopped being about aim."""
    kills = [_kill(ttk=t) for t in (0.3, 0.4, 0.5, 2.5)]
    aim = _calculate_aim_stats([_round_with(kills)])

    assert aim["ttk"]["excluded_outliers"] == 1
    assert aim["ttk"]["n"] == 3
    assert aim["ttk"]["median"] == 0.4
    assert len(aim["ttk"]["outcomes"]) == 3  # kept parallel to the values


def test_reaction_time_never_reaches_the_rating():
    """Reaction is reported but is not a rating input.

    Two matches identical except for reaction time must rate the same, or a
    two-sample metric would be steering the score.
    """
    fast = _calculate_aim_stats([_round_with([_kill(preaim=5.0, reaction=120)] * 6)])
    slow = _calculate_aim_stats([_round_with([_kill(preaim=5.0, reaction=700)] * 6)])

    assert fast["reaction"]["median"] == 120
    assert slow["reaction"]["median"] == 700
    assert fast["reaction"]["diagnostic_only"] is True
    assert fast["aim_rating"] == slow["aim_rating"]
    assert "reaction" not in _AIM_RATING_WEIGHTS
    assert all(i["metric"] != "reaction" for i in fast["aim_rating_inputs"])


def test_missing_components_are_dropped_not_defaulted_to_fifty():
    """A metric that could not be measured must not contribute a placeholder.

    Pre-aim of 0 degrees scores 100. If the absent movement and ttk components
    were still filled in at 50 the rating would land well below 100.
    """
    aim = _calculate_aim_stats([_round_with([_kill(preaim=0.0)] * 30)])

    assert [i["metric"] for i in aim["aim_rating_inputs"]] == ["preaim"]
    assert aim["aim_rating"] == 100.0


def test_rating_is_none_when_nothing_was_measurable():
    aim = _calculate_aim_stats([_round_with([{"weapon": "AK-47"}])])
    assert aim["aim_rating"] is None
    assert aim["aim_rating_inputs"] == []


def test_small_samples_are_shrunk_toward_the_better_evidenced_component():
    """Weight follows evidence: n / (n + k).

    Pre-aim here is measured twice and movement thirty times, so movement must
    dominate despite carrying the smaller nominal weight.
    """
    kills = [_kill(preaim=0.0, movement=200.0) for _ in range(2)]
    kills += [_kill(movement=200.0) for _ in range(28)]
    aim = _calculate_aim_stats([_round_with(kills)])

    by_metric = {i["metric"]: i for i in aim["aim_rating_inputs"]}
    assert by_metric["preaim"]["n"] == 2
    assert by_metric["movement"]["n"] == 30
    assert by_metric["movement"]["weight_share"] > by_metric["preaim"]["weight_share"]

    expected_preaim = _AIM_RATING_WEIGHTS["preaim"] * (2 / (2 + _CONFIDENCE_K))
    assert by_metric["preaim"]["weight"] == round(expected_preaim, 4)


# ---------------------------------------------------------------------------
# Counter-strafe rate gating
# ---------------------------------------------------------------------------


def _mov_kill(weapon, quality, *, crouched=False, peek=200.0):
    return {
        "weapon": weapon,
        "movement": {
            "shot_speed": 0.0, "pre_speed": peek, "stop_ticks": 3,
            "window_span": 32, "crouched": crouched,
            "movement_quality": quality, "movement_direction": "still",
        },
    }


def test_counterstrafe_rate_scores_only_rifle_stops():
    """SMGs and pistols are fired on the move by design.

    Grading them on counter-strafe quality measures the weapon, not the
    player, so they must not reach the denominator.
    """
    kills = [
        _mov_kill("AK-47", "counter-strafed"),
        _mov_kill("AK-47", "stopped"),
        _mov_kill("MP9", "stopped"),          # SMG, excluded
        _mov_kill("Desert Eagle", "stopped"),  # pistol, excluded
        _mov_kill("AWP", "stopped"),           # sniper, excluded
    ]
    mv = _calculate_aim_stats([_round_with(kills)])["movement"]

    assert mv["counterstrafe_attempts"] == 2
    assert mv["counterstrafe_good"] == 1
    assert mv["counterstrafe_rate"] == 50.0


def test_counterstrafe_rate_excludes_crouched_shots():
    """Crouching caps speed on its own — no counter-strafe happened."""
    kills = [
        _mov_kill("AK-47", "counter-strafed"),
        _mov_kill("AK-47", "stopped", crouched=True),
    ]
    mv = _calculate_aim_stats([_round_with(kills)])["movement"]

    assert mv["counterstrafe_attempts"] == 1
    assert mv["counterstrafe_rate"] == 100.0
    assert mv["crouched_pct"] == 50.0


def test_counterstrafe_rate_none_without_qualifying_engagements():
    """Running and standing shots say nothing about stopping ability."""
    kills = [_mov_kill("AK-47", "running"), _mov_kill("AK-47", "standing")]
    mv = _calculate_aim_stats([_round_with(kills)])["movement"]

    assert mv["counterstrafe_attempts"] == 0
    assert mv["counterstrafe_rate"] is None


def test_movement_reports_crouch_state():
    df = _make_velocity_df({t: 0.0 for t in range(68, 101)})
    df["ducked"] = True
    result = _analyze_movement(df, ATTACKER_ID, 100)
    assert result["crouched"] is True


# ---------------------------------------------------------------------------
# Peek speed
# ---------------------------------------------------------------------------


def test_peek_speed_measures_the_run_up_not_the_shot():
    """The speed carried into the duel, which the shot itself no longer shows.

    A clean counter-strafe fires at 0 u/s whether the player walked out or came
    in at full sprint; only the peak beforehand separates the two.
    """
    speeds = {t: 250.0 for t in range(68, 96)}
    speeds.update({96: 200.0, 97: 120.0, 98: 60.0, 99: 20.0, 100: 0.0})
    aim = _calculate_aim_stats([_round_with([{
        "weapon": "AK-47",
        "movement": _analyze_movement(_make_velocity_df(speeds), ATTACKER_ID, 100),
    }])])

    assert aim["peek"]["median"] == 250.0
    assert aim["movement"]["median"] == 0.0     # what the shot alone reports
    assert aim["peek"]["n"] == 1


def test_peek_speed_separates_held_angles_from_full_speed_peeks():
    kills = [
        _mov_kill("AK-47", "standing", peek=20.0),        # held the angle
        _mov_kill("AK-47", "standing", peek=60.0),        # a shuffle, no stop needed
        _mov_kill("AK-47", "counter-strafed", peek=140.0),
        _mov_kill("AK-47", "counter-strafed", peek=220.0),  # full-speed peek
    ]
    peek = _calculate_aim_stats([_round_with(kills)])["peek"]

    assert peek["held_pct"] == 50.0
    assert peek["full_pct"] == 25.0
    assert peek["max"] == 220.0


def test_peek_distribution_covers_the_same_regions_the_chart_shades():
    """The legend beside the chart names the bands drawn on it.

    Every engagement lands in exactly one region, the regions are the ones the
    axis declares, and the two headline percentages are read off those same
    counts rather than measured a second way — otherwise a legend could total
    something other than 100% of the sample it sits under.
    """
    from src.domain.metrics.aim import _AIM_THRESHOLDS

    kills = (
        [_mov_kill("AK-47", "standing", peek=20.0)] * 5          # held
        + [_mov_kill("AK-47", "counter-strafed", peek=100.0)] * 4   # walk
        + [_mov_kill("AK-47", "counter-strafed", peek=150.0)] * 9   # half
        + [_mov_kill("AK-47", "stopped", peek=200.0)] * 12          # full
    )
    peek = _calculate_aim_stats([_round_with(kills)])["peek"]
    by_zone = peek["by_zone"]

    assert [z["label"] for z in by_zone] == [
        z["label"] for z in _AIM_THRESHOLDS["peek"]["zones"]
    ]
    assert [z["n"] for z in by_zone] == [5, 4, 9, 12]
    assert sum(z["n"] for z in by_zone) == peek["n"] == 30
    assert sum(z["pct"] for z in by_zone) == 100.0
    assert peek["held_pct"] == by_zone[0]["pct"]
    assert peek["full_pct"] == by_zone[-1]["pct"]


def test_peek_speed_is_reported_but_never_graded():
    """There is no good or bad peek speed without knowing the intent.

    Walking out is right when holding an angle and wrong when entering a site,
    and the demo does not say which the player meant.  So it ships without
    bands and must not move the rating: two matches identical apart from how
    fast the player entered every duel have to score the same.
    """
    from src.domain.metrics.aim import _AIM_RATING_WEIGHTS, _AIM_THRESHOLDS

    fast = _calculate_aim_stats([_round_with(
        [_mov_kill("AK-47", "counter-strafed", peek=240.0)] * 6
    )])
    slow = _calculate_aim_stats([_round_with(
        [_mov_kill("AK-47", "counter-strafed", peek=100.0)] * 6
    )])

    assert fast["peek"]["median"] == 240.0
    assert slow["peek"]["median"] == 100.0
    assert fast["peek"]["diagnostic_only"] is True
    assert fast["aim_rating"] == slow["aim_rating"]
    assert "peek" not in _AIM_RATING_WEIGHTS
    assert _AIM_THRESHOLDS["peek"]["bounds"] == []


def test_counterstrafe_rate_split_by_peek_speed():
    """The pooled rate hides the case worth acting on.

    Stopping cleanly off a walk and coasting off every full-speed peek averages
    out to the same 50% as being uniformly inconsistent, but only the first has
    a specific thing to practise.
    """
    kills = [
        _mov_kill("AK-47", "counter-strafed", peek=100.0),
        _mov_kill("AK-47", "counter-strafed", peek=120.0),
        _mov_kill("AK-47", "stopped", peek=210.0),
        _mov_kill("AK-47", "stopped", peek=230.0),
    ]
    mv = _calculate_aim_stats([_round_with(kills)])["movement"]
    by_bucket = {b["bucket"]: b for b in mv["counterstrafe_by_peek"]}

    assert mv["counterstrafe_rate"] == 50.0     # the number that hides it
    assert by_bucket["walk"]["rate"] == 100.0
    assert by_bucket["full"]["rate"] == 0.0
    assert by_bucket["full"]["attempts"] == 2
    assert by_bucket["half"]["attempts"] == 0
    assert by_bucket["half"]["rate"] is None    # not zero: nothing was measured


def test_peek_buckets_exclude_engagements_that_needed_no_stop():
    """Below the accuracy threshold nothing had to be cancelled.

    Those are held angles rather than peeks, and the counter-strafe
    classifier never calls them a stop either, so no bucket may claim them.
    """
    from src.domain.metrics.aim import _PEEK_BUCKETS
    from src.processor import _ACCURATE_SPEED

    assert _peek_bucket(_ACCURATE_SPEED - 1) is None
    assert _peek_bucket(_ACCURATE_SPEED) == "walk"
    # The floor is written as a literal because _ACCURATE_SPEED is defined
    # further down the module; this is what keeps the two in step.
    assert _PEEK_BUCKETS[0][2] == _ACCURATE_SPEED


def test_peek_zones_match_the_counterstrafe_buckets():
    """The chart regions and the breakdown rows must mean the same thing.

    A point sitting in the "Full speed" region of the peek axis has to be one
    of the engagements counted in the full-speed row underneath it, so both
    read from one table rather than two sets of numbers that can drift apart.
    """
    from src.domain.metrics.aim import _AIM_THRESHOLDS, _PEEK_BUCKETS

    zones = _AIM_THRESHOLDS["peek"]["zones"]

    # One region below the buckets for the held angles that never needed a stop.
    assert zones[0] == {"at": 0.0, "label": "Held"}
    assert [(z["at"], z["label"]) for z in zones[1:]] == [
        (lo, label) for _key, label, lo, _hi in _PEEK_BUCKETS
    ]
    # Regions are not tiers: the axis stays ungraded.
    assert _AIM_THRESHOLDS["peek"]["bounds"] == []
    assert _AIM_THRESHOLDS["peek"]["range"] == [0, 250]


def test_peek_speed_available_as_a_scatter_axis():
    """Peek speed against counter-strafe is the comparison it exists for."""
    kills = [_mov_kill("AK-47", "counter-strafed", peek=180.0) for _ in range(3)]
    aim = _calculate_aim_stats([_round_with(kills)])

    plotted = [(e["peek"], e["stop_ticks"]) for e in aim["encounters"] if "peek" in e]
    assert plotted == [(180.0, 3)] * 3
    assert "peek" in aim["thresholds"]


# ---------------------------------------------------------------------------
# Damage absorbed
# ---------------------------------------------------------------------------


def _hurt_rows(rows, with_armor=True):
    cols = ["round", "user_steamid", "attacker_steamid", "dmg_health"]
    if with_armor:
        cols.append("dmg_armor")
    return pd.DataFrame(rows, columns=cols)


def test_damage_taken_counts_only_what_the_player_absorbed():
    """The player as victim, never as attacker.

    Reading the wrong side of player_hurt would mark a vest as used in exactly
    the rounds the player was winning fights untouched.
    """
    df = _hurt_rows([
        {"round": 5, "user_steamid": STEAM_ID, "attacker_steamid": "999", "dmg_health": 27, "dmg_armor": 6},
        {"round": 5, "user_steamid": STEAM_ID, "attacker_steamid": "999", "dmg_health": 15, "dmg_armor": 3},
        {"round": 5, "user_steamid": "999", "attacker_steamid": STEAM_ID, "dmg_health": 80, "dmg_armor": 20},
        {"round": 6, "user_steamid": STEAM_ID, "attacker_steamid": "999", "dmg_health": 50, "dmg_armor": 10},
    ])
    assert _get_round_damage_taken(df, STEAM_ID, 5) == {"health": 42, "armor": 9}


def test_damage_taken_reports_an_untouched_round_as_zero():
    df = _hurt_rows([
        {"round": 5, "user_steamid": STEAM_ID, "attacker_steamid": "999", "dmg_health": 27, "dmg_armor": 6},
    ])
    assert _get_round_damage_taken(df, STEAM_ID, 7) == {"health": 0, "armor": 0}


def test_damage_taken_leaves_armour_unknown_when_the_demo_omits_it():
    """None, not 0.

    A vest that absorbed nothing and a demo that cannot say are different
    things, and only the first should draw as a wasted $650.
    """
    df = _hurt_rows([
        {"round": 5, "user_steamid": STEAM_ID, "attacker_steamid": "999", "dmg_health": 27},
    ], with_armor=False)
    taken = _get_round_damage_taken(df, STEAM_ID, 5)

    assert taken["health"] == 27
    assert taken["armor"] is None
    assert _get_round_damage_taken(df, STEAM_ID, 9)["armor"] is None
    assert _get_round_damage_taken(pd.DataFrame(), STEAM_ID, 5)["armor"] is None


# ---------------------------------------------------------------------------
# Benchmark metadata
# ---------------------------------------------------------------------------


def test_benchmarks_declare_they_are_uncalibrated():
    """Tiers come from hand-set constants, and must say so.

    Nothing here has been fitted against a population of real players, so a
    consumer has to be able to tell a heuristic band from a measured one.
    """
    from src.domain.metrics.benchmarks import compute_benchmarks

    aim = _calculate_aim_stats([_round_with([_kill(preaim=4.0, ttk=0.3)] * 10)])
    marks = compute_benchmarks(aim, {}, total_rounds=24, map_name="de_dust2")

    assert marks
    for key, entry in marks.items():
        assert entry["calibration"] == "heuristic", key
        assert "tier" in entry


def test_benchmarks_carry_their_own_sample_size():
    """Counter-strafe is graded on far fewer engagements than movement.

    It must report its own n rather than inherit the movement sample's, or a
    four-engagement rate would look as solid as a fifty-engagement one.
    """
    from src.domain.metrics.benchmarks import compute_benchmarks

    kills = [_mov_kill("AK-47", "counter-strafed") for _ in range(3)]
    kills += [_mov_kill("MP9", "stopped") for _ in range(20)]
    aim = _calculate_aim_stats([_round_with(kills)])
    marks = compute_benchmarks(aim, {}, total_rounds=24, map_name="de_dust2")

    assert marks["counterstrafe"]["n"] == 3
    assert marks["counterstrafe"]["confidence"] == "low"
    assert marks["shot_speed"]["n"] == 23


# ---------------------------------------------------------------------------
# Utility rating
# ---------------------------------------------------------------------------


def _utility_parsed(blinds=(), purchases=(), fires=()):
    """Minimal parsed_data for _calculate_utility_stats."""
    return {
        "player_blind": pd.DataFrame(list(blinds)),
        "player_hurt": pd.DataFrame(),
        "item_purchase": pd.DataFrame(list(purchases)),
        "weapon_fire": pd.DataFrame(list(fires)),
        "smoke_detonate": pd.DataFrame(),
        "molotov_detonate": pd.DataFrame(),
        "player_death": pd.DataFrame(),
    }


def _blind(dur, *, friendly=False):
    return {
        "round": 1, "attacker_steamid": STEAM_ID, "user_steamid": "enemy1",
        "attacker_team_num": 3, "user_team_num": 3 if friendly else 2,
        "blind_duration": dur, "user_name": "Enemy",
    }


def _flash_buy():
    return {"round": 1, "steamid": STEAM_ID, "item_name": "flashbang"}


def _flash_fire():
    return {"round": 1, "user_steamid": STEAM_ID, "weapon": "weapon_flashbang", "tick": 100}


def _utility(blinds, n_flashes):
    from src.domain.metrics.utility import _calculate_utility_stats

    return _calculate_utility_stats(
        [],
        _utility_parsed(
            blinds=blinds,
            purchases=[_flash_buy() for _ in range(n_flashes)],
            fires=[_flash_fire() for _ in range(n_flashes)],
        ),
        STEAM_ID, total_rounds=24, map_name="de_dust2",
    )


def test_flash_quality_tracks_blind_time_not_head_count():
    """Four glances must not score like four real flashes.

    Both players flashed four enemies with two flashbangs; only one actually
    took anyone out of the fight.
    """
    glances = _utility([_blind(0.3) for _ in range(4)], n_flashes=2)
    real = _utility([_blind(3.0) for _ in range(4)], n_flashes=2)

    assert glances["flash"]["enemies_flashed"] == real["flash"]["enemies_flashed"]
    assert glances["flash"]["effective_flashes"] == 0
    assert real["flash"]["effective_flashes"] == 4
    assert real["utility_rating"] > glances["utility_rating"]


def test_smoke_placement_no_longer_scored():
    """It measured callout-map completeness, not smoke quality."""

    assert "smoke" not in _UTILITY_RATING_WEIGHTS
    result = _utility([_blind(2.0)], n_flashes=1)
    assert all(i["metric"] != "smoke" for i in result["utility_rating_inputs"])


def test_utility_rating_none_when_no_utility_used():
    result = _utility([], n_flashes=0)
    assert result["utility_rating"] is None
    assert result["utility_rating_inputs"] == []


def test_utility_rating_weights_by_evidence():
    """One flashbang cannot carry the same weight as a dozen."""
    few = _utility([_blind(3.0)], n_flashes=1)
    many = _utility([_blind(3.0) for _ in range(20)], n_flashes=20)

    w_few = {i["metric"]: i for i in few["utility_rating_inputs"]}["flash"]
    w_many = {i["metric"]: i for i in many["utility_rating_inputs"]}["flash"]
    assert w_many["weight"] > w_few["weight"]


# ---------------------------------------------------------------------------
# Round swing / impact
# ---------------------------------------------------------------------------






def _swing_parsed(deaths, plants=()):
    return {
        "player_death": pd.DataFrame(deaths),
        "bomb_planted": pd.DataFrame(list(plants)),
    }


def _swing_death(tick, attacker, victim, victim_team, rnd=1):
    return {
        "round": rnd, "tick": tick,
        "attacker_steamid": attacker, "user_steamid": victim,
        "user_team_num": victim_team, "attacker_team_num": 3 if victim_team == 2 else 2,
    }


def test_opening_kill_swings_more_than_a_mop_up():
    """5v5 -> 5v4 moves the round; 5v1 -> 5v0 barely does.

    Counting kills cannot express that difference, which is the whole reason
    this metric exists.
    """
    from src.domain.metrics.impact import _calculate_impact_stats

    opening = _swing_parsed([_swing_death(100, STEAM_ID, "e1", 2)])
    early = _calculate_impact_stats(opening, STEAM_ID, 1, {1: "CT"})

    mop_up = _swing_parsed([
        _swing_death(100, "mate1", "e1", 2),
        _swing_death(200, "mate2", "e2", 2),
        _swing_death(300, "mate3", "e3", 2),
        _swing_death(400, "mate4", "e4", 2),
        _swing_death(500, STEAM_ID, "e5", 2),
    ])
    late = _calculate_impact_stats(mop_up, STEAM_ID, 1, {1: "CT"})

    assert early["kill_swing_total"] > late["kill_swing_total"]
    assert late["kills_scored"] == 1


def test_dying_is_scored_as_negative_swing():
    from src.domain.metrics.impact import _calculate_impact_stats

    parsed = _swing_parsed([_swing_death(100, "enemy", STEAM_ID, 3)])
    out = _calculate_impact_stats(parsed, STEAM_ID, 1, {1: "CT"})

    assert out["deaths_scored"] == 1
    assert out["death_swing_total"] < 0
    assert out["net_swing_total"] < 0


def test_swing_is_measured_from_the_players_own_side():
    """The same event is a gain for one team and a loss for the other."""
    from src.domain.metrics.impact import _calculate_impact_stats

    parsed = _swing_parsed([_swing_death(100, STEAM_ID, "e1", 2)])
    as_ct = _calculate_impact_stats(parsed, STEAM_ID, 1, {1: "CT"})

    parsed_t = _swing_parsed([_swing_death(100, STEAM_ID, "e1", 3)])
    as_t = _calculate_impact_stats(parsed_t, STEAM_ID, 1, {1: "T"})

    assert as_ct["kill_swing_total"] > 0
    assert as_t["kill_swing_total"] > 0


def test_impact_empty_without_the_columns_to_reconstruct_a_round():
    from src.domain.metrics.impact import _calculate_impact_stats

    assert _calculate_impact_stats({"player_death": pd.DataFrame()}, STEAM_ID, 1, {}) == {}
    no_tick = pd.DataFrame([{"round": 1, "user_team_num": 2, "user_steamid": "e"}])
    assert _calculate_impact_stats({"player_death": no_tick}, STEAM_ID, 1, {1: "CT"}) == {}


# ---------------------------------------------------------------------------
# Threshold consistency
# ---------------------------------------------------------------------------


def test_one_set_of_bands_drives_buckets_and_benchmarks():
    """The same value must not be graded differently in two places.

    Crosshair placement used to be bucketed at 5/10/20 but benchmarked at
    3/10/25, so a 4-degree median read "excellent" on one card and merely
    "strong" on the badge beside it.
    """
    from src.domain.metrics.aim import _AIM_THRESHOLDS
    from src.domain.metrics.benchmarks import compute_benchmarks
    from src.processor import _analyze_preaim

    exc, good, moderate = _AIM_THRESHOLDS["preaim"]["bounds"]

    # The classifier buckets against those bounds...
    def quality_at(yaw_deg):
        df = _make_aim_df({t: yaw_deg for t in range(60, 101)})
        return _analyze_preaim(df, ATTACKER_ID, VICTIM_ID, 100)["preaim_quality"]

    assert quality_at(exc - 1) == "excellent"
    assert quality_at(exc + 1) == "good"
    assert quality_at(good + 1) == "moderate"
    assert quality_at(moderate + 1) == "poor"

    # ...and the benchmark grades against the very same numbers.
    top = {"preaim": {"median": exc - 1, "n": 10, "confidence": "medium"}}
    assert compute_benchmarks(top, {}, 24, "de_dust2")["preaim_offset"]["tier"] == "pro"
    bottom = {"preaim": {"median": moderate + 1, "n": 10, "confidence": "medium"}}
    assert compute_benchmarks(bottom, {}, 24, "de_dust2")["preaim_offset"]["tier"] == "below_average"


def test_counterstrafe_band_matches_the_classifier():
    """The scatter band and the classifier have to agree on what a stop is."""
    from src.domain.metrics.aim import _AIM_THRESHOLDS
    from src.processor import _COUNTERSTRAFE_MAX_TICKS

    assert _AIM_THRESHOLDS["stop_ticks"]["bounds"][1] == _COUNTERSTRAFE_MAX_TICKS


def test_thresholds_are_shipped_to_the_frontend():
    """The scatter reads these rather than keeping a third copy.

    A metric carries either three bounds or none at all: none marks it as
    ungraded, which the charts render as a plain axis with no tier labels.
    Anything in between would leave a chart guessing how many bands to draw.
    """
    aim = _calculate_aim_stats([_round_with([_kill(preaim=5.0)])])
    assert set(aim["thresholds"]) >= {
        "movement", "peek", "preaim", "ttk", "reaction", "stop_ticks",
    }
    for name, meta in aim["thresholds"].items():
        assert len(meta["bounds"]) in (0, 3), name
        assert "label" in meta and "unit" in meta


def test_scatter_points_agree_with_the_aggregates():
    """A point the median discarded must not still be plotted."""
    kills = [_kill(ttk=t) for t in (0.3, 0.4, 0.5, 2.5)]
    aim = _calculate_aim_stats([_round_with(kills)])

    plotted = [e["ttk"] for e in aim["encounters"] if "ttk" in e]
    assert len(plotted) == aim["ttk"]["n"] == 3
    assert max(plotted) < 1.0


def test_counterstrafe_available_as_a_scatter_axis():
    """Stop time is the per-encounter counter-strafe measurement."""
    kills = [_mov_kill("AK-47", "counter-strafed") for _ in range(3)]
    aim = _calculate_aim_stats([_round_with(kills)])

    stops = [e["stop_ticks"] for e in aim["encounters"] if "stop_ticks" in e]
    assert len(stops) == 3
    assert "stop_ticks" in aim["thresholds"]


# ---------------------------------------------------------------------------
# Damage-only engagements
# ---------------------------------------------------------------------------


def test_separate_fights_with_one_enemy_are_separate_encounters():
    """Hit someone, meet them again twenty seconds later, hit them again.

    Grouping only by victim collapsed that into a single encounter measured at
    the first shot of the first fight. Two exchanges is two duels.
    """
    from src.processor import _get_round_damage_encounters

    hurt = pd.DataFrame([
        {"round": 1, "tick": 1000, "attacker_steamid": STEAM_ID, "user_steamid": "enemy", "weapon": "ak47", "hitgroup": "chest"},
        {"round": 1, "tick": 1030, "attacker_steamid": STEAM_ID, "user_steamid": "enemy", "weapon": "ak47", "hitgroup": "chest"},
        # ~20 s later at 64 tick
        {"round": 1, "tick": 2300, "attacker_steamid": STEAM_ID, "user_steamid": "enemy", "weapon": "ak47", "hitgroup": "head"},
    ])
    out = _get_round_damage_encounters(hurt, pd.DataFrame(), STEAM_ID, 1)

    assert len(out) == 2
    assert out[0]["hits"] == 2
    assert out[1]["hits"] == 1


def test_damage_only_engagements_report_accuracy():
    """Whether the duel ended in a kill has no bearing on hit rate."""
    from src.processor import _get_round_damage_encounters

    hurt = pd.DataFrame([
        {"round": 1, "tick": 1000, "attacker_steamid": STEAM_ID, "user_steamid": "enemy", "weapon": "ak47", "hitgroup": "head"},
        {"round": 1, "tick": 1020, "attacker_steamid": STEAM_ID, "user_steamid": "enemy", "weapon": "ak47", "hitgroup": "chest"},
    ])
    fires = pd.DataFrame([
        {"round": 1, "tick": t, "user_steamid": STEAM_ID, "weapon": "weapon_ak47"}
        for t in (990, 1000, 1010, 1020)
    ])
    out = _get_round_damage_encounters(
        hurt, pd.DataFrame(), STEAM_ID, 1, weapon_fire_df=fires,
    )

    assert len(out) == 1
    acc = out[0]["accuracy"]
    assert acc["hit_pct"] == 50.0     # 2 hits of 4 shots
    assert acc["head"] == 1
    assert acc["upper"] == 1


def test_the_exchange_that_produced_the_kill_is_not_double_counted():
    """The final cluster against a victim you killed belongs to the kill path."""
    from src.processor import _get_round_damage_encounters

    hurt = pd.DataFrame([
        {"round": 1, "tick": 1000, "attacker_steamid": STEAM_ID, "user_steamid": "enemy", "weapon": "ak47", "hitgroup": "chest"},
        {"round": 1, "tick": 2300, "attacker_steamid": STEAM_ID, "user_steamid": "enemy", "weapon": "ak47", "hitgroup": "head"},
    ])
    deaths = pd.DataFrame([
        {"round": 1, "tick": 2310, "attacker_steamid": STEAM_ID, "user_steamid": "enemy"},
    ])
    out = _get_round_damage_encounters(hurt, deaths, STEAM_ID, 1)

    # The early poke survives as its own engagement; the fatal exchange does not.
    assert len(out) == 1
    assert out[0]["last_tick"] == 1000


def test_accuracy_aggregate_includes_lost_and_inconclusive_duels():
    """Accuracy measured only on kills describes the fights that went well."""
    won = _kill(ttk=0.4)
    won["ttd"]["accuracy"] = {
        "hit_pct": 80.0, "first_bullet_hit": True,
        "hitgroups": ["head"], "head": 1, "upper": 0, "lower": 0,
    }
    dmg = [{
        "weapon": "AK-47", "victim_sid": "enemy", "last_tick": 900,
        "accuracy": {"hit_pct": 20.0, "first_bullet_hit": False,
                     "hitgroups": ["chest"], "head": 0, "upper": 1, "lower": 0},
    }]
    aim = _calculate_aim_stats([
        {"round": 1, "kills_detail": [won], "damage_encounters": dmg},
    ])

    assert aim["accuracy"]["n"] == 2
    assert sorted(aim["accuracy"]["outcomes"]) == ["damage", "kill"]
    # The lost duel drags the median off the won one, which is the point.
    assert aim["accuracy"]["median"] == 50.0


def test_reaction_aggregate_includes_damage_only_duels():
    dmg = [{
        "weapon": "AK-47", "victim_sid": "enemy", "last_tick": 900,
        "reaction": {"reaction_ticks": 10, "reaction_ms": 156, "category": "fast"},
    }]
    aim = _calculate_aim_stats([
        {"round": 1, "kills_detail": [], "damage_encounters": dmg},
    ])

    assert aim["reaction"]["n"] == 1
    assert aim["reaction"]["outcomes"] == ["damage"]


def test_accuracy_pools_bullets_rather_than_averaging_engagements():
    """A one-bullet exchange must not outvote a full spray.

    Averaging per-engagement percentages treats 1/1 and 10/30 as two equal
    observations, which pushes the figure well above the rate actually shot.
    """
    def enc(hits, shots):
        return {
            "weapon": "AK-47", "victim_sid": "e", "last_tick": 900,
            "hits": hits, "shots_fired": shots,
            "accuracy": {"hit_pct": round(hits / shots * 100, 1),
                         "first_bullet_hit": True, "hitgroups": [],
                         "head": 0, "upper": 0, "lower": 0},
        }

    dmg = [enc(1, 1), enc(1, 1), enc(10, 30)]
    aim = _calculate_aim_stats([
        {"round": 1, "kills_detail": [], "damage_encounters": dmg},
    ])
    acc = aim["accuracy"]

    assert acc["total_hits"] == 12
    assert acc["total_shots"] == 32
    assert acc["pooled_pct"] == 37.5
    assert acc["median"] == 100.0        # what averaging would have reported
    assert acc["pooled_pct"] < acc["median"]


# ---------------------------------------------------------------------------
# Whiffed engagements
# ---------------------------------------------------------------------------


def _fires(ticks, sid=STEAM_ID, weapon="weapon_ak47", rnd=1):
    return pd.DataFrame([
        {"round": rnd, "tick": t, "user_steamid": sid, "weapon": weapon} for t in ticks
    ])


def _spotted(rows):
    return pd.DataFrame(rows)


def _seen(tick, team_num=2, spotted=True):
    return {"tick": tick, "steamid": "enemy", "team_num": team_num, "spotted": spotted}


def test_burst_that_hits_nothing_counts_against_accuracy():
    """A duel lost without connecting used not to exist at all."""
    from src.processor import _get_round_whiffed_engagements

    out = _get_round_whiffed_engagements(
        _fires([1000, 1005, 1010, 1015]), pd.DataFrame(),
        _spotted([_seen(1000)]), STEAM_ID, 1, player_team=3,
    )
    assert len(out) == 1
    assert out[0]["shots_fired"] == 4
    assert out[0]["hits"] == 0
    assert out[0]["accuracy"]["hit_pct"] == 0.0


def test_spraying_with_no_enemy_visible_is_not_an_aim_duel():
    """Smoke spray, wallbangs and pre-fire must not be charged to aim.

    Around 60% of bursts that hit nothing are fired with nobody on screen.
    """
    from src.processor import _get_round_whiffed_engagements

    out = _get_round_whiffed_engagements(
        _fires([1000, 1005, 1010]), pd.DataFrame(),
        _spotted([_seen(1000, spotted=False)]), STEAM_ID, 1, player_team=3,
    )
    assert out == []


def test_only_enemies_count_as_visible():
    """A spotted teammate is not something to shoot at."""
    from src.processor import _get_round_whiffed_engagements

    out = _get_round_whiffed_engagements(
        _fires([1000, 1005]), pd.DataFrame(),
        _spotted([_seen(1000, team_num=3, spotted=True)]),  # own team
        STEAM_ID, 1, player_team=3,
    )
    assert out == []


def test_a_burst_that_landed_is_left_to_the_damage_path():
    """No double counting: hits mean the engagement is already measured."""
    from src.processor import _get_round_whiffed_engagements

    hurt = pd.DataFrame([
        {"round": 1, "tick": 1012, "attacker_steamid": STEAM_ID,
         "user_steamid": "enemy", "weapon": "ak47", "hitgroup": "chest"},
    ])
    out = _get_round_whiffed_engagements(
        _fires([1000, 1005, 1010]), hurt, _spotted([_seen(1000)]),
        STEAM_ID, 1, player_team=3,
    )
    assert out == []


def test_grenades_are_not_aim():
    from src.processor import _get_round_whiffed_engagements

    out = _get_round_whiffed_engagements(
        _fires([1000, 1100], weapon="weapon_flashbang"), pd.DataFrame(),
        _spotted([_seen(1000), _seen(1100)]), STEAM_ID, 1, player_team=3,
    )
    assert out == []


def test_whiffed_bullets_reach_the_pooled_accuracy():
    """The whole point: the denominator grows, the rate falls."""
    landed = [{
        "weapon": "AK-47", "victim_sid": "e", "last_tick": 900,
        "hits": 3, "shots_fired": 5,
        "accuracy": {"hit_pct": 60.0, "first_bullet_hit": True,
                     "hitgroups": [], "head": 0, "upper": 0, "lower": 0},
    }]
    without = _calculate_aim_stats([
        {"round": 1, "kills_detail": [], "damage_encounters": landed,
         "whiffed_engagements": []},
    ])
    with_whiff = _calculate_aim_stats([
        {"round": 1, "kills_detail": [], "damage_encounters": landed,
         "whiffed_engagements": [{"shots_fired": 5, "hits": 0}]},
    ])

    assert without["accuracy"]["pooled_pct"] == 60.0
    assert with_whiff["accuracy"]["total_shots"] == 10
    assert with_whiff["accuracy"]["pooled_pct"] == 30.0
    assert "whiff" in with_whiff["accuracy"]["outcomes"]


def test_one_tap_kills_count_as_engagement_time():
    """An instant kill is the fastest engagement, not a missing one.

    Requiring a non-zero duration dropped exactly the kills where the first
    bullet ended it, which pulled the median upward.
    """
    kills = [_kill(ttk=0.0), _kill(ttk=0.4), _kill(ttk=0.6)]
    aim = _calculate_aim_stats([_round_with(kills)])

    assert aim["ttk"]["n"] == 3
    assert aim["ttk"]["min"] == 0.0
    assert aim["ttk"]["median"] == 0.4


def test_every_kill_is_accounted_for_in_engagement_time():
    """Kept plus excluded must equal the kills, so the count is explainable."""
    kills = [_kill(ttk=t) for t in (0.0, 0.2, 0.5, 1.4, 2.0)]
    aim = _calculate_aim_stats([_round_with(kills)])

    assert aim["ttk"]["n"] + aim["ttk"]["excluded_outliers"] == len(kills)
    assert aim["ttk"]["n"] == 3
    assert aim["ttk"]["excluded_outliers"] == 2


# ---------------------------------------------------------------------------
# Non-bullet damage
# ---------------------------------------------------------------------------


def test_grenade_and_fire_damage_are_not_gun_accuracy():
    """A molotov ticking is not the player hitting their shots.

    player_hurt carries fire and blast damage with a 'generic' hitgroup. Those
    rows were counted as hits while the grenade was never counted as a shot,
    which is how engagements came out at 100%.
    """
    from src.processor import _get_round_damage_encounters

    hurt = pd.DataFrame([
        {"round": 1, "tick": 1000, "attacker_steamid": STEAM_ID, "user_steamid": "e",
         "weapon": "ak47", "hitgroup": "chest"},
        # Eight molotov ticks on the same enemy
        *[{"round": 1, "tick": 1000 + i, "attacker_steamid": STEAM_ID, "user_steamid": "e",
           "weapon": "inferno", "hitgroup": "generic"} for i in range(1, 9)],
        {"round": 1, "tick": 1010, "attacker_steamid": STEAM_ID, "user_steamid": "e",
         "weapon": "hegrenade", "hitgroup": "generic"},
    ])
    fires = pd.DataFrame([
        {"round": 1, "tick": t, "user_steamid": STEAM_ID, "weapon": "weapon_ak47"}
        for t in (995, 1000, 1005, 1008)
    ])
    out = _get_round_damage_encounters(
        hurt, pd.DataFrame(), STEAM_ID, 1, weapon_fire_df=fires,
    )

    assert len(out) == 1
    assert out[0]["hits"] == 1                       # the bullet, not the fire
    assert out[0]["accuracy"]["hit_pct"] == 25.0     # 1 of 4 bullets
    assert out[0]["accuracy"]["upper"] == 1


def test_hitgroup_split_accounts_for_every_counted_hit():
    """Head + chest + limbs must cover the hits, or the split is over a subset.

    It previously ran over ~70% of them, because grenade damage was in the hit
    total but had no body part to be filed under.
    """
    from src.processor import _get_round_damage_encounters

    hurt = pd.DataFrame([
        {"round": 1, "tick": 1000, "attacker_steamid": STEAM_ID, "user_steamid": "e",
         "weapon": "ak47", "hitgroup": "head"},
        {"round": 1, "tick": 1004, "attacker_steamid": STEAM_ID, "user_steamid": "e",
         "weapon": "ak47", "hitgroup": "left_leg"},
        {"round": 1, "tick": 1008, "attacker_steamid": STEAM_ID, "user_steamid": "e",
         "weapon": "inferno", "hitgroup": "generic"},
    ])
    enc = _get_round_damage_encounters(hurt, pd.DataFrame(), STEAM_ID, 1)[0]
    acc = enc["accuracy"]

    assert acc["head"] + acc["upper"] + acc["lower"] == enc["hits"] == 2


def test_flashing_yourself_is_not_a_team_flash():
    """Same team by definition, so the team check alone charged it as one."""
    blinds = [
        {"round": 1, "attacker_steamid": STEAM_ID, "user_steamid": STEAM_ID,
         "attacker_team_num": 3, "user_team_num": 3, "blind_duration": 2.0,
         "user_name": "Me"},
        {"round": 1, "attacker_steamid": STEAM_ID, "user_steamid": "mate",
         "attacker_team_num": 3, "user_team_num": 3, "blind_duration": 1.5,
         "user_name": "Mate"},
    ]
    fl = _utility(blinds, n_flashes=2)["flash"]

    assert fl["self_flashed"] == 1
    assert fl["team_flashed"] == 1
    assert fl["enemies_flashed"] == 0


def test_inventory_snapshots_are_not_purchases():
    """The game re-emits a whole inventory on one tick with slots from zero.

    Counted as buys, a re-emitted rifle looked like a second one bought to
    drop for a teammate.
    """

    df = pd.DataFrame([
        {"steamid": STEAM_ID, "tick": 6771, "item_name": "XM1014", "inventory_slot": 2, "was_sold": False},
        # inventory re-emission: four rows, one tick, slots restarting at 0
        {"steamid": STEAM_ID, "tick": 8389, "item_name": "XM1014", "inventory_slot": 0, "was_sold": False},
        {"steamid": STEAM_ID, "tick": 8389, "item_name": "Smoke Grenade", "inventory_slot": 1, "was_sold": False},
        {"steamid": STEAM_ID, "tick": 8389, "item_name": "Flashbang", "inventory_slot": 2, "was_sold": False},
        {"steamid": STEAM_ID, "tick": 8389, "item_name": "Flashbang", "inventory_slot": 3, "was_sold": False},
    ])
    kept = _genuine_purchases(df)

    assert len(kept) == 1
    assert int(kept.iloc[0]["tick"]) == 6771


def test_refunds_are_not_purchases():

    df = pd.DataFrame([
        {"steamid": STEAM_ID, "tick": 100, "item_name": "AWP", "inventory_slot": 0, "was_sold": True},
        {"steamid": STEAM_ID, "tick": 200, "item_name": "AK-47", "inventory_slot": 0, "was_sold": False},
    ])
    kept = _genuine_purchases(df)

    assert len(kept) == 1
    assert kept.iloc[0]["item_name"] == "AK-47"
