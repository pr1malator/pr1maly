"""Characterisation tests for the behavioural axes and role classification.

Every expected number here was captured from api.py's `_compute_side_axes` and
`_classify_side_role` before they were merged into one calculation, so a
passing run is evidence the merge changed nothing.

The one intended difference is at the bottom: a utility block carrying an
explicit null used to crash `_classify_side_role` and not `_compute_side_axes`.
"""

from __future__ import annotations

import pytest

from src.metrics.behavior import (
    AXES,
    accumulate,
    classify_archetype,
    empty_side_role,
    match_behavioral_axes,
    side_axes,
    side_role,
)


def _rounds() -> list[dict]:
    """Four rounds covering every counter the calculation reads."""
    return [
        {   # opening kill, AWP long-range, flashes, survives
            "traded": 0, "deaths": 0, "damage": 210,
            "enriched": {
                "side": "CT", "round_winner": "CT",
                "kills_detail": [
                    {"weapon": "AWP", "distance": 42.0},
                    {"weapon": "AK-47", "distance": 12.0},
                ],
                "death_detail": None,
                "opening_duel": {"role": "opening_kill"},
                "utility": {"enemies_flashed": 2, "flash_assists": 1,
                            "he_damage": 40,
                            "molotov_damage": [{"damage": 15}, {"damage": 5}]},
            },
        },
        {   # opening death, traded, dies
            "traded": 1, "deaths": 1, "damage": 80,
            "enriched": {
                "side": "CT", "round_winner": "T",
                "kills_detail": [{"weapon": "M4A4", "distance": 8.0}],
                "death_detail": {"victim_position": "A Site"},
                "opening_duel": {"role": "opening_death"},
                "utility": {"enemies_flashed": 0, "flash_assists": 0,
                            "he_damage": 0, "molotov_damage": []},
            },
        },
        {   # no opening duel, survives, empty utility block
            "traded": 0, "deaths": 0, "damage": 55,
            "enriched": {
                "side": "CT", "round_winner": "CT",
                "kills_detail": [{"weapon": "AWP", "distance": 55.0}],
                "death_detail": None,
                "opening_duel": None,
                "utility": {},
            },
        },
        {   # dies without killing; molotov_damage explicitly null
            "traded": 0, "deaths": 1, "damage": 20,
            "enriched": {
                "side": "CT", "round_winner": "T",
                "kills_detail": [],
                "death_detail": {"victim_position": "B Site"},
                "opening_duel": {"role": "traded"},
                "utility": {"enemies_flashed": 1, "flash_assists": 0,
                            "he_damage": 12, "molotov_damage": None},
            },
        },
    ]


# The values api.py produced before the merge.
_EXPECTED_AXES = {
    "aggression": 62, "trading": 51, "isolation": 58, "survival": 62, "sniper": 50,
}


# ---------------------------------------------------------------------------
# The axes
# ---------------------------------------------------------------------------


def test_axes_match_the_pre_merge_values():
    assert side_axes(_rounds())["axes"] == _EXPECTED_AXES


def test_side_role_reports_the_same_axes_as_side_axes():
    """The whole reason for merging: these were two copies of one calculation."""
    assert side_role(_rounds())["axes"] == side_axes(_rounds())["axes"]


def test_per_axis_success_matches_the_pre_merge_values():
    assert side_axes(_rounds())["success"] == {
        "aggression": {"rounds": 3, "wins": 1, "win_pct": 33.0},
        "isolation": {"rounds": 1, "wins": 1, "win_pct": 100.0},
    }


def test_axes_are_bounded_to_100():
    for value in side_axes(_rounds())["axes"].values():
        assert 0 <= value <= 100


def test_every_axis_is_reported():
    assert set(side_axes(_rounds())["axes"]) == set(AXES)


# ---------------------------------------------------------------------------
# The role
# ---------------------------------------------------------------------------


def test_side_role_matches_the_pre_merge_values():
    role = side_role(_rounds())
    assert role["name"] == "AWPer"
    assert role["icon"] == "precision_manufacturing"
    assert role["kills"] == 4
    assert role["deaths"] == 2
    assert role["rounds"] == 4
    assert role["adr"] == 91.2
    assert role["opening_kills"] == 1
    assert role["opening_deaths"] == 1
    assert role["survival_pct"] == 50.0


# ---------------------------------------------------------------------------
# Empty sides
# ---------------------------------------------------------------------------


def test_empty_side_axes():
    assert side_axes([]) == {
        "axes": dict.fromkeys(AXES, 0),
        "success": {},
    }


def test_empty_side_role():
    assert side_role([]) == empty_side_role()
    assert side_role([])["name"] == "Unknown"
    assert side_role([])["adr"] == 0


def test_empty_side_role_is_a_fresh_dict_each_time():
    """It is handed to callers that mutate it; a shared literal would leak."""
    first = empty_side_role()
    first["axes"]["sniper"] = 99
    assert empty_side_role()["axes"]["sniper"] == 0


# ---------------------------------------------------------------------------
# Archetype thresholds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expected,kwargs", [
    ("AWPer", dict(opening_kill_pct=10, survival_pct=10, util_per_round=0,
                   trade_pct=0, weapon_kills={"AWP": 40}, total_kills=100)),
    ("Entry Fragger", dict(opening_kill_pct=60, survival_pct=10, util_per_round=0,
                           trade_pct=0, weapon_kills={}, total_kills=10)),
    ("Lurker", dict(opening_kill_pct=20, survival_pct=60, util_per_round=0,
                    trade_pct=0, weapon_kills={}, total_kills=10)),
    ("Support", dict(opening_kill_pct=40, survival_pct=40, util_per_round=2.0,
                     trade_pct=35, weapon_kills={}, total_kills=10)),
    ("Anchor", dict(opening_kill_pct=40, survival_pct=52, util_per_round=0,
                    trade_pct=0, weapon_kills={}, total_kills=10)),
    ("Flex", dict(opening_kill_pct=40, survival_pct=10, util_per_round=0,
                  trade_pct=0, weapon_kills={}, total_kills=10)),
])
def test_archetype_thresholds(expected, kwargs):
    assert classify_archetype(**kwargs)["name"] == expected


def test_awp_outranks_entry_fragging():
    """Order matters: a heavy AWPer who also entries is still an AWPer."""
    assert classify_archetype(
        opening_kill_pct=80, survival_pct=10, util_per_round=0, trade_pct=0,
        weapon_kills={"AWP": 50}, total_kills=100,
    )["name"] == "AWPer"


def test_archetype_with_no_kills_does_not_divide_by_zero():
    assert classify_archetype(
        opening_kill_pct=0, survival_pct=0, util_per_round=0, trade_pct=0,
        weapon_kills={}, total_kills=0,
    )["name"] == "Flex"


# ---------------------------------------------------------------------------
# Per-match split
# ---------------------------------------------------------------------------


def test_match_axes_split_by_side():
    rounds = [
        {"enriched_json": '{"side": "CT", "round_winner": "CT", "kills_detail": []}'},
        {"enriched_json": '{"side": "T", "round_winner": "T", "kills_detail": []}'},
    ]
    result = match_behavioral_axes(rounds)
    assert set(result) == {"ct", "t"}
    assert result["ct"]["axes"]["survival"] > 0
    assert result["t"]["axes"]["survival"] > 0


def test_match_axes_tolerates_corrupt_enriched_json():
    rounds = [{"enriched_json": "{not json"}]
    result = match_behavioral_axes(rounds)
    assert result["ct"]["axes"] == dict.fromkeys(AXES, 0)
    assert rounds[0]["enriched"] == {}


def test_match_axes_deserialises_in_place():
    """Callers read r["enriched"] afterwards, so the mutation is load-bearing."""
    rounds = [{"enriched_json": '{"side": "CT"}'}]
    match_behavioral_axes(rounds)
    assert rounds[0]["enriched"] == {"side": "CT"}


# ---------------------------------------------------------------------------
# The drift that is deliberately fixed
# ---------------------------------------------------------------------------


def _null_utility_round() -> list[dict]:
    return [{
        "traded": 0, "deaths": 0, "damage": 0,
        "enriched": {
            "side": "CT", "round_winner": "CT", "kills_detail": [],
            "death_detail": None, "opening_duel": None,
            "utility": {"enemies_flashed": None, "flash_assists": None,
                        "he_damage": None, "molotov_damage": None},
        },
    }]


def test_null_utility_counters_do_not_raise():
    """`_classify_side_role` raised TypeError here; `_compute_side_axes` did not.

    Utility blocks are read back from enriched_json written by older analyzer
    versions, so an explicit null is a shape that can reach this code.
    """
    # Survived the only round and took no opening duel, so: Lurker.
    assert side_role(_null_utility_round())["name"] == "Lurker"
    assert side_axes(_null_utility_round())["axes"]["trading"] == 0


def test_null_distance_on_a_kill_is_not_long_range():
    rounds = [{
        "traded": 0, "deaths": 0, "damage": 0,
        "enriched": {
            "side": "CT", "round_winner": "CT",
            "kills_detail": [{"weapon": "AK-47", "distance": None}],
            "death_detail": None, "opening_duel": None, "utility": {},
        },
    }]
    assert accumulate(rounds).long_range_kills == 0
