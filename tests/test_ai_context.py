"""Tests for the prompt context handed to the model.

Moved here from test_api.py when the builders left api.py — they were never
tests of the HTTP layer, they exercise string assembly over stored match data.
"""

from __future__ import annotations

import json

from src.services.ai_context import (
    OVERALL_KEY,
    build_patterns_context,
    build_role_context,
    matches_with_recorded_play,
    strip_json_fences,
)


def _patterns_fixture():
    match = {
        "match_id": "m1",
        "map_name": "de_mirage",
        "aim_stats": json.dumps({
            "aim_rating": 84.0,
            "movement": {
                "median": 23.7, "n": 16,
                "counterstrafe_attempts": 6, "counterstrafe_good": 5,
                "counterstrafe_by_peek": [
                    {"label": "Walk", "attempts": 2, "good": 1},
                    {"label": "Full speed", "attempts": 4, "good": 4},
                ],
            },
            "peek": {"by_zone": [
                {"label": "Held", "n": 4}, {"label": "Full speed", "n": 12},
            ]},
            "preaim": {"median": 2.8, "n": 16},
            "ttk": {"median": 0.2, "n": 9},
            "reaction": {"median": 375.0, "n": 3},
            "accuracy": {"pooled_pct": 27.6},
        }),
        "utility_data": json.dumps({
            "utility_rating": 59.8,
            "economics": {"total_spent": 13600, "total_wasted": 3900},
            "flash": {
                "thrown": 9, "enemies_flashed": 5, "avg_enemy_blind_duration": 3.5,
                "effective_flash_pct": 100.0, "team_flashed": 0, "self_flashed": 5,
            },
            "he": {"total_damage": 77},
            "molotov": {"total_damage": 0},
        }),
    }
    rounds = [
        {
            "round": 1, "survived": True, "traded": False, "deaths": 0,
            "enriched": {
                "side": "CT", "round_winner": "CT",
                "opening_duel": {"role": "opening_kill"},
                "economy": {"buy_type": "FULL BUY", "items": ["Flashbang", "AK-47"]},
                "utility": {"grenades": [{"type": "flash"}]},
            },
        },
        {
            "round": 2, "survived": False, "traded": True, "deaths": 1,
            "enriched": {
                "side": "T", "round_winner": "CT",
                "opening_duel": {"role": "opening_death"},
                "economy": {"buy_type": "ECO", "items": ["Smoke Grenade"]},
                "utility": {"grenades": []},
                "clutch": {"won": False},
            },
        },
    ]
    return match, rounds


def test_patterns_context_carries_the_numbers_and_their_sample_sizes():
    """A tendency claimed off three engagements is not a tendency.

    The model cannot tell which figures are thin unless the counts travel with
    them, so every measured line has to name its n.
    """
    match, rounds = _patterns_fixture()
    ctx = build_patterns_context("de_mirage", [match], rounds)

    assert "n=16" in ctx                      # aim samples
    assert "23.7 u/s" in ctx                  # shot speed
    assert "83.3% of rifle stops" in ctx      # 5 of 6
    assert "Walk peeks: 50.0%" in ctx         # the per-peek-speed split
    assert "Full speed peeks: 100.0%" in ctx
    assert "28.7% wasted" in ctx              # 3900 of 13600
    assert "CT:" in ctx and "T:" in ctx       # both sides described
    assert "de_mirage" in ctx


def test_patterns_context_uses_a_real_degree_sign():
    """It read `Â°` for a long time — mojibake going straight to the model."""
    match, rounds = _patterns_fixture()
    ctx = build_patterns_context("de_mirage", [match], rounds)
    assert "2.8° off target" in ctx
    assert "Â" not in ctx


def test_patterns_context_survives_matches_with_nothing_stored():
    """Old matches predate some of these blocks and must not break the build."""
    bare = {"match_id": "m0", "map_name": "de_mirage"}
    ctx = build_patterns_context("de_mirage", [bare], [{"round": 1, "enriched": {}}])

    assert "No aim data stored" in ctx
    assert "No utility data stored" in ctx


def test_patterns_context_tolerates_unparseable_blobs():
    """aim_stats is opaque TEXT; a corrupt row must not take the request down."""
    broken = {"match_id": "m0", "map_name": "de_mirage",
              "aim_stats": "{not json", "utility_data": "{also not"}
    ctx = build_patterns_context("de_mirage", [broken], [])
    assert "No aim data stored" in ctx


def test_role_context_describes_both_sides():
    _match, rounds = _patterns_fixture()
    ctx = build_role_context("de_mirage", rounds[:1], rounds[1:])
    assert "Map: de_mirage" in ctx
    assert "=== CT SIDE ROUNDS ===" in ctx
    assert "=== T SIDE ROUNDS ===" in ctx
    assert "CT rounds: 1" in ctx and "T rounds: 1" in ctx


def test_fenced_json_is_unwrapped_not_discarded():
    """A fenced answer is a correct answer badly packaged.

    Both AI assessments ask for raw JSON and some providers fence it anyway;
    dropping those into the prose fallback would lose the structure for no
    reason.
    """
    payload = '{"headline": "peeks fast, stops late"}'
    for wrapped in (
        f"```json\n{payload}\n```",
        f"```\n{payload}\n```",
        f"  ```json\n{payload}\n```  ",
        payload,
    ):
        assert json.loads(strip_json_fences(wrapped))["headline"] == "peeks fast, stops late"

    # Malformed input must fall through to the caller's error handling rather
    # than raising out of the helper.
    assert strip_json_fences("```") == "```"
    assert strip_json_fences("") == ""


def test_matches_the_player_is_absent_from_are_not_assessed():
    """A demo belonging to someone else imports as an all-zero match.

    Averaged in, one of those drags a map's ADR down by a third and invites the
    assessment to call it a weak map on the strength of a match nobody played.
    A genuinely bad match still counts — only the total absence of the player
    is evidence they were not there.
    """
    absent = {"map_name": "de_mirage", "kills": 0, "deaths": 0, "adr": 0.0,
              "match_result": "unknown"}
    played_badly = {"map_name": "de_mirage", "kills": 0, "deaths": 18, "adr": 12.4}
    normal = {"map_name": "de_mirage", "kills": 19, "deaths": 19, "adr": 71.7}

    kept = matches_with_recorded_play([absent, played_badly, normal])

    assert kept == [played_badly, normal]
    assert matches_with_recorded_play([absent]) == []
    # Damage alone is enough to show they were there.
    assert matches_with_recorded_play([{"kills": 0, "deaths": 0, "adr": 3.0}]) != []


def test_overall_key_cannot_collide_with_a_map():
    """The career assessment shares a file with the per-map ones."""
    assert OVERALL_KEY == "__overall__"
    assert not OVERALL_KEY.startswith("de_")
