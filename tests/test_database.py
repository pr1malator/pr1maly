"""
Tests for src/database.py
These tests use an in-memory SQLite database so they leave no side effects.
"""

from __future__ import annotations

import pytest

from src.database import (
    add_tag,
    delete_match,
    get_all_matches,
    get_connection,
    get_match,
    get_match_players,
    get_round_stats,
    get_tags,
    save_match,
    update_context_notes,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    """In-memory SQLite connection, closed after each test."""
    c = get_connection(":memory:")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Minimal stats dict
# ---------------------------------------------------------------------------

_STATS = {
    "map_name": "de_inferno",
    "player_name": "TestPlayer",
    "total_rounds": 24,
    "kills": 18,
    "deaths": 14,
    "assists": 4,
    "kpr": 0.75,
    "dpr": 0.58,
    "adr": 82.5,
    "kast": 72.0,
    "impact": 1.12,
    "hltv_rating": 1.08,
    "kd_ratio": 1.29,
    "rounds_2k": 3,
    "rounds_3k": 1,
    "rounds_4k": 0,
    "rounds_5k": 0,
    "team_score": 13,
    "enemy_score": 11,
    "match_result": "win",
    "round_stats": [
        {"round": 1, "kills": 2, "deaths": 0, "assists": 0, "damage": 150, "survived": 1, "traded": 0},
        {"round": 2, "kills": 0, "deaths": 1, "assists": 1, "damage": 40, "survived": 0, "traded": 0},
    ],
}

# ---------------------------------------------------------------------------
# save_match / get_match
# ---------------------------------------------------------------------------


def test_save_and_retrieve_match(conn):
    mid = save_match(conn, _STATS, "match.dem", "76561198000000001", context_notes="test notes")
    match = get_match(conn, mid)

    assert match is not None
    assert match["match_id"] == mid
    assert match["map_name"] == "de_inferno"
    assert match["kills"] == 18
    assert match["hltv_rating"] == pytest.approx(1.08)
    assert match["context_notes"] == "test notes"
    assert match["kd_ratio"] == pytest.approx(1.29)
    assert match["rounds_2k"] == 3
    assert match["rounds_3k"] == 1
    assert match["rounds_4k"] == 0
    assert match["rounds_5k"] == 0
    assert match["team_score"] == 13
    assert match["enemy_score"] == 11
    assert match["match_result"] == "win"


def test_save_match_returns_uuid_string(conn):
    mid = save_match(conn, _STATS, "match.dem", "76561198000000001")
    assert isinstance(mid, str)
    assert len(mid) == 36  # UUID4 length


def test_get_match_nonexistent_returns_none(conn):
    result = get_match(conn, "nonexistent-id")
    assert result is None


# ---------------------------------------------------------------------------
# get_all_matches
# ---------------------------------------------------------------------------


def test_get_all_matches_returns_all(conn):
    save_match(conn, _STATS, "a.dem", "76561198000000001", match_date="2025-01-01")
    save_match(conn, _STATS, "b.dem", "76561198000000001", match_date="2025-01-02")
    matches = get_all_matches(conn)
    assert len(matches) == 2


def test_get_all_matches_empty_db(conn):
    assert get_all_matches(conn) == []


# ---------------------------------------------------------------------------
# round_stats
# ---------------------------------------------------------------------------


def test_round_stats_are_persisted(conn):
    mid = save_match(conn, _STATS, "match.dem", "76561198000000001")
    rounds = get_round_stats(conn, mid)
    assert len(rounds) == 2
    assert rounds[0]["round_number"] == 1
    assert rounds[0]["kills"] == 2
    assert rounds[1]["deaths"] == 1


def test_round_stats_empty_for_unknown_match(conn):
    assert get_round_stats(conn, "unknown") == []


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_add_and_get_tags(conn):
    mid = save_match(conn, _STATS, "match.dem", "76561198000000001")
    add_tag(conn, mid, "solo")
    add_tag(conn, mid, "warm-up")
    tags = get_tags(conn, mid)
    assert sorted(tags) == ["solo", "warm-up"]


def test_tags_empty_for_unknown_match(conn):
    assert get_tags(conn, "unknown") == []


# ---------------------------------------------------------------------------
# update_context_notes
# ---------------------------------------------------------------------------


def test_update_context_notes(conn):
    mid = save_match(conn, _STATS, "match.dem", "76561198000000001", context_notes="old")
    update_context_notes(conn, mid, "new notes")
    match = get_match(conn, mid)
    assert match["context_notes"] == "new notes"


# ---------------------------------------------------------------------------
# delete_match
# ---------------------------------------------------------------------------


def test_delete_match_removes_all_data(conn):
    mid = save_match(conn, _STATS, "match.dem", "76561198000000001")
    add_tag(conn, mid, "solo")
    delete_match(conn, mid)

    assert get_match(conn, mid) is None
    assert get_round_stats(conn, mid) == []
    assert get_tags(conn, mid) == []
    assert get_match_players(conn, mid) == []


# ---------------------------------------------------------------------------
# match_players
# ---------------------------------------------------------------------------


def test_match_players_are_persisted(conn):
    stats_with_players = {
        **_STATS,
        "all_players": [
            {
                "steam_id": "76561198000000001",
                "name": "TestPlayer",
                "team": 3,
                "is_user": True,
                "kills": 18, "deaths": 14, "assists": 4,
                "kd_ratio": 1.29, "adr": 82.5, "kast": 72.0,
                "hltv_rating": 1.08,
                "rounds_2k": 3, "rounds_3k": 1, "rounds_4k": 0, "rounds_5k": 0,
            },
            {
                "steam_id": "76561198000000002",
                "name": "Teammate",
                "team": 3,
                "is_user": False,
                "kills": 12, "deaths": 16, "assists": 6,
                "kd_ratio": 0.75, "adr": 65.0, "kast": 60.0,
                "hltv_rating": 0.85,
                "rounds_2k": 1, "rounds_3k": 0, "rounds_4k": 0, "rounds_5k": 0,
            },
        ],
    }
    mid = save_match(conn, stats_with_players, "match.dem", "76561198000000001")
    players = get_match_players(conn, mid)
    assert len(players) == 2
    user_rows = [p for p in players if p["is_user"]]
    assert len(user_rows) == 1
    assert user_rows[0]["name"] == "TestPlayer"
    assert user_rows[0]["kills"] == 18


def test_match_players_empty_when_no_players(conn):
    mid = save_match(conn, _STATS, "match.dem", "76561198000000001")
    players = get_match_players(conn, mid)
    assert players == []
