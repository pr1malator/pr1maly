"""Tests for the demo import service.

The pieces here were previously inlined four times over, each copy slightly
different. These tests pin the behaviour the copies agreed on and the edges
where they did not — particularly that a bad .dem.info sidecar costs the match
date but never the import.
"""

from __future__ import annotations

import pytest

from src.database import get_connection, get_match, get_tags
from src.services import import_service
from src.services.import_service import (
    SidecarInfo,
    account_name_for,
    analyse_demo,
    apply_parse_metadata,
    read_sidecar,
    resolve_steam_id,
    store_match,
)

STEAM_ID = "76561198012345678"


# ---------------------------------------------------------------------------
# resolve_steam_id
# ---------------------------------------------------------------------------


def test_first_non_empty_candidate_wins():
    assert resolve_steam_id("", None, "  ", "fallback") == "fallback"
    assert resolve_steam_id("explicit", "active", "legacy") == "explicit"


def test_candidates_are_stripped():
    assert resolve_steam_id("  123  ") == "123"


def test_no_candidates_is_empty_not_none():
    """Callers test falsiness and raise their own 400; None would still be falsy
    but would leak into save_match as a null player_steam_id."""
    assert resolve_steam_id() == ""
    assert resolve_steam_id(None, "", "   ") == ""


# ---------------------------------------------------------------------------
# Sidecar reading
# ---------------------------------------------------------------------------


def test_unreadable_sidecar_is_empty_not_an_error(tmp_path):
    """The whole point: a corrupt sidecar must not fail an importable demo."""
    bad = tmp_path / "x.dem.info"
    bad.write_bytes(b"\x00\x01 not a protobuf \xff")
    assert read_sidecar(bad) == SidecarInfo()


def test_missing_sidecar_is_empty(tmp_path):
    assert read_sidecar(tmp_path / "absent.dem.info") == SidecarInfo()


def test_sidecar_from_bytes(monkeypatch):
    monkeypatch.setattr(
        import_service, "parse_info_file",
        lambda raw: {"match_date": "2026-08-01", "account_ids": [123, 456]},
    )
    info = read_sidecar(b"whatever")
    assert info.match_date == "2026-08-01"
    assert info.account_ids == ("123", "456")


def test_account_ids_are_stringified(monkeypatch):
    """They arrive as ints from the protobuf but are compared to string IDs."""
    monkeypatch.setattr(
        import_service, "parse_info_file",
        lambda raw: {"account_ids": [76561198012345678]},
    )
    assert read_sidecar(b"x").account_ids == ("76561198012345678",)


# ---------------------------------------------------------------------------
# first_known — which of my accounts played this match
# ---------------------------------------------------------------------------


def test_first_known_picks_the_owned_account():
    info = SidecarInfo(account_ids=("stranger", "mine", "other"))
    assert info.first_known({"mine", "unused"}) == "mine"


def test_first_known_is_none_when_no_account_matches():
    assert SidecarInfo(account_ids=("a", "b")).first_known({"c"}) is None


def test_first_known_of_empty_sidecar_is_none():
    assert SidecarInfo().first_known({"anything"}) is None


def test_first_known_respects_sidecar_order():
    info = SidecarInfo(account_ids=("first", "second"))
    assert info.first_known({"first", "second"}) == "first"


# ---------------------------------------------------------------------------
# apply_parse_metadata
# ---------------------------------------------------------------------------


def test_header_only_fallback_marks_the_import_partial():
    stats: dict = {}
    apply_parse_metadata(stats, {"header": {"parse_mode": "header_only_fallback"}})
    assert stats["partial_import"] is True
    assert stats["parse_mode"] == "header_only_fallback"


def test_a_warning_alone_marks_the_import_partial():
    stats: dict = {}
    apply_parse_metadata(stats, {"header": {"parse_warning": "newer schema"}})
    assert stats["partial_import"] is True


def test_clean_parse_is_not_marked_partial():
    stats: dict = {}
    apply_parse_metadata(stats, {"header": {"map_name": "de_mirage"}})
    assert "partial_import" not in stats


def test_unparseable_patch_version_is_dropped_not_fatal():
    stats: dict = {}
    apply_parse_metadata(stats, {"header": {"patch_version": "not-a-number"}})
    assert "source_patch_version" not in stats


def test_patch_version_is_coerced_to_int():
    stats: dict = {}
    apply_parse_metadata(stats, {"header": {"patch_version": "14000"}})
    assert stats["source_patch_version"] == 14000


# ---------------------------------------------------------------------------
# analyse_demo — the three lines that were written out four times
# ---------------------------------------------------------------------------


def test_analyse_demo_parses_calculates_and_stamps_metadata(monkeypatch):
    monkeypatch.setattr(
        import_service, "parse_demo",
        lambda path: {"header": {"parse_warning": "heads up"}, "path": path},
    )
    monkeypatch.setattr(
        import_service, "calculate_match_stats",
        lambda parsed, sid: {"kills": 20, "steam_id": sid},
    )

    stats = analyse_demo("/demos/x.dem", STEAM_ID)

    assert stats["kills"] == 20
    assert stats["steam_id"] == STEAM_ID
    assert stats["parse_warning"] == "heads up"
    assert stats["partial_import"] is True


def test_analyse_demo_lets_parse_failures_propagate(monkeypatch):
    """Callers translate this differently — a 422 for one upload, a per-file
    error entry for a bulk one — so the service must not decide for them."""
    def boom(path):
        raise RuntimeError("corrupt demo")

    monkeypatch.setattr(import_service, "parse_demo", boom)
    with pytest.raises(RuntimeError, match="corrupt demo"):
        analyse_demo("/demos/bad.dem", STEAM_ID)


# ---------------------------------------------------------------------------
# store_match
# ---------------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path):
    c = get_connection(tmp_path / "t.db")
    yield c
    c.close()


_STATS = {
    "player_name": "P", "map_name": "de_mirage", "total_rounds": 2,
    "kills": 5, "deaths": 3, "assists": 1,
    "round_stats": [
        {"round": 1, "kills": 3, "deaths": 0, "assists": 0,
         "damage": 200, "survived": 1, "traded": 0},
    ],
}


def test_store_match_persists_and_returns_id(conn):
    match_id = store_match(conn, dict(_STATS), filename="a.dem", steam_id=STEAM_ID)
    stored = get_match(conn, match_id)
    assert stored["map_name"] == "de_mirage"
    assert stored["player_steam_id"] == STEAM_ID
    assert stored["filename"] == "a.dem"


def test_store_match_writes_tags(conn):
    match_id = store_match(
        conn, dict(_STATS), filename="a.dem", steam_id=STEAM_ID,
        tags=["ranked", "good-game"],
    )
    assert sorted(get_tags(conn, match_id)) == ["good-game", "ranked"]


def test_store_match_skips_blank_tags(conn):
    """The upload route splits a comma-separated field, so blanks are routine."""
    match_id = store_match(
        conn, dict(_STATS), filename="a.dem", steam_id=STEAM_ID,
        tags=["  ", "", "kept", "  spaced  "],
    )
    assert sorted(get_tags(conn, match_id)) == ["kept", "spaced"]


# ---------------------------------------------------------------------------
# account_name_for
# ---------------------------------------------------------------------------


def test_account_name_lookup():
    accounts = [{"steam_id": "1", "name": "Smurf"}, {"steam_id": "2", "name": "Main"}]
    assert account_name_for(accounts, "2") == "Main"


def test_account_name_is_none_when_unknown():
    assert account_name_for([{"steam_id": "1", "name": "Only"}], "999") is None


def test_account_name_of_empty_list_is_none():
    assert account_name_for([], "1") is None
