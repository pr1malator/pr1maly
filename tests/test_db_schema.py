"""Guards on the SQLite shape, because users own their database files.

Demos are deleted by the retention feature once imported, so a match that can
no longer be re-analysed exists *only* as rows in this schema. Anything that
drops a column or breaks the forward migration destroys data that cannot be
regenerated. These tests are the reason a refactor can claim read-compatibility
with an existing install rather than hoping for it.

Regenerate the snapshot only when a schema change is intended:

    UPDATE_SNAPSHOTS=1 python -m pytest tests/test_db_schema.py
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from src.database import (
    add_tag,
    get_analyzer_versions,
    get_connection,
    get_enriched_json_for_map,
    get_imported_filenames,
    get_players_for_matches,
    get_round_stats,
    get_rounds_for_matches,
    get_tags_for_all_matches,
    save_match,
)

_SNAPSHOT = Path(__file__).parent / "snapshots" / "db_schema.json"
_UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"

# The shape the database had before any of the ALTER TABLE migrations in
# _ensure_schema ran: no aim/role/utility/impact blobs on matches, no
# enriched_json or replay_json on round_stats, no rank columns on match_players.
# A user who has not opened the app in a long time still has exactly this.
_LEGACY_DDL = """
CREATE TABLE matches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id      TEXT    UNIQUE NOT NULL,
    filename      TEXT,
    date          TEXT,
    map_name      TEXT,
    player_steam_id TEXT,
    player_name   TEXT,
    total_rounds  INTEGER,
    kills         INTEGER,
    deaths        INTEGER,
    assists       INTEGER,
    kast          REAL,
    adr           REAL,
    kpr           REAL,
    dpr           REAL,
    impact        REAL,
    hltv_rating   REAL,
    kd_ratio      REAL,
    rounds_2k     INTEGER DEFAULT 0,
    rounds_3k     INTEGER DEFAULT 0,
    rounds_4k     INTEGER DEFAULT 0,
    rounds_5k     INTEGER DEFAULT 0,
    team_score    INTEGER DEFAULT 0,
    enemy_score   INTEGER DEFAULT 0,
    match_result  TEXT    DEFAULT 'unknown',
    context_notes TEXT,
    uploaded_at   TEXT
);
CREATE TABLE round_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id     TEXT    NOT NULL,
    round_number INTEGER,
    kills        INTEGER,
    deaths       INTEGER,
    assists      INTEGER,
    damage       INTEGER,
    survived     INTEGER,
    traded       INTEGER
);
CREATE TABLE context_tags (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    tag      TEXT
);
CREATE TABLE match_players (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id     TEXT    NOT NULL,
    steam_id     TEXT,
    name         TEXT,
    team         INTEGER,
    is_user      INTEGER DEFAULT 0,
    kills        INTEGER DEFAULT 0,
    deaths       INTEGER DEFAULT 0,
    assists      INTEGER DEFAULT 0,
    kd_ratio     REAL    DEFAULT 0.0,
    adr          REAL    DEFAULT 0.0,
    kast         REAL    DEFAULT 0.0,
    hltv_rating  REAL    DEFAULT 0.0,
    rounds_2k    INTEGER DEFAULT 0,
    rounds_3k    INTEGER DEFAULT 0,
    rounds_4k    INTEGER DEFAULT 0,
    rounds_5k    INTEGER DEFAULT 0
);
"""


def _describe(conn: sqlite3.Connection) -> dict[str, Any]:
    """Structural fingerprint: tables, their columns, and the indexes."""
    tables = sorted(
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    )
    described: dict[str, Any] = {}
    for table in tables:
        described[table] = [
            {
                "name": row["name"],
                "type": row["type"],
                "notnull": row["notnull"],
                "default": row["dflt_value"],
                "pk": row["pk"],
            }
            for row in conn.execute(f"PRAGMA table_info({table})")
        ]

    indexes = sorted(
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    )
    return {"tables": described, "indexes": indexes}


@pytest.fixture
def fresh(tmp_path):
    conn = get_connection(tmp_path / "fresh.db")
    yield conn
    conn.close()


def test_fresh_schema_matches_snapshot(fresh):
    """Every column a stored match depends on is still declared."""
    live = _describe(fresh)
    if _UPDATE or not _SNAPSHOT.exists():
        _SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT.write_text(json.dumps(live, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stored = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))

    assert set(live["tables"]) == set(stored["tables"]), "a table appeared or vanished"
    for table, columns in stored["tables"].items():
        live_names = [c["name"] for c in live["tables"][table]]
        for col in columns:
            assert col["name"] in live_names, f"{table}.{col['name']} was dropped"
        assert live["tables"][table] == columns, f"column definitions changed on {table}"

    assert live["indexes"] == stored["indexes"]


def test_replay_json_exists_only_via_migration(fresh):
    """A quirk worth pinning: round_stats.replay_json is NOT in the CREATE TABLE.

    It is added by the ALTER at the end of _ensure_schema, so even a brand new
    database reaches its final shape through the migration path. Anyone
    rewriting the DDL needs to keep that column.
    """
    names = [row["name"] for row in fresh.execute("PRAGMA table_info(round_stats)")]
    assert "replay_json" in names
    assert "enriched_json" in names


def test_legacy_database_migrates_forward(tmp_path):
    """The whole point: an old database opens, gains its columns, keeps its rows."""
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    raw.executescript(_LEGACY_DDL)
    raw.execute(
        "INSERT INTO matches (match_id, filename, map_name, player_steam_id, "
        "player_name, total_rounds, kills, deaths, assists, adr, hltv_rating) "
        "VALUES ('legacy-1', 'old.dem', 'de_inferno', '76561198000000000', "
        "'OldPlayer', 24, 20, 15, 4, 82.5, 1.13)"
    )
    raw.execute(
        "INSERT INTO round_stats (match_id, round_number, kills, deaths, damage) "
        "VALUES ('legacy-1', 1, 2, 0, 187)"
    )
    raw.execute(
        "INSERT INTO match_players (match_id, steam_id, name, team, is_user, kills) "
        "VALUES ('legacy-1', '76561198000000000', 'OldPlayer', 3, 1, 20)"
    )
    raw.commit()
    raw.close()

    conn = get_connection(db)
    try:
        # Columns added by the migration chain are all present...
        match_cols = {row["name"] for row in conn.execute("PRAGMA table_info(matches)")}
        assert {
            "aim_stats", "role_data", "utility_data", "impact_stats",
            "partial_import", "parse_mode", "parse_warning",
            "source_patch_version", "analyzer_version",
        } <= match_cols

        round_cols = {row["name"] for row in conn.execute("PRAGMA table_info(round_stats)")}
        assert {"enriched_json", "replay_json"} <= round_cols

        player_cols = {row["name"] for row in conn.execute("PRAGMA table_info(match_players)")}
        assert {
            "rank", "rank_old", "rank_change", "rank_type_id", "comp_wins", "mvps",
        } <= player_cols

        # ...and the pre-existing row is untouched.
        row = conn.execute(
            "SELECT * FROM matches WHERE match_id = 'legacy-1'"
        ).fetchone()
        assert row["player_name"] == "OldPlayer"
        assert row["kills"] == 20
        assert row["adr"] == 82.5
        assert row["map_name"] == "de_inferno"

        # A match written before versioning reads as stale rather than current,
        # which is what drives the re-analysis prompt.
        assert (row["analyzer_version"] or 0) == 0

        rounds = get_round_stats(conn, "legacy-1")
        assert len(rounds) == 1
        assert rounds[0]["damage"] == 187
    finally:
        conn.close()


def test_migration_is_idempotent(tmp_path):
    """Opening the same database repeatedly must not keep altering it."""
    db = tmp_path / "repeat.db"
    first = get_connection(db)
    before = _describe(first)
    first.close()

    for _ in range(3):
        conn = get_connection(db)
        after = _describe(conn)
        conn.close()
        assert after == before


def _save_simple_match(conn, *, map_name: str, rounds: int) -> str:
    return save_match(
        conn,
        {
            "player_name": "P", "map_name": map_name, "total_rounds": rounds,
            "kills": rounds, "deaths": 0, "assists": 0,
            "round_stats": [
                {"round": n, "kills": 1, "deaths": 0, "assists": 0,
                 "damage": 100 * n, "survived": 1, "traded": 0}
                for n in range(1, rounds + 1)
            ],
            "enriched_rounds": [
                {"round": n, "side": "CT" if n <= rounds // 2 else "T"}
                for n in range(1, rounds + 1)
            ],
        },
        filename=f"{map_name}.dem",
        steam_id="76561198000000000",
    )


def test_batch_rounds_decode_enriched_json(tmp_path):
    """The four callers each re-implemented json.loads with a bare except."""
    conn = get_connection(tmp_path / "batch.db")
    try:
        match_id = _save_simple_match(conn, map_name="de_mirage", rounds=4)
        rounds = get_rounds_for_matches(conn, [match_id])
        assert len(rounds) == 4
        assert rounds[0]["enriched"] == {"round": 1, "side": "CT"}
        assert rounds[-1]["enriched"]["side"] == "T"
    finally:
        conn.close()


def test_batch_rounds_preserve_the_order_the_per_match_loop_produced(tmp_path):
    """Grouped by the order of match_ids, then by round number within a match.

    A plain `ORDER BY match_id` would reorder by UUID, which is not the order
    the loop it replaces produced.
    """
    conn = get_connection(tmp_path / "order.db")
    try:
        first = _save_simple_match(conn, map_name="de_nuke", rounds=3)
        second = _save_simple_match(conn, map_name="de_inferno", rounds=2)

        rounds = get_rounds_for_matches(conn, [second, first])
        assert [r["match_id"] for r in rounds] == [second] * 2 + [first] * 3
        assert [r["round_number"] for r in rounds] == [1, 2, 1, 2, 3]
    finally:
        conn.close()


def test_batch_rounds_of_no_matches_is_empty(tmp_path):
    conn = get_connection(tmp_path / "empty.db")
    try:
        assert get_rounds_for_matches(conn, []) == []
        assert get_rounds_for_matches(conn, ["never-existed"]) == []
    finally:
        conn.close()


def test_batch_rounds_exclude_replay_json(tmp_path):
    conn = get_connection(tmp_path / "noreplay.db")
    try:
        match_id = _save_simple_match(conn, map_name="de_dust2", rounds=2)
        for row in get_rounds_for_matches(conn, [match_id]):
            assert "replay_json" not in row
    finally:
        conn.close()


def test_batch_rounds_chunk_boundary(tmp_path, monkeypatch):
    """More matches than SQLite will take parameters for in one statement."""
    import src.database as db

    monkeypatch.setattr(db, "_MAX_PARAMS", 2)
    conn = get_connection(tmp_path / "chunked.db")
    try:
        ids = [
            _save_simple_match(conn, map_name=f"de_map{n}", rounds=1)
            for n in range(5)
        ]
        rounds = get_rounds_for_matches(conn, ids)
        assert [r["match_id"] for r in rounds] == ids
    finally:
        conn.close()


def _scoreboard_match(conn, *, map_name="de_mirage", filename="s.dem") -> str:
    def player(steam_id, is_user, kills, team):
        return {"steam_id": steam_id, "name": steam_id, "team": team,
                "is_user": is_user, "kills": kills, "deaths": 10, "assists": 2,
                "kd_ratio": 1.0, "adr": 70.0, "kast": 65.0, "hltv_rating": 1.0,
                "rounds_2k": 1, "rounds_3k": 0, "rounds_4k": 0, "rounds_5k": 0}

    return save_match(
        conn,
        {
            "player_name": "P", "map_name": map_name, "total_rounds": 2,
            "kills": 20, "deaths": 10, "assists": 2,
            "round_stats": [
                {"round": n, "kills": 1, "deaths": 0, "assists": 0,
                 "damage": 100, "survived": 1, "traded": 0} for n in (1, 2)
            ],
            "all_players": [
                player("me", 1, 20, 3),
                player("them_a", 0, 15, 2),
                player("them_b", 0, 5, 2),
            ],
        },
        filename=filename,
        steam_id="me",
    )


def test_batch_players_group_by_the_order_asked_for(tmp_path):
    conn = get_connection(tmp_path / "players.db")
    try:
        first = _scoreboard_match(conn, filename="a.dem")
        second = _scoreboard_match(conn, filename="b.dem")

        rows = get_players_for_matches(conn, [second, first])
        assert [r["match_id"] for r in rows] == [second] * 3 + [first] * 3
    finally:
        conn.close()


def test_batch_players_can_fetch_only_the_tracked_account(tmp_path):
    """The career aggregates want one row per match, not the whole scoreboard.

    Without this, 200 matches means 2,000 rows fetched and 1,800 discarded.
    """
    conn = get_connection(tmp_path / "user_only.db")
    try:
        match_id = _scoreboard_match(conn)
        everyone = get_players_for_matches(conn, [match_id])
        just_me = get_players_for_matches(conn, [match_id], user_only=True)

        assert len(everyone) == 3
        assert len(just_me) == 1
        assert just_me[0]["is_user"] == 1
    finally:
        conn.close()


def test_batch_players_of_no_matches_is_empty(tmp_path):
    conn = get_connection(tmp_path / "empty_players.db")
    try:
        assert get_players_for_matches(conn, []) == []
        assert get_players_for_matches(conn, ["nope"]) == []
    finally:
        conn.close()


def test_tags_for_all_matches_groups_by_match(tmp_path):
    conn = get_connection(tmp_path / "tags.db")
    try:
        first = _scoreboard_match(conn, filename="a.dem")
        second = _scoreboard_match(conn, filename="b.dem")
        add_tag(conn, first, "ranked")
        add_tag(conn, first, "good-game")
        add_tag(conn, second, "practice")

        tags = get_tags_for_all_matches(conn)
        assert sorted(tags[first]) == ["good-game", "ranked"]
        assert tags[second] == ["practice"]
    finally:
        conn.close()


def test_imported_filenames_can_be_scoped_to_one_account(tmp_path):
    """The sync scan asks "which of these files do I already have?"."""
    conn = get_connection(tmp_path / "names.db")
    try:
        save_match(conn, {"player_name": "A", "map_name": "de_dust2",
                          "total_rounds": 1, "kills": 1, "deaths": 0,
                          "assists": 0, "round_stats": []},
                   filename="mine.dem", steam_id="me")
        save_match(conn, {"player_name": "B", "map_name": "de_dust2",
                          "total_rounds": 1, "kills": 1, "deaths": 0,
                          "assists": 0, "round_stats": []},
                   filename="theirs.dem", steam_id="someone-else")

        assert get_imported_filenames(conn) == {"mine.dem", "theirs.dem"}
        assert get_imported_filenames(conn, "me") == {"mine.dem"}
        assert get_imported_filenames(conn, "  me  ") == {"mine.dem"}
    finally:
        conn.close()


def test_analyzer_versions_lists_every_match(tmp_path):
    conn = get_connection(tmp_path / "versions.db")
    try:
        _scoreboard_match(conn, filename="a.dem")
        _scoreboard_match(conn, filename="b.dem")
        versions = get_analyzer_versions(conn)
        assert len(versions) == 2
        assert all(isinstance(v, int) for v in versions)
    finally:
        conn.close()


def test_enriched_json_for_map_only_returns_that_map(tmp_path):
    conn = get_connection(tmp_path / "bymap.db")
    try:
        save_match(conn, {
            "player_name": "P", "map_name": "de_nuke", "total_rounds": 1,
            "kills": 1, "deaths": 0, "assists": 0,
            "round_stats": [{"round": 1, "kills": 1, "deaths": 0, "assists": 0,
                             "damage": 10, "survived": 1, "traded": 0}],
            "enriched_rounds": [{"round": 1, "side": "CT"}],
        }, filename="nuke.dem", steam_id="me")
        save_match(conn, {
            "player_name": "P", "map_name": "de_mirage", "total_rounds": 1,
            "kills": 1, "deaths": 0, "assists": 0,
            "round_stats": [{"round": 1, "kills": 1, "deaths": 0, "assists": 0,
                             "damage": 10, "survived": 1, "traded": 0}],
            "enriched_rounds": [{"round": 1, "side": "T"}],
        }, filename="mirage.dem", steam_id="me")

        nuke = get_enriched_json_for_map(conn, "de_nuke")
        assert len(nuke) == 1
        assert '"CT"' in nuke[0]
        assert get_enriched_json_for_map(conn, "de_cache") == []
    finally:
        conn.close()


def test_two_open_databases_do_not_share_a_column_cache(tmp_path):
    """The column list belongs to one database, not to the process.

    It used to be a module global with no lock, so it described whichever
    database connected most recently. With an in-memory test database and a
    real one alive at once — which is exactly what the suite does — one could
    be queried using the other's columns.
    """
    from src.database import _round_stats_columns

    first = get_connection(tmp_path / "one.db")
    second = get_connection(tmp_path / "two.db")
    try:
        # Give the second database a column the first does not have.
        second.execute("ALTER TABLE round_stats ADD COLUMN extra_column TEXT")
        second.commit()
        second._round_columns = None  # the ALTER was made behind the loader

        assert "extra_column" not in _round_stats_columns(first)
        assert "extra_column" in _round_stats_columns(second)
        # ...and asking again does not let one leak into the other.
        assert "extra_column" not in _round_stats_columns(first)
    finally:
        first.close()
        second.close()


def test_the_column_cache_is_dropped_when_the_schema_changes(tmp_path):
    """_ensure_schema runs on every open and can ALTER the table."""
    from src.database import _round_stats_columns

    conn = get_connection(tmp_path / "cache.db")
    try:
        first = _round_stats_columns(conn)
        assert conn._round_columns == first
        _ensure_schema_again = get_connection(tmp_path / "cache.db")
        _ensure_schema_again.close()
        assert "replay_json" not in first
    finally:
        conn.close()


def test_round_stats_excludes_replay_json_by_default(tmp_path):
    """replay_json is ~95% of the database; only the replay viewer pays for it."""
    conn = get_connection(tmp_path / "replay.db")
    try:
        match_id = save_match(
            conn,
            {
                "player_name": "P", "map_name": "de_mirage", "total_rounds": 1,
                "kills": 1, "deaths": 0, "assists": 0,
                "round_stats": [{"round": 1, "kills": 1, "deaths": 0,
                                 "assists": 0, "damage": 100, "survived": 1,
                                 "traded": 0}],
                "replay_data": {1: {"frames": ["x" * 1000]}},
            },
            filename="r.dem",
            steam_id="76561198000000000",
        )

        lean = get_round_stats(conn, match_id)
        assert lean, "expected a round row"
        assert "replay_json" not in lean[0]

        full = get_round_stats(conn, match_id, include_replay=True)
        assert "replay_json" in full[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# A readonly database, explained
# ---------------------------------------------------------------------------


def test_no_complaint_about_a_writable_database(tmp_path):
    from src.database import get_connection, writability_problem

    db = tmp_path / "fine.db"
    get_connection(db).close()
    assert writability_problem(db) is None


def test_in_memory_is_never_a_problem():
    from src.database import writability_problem

    assert writability_problem(":memory:") is None


def test_a_readonly_database_is_explained_in_words(tmp_path):
    """The bug this exists for: the container started running as uid 1000, and
    a database written by the earlier root image stayed root-owned. SQLite
    reports "attempt to write a readonly database" — no filename, no mention of
    ownership — attached to whatever the user was doing at the time."""
    import os

    from src.database import get_connection, writability_problem

    db = tmp_path / "locked.db"
    get_connection(db).close()
    os.chmod(db, 0o444)
    try:
        problem = writability_problem(db)
        assert problem, "a read-only database was reported as fine"
        assert str(db) in problem, "the message does not say which file"
        assert "chown" in problem, "the message does not say how to fix it"
    finally:
        os.chmod(db, 0o644)


def test_a_missing_database_in_a_writable_directory_is_fine(tmp_path):
    from src.database import writability_problem

    assert writability_problem(tmp_path / "not-created-yet.db") is None
