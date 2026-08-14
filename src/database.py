"""
Layer 3: Storage / Database
Zero-configuration SQLite persistence for matches, per-round timelines, and
user-supplied context tags.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Falls back to data/, matching both the README and the DB_PATH that
# docker-compose sets. Without the data/ segment a local `uvicorn api:app` run
# would use a different database from the containerised one.
_DEFAULT_DB_PATH = Path(
    os.environ.get("DB_PATH")
    or str(Path(__file__).parent.parent / "data" / "pr1mealazyer.db")
)

_DDL = """
CREATE TABLE IF NOT EXISTS matches (
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
    aim_stats     TEXT,
    role_data     TEXT,
    utility_data  TEXT,
    impact_stats  TEXT,
    partial_import INTEGER DEFAULT 0,
    parse_mode    TEXT,
    parse_warning TEXT,
    source_patch_version INTEGER,
    analyzer_version INTEGER DEFAULT 0,
    context_notes TEXT,
    uploaded_at   TEXT
);

CREATE TABLE IF NOT EXISTS round_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id     TEXT    NOT NULL,
    round_number INTEGER,
    kills        INTEGER,
    deaths       INTEGER,
    assists      INTEGER,
    damage       INTEGER,
    survived     INTEGER,
    traded       INTEGER,
    enriched_json TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE TABLE IF NOT EXISTS context_tags (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    tag      TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE TABLE IF NOT EXISTS match_players (
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
    rank         INTEGER DEFAULT 0,
    rank_old     INTEGER DEFAULT 0,
    rank_change  REAL    DEFAULT 0.0,
    rank_type_id INTEGER DEFAULT 0,
    comp_wins    INTEGER DEFAULT 0,
    mvps         INTEGER DEFAULT 0,
    rounds_2k    INTEGER DEFAULT 0,
    rounds_3k    INTEGER DEFAULT 0,
    rounds_4k    INTEGER DEFAULT 0,
    rounds_5k    INTEGER DEFAULT 0,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE TABLE IF NOT EXISTS ai_chats (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id   TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    provider   TEXT,
    model      TEXT,
    created_at TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

-- Every child table is read by match_id, and none of them had an index, so
-- each lookup was a full scan. That is cheap on context_tags and brutal on
-- round_stats, where the replay frames make the table two orders of magnitude
-- larger than every other table combined.
CREATE INDEX IF NOT EXISTS idx_round_stats_match  ON round_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_match_players_match ON match_players(match_id);
CREATE INDEX IF NOT EXISTS idx_context_tags_match ON context_tags(match_id);
CREATE INDEX IF NOT EXISTS idx_ai_chats_match     ON ai_chats(match_id);
CREATE INDEX IF NOT EXISTS idx_matches_player     ON matches(player_steam_id);
"""


class _Connection(sqlite3.Connection):
    """A connection with room to hang a cache off.

    sqlite3.Connection is a C type that allows neither weak references nor
    attribute assignment, so caching anything per connection needs a subclass.
    Anything cached here dies when the connection does, which is the point:
    the column list it holds describes *this* database.
    """

    _round_columns: str | None = None


def writability_problem(db_path: str | Path = _DEFAULT_DB_PATH) -> str | None:
    """Why the database cannot be written to, in words, or None if it can.

    SQLite says "attempt to write a readonly database" and stops there. It does
    not say which file, or that the reason is ownership, so the error arrives
    attached to whatever the user was doing — importing a demo — and points
    nowhere near the cause. This is checked at startup so the answer is in the
    log before anything fails.
    """
    path = Path(db_path)
    if str(path) == ":memory:":
        return None

    if path.exists():
        if os.access(path, os.W_OK):
            return None
        owner = ""
        if hasattr(os, "getuid"):  # POSIX only; Windows has no uid to report
            try:
                owner = (
                    f" It is owned by uid {path.stat().st_uid}, and this process "
                    f"is uid {os.getuid()}."
                )
            except OSError:
                pass
        return (
            f"The database at {path} is not writable.{owner} In Docker this "
            f"usually means it was created by an older image that ran as root. "
            f"Fix it with:\n"
            f"    docker compose exec -u root api chown -R 1000:1000 /app/data"
        )

    parent = path.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        return f"The directory {parent} is not writable, so the database cannot be created."
    return None


def get_connection(db_path: str | Path = _DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, factory=_Connection)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't already exist."""
    # The migrations below can change round_stats, so drop this connection's
    # cached column list rather than let a later ALTER go unnoticed.
    _forget_round_columns(conn)
    conn.executescript(_DDL)
    conn.commit()
    # Migrate: add enriched_json to round_stats if missing
    cursor = conn.execute("PRAGMA table_info(round_stats)")
    columns = {row["name"] for row in cursor.fetchall()}
    if "enriched_json" not in columns:
        conn.execute("ALTER TABLE round_stats ADD COLUMN enriched_json TEXT")
        conn.commit()
    # Migrate: add rank to match_players if missing
    cursor = conn.execute("PRAGMA table_info(match_players)")
    columns = {row["name"] for row in cursor.fetchall()}
    if "rank" not in columns:
        conn.execute("ALTER TABLE match_players ADD COLUMN rank INTEGER DEFAULT 0")
        conn.commit()
    # Migrate: add rank_old, rank_change, comp_wins, mvps to match_players
    cursor = conn.execute("PRAGMA table_info(match_players)")
    columns = {row["name"] for row in cursor.fetchall()}
    for col, definition in [
        ("rank_old", "INTEGER DEFAULT 0"),
        ("rank_change", "REAL DEFAULT 0.0"),
        ("rank_type_id", "INTEGER DEFAULT 0"),
        ("comp_wins", "INTEGER DEFAULT 0"),
        ("mvps", "INTEGER DEFAULT 0"),
    ]:
        if col not in columns:
            conn.execute(f"ALTER TABLE match_players ADD COLUMN {col} {definition}")
    conn.commit()
    # Migrate: add aim_stats to matches if missing
    cursor = conn.execute("PRAGMA table_info(matches)")
    columns = {row["name"] for row in cursor.fetchall()}
    if "aim_stats" not in columns:
        conn.execute("ALTER TABLE matches ADD COLUMN aim_stats TEXT")
        conn.commit()
    if "role_data" not in columns:
        conn.execute("ALTER TABLE matches ADD COLUMN role_data TEXT")
        conn.commit()
    if "utility_data" not in columns:
        conn.execute("ALTER TABLE matches ADD COLUMN utility_data TEXT")
        conn.commit()
    if "impact_stats" not in columns:
        conn.execute("ALTER TABLE matches ADD COLUMN impact_stats TEXT")
        conn.commit()
    for col, definition in [
        ("partial_import", "INTEGER DEFAULT 0"),
        ("parse_mode", "TEXT"),
        ("parse_warning", "TEXT"),
        ("source_patch_version", "INTEGER"),
        # Rows that predate versioning default to 0, which reads as "older
        # than anything current" and so shows up as stale.
        ("analyzer_version", "INTEGER DEFAULT 0"),
    ]:
        if col not in columns:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {col} {definition}")
    conn.commit()
    # Migrate: add replay_json to round_stats if missing
    cursor = conn.execute("PRAGMA table_info(round_stats)")
    columns = {row["name"] for row in cursor.fetchall()}
    if "replay_json" not in columns:
        conn.execute("ALTER TABLE round_stats ADD COLUMN replay_json TEXT")
        conn.commit()


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------


def save_match(
    conn: sqlite3.Connection,
    stats: dict[str, Any],
    filename: str,
    steam_id: str,
    context_notes: str = "",
    match_date: str | None = None,
) -> str:
    """
    Persist a full match result (aggregate stats + per-round timeline).

    Args:
        conn: Active SQLite connection.
        stats: Dict returned by :func:`src.processor.calculate_match_stats`.
        filename: Original demo filename.
        steam_id: Player's 64-bit Steam ID.
        context_notes: Free-text notes entered by the user.
        match_date: ISO-8601 date string; defaults to current UTC date.

    Returns:
        The generated ``match_id`` (UUID).
    """
    match_id = str(uuid.uuid4())
    uploaded_at = datetime.now(tz=UTC).isoformat()
    if match_date is None:
        match_date = datetime.now(tz=UTC).date().isoformat()

    conn.execute(
        """
        INSERT INTO matches (
            match_id, filename, date, map_name, player_steam_id, player_name,
            total_rounds, kills, deaths, assists,
            kast, adr, kpr, dpr, impact, hltv_rating,
            kd_ratio, rounds_2k, rounds_3k, rounds_4k, rounds_5k,
            team_score, enemy_score, match_result,
            aim_stats, role_data, utility_data, impact_stats,
            partial_import, parse_mode, parse_warning, source_patch_version,
            analyzer_version, context_notes, uploaded_at
        ) VALUES (
            :match_id, :filename, :date, :map_name, :player_steam_id,
            :player_name, :total_rounds, :kills, :deaths, :assists,
            :kast, :adr, :kpr, :dpr, :impact, :hltv_rating,
            :kd_ratio, :rounds_2k, :rounds_3k, :rounds_4k, :rounds_5k,
            :team_score, :enemy_score, :match_result,
            :aim_stats, :role_data, :utility_data, :impact_stats,
            :partial_import, :parse_mode, :parse_warning, :source_patch_version,
            :analyzer_version, :context_notes, :uploaded_at
        )
        """,
        {
            "match_id": match_id,
            "filename": filename,
            "date": match_date,
            "map_name": stats.get("map_name", "unknown"),
            "player_steam_id": steam_id,
            "player_name": stats.get("player_name", "Unknown"),
            "total_rounds": stats.get("total_rounds", 0),
            "kills": stats.get("kills", 0),
            "deaths": stats.get("deaths", 0),
            "assists": stats.get("assists", 0),
            "kast": stats.get("kast", 0.0),
            "adr": stats.get("adr", 0.0),
            "kpr": stats.get("kpr", 0.0),
            "dpr": stats.get("dpr", 0.0),
            "impact": stats.get("impact", 0.0),
            "hltv_rating": stats.get("hltv_rating", 0.0),
            "kd_ratio": stats.get("kd_ratio", 0.0),
            "rounds_2k": stats.get("rounds_2k", 0),
            "rounds_3k": stats.get("rounds_3k", 0),
            "rounds_4k": stats.get("rounds_4k", 0),
            "rounds_5k": stats.get("rounds_5k", 0),
            "team_score": stats.get("team_score", 0),
            "enemy_score": stats.get("enemy_score", 0),
            "match_result": stats.get("match_result", "unknown"),
            "aim_stats": json.dumps(stats.get("aim_stats")) if stats.get("aim_stats") else None,
            "role_data": json.dumps(stats.get("role_data")) if stats.get("role_data") else None,
            "utility_data": json.dumps(stats.get("utility_data")) if stats.get("utility_data") else None,
            "impact_stats": json.dumps(stats.get("impact_stats")) if stats.get("impact_stats") else None,
            "partial_import": int(bool(stats.get("partial_import", False))),
            "parse_mode": stats.get("parse_mode"),
            "parse_warning": stats.get("parse_warning"),
            "source_patch_version": stats.get("source_patch_version"),
            "analyzer_version": int(stats.get("analyzer_version") or 0),
            "context_notes": context_notes,
            "uploaded_at": uploaded_at,
        },
    )

    # Persist per-round stats (with enriched data if available)
    enriched_rounds = stats.get("enriched_rounds", [])
    enriched_by_round = {er["round"]: er for er in enriched_rounds}
    replay_data = stats.get("replay_data") or {}
    for rs in stats.get("round_stats", []):
        enriched = enriched_by_round.get(rs["round"])
        enriched_str = json.dumps(enriched) if enriched else None
        replay = replay_data.get(rs["round"])
        replay_str = json.dumps(replay) if replay else None
        conn.execute(
            """
            INSERT INTO round_stats
                (match_id, round_number, kills, deaths, assists,
                 damage, survived, traded, enriched_json, replay_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                rs["round"],
                rs["kills"],
                rs["deaths"],
                rs["assists"],
                rs["damage"],
                rs["survived"],
                rs["traded"],
                enriched_str,
                replay_str,
            ),
        )

    # Persist all-player scoreboard
    for p in stats.get("all_players", []):
        conn.execute(
            """
            INSERT INTO match_players
                (match_id, steam_id, name, team, is_user,
                 kills, deaths, assists, kd_ratio,
                 adr, kast, hltv_rating, rank,
                 rank_old, rank_change, rank_type_id, comp_wins, mvps,
                 rounds_2k, rounds_3k, rounds_4k, rounds_5k)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                p["steam_id"],
                p["name"],
                p.get("team"),
                int(p.get("is_user", False)),
                p["kills"],
                p["deaths"],
                p["assists"],
                p["kd_ratio"],
                p["adr"],
                p["kast"],
                p["hltv_rating"],
                p.get("rank", 0),
                p.get("rank_old", 0),
                p.get("rank_change", 0.0),
                p.get("rank_type_id", 0),
                p.get("comp_wins", 0),
                p.get("mvps", 0),
                p.get("rounds_2k", 0),
                p.get("rounds_3k", 0),
                p.get("rounds_4k", 0),
                p.get("rounds_5k", 0),
            ),
        )

    conn.commit()
    return match_id


def add_tag(conn: sqlite3.Connection, match_id: str, tag: str) -> None:
    """Add a context tag to a match."""
    conn.execute(
        "INSERT INTO context_tags (match_id, tag) VALUES (?, ?)",
        (match_id, tag.strip()),
    )
    conn.commit()


def update_context_notes(
    conn: sqlite3.Connection, match_id: str, notes: str
) -> None:
    """Update the free-text context notes for a stored match."""
    conn.execute(
        "UPDATE matches SET context_notes = ? WHERE match_id = ?",
        (notes, match_id),
    )
    conn.commit()


def get_all_matches(conn: sqlite3.Connection, player_steam_id: str | None = None) -> list[dict[str, Any]]:
    """Return all matches ordered by date descending, optionally filtered by player."""
    if player_steam_id:
        cursor = conn.execute(
            "SELECT * FROM matches WHERE player_steam_id = ? ORDER BY date DESC, uploaded_at DESC",
            (player_steam_id,),
        )
    else:
        cursor = conn.execute(
            "SELECT * FROM matches ORDER BY date DESC, uploaded_at DESC"
        )
    return [dict(row) for row in cursor.fetchall()]


def get_match(conn: sqlite3.Connection, match_id: str) -> dict[str, Any] | None:
    """Return a single match by its UUID."""
    cursor = conn.execute(
        "SELECT * FROM matches WHERE match_id = ?", (match_id,)
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def _round_stats_columns(conn: sqlite3.Connection) -> str:
    """Column list for round_stats with replay_json left out.

    Read from the schema rather than hardcoded so a future column is included
    automatically — the point is to exclude one known-huge column, not to pin
    the shape of the table.

    Cached per connection rather than per process. It used to be a module
    global with no lock, which meant the column list belonged to whichever
    database connected most recently: with an in-memory test database and a
    real one alive at the same time, one could be queried using the other's
    columns.
    """
    cached: str | None = getattr(conn, "_round_columns", None)
    if cached is None:
        names = [row["name"] for row in conn.execute("PRAGMA table_info(round_stats)")]
        cached = ", ".join(n for n in names if n != "replay_json") or "*"
        try:
            conn._round_columns = cached  # type: ignore[attr-defined]
        except AttributeError:
            pass  # a plain sqlite3.Connection: correct, just uncached
    return cached


def _forget_round_columns(conn: sqlite3.Connection) -> None:
    try:
        conn._round_columns = None  # type: ignore[attr-defined]
    except AttributeError:
        pass


def get_round_stats(
    conn: sqlite3.Connection, match_id: str, include_replay: bool = False
) -> list[dict[str, Any]]:
    """Return per-round stats for *match_id*, ordered by round number.

    ``replay_json`` holds the 2D replay frames — roughly 40 KB a round, and
    about 95% of the database. Only the replay viewer reads it, so it is
    excluded unless *include_replay* is set. Every other caller walks rounds to
    total up kills or damage and would otherwise pay to fetch and discard
    megabytes per request.
    """
    columns = "*" if include_replay else _round_stats_columns(conn)
    cursor = conn.execute(
        f"SELECT {columns} FROM round_stats WHERE match_id = ? ORDER BY round_number",
        (match_id,),
    )
    return [dict(row) for row in cursor.fetchall()]


# SQLite's default limit on host parameters is 999. Batching well under it
# keeps one query per chunk rather than one per match.
_MAX_PARAMS = 500


def get_rounds_for_matches(
    conn: sqlite3.Connection, match_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Rounds for several matches at once, with ``enriched_json`` decoded.

    The analytics endpoints used to loop over match IDs issuing one query each,
    then re-implement the same ``json.loads`` with a bare except in four
    separate places. Five hundred matches meant five hundred round trips.

    Rows come back grouped by the order of *match_ids* and ordered by round
    number within each match, which is the order the per-match loop produced.
    ``replay_json`` is excluded, as in :func:`get_round_stats`.
    """
    ordered_ids = list(match_ids)
    if not ordered_ids:
        return []

    columns = _round_stats_columns(conn)
    by_match: dict[str, list[dict[str, Any]]] = {}

    for start in range(0, len(ordered_ids), _MAX_PARAMS):
        chunk = ordered_ids[start:start + _MAX_PARAMS]
        placeholders = ", ".join("?" * len(chunk))
        cursor = conn.execute(
            f"SELECT {columns} FROM round_stats "
            f"WHERE match_id IN ({placeholders}) ORDER BY match_id, round_number",
            chunk,
        )
        for row in cursor.fetchall():
            record = dict(row)
            raw = record.get("enriched_json")
            try:
                record["enriched"] = json.loads(raw) if raw else {}
            except ValueError:
                record["enriched"] = {}
            by_match.setdefault(record["match_id"], []).append(record)

    return [row for match_id in ordered_ids for row in by_match.get(match_id, [])]


def get_imported_demo_files(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Every demo filename already imported, keyed by name.

    Carries the match date, whose account it belongs to, and how many rounds
    have replay frames stored. That last count is what makes a demo safe to
    delete: without it the 2D viewer could never be populated for the match,
    and the file is not recoverable.

    The query used to sit inline in api.py, which put schema knowledge in the
    route layer.
    """
    cursor = conn.execute(
        """
        SELECT m.filename, m.date, m.player_steam_id,
               COUNT(rs.replay_json) AS replay_rounds
        FROM matches m
        LEFT JOIN round_stats rs ON rs.match_id = m.match_id
        WHERE m.filename IS NOT NULL
        GROUP BY m.filename, m.date, m.player_steam_id
        """
    )
    return {row["filename"]: dict(row) for row in cursor.fetchall()}


def get_tags(conn: sqlite3.Connection, match_id: str) -> list[str]:
    """Return all tags associated with *match_id*."""
    cursor = conn.execute(
        "SELECT tag FROM context_tags WHERE match_id = ?", (match_id,)
    )
    return [row["tag"] for row in cursor.fetchall()]


def delete_match(conn: sqlite3.Connection, match_id: str) -> None:
    """Delete a match and all associated data (rounds, players, tags, chats)."""
    conn.execute("DELETE FROM ai_chats WHERE match_id = ?", (match_id,))
    conn.execute("DELETE FROM match_players WHERE match_id = ?", (match_id,))
    conn.execute("DELETE FROM round_stats WHERE match_id = ?", (match_id,))
    conn.execute("DELETE FROM context_tags WHERE match_id = ?", (match_id,))
    conn.execute("DELETE FROM matches WHERE match_id = ?", (match_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# AI Chat History
# ---------------------------------------------------------------------------


def save_chat_message(
    conn: sqlite3.Connection,
    match_id: str,
    role: str,
    content: str,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Persist a single chat message (user or assistant)."""
    conn.execute(
        "INSERT INTO ai_chats (match_id, role, content, provider, model, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (match_id, role, content, provider, model,
         datetime.now(tz=UTC).isoformat()),
    )
    conn.commit()


def get_chat_history(
    conn: sqlite3.Connection, match_id: str
) -> list[dict[str, Any]]:
    """Return all chat messages for a match, ordered chronologically."""
    cursor = conn.execute(
        "SELECT role, content, provider, model, created_at "
        "FROM ai_chats WHERE match_id = ? ORDER BY id",
        (match_id,),
    )
    return [dict(row) for row in cursor.fetchall()]


def clear_chat_history(conn: sqlite3.Connection, match_id: str) -> None:
    """Delete all chat messages for a match."""
    conn.execute("DELETE FROM ai_chats WHERE match_id = ?", (match_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Batch reads
#
# The analytics pages walk every match the user owns. Asking per match means a
# round trip per match; these ask once. Each returns rows grouped in the order
# the caller passed its ids, which is the order the per-match loops produced.
# ---------------------------------------------------------------------------


def get_players_for_matches(
    conn: sqlite3.Connection, match_ids: Sequence[str], *, user_only: bool = False
) -> list[dict[str, Any]]:
    """Scoreboard rows for several matches at once.

    *user_only* keeps just the tracked player, which is what the career
    aggregates want — otherwise nine rows per match are fetched and discarded.
    """
    ordered_ids = list(match_ids)
    if not ordered_ids:
        return []

    by_match: dict[str, list[dict[str, Any]]] = {}
    condition = " AND is_user = 1" if user_only else ""
    for start in range(0, len(ordered_ids), _MAX_PARAMS):
        chunk = ordered_ids[start:start + _MAX_PARAMS]
        placeholders = ", ".join("?" * len(chunk))
        cursor = conn.execute(
            f"SELECT * FROM match_players WHERE match_id IN ({placeholders})"
            f"{condition} ORDER BY match_id, team, kills DESC",
            chunk,
        )
        for row in cursor.fetchall():
            record = dict(row)
            by_match.setdefault(record["match_id"], []).append(record)

    return [row for match_id in ordered_ids for row in by_match.get(match_id, [])]


def get_tags_for_all_matches(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Every tag, grouped by match. One query rather than one per row."""
    tags: dict[str, list[str]] = {}
    for row in conn.execute("SELECT match_id, tag FROM context_tags"):
        tags.setdefault(row["match_id"], []).append(row["tag"])
    return tags


def get_analyzer_versions(conn: sqlite3.Connection) -> list[int]:
    """The analyzer version of every stored match, for the staleness count."""
    cursor = conn.execute("SELECT COALESCE(analyzer_version, 0) AS v FROM matches")
    return [int(row["v"] or 0) for row in cursor.fetchall()]


def get_imported_filenames(
    conn: sqlite3.Connection, player_steam_id: str = ""
) -> set[str]:
    """Demo filenames already in the database, optionally for one account.

    Used by the sync scan to work out which files on disk are new.
    """
    if player_steam_id.strip():
        cursor = conn.execute(
            "SELECT filename FROM matches WHERE player_steam_id = ?",
            (player_steam_id.strip(),),
        )
    else:
        cursor = conn.execute("SELECT filename FROM matches")
    return {row["filename"] for row in cursor.fetchall() if row["filename"]}


def get_enriched_json_for_map(
    conn: sqlite3.Connection, map_name: str
) -> list[str]:
    """Raw enriched_json for every round played on *map_name*.

    Feeds the minimap heat data, which pools positions across matches.
    """
    cursor = conn.execute(
        "SELECT rs.enriched_json FROM round_stats rs "
        "JOIN matches m ON rs.match_id = m.match_id "
        "WHERE m.map_name = ? AND rs.enriched_json IS NOT NULL",
        (map_name,),
    )
    return [row["enriched_json"] for row in cursor.fetchall()]


def move_chat_history(
    conn: sqlite3.Connection, old_match_id: str, new_match_id: str
) -> int:
    """Re-point a match's chat history at a new match_id, returning the count.

    Re-analysis writes a new match row and deletes the old one, and
    :func:`delete_match` takes the chat history with it.  Re-pointing the rows
    keeps the original ``created_at`` values and ordering, which re-inserting
    them through :func:`save_chat_message` would not.  Must run before the old
    match is deleted.
    """
    cursor = conn.execute(
        "UPDATE ai_chats SET match_id = ? WHERE match_id = ?",
        (new_match_id, old_match_id),
    )
    conn.commit()
    return cursor.rowcount or 0


def get_match_players(
    conn: sqlite3.Connection, match_id: str
) -> list[dict[str, Any]]:
    """Return all player rows for a match, ordered by team then kills desc."""
    cursor = conn.execute(
        "SELECT * FROM match_players WHERE match_id = ? "
        "ORDER BY is_user DESC, team, kills DESC",
        (match_id,),
    )
    return [dict(row) for row in cursor.fetchall()]
