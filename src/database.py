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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_DB_PATH = Path(
    os.environ.get("DB_PATH") or str(Path(__file__).parent.parent / "pr1mealazyer.db")
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
"""


def get_connection(db_path: str | Path = _DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't already exist."""
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
    uploaded_at = datetime.now(tz=timezone.utc).isoformat()
    if match_date is None:
        match_date = datetime.now(tz=timezone.utc).date().isoformat()

    conn.execute(
        """
        INSERT INTO matches (
            match_id, filename, date, map_name, player_steam_id, player_name,
            total_rounds, kills, deaths, assists,
            kast, adr, kpr, dpr, impact, hltv_rating,
            kd_ratio, rounds_2k, rounds_3k, rounds_4k, rounds_5k,
            team_score, enemy_score, match_result,
            aim_stats, role_data, utility_data, context_notes, uploaded_at
        ) VALUES (
            :match_id, :filename, :date, :map_name, :player_steam_id,
            :player_name, :total_rounds, :kills, :deaths, :assists,
            :kast, :adr, :kpr, :dpr, :impact, :hltv_rating,
            :kd_ratio, :rounds_2k, :rounds_3k, :rounds_4k, :rounds_5k,
            :team_score, :enemy_score, :match_result,
            :aim_stats, :role_data, :utility_data, :context_notes, :uploaded_at
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


def get_round_stats(
    conn: sqlite3.Connection, match_id: str
) -> list[dict[str, Any]]:
    """Return per-round stats for *match_id*, ordered by round number."""
    cursor = conn.execute(
        "SELECT * FROM round_stats WHERE match_id = ? ORDER BY round_number",
        (match_id,),
    )
    return [dict(row) for row in cursor.fetchall()]


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
         datetime.now(tz=timezone.utc).isoformat()),
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
