"""Turning a .dem file into a stored match.

Four routes did this — single upload, bulk upload, sync folder, and the
reimport/reanalyze pair — and each had written out the same three lines:

    parsed = parse_demo(path)
    stats = calculate_match_stats(parsed, sid)
    _apply_parse_metadata(stats, parsed)

along with its own copy of "resolve which Steam ID this is", "read the .dem.info
sidecar", and "find the account name for the response". The copies had begun to
disagree about details; the sidecar readers, for instance, caught different
exceptions.

Exceptions are deliberately not translated here. The single upload turns a parse
failure into a 422 with a logged traceback; the bulk and folder routes record it
against one file and carry on with the rest. Both are right for their caller, so
this module raises and lets them decide.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.database import add_tag, save_match
from src.parser import parse_demo, parse_info_file
from src.processor import calculate_match_stats


@dataclass(frozen=True)
class SidecarInfo:
    """What a .dem.info sidecar tells us. Absent or unreadable reads as empty."""

    match_date: str | None = None
    account_ids: tuple[str, ...] = ()

    def first_known(self, known: Collection[str]) -> str | None:
        """The first account in this match that the user actually owns.

        This is how a bulk import works out which of several configured
        accounts a demo belongs to, rather than attributing it to whichever is
        currently active.
        """
        for account_id in self.account_ids:
            if account_id in known:
                return account_id
        return None


def read_sidecar(source: bytes | str | Path) -> SidecarInfo:
    """Read a .dem.info sidecar from bytes or a path. Never raises.

    A missing or corrupt sidecar costs the match date and account detection,
    which the caller can fall back from; it is not a reason to refuse a demo
    that parses perfectly well.
    """
    try:
        raw = source if isinstance(source, bytes) else Path(source).read_bytes()
        data = parse_info_file(raw)
    except Exception:
        return SidecarInfo()

    return SidecarInfo(
        match_date=data.get("match_date"),
        account_ids=tuple(str(a) for a in data.get("account_ids", [])),
    )


def resolve_steam_id(*candidates: str | None) -> str:
    """The first candidate with something in it, stripped.

    Callers pass their preference order — usually the request field, then the
    active account, then the legacy data/steamID file.
    """
    for candidate in candidates:
        if candidate and candidate.strip():
            return candidate.strip()
    return ""


def apply_parse_metadata(stats: dict[str, Any], parsed: dict[str, Any]) -> None:
    """Copy parser compatibility metadata into stats for persistence and UI."""
    header = parsed.get("header", {}) if isinstance(parsed, dict) else {}
    parse_mode = header.get("parse_mode")
    parse_warning = header.get("parse_warning")
    patch_version = header.get("patch_version")

    if parse_mode:
        stats["parse_mode"] = str(parse_mode)
    if parse_warning:
        stats["parse_warning"] = str(parse_warning)
    if parse_mode == "header_only_fallback" or parse_warning:
        stats["partial_import"] = True

    if patch_version is not None:
        try:
            stats["source_patch_version"] = int(patch_version)
        except (TypeError, ValueError):
            pass


def analyse_demo(demo_path: str | Path, steam_id: str) -> dict[str, Any]:
    """Parse a demo and compute the full statistics for *steam_id*.

    Raises whatever the parser or processor raises.
    """
    parsed = parse_demo(str(demo_path))
    stats = calculate_match_stats(parsed, steam_id)
    apply_parse_metadata(stats, parsed)
    return stats


def store_match(
    conn: sqlite3.Connection,
    stats: dict[str, Any],
    *,
    filename: str,
    steam_id: str,
    match_date: str | None = None,
    context_notes: str = "",
    tags: Iterable[str] = (),
) -> str:
    """Persist an analysed match and its tags. Returns the new match_id."""
    match_id = save_match(
        conn,
        stats,
        filename=filename,
        steam_id=steam_id,
        context_notes=context_notes,
        match_date=match_date,
    )
    for tag in tags:
        cleaned = tag.strip()
        if cleaned:
            add_tag(conn, match_id, cleaned)
    return match_id


def account_name_for(accounts: Iterable[dict[str, Any]], steam_id: str) -> str | None:
    """The configured name for *steam_id*, so a result can say whose match it is."""
    for account in accounts:
        if account.get("steam_id") == steam_id:
            return account.get("name")
    return None
