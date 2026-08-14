"""Shared application state and the accessors the routers build on.

The composition root: the database connection factory, the JSON config stores,
the single Steam job runner, the Auto-Sync loop and its wiring. Everything here
is used by more than one router, or is a singleton that must not be constructed
twice.

Router modules import from here. Nothing here imports a router — with one
documented exception, the Auto-Sync import step, which is resolved lazily
because the work it does is a route.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from src.ai_service import (
    PROVIDERS as AI_PROVIDERS,
    load_config as load_ai_config,
)
from src.config import store as store_module
from src.config.settings import DATA_DIR, FETCHER_DIR
from src.config.store import JsonStore
from src.database import get_connection, get_imported_demo_files
from src.services.auto_sync import (
    DEFAULTS as auto_sync_defaults,
    AutoSync,
    AutoSyncOps,
    backoff as auto_sync_backoff,
)
from src.services.demo_files import (
    FETCHED_PREFIX,
    STORAGE_DEFAULTS,
    DemoFolderUnusable,
    available_demo_names,
    classify,
    delete_deletable,
    next_unimported,
    resolve_demo,
    scan_folder,
    search_dirs,
)
from src.services.import_service import resolve_steam_id
from src.services.steam_jobs import (
    FetcherUnavailable,
    JobBusy,
    SteamJobRunner,
    utc_now,
)


# ---------------------------------------------------------------------------
# Config files under data/
#
# Each is a JsonStore: reads merge the declared defaults under whatever is on
# disk, writes go via a temporary file so a crash cannot leave a half-written
# config, and `private` restricts the file to its owner. See src/config/store.py.
# ---------------------------------------------------------------------------
def steam_id_path() -> Path:
    """The legacy plain-text account file.

    A function rather than a constant so it resolves the data directory when it
    is used, the way JsonStore does. As a constant it was bound at import, so
    anything redirecting the data directory afterwards — a test, or a future
    per-profile layout — silently kept writing to the original location.

    Resolved through src.config.store rather than this module's own import, so
    there is one runtime authority for where data lives instead of a copy per
    module that imported it.
    """
    return store_module.DATA_DIR / "steamID"


_DEFAULT_STEAM_ID: str | None = None


_accounts_store = JsonStore("accounts.json")


_friends_store = JsonStore("friends.json")


def _load_accounts() -> list[dict]:
    """Load accounts from data/accounts.json."""
    return _accounts_store.read_list("accounts")


def _save_accounts(accounts: list[dict]) -> None:
    """Persist accounts to data/accounts.json."""
    _accounts_store.write_list("accounts", accounts)


def _get_active_account() -> dict | None:
    """Return the currently active account, if any."""
    for acct in _load_accounts():
        if acct.get("active"):
            return acct
    return None


def _legacy_steam_id() -> str:
    """The pre-accounts data/steamID file, still honoured as a last resort."""
    if steam_id_path().exists():
        return steam_id_path().read_text().strip()
    return ""


def _resolve_request_steam_id(explicit: str = "") -> str:
    """Whose match this is: the request says, else the active account, else the file."""
    active = _get_active_account()
    return resolve_steam_id(
        explicit,
        active["steam_id"] if active else None,
        _legacy_steam_id(),
    )


def _load_friends() -> list[dict]:
    """Load friends from data/friends.json."""
    return _friends_store.read_list("friends")


def _save_friends(friends: list[dict]) -> None:
    """Persist friends to data/friends.json."""
    _friends_store.write_list("friends", friends)


def _db():
    return get_connection()


def _match_summary(stats: dict[str, Any]) -> dict[str, Any]:
    """Build a concise response from processor stats."""
    return {
        "player_name": stats.get("player_name"),
        "map_name": stats.get("map_name"),
        "total_rounds": stats.get("total_rounds"),
        "kills": stats.get("kills"),
        "deaths": stats.get("deaths"),
        "assists": stats.get("assists"),
        "kd_ratio": stats.get("kd_ratio"),
        "adr": stats.get("adr"),
        "kast": stats.get("kast"),
        "impact": stats.get("impact"),
        "hltv_rating": stats.get("hltv_rating"),
        "team_score": stats.get("team_score"),
        "enemy_score": stats.get("enemy_score"),
        "match_result": stats.get("match_result"),
        "rounds_2k": stats.get("rounds_2k"),
        "rounds_3k": stats.get("rounds_3k"),
        "rounds_4k": stats.get("rounds_4k"),
        "rounds_5k": stats.get("rounds_5k"),
        "partial_import": bool(stats.get("partial_import", False)),
        "parse_mode": stats.get("parse_mode"),
        "parse_warning": stats.get("parse_warning"),
    }


# ---------------------------------------------------------------------------
# Folder sync
# ---------------------------------------------------------------------------
_sync_config_store = JsonStore("sync_config.json")


def _load_sync_config() -> dict:
    return _sync_config_store.read()


def _save_sync_config(cfg: dict) -> None:
    _sync_config_store.write(cfg)


# ---------------------------------------------------------------------------
# Onboarding state
# ---------------------------------------------------------------------------
_onboarding_store = JsonStore("onboarding.json", defaults={"completed": False})


def _load_onboarding() -> dict:
    return _onboarding_store.read()


def _save_onboarding(state: dict) -> None:
    _onboarding_store.write(state)


# ---------------------------------------------------------------------------
# Demo files on disk
#
# The logic lives in src/services/demo_files.py. These wrappers supply what it
# needs from config and the database, and translate DemoFolderUnusable into the
# 400 the API has always returned for an unset or missing folder.
# ---------------------------------------------------------------------------
_STORAGE_DEFAULTS = STORAGE_DEFAULTS


_FETCHED_PREFIX = FETCHED_PREFIX


def _demo_search_dirs() -> list[Path]:
    return search_dirs(_load_sync_config().get("folder"), DATA_DIR)


def _resolve_demo(filename: str | None) -> Path | None:
    return resolve_demo(filename, _demo_search_dirs())


def _available_demo_names() -> set[str]:
    return available_demo_names(_demo_search_dirs())


def _analyse_demo_folder(overrides: dict[str, Any] | None = None) -> dict:
    """Classify every .dem in the sync folder as protected, deletable, or neither.

    ``overrides`` applies settings for this calculation only, without saving
    them, so the UI can preview what a retention number would free before the
    user commits to it.
    """
    cfg = _load_storage_config()
    # Only keys actually supplied count as an override — a dict of all-None
    # values means the caller passed no query parameters at all.
    applied = {k: v for k, v in (overrides or {}).items() if v is not None}
    cfg.update(applied)

    try:
        base, found = scan_folder(_load_sync_config().get("folder"))
    except DemoFolderUnusable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    conn = _db()
    try:
        imported = get_imported_demo_files(conn)
    finally:
        conn.close()

    return classify(
        base=base,
        found=found,
        config=cfg,
        imported=imported,
        account_names={a["steam_id"]: a.get("name") for a in _load_accounts()},
        preview=bool(applied),
    )


def _next_unimported_demo(set_aside: set[str] | None = None) -> str | None:
    """The newest fetched demo not yet in the database.

    Raises HTTPException if the demo folder is unusable — swallowing that would
    leave auto-sync downloading match after match it can never import.
    """
    return next_unimported(_analyse_demo_folder(), set_aside or set())


def _run_demo_cleanup(dry_run: bool = False) -> dict:
    """Delete demos the analysis marked deletable, plus their sidecars."""
    return delete_deletable(_analyse_demo_folder(), dry_run)


_storage_config_store = JsonStore("storage_config.json", defaults=_STORAGE_DEFAULTS)


def _load_storage_config() -> dict:
    return _storage_config_store.read()


def _save_storage_config(cfg: dict) -> None:
    _storage_config_store.write(cfg)


# ---------------------------------------------------------------------------
# Steam fetcher — optional Node companion in fetcher/
#
# Deliberately separate from the Sync Folder path above. Sync Folder imports
# .dem files that are already on disk; the fetcher downloads them from Valve
# first. The two meet at the demo folder: the fetcher writes there, Sync Folder
# reads from there. Neither depends on the other.
#
# Authentication is not exposed here on purpose — it is interactive (QR scan or
# Steam Guard) and would mean handling Steam passwords in a web form. It stays
# a one-off terminal step; this API only reports whether it has been done.
# ---------------------------------------------------------------------------
_FETCHER_DIR = FETCHER_DIR


# Both hold credentials — Steam refresh tokens are password-equivalent, and the
# ledger carries the Web API key — so both are owner-only on disk.
_steam_tokens_store = JsonStore("steam_tokens.json", private=True)


_steam_ledger_store = JsonStore("steam_sharecodes.json", private=True)


_STEAM_TOKENS_FILE = _steam_tokens_store.path


_STEAM_LEDGER_FILE = _steam_ledger_store.path


_AUTH_CODE_RE = re.compile(r"^[A-Za-z0-9]{4}-[A-Za-z0-9]{5}-[A-Za-z0-9]{4}$")


_SHARE_CODE_RE = re.compile(
    r"^CSGO(-[ABCDEFGHJKLMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789]{5}){5}$"
)


# One fetcher job at a time — see src/services/steam_jobs.py. The UI polls
# /api/steam/job for its output.
_steam_jobs = SteamJobRunner(FETCHER_DIR)


_utc_now = utc_now


def _read_steam_ledger() -> dict:
    return _steam_ledger_store.read()


def _write_steam_ledger(data: dict) -> None:
    """Persist the ledger. Contains an API key and auth codes, so keep it tight."""
    _steam_ledger_store.write(data)


def _job_snapshot() -> dict:
    return _steam_jobs.snapshot()


def _job_log(text: str) -> None:
    _steam_jobs.log(text)


def _last_job_error() -> str | None:
    return _steam_jobs.last_error()


def _require_fetcher() -> None:
    try:
        _steam_jobs.require_available()
    except FetcherUnavailable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _job_claim(job_type: str, auto: bool = False) -> None:
    try:
        _steam_jobs.claim(job_type, auto=auto)
    except JobBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _job_release(exit_code: int) -> None:
    _steam_jobs.release(exit_code)


def _start_steam_job(job_type: str, args: list[str]) -> dict:
    _require_fetcher()
    _job_claim(job_type)
    threading.Thread(target=_steam_jobs.run, args=(args,), daemon=True).start()
    return _job_snapshot()


def _run_steam_job_sync(job_type: str, args: list[str]) -> int:
    """Same job, run to completion on the calling thread. Used by auto-sync."""
    _require_fetcher()
    _job_claim(job_type, auto=True)
    return _steam_jobs.run(args)


def _steam_account_rows() -> tuple[list[dict[str, Any]], int]:
    """Per-account fetcher state, and how many matches are outstanding overall.

    Shared by /api/steam/status and the auto-sync loop so both agree on what
    "outstanding" means.
    """
    accounts = _load_accounts()
    stored_tokens = _steam_tokens_store.read()
    ledger = _read_steam_ledger()
    ledger_accounts = ledger.get("accounts", {}) or {}

    rows: list[dict[str, Any]] = []
    pending_total = 0
    for account in accounts:
        name = account.get("name")
        entry = ledger_accounts.get(name, {}) or {}
        matches = entry.get("matches", {}) or {}

        tally = {"total": len(matches), "pending": 0, "downloaded": 0, "expired": 0, "failed": 0}
        for match in matches.values():
            status = match.get("status")
            if status in tally:
                tally[status] += 1

        # Absent means enabled, so ledgers written before the flags existed
        # keep behaving the way they did.
        walk_enabled = entry.get("walkEnabled") is not False
        download_enabled = entry.get("downloadEnabled") is not False

        # "failed" is retryable, so it counts as outstanding work alongside
        # pending — but only for accounts we are actually downloading.
        outstanding = tally["pending"] + tally["failed"]
        if download_enabled:
            pending_total += outstanding

        rows.append(
            {
                "name": name,
                "steam_id": account.get("steam_id"),
                "authenticated": name in stored_tokens,
                "configured": bool(entry.get("authCode")),
                "walk_enabled": walk_enabled,
                "download_enabled": download_enabled,
                "outstanding": outstanding,
                **tally,
            }
        )

    return rows, pending_total


# ---------------------------------------------------------------------------
# Auto-Sync
#
# The loop itself lives in src/services/auto_sync.py. Everything it needs the
# rest of the app to do arrives through AutoSyncOps, wired below. The lambdas
# resolve their targets at call time rather than binding them here, so tests
# can substitute any one of them.
# ---------------------------------------------------------------------------
_AUTO_SYNC_DEFAULTS = auto_sync_defaults


def _next_unimported_for(set_aside: set[str]) -> str | None:
    """Adapter: the loop asks in terms of a skiplist, and cannot see HTTP."""
    try:
        return _next_unimported_demo(set_aside)
    except HTTPException as exc:
        raise DemoFolderUnusable(str(exc.detail)) from exc


def _auto_import_one(filename: str) -> bool:
    """Parse and store a single demo, mirroring what Sync Folder does.

    Runs on the auto-sync thread and writes into the job log, so the modal
    shows the import in the same place as the download that preceded it.
    """
    from src.api.routers.imports import sync_process

    _job_claim("import", auto=True)
    _job_log(f"Importing {filename}")
    ok = False
    try:
        outcome = sync_process({"filenames": [filename]})
        entry = (outcome.get("results") or [{}])[0]
        if entry.get("status") == "ok":
            ok = True
            summary = " ".join(
                part for part in (entry.get("map_name"), entry.get("player_name")) if part
            )
            _job_log(f"Imported {filename}{' — ' + summary if summary else ''}")
            _auto_note(f"Imported {summary or filename}")
            if entry.get("parse_warning"):
                _job_log(f"Warning: {entry['parse_warning']}")
            cleanup = outcome.get("cleanup") or {}
            if cleanup.get("deleted_count"):
                freed = cleanup["freed_bytes"] / 1024 / 1024 / 1024
                _job_log(f"Cleanup removed {cleanup['deleted_count']} demo(s), {freed:.1f} GB freed")
        else:
            detail = entry.get("detail") or "import failed"
            _job_log(f"Failed: {detail}")
            _auto_note(f"Import failed for {filename}: {detail}")
    except HTTPException as exc:
        _job_log(f"Failed: {exc.detail}")
        _auto_note(f"Import failed: {exc.detail}")
    except Exception as exc:
        _job_log(f"Failed: {exc}")
        _auto_note(f"Import failed: {exc}")
    finally:
        _job_release(0 if ok else 1)
    return ok


_auto_sync = AutoSync(
    AutoSyncOps(
        load_config=lambda: _load_auto_sync_config(),
        is_playing=lambda: _check_cs2_presence(),
        next_unimported=lambda set_aside: _next_unimported_for(set_aside),
        import_one=lambda filename: _auto_import_one(filename),
        outstanding=lambda: _steam_account_rows()[1],
        run_job=lambda job_type, args: _run_steam_job_sync(job_type, args),
        last_job_error=lambda: _steam_jobs.last_error(),
    )
)


def _auto_note(text: str) -> None:
    _auto_sync.note(text)


def _auto_phase(phase: str, detail: str = "", wait_seconds: float | None = None) -> None:
    _auto_sync.set_phase(phase, detail, wait_seconds)


def _auto_sync_step(cfg: dict) -> float:
    return _auto_sync.step(cfg)


def _auto_backoff(streak: int) -> float:
    return auto_sync_backoff(streak)


def _start_auto_sync() -> None:
    _auto_sync.start()


def _auto_sync_snapshot() -> dict:
    cfg = _load_auto_sync_config()
    state = _auto_sync.snapshot()
    state["config"] = cfg
    state["enabled"] = bool(cfg["enabled"])
    state["running"] = _auto_sync.is_running()
    state["presence"] = _presence_cache["value"]
    return state


# CS2 presence is polled from the Steam Web API, which is rate limited and
# rarely changes between cycles, so the last answer is reused for a while.
_PRESENCE_TTL_SECONDS = 45


_presence_cache: dict[str, Any] = {"at": 0.0, "value": None}


_auto_sync_config_store = JsonStore("auto_sync.json", defaults=_AUTO_SYNC_DEFAULTS)


def _load_auto_sync_config() -> dict:
    return _auto_sync_config_store.read()


def _save_auto_sync_config(cfg: dict) -> None:
    _auto_sync_config_store.write(cfg)


# ─── CS2 presence ───────────────────────────────────────────────────────────
def _steam_api_key() -> str | None:
    """The Web API key, from the environment or the ledger."""
    return os.environ.get("STEAM_API_KEY") or _read_steam_ledger().get("apiKey")


def _check_cs2_presence(force: bool = False) -> dict:
    """Is any tracked account in CS2 right now?

    The app runs in a container and cannot see host processes, so this asks
    Steam instead: GetPlayerSummaries reports ``gameid`` for accounts whose
    "game details" privacy is public. When it is not public the field is simply
    absent, which is indistinguishable from "not playing" — so an unknown
    answer is reported as unknown rather than guessed at, and the loop carries
    on. Half a signal is still worth having; most people leave it public.

    Returns ``playing``: True, False, or None when it could not be determined.
    """

    now = time.monotonic()
    if not force and _presence_cache["value"] and now - _presence_cache["at"] < _PRESENCE_TTL_SECONDS:
        return _presence_cache["value"]

    result: dict[str, Any] = {
        "playing": None,
        "in_game": [],
        "checked_at": _utc_now(),
        "detail": "",
    }

    accounts = _load_accounts()
    key = _steam_api_key()
    steam_ids = [a["steam_id"] for a in accounts if a.get("steam_id")]
    if not key:
        result["detail"] = "No Steam Web API key stored, so CS2 cannot be detected."
    elif not steam_ids:
        result["detail"] = "No accounts to check."
    else:
        try:
            import httpx

            response = httpx.get(
                "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/",
                params={"key": key, "steamids": ",".join(steam_ids[:100])},
                timeout=10.0,
            )
            response.raise_for_status()
            players = (response.json().get("response") or {}).get("players") or []

            names = {a["steam_id"]: a.get("name") for a in accounts}
            visible = 0
            for player in players:
                # communityvisibilitystate 3 == public. Anything less and the
                # game field is withheld, so this account tells us nothing.
                if player.get("communityvisibilitystate") == 3:
                    visible += 1
                if str(player.get("gameid") or "") == "730":
                    sid = str(player.get("steamid"))
                    result["in_game"].append(names.get(sid) or sid)

            if result["in_game"]:
                result["playing"] = True
                result["detail"] = ", ".join(result["in_game"]) + " in CS2"
            elif visible:
                result["playing"] = False
                result["detail"] = "No tracked account is in CS2."
            else:
                result["detail"] = (
                    "Steam did not report game details for any account — set "
                    "\"Game details\" to Public on the Steam profile to enable detection."
                )
        except Exception as exc:
            result["detail"] = f"Could not reach the Steam Web API: {exc}"

    _presence_cache.update(at=now, value=result)
    return result


def _stop_auto_sync() -> None:
    """Signal the loop to end, and stop its child process if one is mid-run.

    Without the terminate, switching off during a download would appear to do
    nothing until the demo finished — potentially several minutes.
    """
    _auto_sync.request_stop()
    _steam_jobs.cancel(only_auto=True)


def _resume_auto_sync() -> None:
    """Pick auto-sync back up after a restart if it was left on.

    Called from the app lifespan handler at the top of this file.
    """
    try:
        if _load_auto_sync_config().get("enabled") and _steam_jobs.node_path():
            _start_auto_sync()
    except Exception:
        logging.exception("could not resume auto-sync")


# The merged assessment, plus the two stores it replaced. Those are still read
# on the way out, so an assessment made before the merge is not lost.
_ai_assessments_store = JsonStore("ai_assessment.json")


_ai_roles_store = JsonStore("ai_roles.json")


_ai_patterns_store = JsonStore("ai_patterns.json")


def _load_ai_roles() -> dict:
    return _ai_roles_store.read()


def _load_ai_assessments() -> dict:
    return _ai_assessments_store.read()


def _save_ai_assessments(data: dict) -> None:
    _ai_assessments_store.write(data)


def _load_ai_patterns() -> dict:
    return _ai_patterns_store.read()


def _resolve_ai_target(provider: str = "", model: str = "") -> tuple[str, str, str]:
    """Pick the provider, model and key to use, and check they are usable.

    A caller may name a provider and model explicitly — the trends page has a
    selector for exactly that — and anything it leaves blank falls back to
    whatever is configured as active.  The key always comes from the stored
    config; it is never accepted over the wire.
    """
    ai_config = load_ai_config()
    provider = (provider or ai_config.get("active_provider", "")).strip()
    model = (model or ai_config.get("active_model", "")).strip()
    if not provider or not model:
        raise HTTPException(
            status_code=400,
            detail="No AI provider/model configured. Set up AI in settings first.",
        )
    if provider not in AI_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown AI provider: {provider}")
    api_key = ai_config.get("providers", {}).get(provider, {}).get("api_key", "")
    if not api_key:
        raise HTTPException(status_code=400, detail=f"No API key set for {provider}")
    return provider, model, api_key
