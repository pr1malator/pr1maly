"""
pr1mealazyer — CS2 Local Analytics & Trend Tracker
FastAPI REST Backend

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from src.database import (
    add_tag,
    clear_chat_history,
    delete_match,
    get_all_matches,
    get_chat_history,
    get_connection,
    get_match,
    get_match_players,
    get_round_stats,
    get_tags,
    save_chat_message,
    update_context_notes,
    save_match,
)
from src.parser import parse_demo, parse_info_file, extract_player_names
from src.processor import calculate_match_stats, compute_benchmarks
from src.callouts import get_radar_config, game_to_pixel, get_zone_center, get_all_zones_pixel, is_map_supported, get_callout
from src.ai_service import (
    PROVIDERS as AI_PROVIDERS,
    build_match_context,
    chat_completion,
    load_config as load_ai_config,
    mask_key,
    save_config as save_ai_config,
    _format_round_narrative,
)

# ---------------------------------------------------------------------------
# Steam ID — read from data/steamID if available
# ---------------------------------------------------------------------------
_STEAM_ID_FILE = Path(__file__).parent / "data" / "steamID"
_DEFAULT_STEAM_ID: str | None = None
if _STEAM_ID_FILE.exists():
    _DEFAULT_STEAM_ID = _STEAM_ID_FILE.read_text().strip() or None

# ---------------------------------------------------------------------------
# Accounts config
# ---------------------------------------------------------------------------
_ACCOUNTS_FILE = Path(__file__).parent / "data" / "accounts.json"


def _load_accounts() -> list[dict]:
    """Load accounts from data/accounts.json."""
    if _ACCOUNTS_FILE.exists():
        try:
            data = json.loads(_ACCOUNTS_FILE.read_text(encoding="utf-8"))
            return data.get("accounts", [])
        except (json.JSONDecodeError, KeyError):
            pass
    return []


def _save_accounts(accounts: list[dict]) -> None:
    """Persist accounts to data/accounts.json."""
    _ACCOUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ACCOUNTS_FILE.write_text(
        json.dumps({"accounts": accounts}, indent=2) + "\n",
        encoding="utf-8",
    )


def _get_active_account() -> dict | None:
    """Return the currently active account, if any."""
    for acct in _load_accounts():
        if acct.get("active"):
            return acct
    return None

# ---------------------------------------------------------------------------
# Friends config
# ---------------------------------------------------------------------------
_FRIENDS_FILE = Path(__file__).parent / "data" / "friends.json"


def _load_friends() -> list[dict]:
    """Load friends from data/friends.json."""
    if _FRIENDS_FILE.exists():
        try:
            data = json.loads(_FRIENDS_FILE.read_text(encoding="utf-8"))
            return data.get("friends", [])
        except (json.JSONDecodeError, KeyError):
            pass
    return []


def _save_friends(friends: list[dict]) -> None:
    """Persist friends to data/friends.json."""
    _FRIENDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FRIENDS_FILE.write_text(
        json.dumps({"friends": friends}, indent=2) + "\n",
        encoding="utf-8",
    )

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="pr1mealazyer API",
    description="CS2 demo analysis and match statistics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
_FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/frontend/breakdown.html")


if _FRONTEND_DIR.is_dir():
    app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")


def _db():
    return get_connection()





# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class NotesUpdate(BaseModel):
    notes: str


class TagCreate(BaseModel):
    tag: str


class ConfigResponse(BaseModel):
    steam_id: str | None


class ConfigUpdate(BaseModel):
    steam_id: str


class ChatMessageIn(BaseModel):
    message: str
    provider: str | None = None
    model: str | None = None


class AIConfigUpdate(BaseModel):
    providers: dict[str, dict] | None = None
    active_provider: str | None = None
    active_model: str | None = None
    system_instructions: str | None = None
    prompts: list[dict] | None = None


class AccountCreate(BaseModel):
    name: str
    steam_id: str
    display_name: str = ""
    rank: str = ""


class AccountUpdate(BaseModel):
    name: str | None = None
    display_name: str | None = None
    rank: str | None = None


class FriendCreate(BaseModel):
    steam_id: str
    name: str = ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@app.get("/api/config", response_model=ConfigResponse)
def get_config():
    """Return the currently configured Steam ID (from active account)."""
    active = _get_active_account()
    if active:
        return {"steam_id": active["steam_id"]}
    # Fallback to legacy steamID file
    steam_id = None
    if _STEAM_ID_FILE.exists():
        steam_id = _STEAM_ID_FILE.read_text().strip() or None
    return {"steam_id": steam_id}


@app.put("/api/config")
def update_config(body: ConfigUpdate):
    """Update the configured Steam ID (also activates matching account)."""
    sid = body.steam_id.strip()
    # If an account with this steam_id exists, activate it
    accounts = _load_accounts()
    found = False
    for acct in accounts:
        acct["active"] = acct["steam_id"] == sid
        if acct["steam_id"] == sid:
            found = True
    if found:
        _save_accounts(accounts)
    # Also update legacy file for backward compat
    _STEAM_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STEAM_ID_FILE.write_text(sid)
    return {"steam_id": sid}


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------
@app.get("/api/accounts")
def list_accounts():
    """Return all configured accounts."""
    return _load_accounts()


@app.post("/api/accounts", status_code=201)
def create_account(body: AccountCreate):
    """Add a new account. First account is automatically active."""
    accounts = _load_accounts()
    sid = body.steam_id.strip().rstrip('/')
    if any(a["steam_id"] == sid for a in accounts):
        raise HTTPException(status_code=409, detail="Account with this Steam ID already exists")
    is_first = len(accounts) == 0
    accounts.append({
        "name": body.name.strip(),
        "steam_id": sid,
        "display_name": body.display_name.strip() or body.name.strip().upper(),
        "rank": body.rank.strip(),
        "active": is_first,
    })
    _save_accounts(accounts)
    # Sync legacy file if this is the first / active account
    if is_first:
        _STEAM_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STEAM_ID_FILE.write_text(sid)
    return accounts[-1]


@app.put("/api/accounts/{steam_id}")
def update_account(steam_id: str, body: AccountUpdate):
    """Update an existing account's name, display_name, or rank."""
    accounts = _load_accounts()
    for acct in accounts:
        if acct["steam_id"] == steam_id:
            if body.name is not None:
                acct["name"] = body.name.strip()
            if body.display_name is not None:
                acct["display_name"] = body.display_name.strip()
            if body.rank is not None:
                acct["rank"] = body.rank.strip()
            _save_accounts(accounts)
            return acct
    raise HTTPException(status_code=404, detail="Account not found")


@app.put("/api/accounts/{steam_id}/activate")
def activate_account(steam_id: str):
    """Set an account as the active account."""
    accounts = _load_accounts()
    found = False
    for acct in accounts:
        if acct["steam_id"] == steam_id:
            acct["active"] = True
            found = True
        else:
            acct["active"] = False
    if not found:
        raise HTTPException(status_code=404, detail="Account not found")
    _save_accounts(accounts)
    # Sync legacy steamID file
    _STEAM_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STEAM_ID_FILE.write_text(steam_id)
    return {"activated": steam_id}


@app.delete("/api/accounts/{steam_id}")
def delete_account(steam_id: str):
    """Remove an account. If it was active, the first remaining account becomes active."""
    accounts = _load_accounts()
    # Normalize: strip slashes that may have been stored accidentally
    steam_id = steam_id.strip().rstrip('/')
    before = len(accounts)
    was_active = any(a["steam_id"].rstrip('/') == steam_id and a.get("active") for a in accounts)
    accounts = [a for a in accounts if a["steam_id"].rstrip('/') != steam_id]
    if len(accounts) == before:
        raise HTTPException(status_code=404, detail="Account not found")
    if was_active and accounts:
        accounts[0]["active"] = True
        _STEAM_ID_FILE.write_text(accounts[0]["steam_id"])
    _save_accounts(accounts)
    return {"deleted": steam_id}


# ---------------------------------------------------------------------------
# Friends
# ---------------------------------------------------------------------------
@app.get("/api/friends")
def list_friends():
    """Return all configured friends."""
    return _load_friends()


@app.post("/api/friends", status_code=201)
def create_friend(body: FriendCreate):
    """Add a new friend by Steam ID."""
    friends = _load_friends()
    sid = body.steam_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Steam ID is required")
    if any(f["steam_id"] == sid for f in friends):
        raise HTTPException(status_code=409, detail="Friend with this Steam ID already exists")
    friends.append({"steam_id": sid, "name": body.name.strip()})
    _save_friends(friends)
    return friends[-1]


@app.delete("/api/friends/{steam_id}")
def delete_friend(steam_id: str):
    """Remove a friend."""
    friends = _load_friends()
    before = len(friends)
    friends = [f for f in friends if f["steam_id"] != steam_id]
    if len(friends) == before:
        raise HTTPException(status_code=404, detail="Friend not found")
    _save_friends(friends)
    return {"deleted": steam_id}


# ---------------------------------------------------------------------------
# Detect player from .info file
# ---------------------------------------------------------------------------
@app.post("/api/matches/detect-player")
def detect_player(
    info_file: UploadFile = File(...),
    demo_file: UploadFile | None = File(default=None),
):
    """Read a .dem.info sidecar and match account IDs against configured accounts.

    Returns ``matched`` (list of known accounts found in the match) and
    ``unmatched`` (steam IDs present in the demo but not in accounts).
    If a .dem file is also provided, player names are resolved from it.
    """
    if not info_file.filename:
        raise HTTPException(status_code=400, detail="Info file is required")
    try:
        info_data = parse_info_file(info_file.file.read())
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse info file: {exc}")

    demo_ids: list[str] = info_data.get("account_ids", [])
    if not demo_ids:
        raise HTTPException(status_code=422, detail="No player accounts found in info file")

    # Resolve names from .dem file if provided
    name_map: dict[str, str] = {}
    if demo_file and demo_file.filename:
        try:
            with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as tmp:
                tmp.write(demo_file.file.read())
                tmp_path = tmp.name
            name_map = extract_player_names(tmp_path)
        except Exception:
            pass
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    accounts = _load_accounts()
    acct_by_sid = {a["steam_id"]: a for a in accounts}
    matched = [acct_by_sid[sid] for sid in demo_ids if sid in acct_by_sid]
    unmatched = [
        {"steam_id": sid, "name": name_map.get(sid, "")}
        for sid in demo_ids
        if sid not in acct_by_sid
    ]

    return {
        "matched": matched,
        "unmatched": unmatched,
        "match_date": info_data.get("match_date"),
    }


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
@app.post("/api/matches/upload")
def upload_demo(
    file: UploadFile = File(...),
    info_file: UploadFile | None = File(default=None),
    steam_id: str = Form(default=""),
    match_date: str = Form(default=""),
    context_notes: str = Form(default=""),
    tags: str = Form(default=""),
):
    """Upload and parse a .dem file. Returns the new match summary."""
    if not file.filename or not file.filename.endswith(".dem"):
        raise HTTPException(status_code=400, detail="File must be a .dem demo file")

    # Resolve steam ID: form field > active account > legacy file
    sid = steam_id.strip()
    if not sid:
        active = _get_active_account()
        if active:
            sid = active["steam_id"]
    if not sid:
        if _STEAM_ID_FILE.exists():
            sid = _STEAM_ID_FILE.read_text().strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Steam ID is required")

    # Parse .info sidecar for match date if provided
    info_date: str | None = None
    if info_file and info_file.filename:
        try:
            info_data = parse_info_file(info_file.file.read())
            info_date = info_data.get("match_date")
        except Exception:
            pass  # .info parsing failure is non-fatal

    # Resolve match date: form field > .info file
    resolved_date = match_date.strip() or info_date or None

    # Write to temp file and parse
    with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        parsed = parse_demo(tmp_path)
        stats = calculate_match_stats(parsed, sid)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse demo: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    conn = _db()
    match_id = save_match(
        conn,
        stats,
        filename=file.filename,
        steam_id=sid,
        context_notes=context_notes.strip(),
        match_date=resolved_date,
    )

    # Add tags
    for tag in tags.split(","):
        tag = tag.strip()
        if tag:
            add_tag(conn, match_id, tag)

    conn.close()
    return {"match_id": match_id, "stats": _match_summary(stats)}


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
    }


# ---------------------------------------------------------------------------
# Bulk upload
# ---------------------------------------------------------------------------
@app.post("/api/matches/upload-bulk")
def upload_demos_bulk(
    files: List[UploadFile] = File(...),
    info_files: List[UploadFile] = File(default=[]),
    steam_id: str = Form(default=""),
):
    """Upload and parse multiple .dem files at once.

    Each .dem can be auto-matched to a .info sidecar by filename prefix.
    If an .info sidecar contains a known account, that account's steam_id
    is used for that demo.  Otherwise, falls back to the form steam_id or
    the active account.
    """
    # Build a lookup of info files by base name
    info_lookup: dict[str, UploadFile] = {}
    for inf in info_files:
        if inf.filename:
            info_lookup[inf.filename] = inf

    accounts = _load_accounts()
    account_ids = {a["steam_id"] for a in accounts}

    # Resolve fallback steam_id
    fallback_sid = steam_id.strip()
    if not fallback_sid:
        active = _get_active_account()
        if active:
            fallback_sid = active["steam_id"]
    if not fallback_sid:
        if _STEAM_ID_FILE.exists():
            fallback_sid = _STEAM_ID_FILE.read_text().strip()

    results: list[dict[str, Any]] = []

    for f in files:
        fname = f.filename or ""
        entry: dict[str, Any] = {"filename": fname}

        if not fname.endswith(".dem"):
            entry["status"] = "skipped"
            entry["detail"] = "Not a .dem file"
            results.append(entry)
            continue

        # Try matching info file: <name>.dem → <name>.dem.info
        info_uf = info_lookup.get(fname + ".info")
        info_date: str | None = None
        detected_sid: str | None = None

        if info_uf:
            try:
                info_uf.file.seek(0)
                info_data = parse_info_file(info_uf.file.read())
                info_date = info_data.get("match_date")
                # Check if any account_id from info is a known account
                for aid in info_data.get("account_ids", []):
                    aid_str = str(aid)
                    if aid_str in account_ids:
                        detected_sid = aid_str
                        break
            except Exception:
                pass

        sid = detected_sid or fallback_sid
        if not sid:
            entry["status"] = "error"
            entry["detail"] = "No Steam ID available"
            results.append(entry)
            continue

        # Write to temp file and parse
        with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as tmp:
            tmp.write(f.file.read())
            tmp_path = tmp.name

        try:
            parsed = parse_demo(tmp_path)
            stats = calculate_match_stats(parsed, sid)
        except Exception as exc:
            entry["status"] = "error"
            entry["detail"] = f"Parse failed: {exc}"
            results.append(entry)
            Path(tmp_path).unlink(missing_ok=True)
            continue
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        conn = _db()
        match_id = save_match(
            conn,
            stats,
            filename=fname,
            steam_id=sid,
            match_date=info_date,
        )
        conn.close()

        # Find the account name for response
        acct_name = None
        for a in accounts:
            if a["steam_id"] == sid:
                acct_name = a.get("name")
                break

        entry["status"] = "ok"
        entry["match_id"] = match_id
        entry["player_name"] = acct_name or stats.get("player_name", "?")
        entry["player_steam_id"] = sid
        entry["map_name"] = stats.get("map_name")
        entry["stats"] = _match_summary(stats)
        results.append(entry)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    return {"processed": ok_count, "total": len(results), "results": results}


# ---------------------------------------------------------------------------
# Folder sync
# ---------------------------------------------------------------------------
_SYNC_CONFIG_FILE = Path(__file__).parent / "data" / "sync_config.json"


def _load_sync_config() -> dict:
    if _SYNC_CONFIG_FILE.exists():
        try:
            return json.loads(_SYNC_CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def _save_sync_config(cfg: dict) -> None:
    _SYNC_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SYNC_CONFIG_FILE.write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
    )


@app.get("/api/sync/config")
def get_sync_config():
    return _load_sync_config()


@app.put("/api/sync/config")
def set_sync_config(body: dict):
    folder = body.get("folder", "").strip()
    if not folder:
        raise HTTPException(status_code=400, detail="Folder path is required")
    p = Path(folder)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder does not exist: {folder}")
    cfg = _load_sync_config()
    cfg["folder"] = str(p)
    _save_sync_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Onboarding state
# ---------------------------------------------------------------------------
_ONBOARDING_FILE = Path(__file__).parent / "data" / "onboarding.json"


def _load_onboarding() -> dict:
    if _ONBOARDING_FILE.exists():
        try:
            return json.loads(_ONBOARDING_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            pass
    return {"completed": False}


def _save_onboarding(state: dict) -> None:
    _ONBOARDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ONBOARDING_FILE.write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )


@app.get("/api/onboarding")
def get_onboarding():
    return _load_onboarding()


@app.put("/api/onboarding")
def set_onboarding(body: dict):
    state = _load_onboarding()
    if "completed" in body:
        state["completed"] = bool(body["completed"])
    _save_onboarding(state)
    return state


# ---------------------------------------------------------------------------
# Factory Reset
# ---------------------------------------------------------------------------
@app.post("/api/factory-reset")
def factory_reset():
    """Delete all user data: database, accounts, friends, AI roles, sync config, steamID.

    Returns the application to a fresh-install state.
    """
    from src.database import _DEFAULT_DB_PATH as db_path

    errors: list[str] = []

    # 1. Delete the SQLite database
    try:
        if db_path.exists():
            db_path.unlink()
    except Exception as exc:
        errors.append(f"Database: {exc}")

    # 2. Reset accounts
    try:
        _save_accounts([])
    except Exception as exc:
        errors.append(f"Accounts: {exc}")

    # 3. Reset friends
    try:
        _save_friends([])
    except Exception as exc:
        errors.append(f"Friends: {exc}")

    # 4. Clear steamID file
    try:
        if _STEAM_ID_FILE.exists():
            _STEAM_ID_FILE.write_text("", encoding="utf-8")
    except Exception as exc:
        errors.append(f"SteamID: {exc}")

    # 5. Reset sync config
    try:
        _save_sync_config({"folder": ""})
    except Exception as exc:
        errors.append(f"Sync config: {exc}")

    # 6. Delete AI roles
    try:
        ai_roles_file = Path(__file__).parent / "data" / "ai_roles.json"
        if ai_roles_file.exists():
            ai_roles_file.unlink()
    except Exception as exc:
        errors.append(f"AI roles: {exc}")

    # 7. Reset AI config (keep structure, clear keys)
    try:
        ai_cfg_file = Path(__file__).parent / "data" / "ai_config.json"
        if ai_cfg_file.exists():
            cfg = json.loads(ai_cfg_file.read_text(encoding="utf-8"))
            for prov in cfg.get("providers", {}).values():
                prov["api_key"] = ""
            cfg["system_instructions"] = ""
            ai_cfg_file.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        errors.append(f"AI config: {exc}")

    # 8. Reset onboarding state
    try:
        _save_onboarding({"completed": False})
    except Exception as exc:
        errors.append(f"Onboarding: {exc}")

    if errors:
        return {"status": "partial", "errors": errors}
    return {"status": "ok"}


@app.get("/api/sync/scan")
def sync_scan(steam_id: str = ""):
    """Scan the sync folder and return new .dem files not yet in the database.

    If ``steam_id`` is provided, only demos whose .dem.info sidecar lists
    that player are returned.  Demos without a .info sidecar are always
    included (they cannot be filtered).
    """
    cfg = _load_sync_config()
    folder = cfg.get("folder")
    if not folder:
        raise HTTPException(status_code=400, detail="No sync folder configured")
    p = Path(folder)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder does not exist: {folder}")

    # Find all .dem files
    dem_files = sorted(p.glob("*.dem"))

    # Get existing filenames from DB for the selected player (or all)
    conn = _db()
    if steam_id.strip():
        rows = conn.execute(
            "SELECT filename FROM matches WHERE player_steam_id = ?",
            (steam_id.strip(),),
        ).fetchall()
    else:
        rows = conn.execute("SELECT filename FROM matches").fetchall()
    conn.close()
    existing = {r["filename"] for r in rows if r["filename"]}

    filter_sid = steam_id.strip()

    new_demos = []
    for df in dem_files:
        fname = df.name
        if fname in existing:
            continue
        info_path = df.with_suffix(".dem.info")
        has_info = info_path.exists()

        # Filter by player if requested
        if filter_sid and has_info:
            try:
                info_data = parse_info_file(info_path.read_bytes())
                if filter_sid not in info_data.get("account_ids", []):
                    continue
            except Exception:
                pass  # include on parse failure

        new_demos.append({
            "filename": fname,
            "size_mb": round(df.stat().st_size / 1024 / 1024, 1),
            "has_info": has_info,
        })

    return {"folder": folder, "total_found": len(dem_files), "new": new_demos}


@app.post("/api/sync/process")
def sync_process(body: dict):
    """Process selected .dem files from the sync folder."""
    cfg = _load_sync_config()
    folder = cfg.get("folder")
    if not folder:
        raise HTTPException(status_code=400, detail="No sync folder configured")
    p = Path(folder)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder does not exist: {folder}")

    filenames: list[str] = body.get("filenames", [])
    if not filenames:
        raise HTTPException(status_code=400, detail="No filenames provided")

    steam_id = body.get("steam_id", "").strip()
    if not steam_id:
        active = _get_active_account()
        if active:
            steam_id = active["steam_id"]
    if not steam_id and _STEAM_ID_FILE.exists():
        steam_id = _STEAM_ID_FILE.read_text().strip()
    if not steam_id:
        raise HTTPException(status_code=400, detail="No Steam ID available")

    accounts = _load_accounts()
    account_ids = {a["steam_id"] for a in accounts}

    results: list[dict[str, Any]] = []
    for fname in filenames:
        entry: dict[str, Any] = {"filename": fname}
        dem_path = p / fname
        if not dem_path.exists() or not fname.endswith(".dem"):
            entry["status"] = "error"
            entry["detail"] = "File not found or not a .dem"
            results.append(entry)
            continue

        # Try .info sidecar
        info_path = dem_path.with_suffix(".dem.info")
        info_date: str | None = None
        detected_sid: str | None = None
        if info_path.exists():
            try:
                info_data = parse_info_file(info_path.read_bytes())
                info_date = info_data.get("match_date")
                for aid in info_data.get("account_ids", []):
                    if str(aid) in account_ids:
                        detected_sid = str(aid)
                        break
            except Exception:
                pass

        sid = detected_sid or steam_id

        try:
            parsed = parse_demo(str(dem_path))
            stats = calculate_match_stats(parsed, sid)
        except Exception as exc:
            entry["status"] = "error"
            entry["detail"] = f"Parse failed: {exc}"
            results.append(entry)
            continue

        conn = _db()
        match_id = save_match(
            conn, stats, filename=fname, steam_id=sid, match_date=info_date,
        )
        conn.close()

        acct_name = None
        for a in accounts:
            if a["steam_id"] == sid:
                acct_name = a.get("name")
                break

        entry["status"] = "ok"
        entry["match_id"] = match_id
        entry["player_name"] = acct_name or stats.get("player_name", "?")
        entry["map_name"] = stats.get("map_name")
        entry["stats"] = _match_summary(stats)
        results.append(entry)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    return {"processed": ok_count, "total": len(results), "results": results}


# ---------------------------------------------------------------------------
# Match list
# ---------------------------------------------------------------------------
@app.get("/api/matches")
def list_matches(player_steam_id: str = None):
    """Return all matches ordered by date descending, optionally filtered by player."""
    conn = _db()
    matches = get_all_matches(conn, player_steam_id=player_steam_id)
    # Attach tags to each match
    for m in matches:
        m["tags"] = get_tags(conn, m["match_id"])
    conn.close()
    return matches


# ---------------------------------------------------------------------------
# Career averages for match-level KPI comparison
# ---------------------------------------------------------------------------
@app.get("/api/matches/career-averages")
def get_career_averages():
    """Compute per-match KPI averages across all matches for trend comparison.

    Returns averages for: HS%, K/D, KAST, enemies flashed, avg blind duration,
    HE damage, Molotov damage, clutch win %, trade %, opening kill rate,
    multi-kill rounds, aim score, utility use rate, utility rating.
    """
    import json as _json

    conn = _db()
    matches = get_all_matches(conn)
    if not matches:
        conn.close()
        return {"total_matches": 0, "averages": {}}

    # Per-match accumulators
    hs_pcts: list[float] = []
    kd_ratios: list[float] = []
    kast_vals: list[float] = []
    flash_counts: list[int] = []
    blind_durs: list[float] = []
    he_dmgs: list[int] = []
    molly_dmgs: list[int] = []
    clutch_pcts: list[float] = []
    trade_pcts: list[float] = []
    opening_rates: list[float] = []
    multikill_counts: list[int] = []
    aim_ratings: list[float] = []
    movement_avgs: list[float] = []
    ttk_avgs: list[float] = []
    preaim_avgs: list[float] = []
    reaction_avgs: list[float] = []
    use_rates: list[float] = []
    utility_ratings: list[float] = []

    for m in matches:
        # Top-level match stats
        kd_ratios.append(m.get("kd_ratio") or 0)
        kast_vals.append(m.get("kast") or 0)
        multikill_counts.append(
            (m.get("rounds_2k") or 0) + (m.get("rounds_3k") or 0) +
            (m.get("rounds_4k") or 0) + (m.get("rounds_5k") or 0)
        )

        # Aim rating from stored JSON
        aim_raw = m.get("aim_stats")
        if aim_raw:
            try:
                aim_obj = _json.loads(aim_raw) if isinstance(aim_raw, str) else aim_raw
                ar = aim_obj.get("aim_rating")
                if ar is not None:
                    aim_ratings.append(ar)
                mov = aim_obj.get("movement", {})
                if mov.get("avg") is not None:
                    movement_avgs.append(mov["avg"])
                ttk_o = aim_obj.get("ttk", {})
                if ttk_o.get("avg") is not None:
                    ttk_avgs.append(ttk_o["avg"])
                pa = aim_obj.get("preaim", {})
                if pa.get("avg") is not None:
                    preaim_avgs.append(pa["avg"])
                rxn = aim_obj.get("reaction", {})
                if rxn.get("avg") is not None:
                    reaction_avgs.append(rxn["avg"])
            except Exception:
                pass

        # Utility data from stored JSON
        util_raw = m.get("utility_data")
        if util_raw:
            try:
                util_obj = _json.loads(util_raw) if isinstance(util_raw, str) else util_raw
                ur = util_obj.get("utility_rating")
                if ur is not None:
                    utility_ratings.append(ur)
                eco = util_obj.get("economics") or {}
                use_r = eco.get("use_rate")
                if use_r is not None:
                    use_rates.append(use_r)
            except Exception:
                pass

        # Round-level stats
        rounds = get_round_stats(conn, m["match_id"])
        total_kills = 0
        hs_kills = 0
        m_flashed = 0
        m_blind_dur = 0.0
        m_blind_hits = 0
        m_he = 0
        m_molly = 0
        clutch_won = 0
        clutch_total = 0
        deaths = 0
        traded = 0
        open_kills = 0
        open_deaths = 0

        for r in rounds:
            ej = r.get("enriched_json")
            if not ej:
                continue
            try:
                e = _json.loads(ej) if isinstance(ej, str) else ej
            except Exception:
                continue

            # HS%
            for k in e.get("kills_detail", []):
                total_kills += 1
                if k.get("headshot"):
                    hs_kills += 1

            # Utility
            u = e.get("utility", {})
            ef = u.get("enemies_flashed", 0) or 0
            m_flashed += ef
            abd = u.get("avg_blind_duration", 0) or 0
            if ef > 0 and abd > 0:
                m_blind_dur += abd * ef
                m_blind_hits += ef
            m_he += u.get("he_damage", 0) or 0
            for md in u.get("molotov_damage", []):
                m_molly += md.get("damage", 0) or 0

            # Clutch
            c = e.get("clutch")
            if c:
                clutch_total += 1
                if c.get("won"):
                    clutch_won += 1

            # Trade
            if r.get("deaths", 0) > 0:
                deaths += 1
                if r.get("traded"):
                    traded += 1

            # Opening duels
            od = e.get("opening_duel")
            if od:
                if od.get("role") == "opening_kill":
                    open_kills += 1
                elif od.get("role") == "opening_death":
                    open_deaths += 1

        hs_pcts.append((hs_kills / total_kills * 100) if total_kills else 0)
        flash_counts.append(m_flashed)
        blind_durs.append((m_blind_dur / m_blind_hits) if m_blind_hits else 0)
        he_dmgs.append(m_he)
        molly_dmgs.append(m_molly)
        clutch_pcts.append((clutch_won / clutch_total * 100) if clutch_total else 0)
        trade_pcts.append((traded / deaths * 100) if deaths else 0)
        od_total = open_kills + open_deaths
        opening_rates.append((open_kills / od_total * 100) if od_total else 0)

    conn.close()

    def _avg(vals: list) -> float | None:
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "total_matches": len(matches),
        "averages": {
            "hs_pct": _avg(hs_pcts),
            "kd_ratio": _avg(kd_ratios),
            "kast": _avg(kast_vals),
            "enemies_flashed": _avg(flash_counts),
            "avg_blind_duration": _avg(blind_durs),
            "he_damage": _avg(he_dmgs),
            "molotov_damage": _avg(molly_dmgs),
            "clutch_win_pct": _avg(clutch_pcts),
            "trade_pct": _avg(trade_pcts),
            "opening_kill_rate": _avg(opening_rates),
            "multikill_rounds": _avg(multikill_counts),
            "aim_rating": _avg(aim_ratings),
            "movement_avg": _avg(movement_avgs),
            "ttk_avg": _avg(ttk_avgs),
            "preaim_avg": _avg(preaim_avgs),
            "reaction_avg": _avg(reaction_avgs),
            "use_rate": _avg(use_rates),
            "utility_rating": _avg(utility_ratings),
        },
    }


# ---------------------------------------------------------------------------
# 2D Replay data
# ---------------------------------------------------------------------------
@app.get("/api/matches/{match_id}/replay")
def get_replay_data(match_id: str, round_number: int = 0):
    """Return 2D replay frames for a match.

    If *round_number* is 0, returns metadata only (round list with frame
    counts). If a specific round is given, returns full frame data with
    pixel-space coordinates.
    """
    import json as _json

    conn = _db()
    match = get_match(conn, match_id)
    if not match:
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")

    map_name = match.get("map_name", "")
    radar = get_radar_config(map_name)
    if not radar:
        conn.close()
        raise HTTPException(status_code=400, detail=f"No radar data for {map_name}")

    rounds = get_round_stats(conn, match_id)
    conn.close()

    # Check if replay data exists at all
    has_replay = any(r.get("replay_json") for r in rounds)
    if not has_replay:
        return {
            "match_id": match_id,
            "map_name": map_name,
            "has_replay": False,
            "message": "No replay data available. Re-upload the demo to generate replay data.",
        }

    # Determine radar image URL
    clean = map_name.replace("de_", "").replace("cs_", "")
    import time as _time
    radar_image = f"/frontend/img/radar/{clean}.png?v={int(_time.time())}"

    if round_number == 0:
        # Return round list overview (no frames)
        round_list = []
        for r in rounds:
            rj = r.get("replay_json")
            if not rj:
                round_list.append({"round": r["round_number"], "frames": 0})
                continue
            try:
                rd = _json.loads(rj)
                round_list.append({
                    "round": r["round_number"],
                    "frames": len(rd.get("frames", [])),
                })
            except Exception:
                round_list.append({"round": r["round_number"], "frames": 0})
        return {
            "match_id": match_id,
            "map_name": map_name,
            "has_replay": True,
            "radar_image": radar_image,
            "radar": {"pos_x": radar["pos_x"], "pos_y": radar["pos_y"], "scale": radar["scale"]},
            "total_rounds": len(rounds),
            "rounds": round_list,
        }

    # Return full frame data for a specific round
    target = None
    for r in rounds:
        if r["round_number"] == round_number:
            target = r
            break
    if not target or not target.get("replay_json"):
        raise HTTPException(status_code=404, detail=f"No replay data for round {round_number}")

    try:
        rd = _json.loads(target["replay_json"])
    except Exception:
        raise HTTPException(status_code=500, detail="Corrupt replay data")

    # Convert game coordinates to pixel coordinates
    players = rd.get("players", {})
    raw_frames = rd.get("frames", [])
    events = rd.get("events", [])

    pixel_frames = []
    for frame in raw_frames:
        tick_offset = frame[0]
        positions = frame[1]
        px_positions = {}
        for sid, coords in positions.items():
            gx, gy, hp = coords[0], coords[1], coords[2]
            pix = game_to_pixel(map_name, gx, gy)
            if pix:
                px_positions[sid] = [round(pix[0], 1), round(pix[1], 1), hp]
            else:
                px_positions[sid] = [0, 0, hp]
        pixel_frames.append([tick_offset, px_positions])

    # Convert kill event positions to pixel coords (attacker/victim from frames)
    # Convert grenade event game coordinates to pixel coordinates
    for ev in events:
        if ev.get("type") == "grenade" and "x" in ev and "y" in ev:
            pix = game_to_pixel(map_name, ev["x"], ev["y"])
            if pix:
                ev["px"] = round(pix[0], 1)
                ev["py"] = round(pix[1], 1)
            del ev["x"]
            del ev["y"]

    return {
        "match_id": match_id,
        "map_name": map_name,
        "has_replay": True,
        "radar_image": radar_image,
        "radar": {"pos_x": radar["pos_x"], "pos_y": radar["pos_y"], "scale": radar["scale"]},
        "round_number": round_number,
        "players": players,
        "frames": pixel_frames,
        "events": events,
        "sample_interval": 32,
        "tick_rate": 64,
    }


# ---------------------------------------------------------------------------
# Single match detail
# ---------------------------------------------------------------------------
@app.get("/api/matches/{match_id}")
def get_match_detail(match_id: str):
    """Return full match detail including players, rounds, and tags."""
    conn = _db()
    match = get_match(conn, match_id)
    if not match:
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")

    players = get_match_players(conn, match_id)
    rounds = get_round_stats(conn, match_id)
    tags = get_tags(conn, match_id)
    conn.close()

    # Split players by team
    user_team = None
    for p in players:
        if p.get("is_user"):
            user_team = p.get("team")
            break

    my_team = [p for p in players if p.get("team") == user_team]
    enemy_team = [p for p in players if p.get("team") != user_team]

    # Annotate friends
    friend_ids = {f["steam_id"] for f in _load_friends()}
    for p in my_team + enemy_team:
        p["is_friend"] = p.get("steam_id", "") in friend_ids

    # Deserialize aim_stats JSON
    aim_stats = None
    if match.get("aim_stats"):
        try:
            aim_stats = json.loads(match["aim_stats"])
        except (json.JSONDecodeError, TypeError):
            pass

    # Deserialize role_data JSON
    role_data = None
    if match.get("role_data"):
        try:
            role_data = json.loads(match["role_data"])
        except (json.JSONDecodeError, TypeError):
            pass

    # Deserialize utility_data JSON
    utility_data = None
    if match.get("utility_data"):
        try:
            utility_data = json.loads(match["utility_data"])
        except (json.JSONDecodeError, TypeError):
            pass

    # Compute benchmark tier labels from stored stats
    benchmarks = compute_benchmarks(
        aim_stats or {},
        utility_data or {},
        match.get("total_rounds", 0),
        match.get("map_name", ""),
    )

    return {
        **match,
        "aim_stats": aim_stats,
        "role_data": role_data,
        "utility_data": utility_data,
        "benchmarks": benchmarks,
        "behavioral_axes": _compute_match_behavioral_axes(rounds),
        "tags": tags,
        "round_stats": rounds,
        "my_team": my_team,
        "enemy_team": enemy_team,
    }


# ---------------------------------------------------------------------------
# Match mutations
# ---------------------------------------------------------------------------
@app.put("/api/matches/{match_id}/notes")
def update_notes(match_id: str, body: NotesUpdate):
    """Update context notes for a match."""
    conn = _db()
    if not get_match(conn, match_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")
    update_context_notes(conn, match_id, body.notes)
    conn.close()
    return {"match_id": match_id, "notes": body.notes}


@app.post("/api/matches/{match_id}/tags")
def create_tag(match_id: str, body: TagCreate):
    """Add a tag to a match."""
    conn = _db()
    if not get_match(conn, match_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")
    add_tag(conn, match_id, body.tag)
    conn.close()
    return {"match_id": match_id, "tag": body.tag}


@app.delete("/api/matches/{match_id}")
def remove_match(match_id: str):
    """Delete a match and all associated data."""
    conn = _db()
    if not get_match(conn, match_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")
    delete_match(conn, match_id)
    conn.close()
    return {"deleted": match_id}


# ---------------------------------------------------------------------------
# AI Config
# ---------------------------------------------------------------------------
@app.get("/api/ai/providers")
def list_ai_providers():
    """Return available AI providers with their model lists."""
    return AI_PROVIDERS


@app.get("/api/ai/config")
def get_ai_config():
    """Return current AI configuration (API keys masked)."""
    config = load_ai_config()
    safe = {
        "providers": {},
        "active_provider": config.get("active_provider", ""),
        "active_model": config.get("active_model", ""),
        "system_instructions": config.get("system_instructions", ""),
        "prompts": config.get("prompts", []),
    }
    for name, prov in config.get("providers", {}).items():
        safe["providers"][name] = {
            "api_key_set": bool(prov.get("api_key", "").strip()),
            "api_key_masked": mask_key(prov.get("api_key", "")),
            "default_model": prov.get("default_model", ""),
        }
    return safe


@app.put("/api/ai/config")
def update_ai_config(body: AIConfigUpdate):
    """Update AI configuration (providers, keys, prompts)."""
    config = load_ai_config()
    if body.providers is not None:
        existing = config.get("providers", {})
        for name, prov in body.providers.items():
            key = prov.get("api_key", "").strip()
            if key:
                existing.setdefault(name, {})["api_key"] = key
            if "default_model" in prov:
                existing.setdefault(name, {})["default_model"] = prov["default_model"]
        config["providers"] = existing
    if body.active_provider is not None:
        config["active_provider"] = body.active_provider
    if body.active_model is not None:
        config["active_model"] = body.active_model
    if body.system_instructions is not None:
        config["system_instructions"] = body.system_instructions
    if body.prompts is not None:
        config["prompts"] = body.prompts
    save_ai_config(config)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Match Chat
# ---------------------------------------------------------------------------
@app.get("/api/matches/{match_id}/chat")
def get_match_chat(match_id: str):
    """Return chat history for a match."""
    conn = _db()
    if not get_match(conn, match_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")
    history = get_chat_history(conn, match_id)
    conn.close()
    return {"match_id": match_id, "messages": history}


@app.post("/api/matches/{match_id}/chat")
async def send_match_chat(match_id: str, body: ChatMessageIn):
    """Send a message, get AI response, persist both."""
    conn = _db()
    match = get_match(conn, match_id)
    if not match:
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")

    # Resolve provider/model
    ai_config = load_ai_config()
    provider = body.provider or ai_config.get("active_provider", "")
    model = body.model or ai_config.get("active_model", "")
    if not provider or not model:
        conn.close()
        raise HTTPException(status_code=400, detail="No AI provider/model configured")

    prov_config = ai_config.get("providers", {}).get(provider, {})
    api_key = prov_config.get("api_key", "")
    if not api_key:
        conn.close()
        raise HTTPException(status_code=400, detail=f"No API key set for {provider}")

    # Build context
    players = get_match_players(conn, match_id)
    rounds = get_round_stats(conn, match_id)
    system_prompt = build_match_context(
        match, players, rounds,
        custom_instructions=ai_config.get("system_instructions", ""),
    )

    # Load existing history
    history = get_chat_history(conn, match_id)
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": body.message})

    # Call AI
    try:
        response = await chat_completion(provider, model, api_key, messages, system_prompt)
    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=502, detail=f"AI request failed: {exc}")

    # Persist both messages
    save_chat_message(conn, match_id, "user", body.message, provider, model)
    save_chat_message(conn, match_id, "assistant", response, provider, model)
    conn.close()

    return {"role": "assistant", "content": response, "provider": provider, "model": model}


@app.delete("/api/matches/{match_id}/chat")
def clear_match_chat(match_id: str):
    """Clear all chat history for a match."""
    conn = _db()
    if not get_match(conn, match_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")
    clear_chat_history(conn, match_id)
    conn.close()
    return {"match_id": match_id, "cleared": True}


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------
@app.get("/api/trends")
def get_trends(maps: str = ""):
    """Return trend data for charts (rating, ADR, KAST, K/D/A over time).

    Optional ``maps`` query param: comma-separated map filter.
    """
    conn = _db()
    matches = get_all_matches(conn)
    conn.close()

    # Filter by maps if provided
    if maps.strip():
        allowed = {m.strip().lower() for m in maps.split(",")}
        matches = [m for m in matches if m.get("map_name", "").lower() in allowed]

    # Sort chronologically (oldest first for charts)
    matches.sort(key=lambda m: m.get("date", ""))

    data_points = []
    for m in matches:
        # Extract aim_rating from stored JSON
        aim_rating = None
        aim_raw = m.get("aim_stats")
        if aim_raw:
            try:
                import json as _json
                aim_obj = _json.loads(aim_raw) if isinstance(aim_raw, str) else aim_raw
                aim_rating = aim_obj.get("aim_rating")
            except Exception:
                pass
        # Extract utility_rating from stored JSON
        utility_rating = None
        util_raw = m.get("utility_data")
        if util_raw:
            try:
                import json as _json
                util_obj = _json.loads(util_raw) if isinstance(util_raw, str) else util_raw
                utility_rating = util_obj.get("utility_rating")
            except Exception:
                pass
        data_points.append({
            "match_id": m["match_id"],
            "date": m.get("date"),
            "map_name": m.get("map_name"),
            "hltv_rating": m.get("hltv_rating"),
            "adr": m.get("adr"),
            "kast": m.get("kast"),
            "kills": m.get("kills"),
            "deaths": m.get("deaths"),
            "assists": m.get("assists"),
            "match_result": m.get("match_result"),
            "team_score": m.get("team_score"),
            "enemy_score": m.get("enemy_score"),
            "aim_rating": aim_rating,
            "utility_rating": utility_rating,
        })

    # Career averages
    if data_points:
        n = len(data_points)
        aim_vals = [d["aim_rating"] for d in data_points if d["aim_rating"] is not None]
        util_vals = [d["utility_rating"] for d in data_points if d["utility_rating"] is not None]
        averages = {
            "avg_rating": round(sum(d["hltv_rating"] or 0 for d in data_points) / n, 4),
            "avg_adr": round(sum(d["adr"] or 0 for d in data_points) / n, 2),
            "avg_kast": round(sum(d["kast"] or 0 for d in data_points) / n, 2),
            "avg_kills": round(sum(d["kills"] or 0 for d in data_points) / n, 1),
            "avg_deaths": round(sum(d["deaths"] or 0 for d in data_points) / n, 1),
            "avg_aim_rating": round(sum(aim_vals) / len(aim_vals), 1) if aim_vals else None,
            "avg_utility_rating": round(sum(util_vals) / len(util_vals), 1) if util_vals else None,
            "total_matches": n,
        }
    else:
        averages = {
            "avg_rating": 0, "avg_adr": 0, "avg_kast": 0,
            "avg_kills": 0, "avg_deaths": 0,
            "avg_aim_rating": None, "avg_utility_rating": None,
            "total_matches": 0,
        }

    return {
        "data_points": data_points,
        "averages": averages,
        "available_maps": sorted({m.get("map_name", "") for m in matches}),
    }


# ---------------------------------------------------------------------------
# Performance analytics (powers the breakdown / stats page)
# ---------------------------------------------------------------------------
@app.get("/api/performance")
def get_performance(maps: str = ""):
    """Aggregate enriched round data into role / mechanic / phase stats."""
    import json

    conn = _db()
    matches = get_all_matches(conn)

    if maps.strip():
        allowed = {m.strip().lower() for m in maps.split(",")}
        matches = [m for m in matches if m.get("map_name", "").lower() in allowed]

    if not matches:
        conn.close()
        return _empty_performance()

    match_ids = [m["match_id"] for m in matches]

    # Gather all round_stats with enriched data
    all_rounds = []
    for mid in match_ids:
        rounds = get_round_stats(conn, mid)
        for r in rounds:
            ej = r.get("enriched_json")
            if ej:
                try:
                    r["enriched"] = json.loads(ej)
                except Exception:
                    r["enriched"] = {}
            else:
                r["enriched"] = {}
            all_rounds.append(r)

    # Gather user player rows for multi-kill data
    user_players = []
    for mid in match_ids:
        players = get_match_players(conn, mid)
        for p in players:
            if p.get("is_user"):
                user_players.append(p)
    conn.close()

    total_rounds = len(all_rounds)
    if total_rounds == 0:
        return _empty_performance()

    # --- Headshot % ---
    total_kills_hs = 0
    total_kills_count = 0
    weapon_kills: dict[str, int] = {}
    for r in all_rounds:
        kd = r["enriched"].get("kills_detail", [])
        for k in kd:
            total_kills_count += 1
            if k.get("headshot"):
                total_kills_hs += 1
            wep = k.get("weapon", "Unknown")
            weapon_kills[wep] = weapon_kills.get(wep, 0) + 1
    hs_pct = round((total_kills_hs / total_kills_count * 100) if total_kills_count else 0, 1)

    # --- Side stats ---
    ct_rounds = [r for r in all_rounds if r["enriched"].get("side") == "CT"]
    t_rounds = [r for r in all_rounds if r["enriched"].get("side") == "T"]
    ct_wins = sum(1 for r in ct_rounds if r["enriched"].get("side") == r["enriched"].get("round_winner"))
    t_wins = sum(1 for r in t_rounds if r["enriched"].get("side") == r["enriched"].get("round_winner"))
    ct_win_pct = round((ct_wins / len(ct_rounds) * 100) if ct_rounds else 0, 1)
    t_win_pct = round((t_wins / len(t_rounds) * 100) if t_rounds else 0, 1)

    # Pistol rounds (round 1 and first round of second half)
    half_start = 13  # MR12 → round 13 is second pistol
    ct_pistol = [r for r in ct_rounds if r.get("round_number") in (1, half_start)]
    t_pistol = [r for r in t_rounds if r.get("round_number") in (1, half_start)]
    ct_pistol_wins = sum(1 for r in ct_pistol if r["enriched"].get("side") == r["enriched"].get("round_winner"))
    t_pistol_wins = sum(1 for r in t_pistol if r["enriched"].get("side") == r["enriched"].get("round_winner"))
    ct_pistol_pct = round((ct_pistol_wins / len(ct_pistol) * 100) if ct_pistol else 0, 0)
    t_pistol_pct = round((t_pistol_wins / len(t_pistol) * 100) if t_pistol else 0, 0)

    # --- Opening duels ---
    opening_kills = 0
    opening_deaths = 0
    for r in all_rounds:
        od = r["enriched"].get("opening_duel")
        if od:
            if od.get("role") == "opening_kill":
                opening_kills += 1
            elif od.get("role") == "opening_death":
                opening_deaths += 1
    opening_total = opening_kills + opening_deaths
    opening_kill_pct = round((opening_kills / opening_total * 100) if opening_total else 0, 1)

    # --- Survival rate ---
    survived = sum(1 for r in all_rounds if r.get("survived"))
    survival_pct = round((survived / total_rounds * 100), 1)

    # --- Clutch stats ---
    clutch_attempts = 0
    clutch_wins = 0
    for r in all_rounds:
        c = r["enriched"].get("clutch")
        if c:
            clutch_attempts += 1
            if c.get("won"):
                clutch_wins += 1
    clutch_win_pct = round((clutch_wins / clutch_attempts * 100) if clutch_attempts else 0, 0)

    # --- Utility stats ---
    total_flashed = 0
    total_flash_assists = 0
    total_he_dmg = 0
    total_molly_dmg = 0
    for r in all_rounds:
        u = r["enriched"].get("utility", {})
        total_flashed += u.get("enemies_flashed", 0)
        total_flash_assists += u.get("flash_assists", 0)
        total_he_dmg += u.get("he_damage", 0)
        total_molly_dmg += sum(m.get("damage", 0) for m in u.get("molotov_damage", []))
    util_per_round = round((total_flashed + total_flash_assists) / total_rounds, 2) if total_rounds else 0
    util_dmg_per_round = round((total_he_dmg + total_molly_dmg) / total_rounds, 1) if total_rounds else 0

    # --- Trade stats ---
    traded_deaths = sum(1 for r in all_rounds if r.get("traded"))
    total_deaths = sum(1 for r in all_rounds if r.get("deaths", 0) > 0)
    trade_pct = round((traded_deaths / total_deaths * 100) if total_deaths else 0, 1)

    # --- Multi-kill stats ---
    total_2k = sum(p.get("rounds_2k", 0) for p in user_players)
    total_3k = sum(p.get("rounds_3k", 0) for p in user_players)
    total_4k = sum(p.get("rounds_4k", 0) for p in user_players)
    total_5k = sum(p.get("rounds_5k", 0) for p in user_players)
    multikill_rounds = total_2k + total_3k + total_4k + total_5k
    multikill_pct = round((multikill_rounds / total_rounds * 100) if total_rounds else 0, 1)

    # --- Role classification ---
    role = _classify_role(
        opening_kill_pct=opening_kill_pct,
        survival_pct=survival_pct,
        util_per_round=util_per_round,
        trade_pct=trade_pct,
        weapon_kills=weapon_kills,
        total_kills=total_kills_count,
    )

    # --- Per-side role classification ---
    ct_role = _classify_side_role(ct_rounds, "CT")
    t_role = _classify_side_role(t_rounds, "T")

    # --- Win/Loss streak (last 10) ---
    matches.sort(key=lambda m: m.get("date", ""), reverse=True)
    recent_results = [m.get("match_result", "")[:1] for m in matches[:10]]

    # Top weapon
    top_weapon = max(weapon_kills, key=weapon_kills.get) if weapon_kills else "Unknown"

    return {
        "total_rounds": total_rounds,
        "total_matches": len(matches),
        "hs_pct": hs_pct,
        "ct_win_pct": ct_win_pct,
        "t_win_pct": t_win_pct,
        "ct_pistol_pct": ct_pistol_pct,
        "t_pistol_pct": t_pistol_pct,
        "opening_kill_pct": opening_kill_pct,
        "opening_kills": opening_kills,
        "opening_deaths": opening_deaths,
        "survival_pct": survival_pct,
        "clutch_attempts": clutch_attempts,
        "clutch_wins": clutch_wins,
        "clutch_win_pct": clutch_win_pct,
        "util_per_round": util_per_round,
        "util_dmg_per_round": util_dmg_per_round,
        "trade_pct": trade_pct,
        "multikill_pct": multikill_pct,
        "multikill_rounds": multikill_rounds,
        "rounds_2k": total_2k,
        "rounds_3k": total_3k,
        "rounds_4k": total_4k,
        "rounds_5k": total_5k,
        "role": role,
        "ct_role": ct_role,
        "t_role": t_role,
        "recent_results": recent_results,
        "top_weapon": top_weapon,
        "weapon_kills": dict(sorted(weapon_kills.items(), key=lambda x: -x[1])[:5]),
    }


def _compute_side_axes(side_rounds: list[dict]) -> dict:
    """Compute 5-axis behavioral scores (0-100) for a set of rounds on one side."""
    n = len(side_rounds)
    if n == 0:
        return {"axes": {"aggression": 0, "trading": 0, "isolation": 0, "survival": 0, "sniper": 0},
                "success": {}}

    kills = 0
    deaths = 0
    opening_kills = 0
    opening_deaths = 0
    opening_duel_involved = 0
    survived = 0
    weapon_kills: dict[str, int] = {}
    total_flashed = 0
    total_flash_assists = 0
    traded_deaths = 0
    total_deaths_count = 0
    total_util_dmg = 0
    awp_kills = 0
    long_range_kills = 0

    # Per-round tracking for success assessment
    round_axes: list[dict] = []  # per-round: dominant axis + won?

    for r in side_rounds:
        e = r.get("enriched", {})
        kd = e.get("kills_detail", [])
        round_kills = len(kd)
        kills += round_kills
        round_awp = 0
        round_lr = 0
        for k in kd:
            wep = k.get("weapon", "Unknown")
            weapon_kills[wep] = weapon_kills.get(wep, 0) + 1
            if wep == "AWP":
                awp_kills += 1
                round_awp += 1
            if (k.get("distance") or 0) >= 30:
                long_range_kills += 1
                round_lr += 1

        round_died = bool(e.get("death_detail"))
        if round_died:
            deaths += 1
        else:
            survived += 1

        round_od_involved = False
        round_od_kill = False
        od = e.get("opening_duel")
        if od:
            opening_duel_involved += 1
            round_od_involved = True
            if od.get("role") == "opening_kill":
                opening_kills += 1
                round_od_kill = True
            elif od.get("role") == "opening_death":
                opening_deaths += 1

        u = e.get("utility", {})
        round_flashed = u.get("enemies_flashed", 0) or 0
        round_fa = u.get("flash_assists", 0) or 0
        total_flashed += round_flashed
        total_flash_assists += round_fa
        he_dmg = u.get("he_damage", 0) or 0
        molly_dmg = sum(m.get("damage", 0) for m in (u.get("molotov_damage") or []))
        round_util_dmg = he_dmg + molly_dmg
        total_util_dmg += round_util_dmg

        round_traded = bool(r.get("traded"))
        if round_traded:
            traded_deaths += 1
        if r.get("deaths", 0) > 0:
            total_deaths_count += 1

        # Per-round dominant behavior (quick heuristic)
        scores = {
            "aggression": (1.0 if round_od_involved else 0) + (0.5 if round_od_kill else 0),
            "trading": (0.5 if round_traded else 0) + min(round_fa * 0.5, 1.0) + min(round_flashed * 0.3, 0.6),
            "isolation": (0.8 if not round_died and not round_od_involved else 0) + min(round_kills * 0.3, 0.6),
            "survival": (0.7 if not round_died else 0) + min(round_util_dmg * 0.02, 0.3),
            "sniper": min(round_awp * 0.8, 1.5) + min(round_lr * 0.3, 0.5),
        }
        dominant = max(scores, key=scores.get)
        side = e.get("side", "")
        won = (side == e.get("round_winner", ""))
        round_axes.append({"dominant": dominant, "won": won})

    # Aggregate axes (same formulas as _classify_side_role)
    survival_pct = (survived / n * 100) if n else 0
    od_total = opening_kills + opening_deaths
    ok_pct = (opening_kills / od_total * 100) if od_total else 0
    involvement_rate = (opening_duel_involved / n * 100) if n else 0

    aggression = min(100, round(involvement_rate * 0.5 + ok_pct * 0.5))

    flash_assist_pr = total_flash_assists / n if n else 0
    enemies_flashed_pr = total_flashed / n if n else 0
    trade_pct = (traded_deaths / total_deaths_count * 100) if total_deaths_count else 0
    trading = min(100, round(
        trade_pct * 0.4
        + min(flash_assist_pr * 50, 30)
        + min(enemies_flashed_pr * 25, 30)
    ))

    non_involvement = 100 - involvement_rate
    isolation = min(100, round(
        survival_pct * 0.4
        + non_involvement * 0.3
        + (min((kills / n) * 40, 30) if n else 0)
    ))

    util_dmg_pr = total_util_dmg / n if n else 0
    death_rate = (deaths / n * 100) if n else 0
    low_death_score = max(0, 100 - death_rate)
    survival_axis = min(100, round(
        survival_pct * 0.5
        + min(util_dmg_pr * 3, 25)
        + low_death_score * 0.25
    ))

    awp_ratio = (awp_kills / kills * 100) if kills else 0
    lr_ratio = (long_range_kills / kills * 100) if kills else 0
    sniper = min(100, round(awp_ratio * 0.7 + lr_ratio * 0.3))

    axes = {
        "aggression": aggression,
        "trading": trading,
        "isolation": isolation,
        "survival": survival_axis,
        "sniper": sniper,
    }

    # Success per dominant-axis: win rate when that axis was dominant
    success: dict[str, dict] = {}
    for axis in axes:
        axis_rounds = [ra for ra in round_axes if ra["dominant"] == axis]
        total_ax = len(axis_rounds)
        if total_ax > 0:
            wins_ax = sum(1 for ra in axis_rounds if ra["won"])
            success[axis] = {
                "rounds": total_ax,
                "wins": wins_ax,
                "win_pct": round(wins_ax / total_ax * 100, 0),
            }

    return {"axes": axes, "success": success}


def _compute_match_behavioral_axes(rounds: list[dict]) -> dict:
    """Compute per-match 5-axis behavioral assessment for CT and T sides."""
    # Parse enriched_json for each round
    for r in rounds:
        ej = r.get("enriched_json")
        if ej and isinstance(ej, str):
            try:
                r["enriched"] = json.loads(ej)
            except Exception:
                r["enriched"] = {}
        elif not r.get("enriched"):
            r["enriched"] = {}

    ct_rounds = [r for r in rounds if r.get("enriched", {}).get("side") == "CT"]
    t_rounds = [r for r in rounds if r.get("enriched", {}).get("side") == "T"]

    return {
        "ct": _compute_side_axes(ct_rounds),
        "t": _compute_side_axes(t_rounds),
    }


def _classify_role(
    opening_kill_pct: float,
    survival_pct: float,
    util_per_round: float,
    trade_pct: float,
    weapon_kills: dict,
    total_kills: int,
) -> dict:
    """Heuristic role classification based on playstyle indicators."""
    awp_kills = weapon_kills.get("AWP", 0)
    awp_ratio = (awp_kills / total_kills * 100) if total_kills else 0

    if awp_ratio >= 30:
        return {
            "name": "AWPer",
            "icon": "precision_manufacturing",
            "description": "Primary AWP player. High-impact picks from long-range angles, "
            "controlling sightlines and creating openings for the team.",
        }
    if opening_kill_pct >= 55:
        return {
            "name": "Entry Fragger",
            "icon": "bolt",
            "description": "Aggressive entry style. Frequently takes the first duel of the "
            "round, creating space for teammates to trade and execute.",
        }
    if survival_pct >= 55 and opening_kill_pct < 35:
        return {
            "name": "Lurker",
            "icon": "visibility",
            "description": "Patient rotator who picks off distracted enemies. High survival "
            "rate indicates good timing and map awareness.",
        }
    if util_per_round >= 1.5 and trade_pct >= 30:
        return {
            "name": "Support",
            "icon": "shield_with_heart",
            "description": "Team-oriented playstyle with strong utility usage and trade discipline. "
            "Enables teammates through flashes, trades, and info plays.",
        }
    if survival_pct >= 50:
        return {
            "name": "Anchor",
            "icon": "anchor",
            "description": "Reliable site holder with strong survival instincts. Holds positions "
            "patiently and trades effectively during retakes.",
        }
    return {
        "name": "Flex",
        "icon": "sync_alt",
        "description": "Versatile player who adapts role based on the round situation. "
        "Balanced across entry, utility, and trading metrics.",
    }


def _empty_performance() -> dict:
    _side_empty = {"name": "Unknown", "icon": "help", "description": "No data.",
                   "kills": 0, "deaths": 0, "rounds": 0, "adr": 0,
                   "opening_kills": 0, "opening_deaths": 0, "survival_pct": 0,
                   "axes": {"aggression": 0, "trading": 0, "isolation": 0, "survival": 0, "sniper": 0}}
    return {
        "total_rounds": 0, "total_matches": 0, "hs_pct": 0,
        "ct_win_pct": 0, "t_win_pct": 0, "ct_pistol_pct": 0, "t_pistol_pct": 0,
        "opening_kill_pct": 0, "opening_kills": 0, "opening_deaths": 0,
        "survival_pct": 0, "clutch_attempts": 0, "clutch_wins": 0, "clutch_win_pct": 0,
        "util_per_round": 0, "util_dmg_per_round": 0, "trade_pct": 0,
        "multikill_pct": 0, "multikill_rounds": 0,
        "rounds_2k": 0, "rounds_3k": 0, "rounds_4k": 0, "rounds_5k": 0,
        "role": {"name": "Unknown", "icon": "help", "description": "Upload demos to analyze your playstyle."},
        "ct_role": _side_empty,
        "t_role": _side_empty,
        "recent_results": [], "top_weapon": "Unknown", "weapon_kills": {},
    }


def _classify_side_role(side_rounds: list, side: str) -> dict:
    """Classify role for a specific side (CT/T) and return stats."""
    n = len(side_rounds)
    if n == 0:
        return {"name": "Unknown", "icon": "help", "description": "No data.",
                "kills": 0, "deaths": 0, "rounds": 0, "adr": 0,
                "opening_kills": 0, "opening_deaths": 0, "survival_pct": 0,
                "axes": {"aggression": 0, "trading": 0, "isolation": 0, "survival": 0, "sniper": 0}}

    kills = 0
    deaths = 0
    opening_kills = 0
    opening_deaths = 0
    survived = 0
    weapon_kills: dict[str, int] = {}
    total_flashed = 0
    total_flash_assists = 0
    traded_deaths = 0
    total_deaths_count = 0
    total_damage = 0
    # Extra counters for 5-axis assessment
    total_util_dmg = 0       # HE + molotov damage
    awp_kills = 0
    opening_duel_involved = 0  # rounds with an opening duel involvement
    trade_kills = 0            # rounds where the player's death was traded by a teammate
    long_range_kills = 0       # kills at distance >= 30m (3000 units ≈ 30m in CS2)

    for r in side_rounds:
        e = r.get("enriched", {})
        kd = e.get("kills_detail", [])
        kills += len(kd)
        for k in kd:
            wep = k.get("weapon", "Unknown")
            weapon_kills[wep] = weapon_kills.get(wep, 0) + 1
            if wep == "AWP":
                awp_kills += 1
            dist = k.get("distance", 0) or 0
            if dist >= 30:
                long_range_kills += 1
        if e.get("death_detail"):
            deaths += 1
        else:
            survived += 1
        od = e.get("opening_duel")
        if od:
            opening_duel_involved += 1
            if od.get("role") == "opening_kill":
                opening_kills += 1
            elif od.get("role") == "opening_death":
                opening_deaths += 1
        u = e.get("utility", {})
        total_flashed += u.get("enemies_flashed", 0)
        total_flash_assists += u.get("flash_assists", 0)
        he_dmg = u.get("he_damage", 0) or 0
        molly_dmg = sum(m.get("damage", 0) for m in (u.get("molotov_damage") or []))
        total_util_dmg += he_dmg + molly_dmg
        if r.get("traded"):
            traded_deaths += 1
            trade_kills += 1  # teammate traded the player's death
        if r.get("deaths", 0) > 0:
            total_deaths_count += 1
        total_damage += r.get("damage", 0)

    survival_pct = round(survived / n * 100, 1)
    od_total = opening_kills + opening_deaths
    ok_pct = round(opening_kills / od_total * 100, 1) if od_total else 0
    util_pr = round((total_flashed + total_flash_assists) / n, 2)
    trade_pct = round(traded_deaths / total_deaths_count * 100, 1) if total_deaths_count else 0
    adr = round(total_damage / n, 1) if n else 0

    # --- 5-Axis Role Assessment (each 0-100) ---
    # Aggression (Entry Score): opening duel involvement rate + opening kill win rate
    involvement_rate = (opening_duel_involved / n * 100) if n else 0
    aggression = min(100, round(involvement_rate * 0.5 + ok_pct * 0.5))

    # Trading (Support Score): trade %, flash assists per round, enemies flashed per round
    flash_assist_pr = total_flash_assists / n if n else 0
    enemies_flashed_pr = total_flashed / n if n else 0
    trading = min(100, round(
        trade_pct * 0.4
        + min(flash_assist_pr * 50, 30) * 1.0       # ~0.6 fa/r → 30 pts
        + min(enemies_flashed_pr * 25, 30) * 1.0     # ~1.2 ef/r → 30 pts
    ))

    # Isolation (Lurker Score): high survival + low opening duel involvement + high kill count
    non_involvement = 100 - involvement_rate
    isolation = min(100, round(
        survival_pct * 0.4
        + non_involvement * 0.3
        + min((kills / n) * 40, 30) if n else 0       # reward getting kills while lurking
    ))

    # Survival (Passive Score): survival rate + utility damage contribution + low deaths
    util_dmg_pr = total_util_dmg / n if n else 0
    death_rate = (deaths / n * 100) if n else 0
    low_death_score = max(0, 100 - death_rate)
    survival_axis = min(100, round(
        survival_pct * 0.5
        + min(util_dmg_pr * 3, 25)               # ~8 util dmg/r → 25 pts
        + low_death_score * 0.25
    ))

    # Sniper (AWP Score): AWP kill ratio + long-range kill ratio
    awp_ratio = (awp_kills / kills * 100) if kills else 0
    lr_ratio = (long_range_kills / kills * 100) if kills else 0
    sniper = min(100, round(
        awp_ratio * 0.7
        + lr_ratio * 0.3
    ))

    axes = {
        "aggression": aggression,
        "trading": trading,
        "isolation": isolation,
        "survival": survival_axis,
        "sniper": sniper,
    }

    role = _classify_role(
        opening_kill_pct=ok_pct,
        survival_pct=survival_pct,
        util_per_round=util_pr,
        trade_pct=trade_pct,
        weapon_kills=weapon_kills,
        total_kills=kills,
    )
    role.update({
        "kills": kills,
        "deaths": deaths,
        "rounds": n,
        "adr": adr,
        "opening_kills": opening_kills,
        "opening_deaths": opening_deaths,
        "survival_pct": survival_pct,
        "axes": axes,
    })
    return role


# ---------------------------------------------------------------------------
# AI-powered role assessment
# ---------------------------------------------------------------------------

_AI_ROLES_FILE = Path(__file__).parent / "data" / "ai_roles.json"


def _load_ai_roles() -> dict:
    if _AI_ROLES_FILE.exists():
        return json.loads(_AI_ROLES_FILE.read_text(encoding="utf-8"))
    return {}


def _save_ai_roles(data: dict) -> None:
    _AI_ROLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _AI_ROLES_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_role_context(map_name: str, ct_rounds: list, t_rounds: list) -> str:
    """Build a concise prompt with round-by-round positional data for AI role assessment."""
    lines = [
        f"Map: {map_name}",
        f"CT rounds: {len(ct_rounds)}  |  T rounds: {len(t_rounds)}",
        "",
        "=== CT SIDE ROUNDS ===",
    ]
    for r in ct_rounds:
        lines.append(_format_round_narrative(r))
    lines.extend(["", "=== T SIDE ROUNDS ==="])
    for r in t_rounds:
        lines.append(_format_round_narrative(r))
    return "\n".join(lines)


@app.post("/api/performance/ai-roles")
async def ai_role_assessment(maps: str = ""):
    """Use AI to assess player's CT and T roles on a specific map."""
    import json as _json

    if not maps.strip():
        raise HTTPException(status_code=400, detail="Map filter required for AI role assessment")

    # Check AI config
    ai_config = load_ai_config()
    provider = ai_config.get("active_provider", "")
    model = ai_config.get("active_model", "")
    if not provider or not model:
        raise HTTPException(status_code=400, detail="No AI provider/model configured. Set up AI in settings first.")
    prov_config = ai_config.get("providers", {}).get(provider, {})
    api_key = prov_config.get("api_key", "")
    if not api_key:
        raise HTTPException(status_code=400, detail=f"No API key set for {provider}")

    # Gather round data
    conn = _db()
    matches = get_all_matches(conn)
    allowed = {m.strip().lower() for m in maps.split(",")}
    matches = [m for m in matches if m.get("map_name", "").lower() in allowed]
    if not matches:
        conn.close()
        raise HTTPException(status_code=404, detail="No matches found for this map")

    all_rounds = []
    for mid in [m["match_id"] for m in matches]:
        for r in get_round_stats(conn, mid):
            ej = r.get("enriched_json")
            if ej:
                try:
                    r["enriched"] = _json.loads(ej)
                except Exception:
                    r["enriched"] = {}
            else:
                r["enriched"] = {}
            all_rounds.append(r)
    conn.close()

    ct_rounds = [r for r in all_rounds if r["enriched"].get("side") == "CT"]
    t_rounds = [r for r in all_rounds if r["enriched"].get("side") == "T"]

    if not ct_rounds and not t_rounds:
        raise HTTPException(status_code=404, detail="No enriched round data for this map")

    map_display = maps.strip().split(",")[0]
    round_context = _build_role_context(map_display, ct_rounds, t_rounds)

    system_prompt = (
        "You are PULSE_AI, an expert CS2 analyst. You are given detailed round-by-round data "
        "from competitive matches on a specific map. Your task is to identify the player's "
        "ROLE on each side (CT and T) based on their positioning, kill locations, death locations, "
        "utility usage, opening duel tendencies, and economy patterns.\n\n"
        "IMPORTANT:\n"
        "- Analyze WHERE the player positions (callout names in the data) across rounds.\n"
        "- Look for PATTERNS: do they consistently play the same site/area? Do they rotate?\n"
        "- Consider their aggression (opening duels), utility (flashes, molotovs), and survival.\n"
        "- Be specific to the map. E.g. on Inferno CT side: 'Pit Anchor' or 'B Rotator from CT'.\n"
        "- On T side, identify if they entry, lurk, or support based on kill/death positions.\n\n"
        "Respond in EXACTLY this JSON format (no markdown, no extra text):\n"
        "{\n"
        '  "ct_role": {\n'
        '    "name": "<short role name, 2-4 words, map-specific e.g. Pit Anchor, B Anchor, AWP Mid>",\n'
        '    "icon": "<one of: shield, bolt, visibility, anchor, shield_with_heart, sync_alt, precision_manufacturing, swords>",\n'
        '    "description": "<2-3 sentences explaining the role based on actual round data and positions>"\n'
        "  },\n"
        '  "t_role": {\n'
        '    "name": "<short role name, 2-4 words, map-specific e.g. Banana Entry, Lurk Apartments>",\n'
        '    "icon": "<one of: shield, bolt, visibility, anchor, shield_with_heart, sync_alt, precision_manufacturing, swords>",\n'
        '    "description": "<2-3 sentences explaining the role based on actual round data and positions>"\n'
        "  }\n"
        "}"
    )

    messages = [{"role": "user", "content": round_context}]
    try:
        response = await chat_completion(provider, model, api_key, messages, system_prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI request failed: {exc}")

    # Parse AI response JSON
    try:
        # Strip markdown fences if the model wraps in ```json
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        result = _json.loads(text)
    except (_json.JSONDecodeError, IndexError):
        # Fallback: return raw text so frontend can still display something
        result = {
            "ct_role": {"name": "AI Analysis", "icon": "smart_toy", "description": response[:500]},
            "t_role": {"name": "AI Analysis", "icon": "smart_toy", "description": response[:500]},
        }

    # Persist the assessment
    ai_roles = _load_ai_roles()
    map_key = maps.strip().split(",")[0].lower()
    ai_roles[map_key] = {
        "ct_role": result.get("ct_role", {}),
        "t_role": result.get("t_role", {}),
        "provider": provider,
        "model": model,
    }
    _save_ai_roles(ai_roles)

    return {
        "ct_role": result.get("ct_role", {}),
        "t_role": result.get("t_role", {}),
        "provider": provider,
        "model": model,
    }


@app.get("/api/performance/ai-roles")
def get_persisted_ai_roles(maps: str = ""):
    """Return persisted AI role assessments for a map (if any)."""
    roles = _load_ai_roles()
    if maps.strip():
        map_key = maps.strip().split(",")[0].lower()
        entry = roles.get(map_key)
        if entry:
            return entry
    return {}


# ---------------------------------------------------------------------------
# Minimap
# ---------------------------------------------------------------------------
@app.get("/api/matches/{match_id}/minimap")
def get_minimap_data(match_id: str, round_number: int = 0):
    """Return position data for minimap rendering.

    If *round_number* is 0, returns all rounds. Otherwise, returns data
    for the specified round only.
    """
    import json as _json

    conn = _db()
    match = get_match(conn, match_id)
    if not match:
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")

    map_name = match.get("map_name", "")
    radar = get_radar_config(map_name)
    if not radar:
        conn.close()
        raise HTTPException(status_code=400, detail=f"No radar data for {map_name}")

    rounds = get_round_stats(conn, match_id)
    conn.close()

    result_rounds = []
    for r in rounds:
        rn = r.get("round_number", 0)
        if round_number and rn != round_number:
            continue

        enriched = {}
        ej = r.get("enriched_json")
        if ej:
            try:
                enriched = _json.loads(ej)
            except Exception:
                pass

        events = []
        # Kill events (attacker + victim positions)
        for k in enriched.get("kills_detail", []):
            if k.get("attacker_xy"):
                px = game_to_pixel(map_name, k["attacker_xy"][0], k["attacker_xy"][1])
                if px:
                    events.append({
                        "type": "kill",
                        "role": "attacker",
                        "px": px[0], "py": px[1],
                        "victim": k.get("victim", "?"),
                        "weapon": k.get("weapon", "?"),
                        "headshot": k.get("headshot", False),
                    })
            if k.get("victim_xy"):
                px = game_to_pixel(map_name, k["victim_xy"][0], k["victim_xy"][1])
                if px:
                    events.append({
                        "type": "death",
                        "role": "victim_of_kill",
                        "px": px[0], "py": px[1],
                        "name": k.get("victim", "?"),
                    })

        # Player death (user died)
        dd = enriched.get("death_detail")
        if dd and dd.get("victim_xy"):
            px = game_to_pixel(map_name, dd["victim_xy"][0], dd["victim_xy"][1])
            if px:
                events.append({
                    "type": "player_death",
                    "role": "user_died",
                    "px": px[0], "py": px[1],
                    "killer": dd.get("killer", "?"),
                    "weapon": dd.get("weapon", "?"),
                })
        if dd and dd.get("killer_xy"):
            px = game_to_pixel(map_name, dd["killer_xy"][0], dd["killer_xy"][1])
            if px:
                events.append({
                    "type": "killer_pos",
                    "role": "killed_user",
                    "px": px[0], "py": px[1],
                    "name": dd.get("killer", "?"),
                })

        # Grenade events (throw → land positions)
        for g in enriched.get("utility", {}).get("grenades", []):
            throw_px = None
            land_px = None
            if g.get("throw_xy"):
                throw_px = game_to_pixel(map_name, g["throw_xy"][0], g["throw_xy"][1])
            if g.get("land_xy"):
                land_px = game_to_pixel(map_name, g["land_xy"][0], g["land_xy"][1])
            if throw_px or land_px:
                ev = {
                    "type": "grenade",
                    "nade_type": g.get("type", "?"),
                    "throw_callout": g.get("throw_callout", ""),
                    "land_callout": g.get("land_callout", ""),
                }
                if throw_px:
                    ev["throw_px"] = throw_px[0]
                    ev["throw_py"] = throw_px[1]
                if land_px:
                    ev["land_px"] = land_px[0]
                    ev["land_py"] = land_px[1]
                events.append(ev)

        # Flash victim positions
        for fv in enriched.get("utility", {}).get("flash_instances", []):
            if fv.get("victim_xy") and not fv.get("is_friendly", False):
                px = game_to_pixel(map_name, fv["victim_xy"][0], fv["victim_xy"][1])
                if px:
                    events.append({
                        "type": "flash_victim",
                        "px": px[0], "py": px[1],
                        "name": fv.get("name", "?"),
                        "duration": fv.get("duration", 0),
                    })

        # HE victim positions
        for hv in enriched.get("utility", {}).get("he_victims", []):
            if hv.get("victim_xy"):
                px = game_to_pixel(map_name, hv["victim_xy"][0], hv["victim_xy"][1])
                if px:
                    events.append({
                        "type": "he_victim",
                        "px": px[0], "py": px[1],
                        "name": hv.get("name", "?"),
                        "damage": hv.get("damage", 0),
                    })

        # Molotov/incendiary victim positions
        for mv in enriched.get("utility", {}).get("molotov_damage", []):
            if mv.get("victim_xy"):
                px = game_to_pixel(map_name, mv["victim_xy"][0], mv["victim_xy"][1])
                if px:
                    events.append({
                        "type": "molotov_victim",
                        "px": px[0], "py": px[1],
                        "name": mv.get("victim", "?"),
                        "damage": mv.get("damage", 0),
                    })

        result_rounds.append({
            "round": rn,
            "side": enriched.get("side", "?"),
            "won": enriched.get("side") == enriched.get("round_winner"),
            "events": events,
        })

    return {
        "map_name": map_name,
        "radar": radar,
        "radar_image": f"/frontend/img/radar/{map_name.removeprefix('de_').removeprefix('cs_')}.png?v={int(__import__('time').time())}",
        "rounds": result_rounds,
    }


@app.post("/api/minimap/zones")
def resolve_minimap_zones(body: dict):
    """Convert callout names to pixel coordinates for highlighting.

    Body: {"map_name": "de_mirage", "callouts": ["B Apartments", "A Ramp"]}
    """
    map_name = body.get("map_name", "")
    callouts = body.get("callouts", [])

    if not is_map_supported(map_name):
        raise HTTPException(status_code=400, detail=f"Map {map_name} not supported")

    zones = []
    for name in callouts:
        center = get_zone_center(map_name, name)
        if center:
            zones.append({"callout": name, "px": center[0], "py": center[1]})
    return {"map_name": map_name, "zones": zones}


@app.get("/api/minimap/{map_name}/schematic")
def get_minimap_schematic(map_name: str):
    """Return all zone rectangles in pixel-space for the schematic renderer."""
    rects = get_all_zones_pixel(map_name)
    if rects is None:
        raise HTTPException(status_code=400, detail=f"Map {map_name} not supported")

    return {"map_name": map_name, "zones": rects}


@app.get("/api/minimap/{map_name}/debug-positions")
def get_debug_positions(map_name: str):
    """Return every event position from all matches on this map.

    Used for visual diagnostics: plot all dots on the radar to verify
    that the game-to-pixel transform is correct.
    """
    import json as _json

    if not is_map_supported(map_name):
        raise HTTPException(status_code=400, detail=f"Map {map_name} not supported")

    conn = _db()
    rows = conn.execute(
        "SELECT rs.enriched_json FROM round_stats rs "
        "JOIN matches m ON rs.match_id = m.match_id "
        "WHERE m.map_name = ?",
        (map_name,),
    ).fetchall()
    conn.close()

    points: list[dict] = []
    for (ej_raw,) in rows:
        if not ej_raw:
            continue
        data = _json.loads(ej_raw)
        for k in data.get("kills_detail", []):
            for role in ("attacker", "victim"):
                xy = k.get(f"{role}_xy")
                if xy:
                    px = game_to_pixel(map_name, xy[0], xy[1])
                    if px:
                        points.append({
                            "gx": xy[0], "gy": xy[1],
                            "px": px[0], "py": px[1],
                            "zone": get_callout(map_name, xy[0], xy[1]),
                            "role": role,
                        })
        dd = data.get("death_detail")
        if dd:
            for role in ("victim", "killer"):
                xy = dd.get(f"{role}_xy")
                if xy:
                    px = game_to_pixel(map_name, xy[0], xy[1])
                    if px:
                        points.append({
                            "gx": xy[0], "gy": xy[1],
                            "px": px[0], "py": px[1],
                            "zone": get_callout(map_name, xy[0], xy[1]),
                            "role": role,
                        })
    return {"map_name": map_name, "count": len(points), "positions": points}
