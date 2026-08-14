"""Accounts, friends, the active Steam ID, onboarding, and factory reset."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.deps import (
    JsonStore,
    _ai_assessments_store,
    _ai_patterns_store,
    _ai_roles_store,
    _auto_sync_config_store,
    _get_active_account,
    _load_accounts,
    _load_friends,
    _load_onboarding,
    _save_accounts,
    _save_friends,
    _save_onboarding,
    _save_sync_config,
    _steam_ledger_store,
    _steam_tokens_store,
    _stop_auto_sync,
    _storage_config_store,
    steam_id_path,
)
from src.api.schemas import (
    AccountCreate,
    AccountUpdate,
    ConfigResponse,
    ConfigUpdate,
    FriendCreate,
    UpdateConfig,
    UpdateStatusResponse,
)
from src.services import updates

router = APIRouter()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@router.get("/api/config", response_model=ConfigResponse)
def get_config():
    """Return the currently configured Steam ID (from active account)."""
    active = _get_active_account()
    if active:
        return {"steam_id": active["steam_id"]}
    # Fallback to legacy steamID file
    steam_id = None
    if steam_id_path().exists():
        steam_id = steam_id_path().read_text().strip() or None
    return {"steam_id": steam_id}

@router.put("/api/config")
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
    steam_id_path().parent.mkdir(parents=True, exist_ok=True)
    steam_id_path().write_text(sid)
    return {"steam_id": sid}

# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------
@router.get("/api/accounts")
def list_accounts():
    """Return all configured accounts."""
    return _load_accounts()

@router.post("/api/accounts", status_code=201)
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
        steam_id_path().parent.mkdir(parents=True, exist_ok=True)
        steam_id_path().write_text(sid)
    return accounts[-1]

@router.put("/api/accounts/{steam_id}")
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

@router.put("/api/accounts/{steam_id}/activate")
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
    steam_id_path().parent.mkdir(parents=True, exist_ok=True)
    steam_id_path().write_text(steam_id)
    return {"activated": steam_id}

@router.delete("/api/accounts/{steam_id}")
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
        steam_id_path().write_text(accounts[0]["steam_id"])
    _save_accounts(accounts)
    return {"deleted": steam_id}

# ---------------------------------------------------------------------------
# Friends
# ---------------------------------------------------------------------------
@router.get("/api/friends")
def list_friends():
    """Return all configured friends."""
    return _load_friends()

@router.post("/api/friends", status_code=201)
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

@router.delete("/api/friends/{steam_id}")
def delete_friend(steam_id: str):
    """Remove a friend."""
    friends = _load_friends()
    before = len(friends)
    friends = [f for f in friends if f["steam_id"] != steam_id]
    if len(friends) == before:
        raise HTTPException(status_code=404, detail="Friend not found")
    _save_friends(friends)
    return {"deleted": steam_id}

@router.get("/api/onboarding")
def get_onboarding():
    return _load_onboarding()

@router.put("/api/onboarding")
def set_onboarding(body: dict):
    state = _load_onboarding()
    if "completed" in body:
        state["completed"] = bool(body["completed"])
    _save_onboarding(state)
    return state

# ---------------------------------------------------------------------------
# Updates
#
# The only outbound request the app makes on its own behalf, and it makes it
# only when told to. See src/services/updates.py for what is and is not sent.
# ---------------------------------------------------------------------------
@router.get("/api/updates", response_model=UpdateStatusResponse)
def get_updates():
    """Cached status. Contacts nothing unless the setting is on and stale."""
    return updates.check().as_dict()


@router.post("/api/updates/check", response_model=UpdateStatusResponse)
def check_updates():
    """Ask now. Works with the setting off — pressing the button is the ask."""
    return updates.check(force=True).as_dict()


@router.put("/api/updates/config", response_model=UpdateStatusResponse)
def set_update_config(body: UpdateConfig):
    return updates.set_enabled(body.enabled).as_dict()


# ---------------------------------------------------------------------------
# Factory Reset
# ---------------------------------------------------------------------------
@router.post("/api/factory-reset")
def factory_reset():
    """Delete everything the user owns and return to a fresh install.

    It used to leave four files behind, two of which are credentials:
    steam_tokens.json holds Steam refresh tokens, which are
    password-equivalent, and steam_sharecodes.json holds the Steam Web API key
    and the match ledger. Someone who reset the app before handing over the
    machine, or before publishing a container, still had both on disk. The
    other two were the Auto-Sync and retention settings.

    Auto-Sync is stopped first. Left running it would carry on downloading
    with the tokens this is about to delete, and write a new ledger behind it.
    """
    from src.database import _DEFAULT_DB_PATH as db_path

    errors: list[str] = []

    def attempt(what: str, action) -> None:
        try:
            action()
        except Exception as exc:  # one failure must not skip the rest
            errors.append(f"{what}: {exc}")

    # The loop holds the job slot and would keep writing after the wipe.
    attempt("Auto-Sync", _stop_auto_sync)

    # ":memory:" and an already-absent file are both "nothing to delete",
    # not failures worth reporting as a partial reset.
    attempt("Database", lambda: db_path.unlink() if db_path.is_file() else None)
    attempt("Accounts", lambda: _save_accounts([]))
    attempt("Friends", lambda: _save_friends([]))
    attempt("Sync config", lambda: _save_sync_config({"folder": ""}))
    attempt("Onboarding", lambda: _save_onboarding({"completed": False}))

    # Credentials. Deleted rather than emptied — an empty file that used to
    # hold a token is still a file the user has to be told about.
    for store in (_steam_tokens_store, _steam_ledger_store):
        attempt(f"Steam credentials ({store.name})", store.delete)

    # Assessments, including the two files the merged one replaced.
    for store in (_ai_assessments_store, _ai_roles_store, _ai_patterns_store):
        attempt(f"AI assessment ({store.name})", store.delete)

    # Settings with defaults: removing the file is the reset. The update
    # setting goes too — a fresh install must not inherit permission to reach
    # the network from the one before it.
    for store in (_auto_sync_config_store, _storage_config_store,
                  updates.config_store, updates.cache_store):
        attempt(f"Settings ({store.name})", store.delete)

    attempt("SteamID", lambda: steam_id_path().unlink(missing_ok=True))

    def clear_ai_keys() -> None:
        # Kept rather than deleted so a user's prompts and provider choice
        # survive; only the secrets and instructions go.
        store = JsonStore("ai_config.json")
        if not store.exists():
            return
        cfg = store.read()
        for provider in cfg.get("providers", {}).values():
            provider["api_key"] = ""
        cfg["system_instructions"] = ""
        store.write(cfg)

    attempt("AI config", clear_ai_keys)

    if errors:
        return {"status": "partial", "errors": errors}
    return {"status": "ok"}
