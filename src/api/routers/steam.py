"""The Steam fetcher: setup, manual jobs, and Auto-Sync."""

from __future__ import annotations

import os
import re

from fastapi import APIRouter, HTTPException

from src.api.deps import (
    _AUTH_CODE_RE,
    _FETCHER_DIR,
    _SHARE_CODE_RE,
    _auto_sync_snapshot,
    _check_cs2_presence,
    _job_snapshot,
    _load_accounts,
    _load_auto_sync_config,
    _read_steam_ledger,
    _require_fetcher,
    _save_auto_sync_config,
    _start_auto_sync,
    _start_steam_job,
    _steam_account_rows,
    _steam_jobs,
    _stop_auto_sync,
    _utc_now,
    _write_steam_ledger,
)
from src.api.schemas import (
    AutoSyncIn,
    SteamApiKeyIn,
    SteamCodesIn,
    SteamDownloadIn,
    SteamTogglesIn,
)

router = APIRouter()


@router.get("/api/steam/status")
def steam_status():
    """Setup state for the fetcher path: what is installed, authenticated, configured."""
    ledger = _read_steam_ledger()
    rows, pending_total = _steam_account_rows()

    node = _steam_jobs.node_path()
    deps_installed = (_FETCHER_DIR / "node_modules").is_dir()

    # The fetcher prefers STEAM_API_KEY over the stored one, so say which is
    # actually in play — otherwise "issued by pr1me" would be a lie whenever
    # the environment overrides it.
    env_key = os.environ.get("STEAM_API_KEY")
    stored_key = ledger.get("apiKey")
    key_in_use = env_key or stored_key

    return {
        "available": bool(node) and deps_installed,
        "node_installed": bool(node),
        "deps_installed": deps_installed,
        "fetcher_present": _FETCHER_DIR.is_dir(),
        "api_key_set": bool(key_in_use),
        "api_key_source": "environment" if env_key else ("stored" if stored_key else None),
        "api_key_account": None if env_key else ledger.get("apiKeyAccount"),
        # Last 4 characters only — enough to tell one key from another without
        # putting the credential back on the wire.
        "api_key_tail": key_in_use[-4:] if key_in_use else None,
        "authenticated_count": sum(1 for r in rows if r["authenticated"]),
        "accounts": rows,
        "pending_total": pending_total,
        "job": _job_snapshot(),
        # Folded in so the modal's single poll covers the auto-sync panel too.
        "auto_sync": _auto_sync_snapshot(),
    }

@router.put("/api/steam/api-key")
def set_steam_api_key(body: SteamApiKeyIn):
    """Store the Steam Web API key.

    One key covers every account — it is an API credential, not per-account
    authentication (that is the steamidkey). But a key is *issued* by one
    specific Steam account, and nothing in the key itself says which. That
    matters when it stops working: you have to know whose profile to go and
    regenerate it from. So the issuing account is recorded alongside it.
    """
    key = body.api_key.strip()
    if not re.fullmatch(r"[A-Fa-f0-9]{32}", key):
        raise HTTPException(
            status_code=400,
            detail="That does not look like a Steam Web API key (32 hexadecimal characters).",
        )

    owner = (body.account or "").strip() or None
    if owner and not any(a.get("name") == owner for a in _load_accounts()):
        raise HTTPException(status_code=404, detail=f"No account named '{owner}'.")

    ledger = _read_steam_ledger()
    ledger.setdefault("accounts", {})
    ledger["apiKey"] = key
    ledger["apiKeyAccount"] = owner
    _write_steam_ledger(ledger)
    return {"ok": True, "account": owner, "tail": key[-4:]}

@router.put("/api/steam/accounts/{name}")
def set_steam_account_codes(name: str, body: SteamCodesIn):
    """Store an account's match-sharing auth code and its starting share code."""
    auth_code = body.auth_code.strip()
    share_code = body.share_code.strip()

    if not _AUTH_CODE_RE.fullmatch(auth_code):
        raise HTTPException(
            status_code=400, detail="Auth code should look like XXXX-XXXXX-XXXX."
        )
    if not _SHARE_CODE_RE.fullmatch(share_code):
        raise HTTPException(
            status_code=400,
            detail="Share code should look like CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx.",
        )
    if not any(a.get("name") == name for a in _load_accounts()):
        raise HTTPException(status_code=404, detail=f"No account named '{name}'.")

    ledger = _read_steam_ledger()
    accounts = ledger.setdefault("accounts", {})
    entry = accounts.setdefault(
        name, {"authCode": None, "seedShareCode": None, "cursor": None, "matches": {}}
    )

    entry["authCode"] = auth_code
    entry["seedShareCode"] = share_code
    if not entry.get("cursor"):
        entry["cursor"] = share_code

    # The seed is itself a real match, so record it as work to do.
    entry.setdefault("matches", {})
    entry["matches"].setdefault(
        share_code, {"discoveredAt": _utc_now(), "status": "pending", "filename": None}
    )

    _write_steam_ledger(ledger)
    return {"ok": True}

@router.put("/api/steam/accounts/{name}/toggles")
def set_steam_account_toggles(name: str, body: SteamTogglesIn):
    """Choose which accounts are tracked, and which have their demos downloaded.

    The two are independent: an account can stay in the ledger — so you always
    know what it played — without its demos being fetched.
    """
    if not any(a.get("name") == name for a in _load_accounts()):
        raise HTTPException(status_code=404, detail=f"No account named '{name}'.")

    ledger = _read_steam_ledger()
    accounts = ledger.setdefault("accounts", {})
    entry = accounts.setdefault(
        name,
        {
            "authCode": None,
            "seedShareCode": None,
            "cursor": None,
            "walkEnabled": True,
            "downloadEnabled": True,
            "matches": {},
        },
    )

    if body.walk_enabled is not None:
        entry["walkEnabled"] = body.walk_enabled
    if body.download_enabled is not None:
        entry["downloadEnabled"] = body.download_enabled

    _write_steam_ledger(ledger)
    return {
        "name": name,
        "walk_enabled": entry.get("walkEnabled") is not False,
        "download_enabled": entry.get("downloadEnabled") is not False,
    }

@router.post("/api/steam/auth/{name}")
def steam_auth(name: str):
    """Start a QR sign-in for one account.

    Only the QR flow is exposed here. QR is a device authorisation — no
    password is involved — so it is safe in a browser. The credential flow
    would mean a Steam password in a web form, and stays a terminal command.
    """
    if not any(a.get("name") == name for a in _load_accounts()):
        raise HTTPException(status_code=404, detail=f"No account named '{name}'.")
    return _start_steam_job("auth", ["auth-qr.js", name])

@router.post("/api/steam/check")
def steam_check_for_new():
    """Ask Steam which matches have been played since the stored cursor.

    Needs no Steam client session, so it is safe to run while Steam is open.
    """
    return _start_steam_job("check", ["sharecodes.js", "--walk"])

@router.post("/api/steam/download")
def steam_download(body: SteamDownloadIn | None = None):
    """Download outstanding demos.

    ``limit`` takes only the newest N, which keeps a first run over a large
    backlog to a sensible size. Older matches are likelier to have expired
    anyway, so newest-first is the useful order.
    """
    args = ["fetch.js"]

    if body and body.limit is not None:
        if body.limit < 1:
            raise HTTPException(status_code=400, detail="limit must be at least 1")
        args += ["--limit", str(body.limit)]

    return _start_steam_job("download", args)

@router.get("/api/steam/job")
def steam_job():
    """Poll the running (or last) fetcher job."""
    return _job_snapshot()

@router.post("/api/steam/job/cancel")
def steam_job_cancel():
    """Stop the running job.

    Mainly for QR sign-in: dismissing the code should end the attempt straight
    away rather than leaving it pending until Steam's own timeout.
    """
    outcome = _steam_jobs.cancel()
    if outcome["cancelled"] is False and "Could not stop it" in outcome.get("detail", ""):
        raise HTTPException(status_code=500, detail=outcome["detail"])
    return outcome

@router.get("/api/steam/auto-sync")
def get_auto_sync():
    return _auto_sync_snapshot()

@router.put("/api/steam/auto-sync")
def set_auto_sync(body: AutoSyncIn):
    """Change auto-sync settings, and start or stop the loop to match.

    Settings can be changed while it runs; the loop re-reads them every cycle,
    so a new interval takes effect from the next match rather than needing a
    restart.
    """
    cfg = _load_auto_sync_config()

    if body.interval_minutes is not None:
        if not 0 <= body.interval_minutes <= 1440:
            raise HTTPException(
                status_code=400, detail="Interval must be between 0 and 1440 minutes."
            )
        cfg["interval_minutes"] = body.interval_minutes
    if body.idle_check_minutes is not None:
        if not 1 <= body.idle_check_minutes <= 1440:
            raise HTTPException(
                status_code=400, detail="Check interval must be between 1 and 1440 minutes."
            )
        cfg["idle_check_minutes"] = body.idle_check_minutes
    if body.pause_when_playing is not None:
        cfg["pause_when_playing"] = bool(body.pause_when_playing)

    if body.enabled is not None:
        if body.enabled and not cfg["enabled"]:
            _require_fetcher()  # fail loudly now rather than in a background thread
        cfg["enabled"] = bool(body.enabled)

    _save_auto_sync_config(cfg)

    if cfg["enabled"]:
        _start_auto_sync()
    else:
        _stop_auto_sync()

    return _auto_sync_snapshot()

@router.get("/api/steam/presence")
def steam_presence(force: bool = False):
    """Whether a tracked account is in CS2, as far as Steam will say."""
    return _check_cs2_presence(force=force)
