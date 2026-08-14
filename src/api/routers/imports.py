"""Getting demos in: drag-and-drop, bulk upload, and the sync folder."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.deps import (
    _db,
    _get_active_account,
    _load_accounts,
    _load_storage_config,
    _load_sync_config,
    _match_summary,
    _resolve_request_steam_id,
    _run_demo_cleanup,
    _save_sync_config,
    steam_id_path,
)
from src.database import (
    add_tag,
    get_imported_filenames,
    save_match,
)
from src.parser import extract_player_names, parse_info_file, read_demo_map
from src.services.import_service import (
    SidecarInfo,
    account_name_for,
    analyse_demo,
    read_sidecar,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Detect player from .info file
# ---------------------------------------------------------------------------
@router.post("/api/matches/detect-player")
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
@router.post("/api/matches/upload")
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

    sid = _resolve_request_steam_id(steam_id)
    if not sid:
        raise HTTPException(status_code=400, detail="Steam ID is required")

    info_date: str | None = None
    if info_file and info_file.filename:
        info_date = read_sidecar(info_file.file.read()).match_date

    # Resolve match date: form field > .info file
    resolved_date = match_date.strip() or info_date or None

    # Write to temp file and parse
    with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        stats = analyse_demo(tmp_path, sid)
    except Exception as exc:
        # Log the full traceback so the failing line is visible in server logs
        # (the 422 body only carries the message, not the stack).
        logging.getLogger("uvicorn.error").exception("Demo parse failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse demo: {type(exc).__name__}: {exc}",
        )
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

# ---------------------------------------------------------------------------
# Bulk upload
# ---------------------------------------------------------------------------
@router.post("/api/matches/upload-bulk")
def upload_demos_bulk(
    files: list[UploadFile] = File(...),
    info_files: list[UploadFile] = File(default=[]),
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

    fallback_sid = _resolve_request_steam_id(steam_id)

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
        sidecar = SidecarInfo()
        if info_uf:
            info_uf.file.seek(0)
            sidecar = read_sidecar(info_uf.file.read())
        info_date = sidecar.match_date

        sid = sidecar.first_known(account_ids) or fallback_sid
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
            stats = analyse_demo(tmp_path, sid)
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

        acct_name = account_name_for(accounts, sid)

        entry["status"] = "ok"
        entry["match_id"] = match_id
        entry["player_name"] = acct_name or stats.get("player_name", "?")
        entry["player_steam_id"] = sid
        entry["map_name"] = stats.get("map_name")
        entry["stats"] = _match_summary(stats)
        if stats.get("partial_import"):
            entry["partial_import"] = True
        if stats.get("parse_warning"):
            entry["parse_warning"] = stats["parse_warning"]
        results.append(entry)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    return {"processed": ok_count, "total": len(results), "results": results}

@router.get("/api/sync/config")
def get_sync_config():
    return _load_sync_config()

@router.put("/api/sync/config")
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

@router.get("/api/sync/scan")
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
    try:
        existing = get_imported_filenames(conn, steam_id)
    finally:
        conn.close()

    filter_sid = steam_id.strip()

    new_demos = []
    for df in dem_files:
        fname = df.name
        if fname in existing:
            continue
        info_path = df.with_suffix(".dem.info")
        has_info = info_path.exists()

        # Parse info file once — used for both filtering and metadata
        info_data: dict[str, Any] = {}
        if has_info:
            try:
                info_data = parse_info_file(info_path.read_bytes())
            except Exception:
                pass

        # Filter by player if requested
        if filter_sid and has_info and info_data:
            if filter_sid not in info_data.get("account_ids", []):
                continue

        # Prefer map name from .dem.info; fall back to filename pattern, then demo header
        map_name = info_data.get("map_name") or None
        if not map_name:
            map_match = re.search(r'\b(de|cs|ar|gg|dm)_\w+', fname)
            map_name = map_match.group(0) if map_match else None
        if not map_name:
            map_name = read_demo_map(df)

        # Date: prefer .dem.info, fall back to timestamp embedded in filename
        match_date = info_data.get("match_date")
        if not match_date:
            import datetime
            ts_match = re.search(r'_(\d{10})_', fname)
            if ts_match:
                try:
                    ts = int(ts_match.group(1))
                    if 1_000_000_000 < ts < 2_000_000_000:
                        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC)
                        match_date = dt.date().isoformat()
                except Exception:
                    pass

        new_demos.append({
            "filename": fname,
            "size_mb": round(df.stat().st_size / 1024 / 1024, 1),
            "has_info": has_info,
            "map_name": map_name,
            "match_date": match_date,
        })

    return {"folder": folder, "total_found": len(dem_files), "new": new_demos}

@router.post("/api/sync/process")
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
    if not steam_id and steam_id_path().exists():
        steam_id = steam_id_path().read_text().strip()
    if not steam_id:
        raise HTTPException(status_code=400, detail="No Steam ID available")

    accounts = _load_accounts()
    account_ids = {a["steam_id"] for a in accounts}

    base = p.resolve()
    results: list[dict[str, Any]] = []
    for fname in filenames:
        entry: dict[str, Any] = {"filename": fname}
        dem_path = p / fname
        # The name comes from the request, so "../.." would otherwise reach any
        # .dem on the host. Same rule the cleanup pass uses: the file has to sit
        # directly in the configured folder.
        try:
            resolved = dem_path.resolve()
        except OSError:
            resolved = None
        if resolved is None or resolved.parent != base:
            entry["status"] = "error"
            entry["detail"] = "Refused: outside the sync folder"
            results.append(entry)
            continue
        if not dem_path.exists() or not fname.endswith(".dem"):
            entry["status"] = "error"
            entry["detail"] = "File not found or not a .dem"
            results.append(entry)
            continue

        # Try .info sidecar
        sidecar = read_sidecar(dem_path.with_suffix(".dem.info"))
        info_date = sidecar.match_date
        sid = sidecar.first_known(account_ids) or steam_id

        try:
            stats = analyse_demo(dem_path, sid)
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

        acct_name = account_name_for(accounts, sid)

        entry["status"] = "ok"
        entry["match_id"] = match_id
        entry["player_name"] = acct_name or stats.get("player_name", "?")
        entry["map_name"] = stats.get("map_name")
        entry["stats"] = _match_summary(stats)
        if stats.get("partial_import"):
            entry["partial_import"] = True
        if stats.get("parse_warning"):
            entry["parse_warning"] = stats["parse_warning"]
        results.append(entry)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    response: dict[str, Any] = {
        "processed": ok_count,
        "total": len(results),
        "results": results,
    }

    # Opt-in: reclaim disk space now that these demos are safely in the database.
    if ok_count and _load_storage_config().get("auto_cleanup"):
        try:
            response["cleanup"] = _run_demo_cleanup(dry_run=False)
        except Exception as exc:  # cleanup must never fail an otherwise good import
            response["cleanup"] = {"error": str(exc)}

    return response
