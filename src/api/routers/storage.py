"""Demo retention: what is on disk and what is safe to delete."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.deps import (
    _analyse_demo_folder,
    _load_storage_config,
    _run_demo_cleanup,
    _save_storage_config,
)

router = APIRouter()


@router.get("/api/storage/status")
def storage_status(
    keep_recent: int | None = None,
    per_account: bool | None = None,
    fetched_only: bool | None = None,
):
    """What demos cost on disk, and which are safe to remove.

    Passing any setting previews it without saving, so the retention number can
    be tuned against real figures before committing.
    """
    if keep_recent is not None and keep_recent < 0:
        raise HTTPException(status_code=400, detail="keep_recent cannot be negative")

    return _analyse_demo_folder(
        {
            "keep_recent": keep_recent,
            "per_account": per_account,
            "fetched_only": fetched_only,
        }
    )

@router.get("/api/storage/config")
def get_storage_config():
    return _load_storage_config()

@router.put("/api/storage/config")
def set_storage_config(body: dict):
    cfg = _load_storage_config()

    if "keep_recent" in body:
        try:
            keep = int(body["keep_recent"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="keep_recent must be a whole number")
        if keep < 0:
            raise HTTPException(status_code=400, detail="keep_recent cannot be negative")
        cfg["keep_recent"] = keep

    if "per_account" in body:
        cfg["per_account"] = bool(body["per_account"])
    if "auto_cleanup" in body:
        cfg["auto_cleanup"] = bool(body["auto_cleanup"])
    if "fetched_only" in body:
        cfg["fetched_only"] = bool(body["fetched_only"])

    _save_storage_config(cfg)
    return cfg

@router.post("/api/storage/cleanup")
def storage_cleanup(body: dict | None = None):
    """Delete imported demos outside the retention window.

    Pass ``{"dry_run": true}`` to see what would go without touching anything.
    """
    return _run_demo_cleanup(dry_run=bool((body or {}).get("dry_run")))
