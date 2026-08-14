"""Finding demos on disk, and deciding which are safe to delete.

A demo costs roughly 280 MB and about 1.3 MB once analysed, so a library gets
pruned. Deciding what to prune is the delicate part: deleting a demo whose
replay frames never made it into the database means the 2D viewer can never be
populated for that match, and the file is not recoverable.

This was 254 lines inside api.py, reading module globals and a raw SQL join, and
consequently untestable. The classification is now a pure function — hand it the
folder, the settings, the imported rows and the account names, and it returns the
verdict — so the retention rules can be exercised without a disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Demos this app downloaded are prefixed, so retention can be limited to them.
FETCHED_PREFIX = "pr1maly_"

# Where CS2 puts demos you download in-game. Mirrors DEFAULT_REPLAY_DIRS in
# fetcher/lib/paths.js — the fetcher falls back to these when nothing is
# configured, so anything it downloaded on a default Steam install lands here
# rather than in data/.
DEFAULT_REPLAY_DIRS = (
    "C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays",
    str(Path.home() / ".steam/steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/replays"),
)

STORAGE_DEFAULTS: dict[str, Any] = {
    "keep_recent": 30,      # newest N demos are never deleted
    "per_account": True,    # count that N per account rather than overall
    "auto_cleanup": False,  # run cleanup automatically after a successful sync
    "fetched_only": True,   # only touch demos this app downloaded
}


class DemoFolderUnusable(Exception):
    """The sync folder is not configured, or is not a directory.

    Worth stopping for rather than swallowing: Auto-Sync would otherwise
    download match after match it can never import.
    """


def search_dirs(sync_folder: str | None, data_dir: Path) -> list[Path]:
    """Directories a stored demo may live in, most specific first.

    Deliberately the same resolution order the fetcher uses to *write* demos
    (DEMO_DIR, then sync_config.json, then the default CS2 replays folder), plus
    data/ for demos the app downloaded itself. Unlike the fetcher this keeps
    every directory that exists rather than stopping at the first — a library
    built up over time can be spread across more than one of them.

    sync_config.json holds the container path (/demos) under Docker, which is
    why entries that do not resolve on this machine are skipped rather than
    treated as an error.
    """
    candidates: list[Path] = []
    env_dir = os.environ.get("DEMO_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    if sync_folder:
        candidates.append(Path(sync_folder))
    candidates.extend(Path(d) for d in DEFAULT_REPLAY_DIRS if d)
    candidates.append(data_dir)

    seen: set[Path] = set()
    dirs: list[Path] = []
    for directory in candidates:
        try:
            if not directory.is_dir():
                continue
            key = directory.resolve()
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        dirs.append(directory)
    return dirs


def resolve_demo(filename: str | None, dirs: list[Path]) -> Path | None:
    """Locate a stored .dem by name, or None when it is not on disk.

    *filename* comes from the database but originated as user input, so it is
    treated as untrusted: only a bare filename is accepted, and the resolved
    path has to sit directly inside one of the search directories. This is the
    same rule /api/sync/process applies to the names it is handed.
    """
    if not filename or not filename.endswith(".dem"):
        return None
    if Path(filename).name != filename:
        return None  # rejects separators and traversal segments

    for directory in dirs:
        try:
            base = directory.resolve()
            resolved = (directory / filename).resolve()
        except OSError:
            continue
        if resolved.parent != base:
            continue
        if resolved.is_file():
            return resolved
    return None


def available_demo_names(dirs: list[Path]) -> set[str]:
    """Every .dem filename currently on disk, for bulk availability checks.

    One glob per search directory beats one stat per match row.
    """
    names: set[str] = set()
    for directory in dirs:
        try:
            names.update(p.name for p in directory.glob("*.dem") if p.is_file())
        except OSError:
            continue
    return names


def scan_folder(folder: str | None) -> tuple[Path, list[dict[str, Any]]]:
    """Every .dem in *folder* with its size (including sidecar) and mtime.

    Raises :class:`DemoFolderUnusable` when the folder is unset or missing.
    """
    if not folder:
        raise DemoFolderUnusable("No sync folder configured")

    base = Path(folder)
    if not base.is_dir():
        raise DemoFolderUnusable(f"Folder does not exist: {folder}")

    found: list[dict[str, Any]] = []
    for path in base.glob("*.dem"):
        try:
            stat = path.stat()
            size, mtime = stat.st_size, stat.st_mtime
        except OSError:
            continue

        sidecar = Path(f"{path}.info")
        if sidecar.exists():
            size += sidecar.stat().st_size

        found.append({"filename": path.name, "bytes": size, "mtime": mtime})
    return base, found


def classify(
    *,
    base: Path,
    found: list[dict[str, Any]],
    config: dict[str, Any],
    imported: dict[str, Any],
    account_names: dict[str | None, str | None],
    preview: bool = False,
) -> dict[str, Any]:
    """Decide which demos are protected, deletable, or neither.

    Pure: everything it needs is an argument. *imported* maps filename to a row
    carrying ``date``, ``player_steam_id`` and ``replay_rounds``.
    """
    entries: list[dict[str, Any]] = []
    for item in found:
        record = imported.get(item["filename"])
        owner = record["player_steam_id"] if record else None
        entries.append({
            **item,
            "match_date": record["date"] if record else None,
            "imported": record is not None,
            "has_replay": bool(record and record["replay_rounds"]),
            "fetched": item["filename"].startswith(FETCHED_PREFIX),
            "owner_steam_id": owner,
            "owner": account_names.get(owner) or owner,
        })

    # Newest first, then protect by recency.
    entries.sort(key=lambda e: (e["match_date"] or "", e["mtime"]), reverse=True)
    keep_recent = max(0, int(config.get("keep_recent", 30)))
    per_account = bool(config.get("per_account", True))

    # Counting per account stops a heavily-played account from pushing every
    # other account's demos out of the window.
    seen: dict[Any, int] = {}
    for entry in entries:
        group = entry["owner_steam_id"] if per_account else "__all__"
        rank = seen.get(group, 0)
        seen[group] = rank + 1
        entry["protected"] = rank < keep_recent

        if entry["protected"]:
            entry["reason"] = (
                f"within the newest {keep_recent} for {entry['owner'] or 'this account'}"
                if per_account
                else f"within the newest {keep_recent}"
            )
        elif not entry["imported"]:
            entry["reason"] = "not imported yet"
        elif not entry["has_replay"]:
            # The 2D viewer could never be populated for this match afterwards.
            entry["reason"] = "no replay data stored"
        elif config.get("fetched_only") and not entry["fetched"]:
            entry["reason"] = "not downloaded by this app"
        else:
            entry["reason"] = "safe to delete"

        entry["deletable"] = entry["reason"] == "safe to delete"

    deletable = [e for e in entries if e["deletable"]]
    return {
        "folder": str(base),
        "config": config,
        "preview": preview,
        "protected_bytes": sum(e["bytes"] for e in entries if e["protected"]),
        "files": entries,
        "total_files": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "imported_files": sum(1 for e in entries if e["imported"]),
        "protected_files": sum(1 for e in entries if e["protected"]),
        "deletable_files": len(deletable),
        "deletable_bytes": sum(e["bytes"] for e in deletable),
    }


def next_unimported(analysis: dict[str, Any], set_aside: set[str]) -> str | None:
    """The newest demo this app downloaded that is not yet in the database.

    Restricted to fetched demos: Auto-Sync imports what it downloaded, and
    silently importing whatever else the user happens to have dropped in the
    folder would be a surprise. Sync Folder still covers those.
    """
    candidates = [
        e for e in analysis["files"]
        if e["fetched"] and not e["imported"] and e["filename"] not in set_aside
    ]
    if not candidates:
        return None
    # Newest first, so a long backlog produces useful matches straight away.
    candidates.sort(key=lambda e: e["mtime"], reverse=True)
    return candidates[0]["filename"]


def delete_deletable(analysis: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Delete the demos the analysis marked deletable, plus their sidecars."""
    base = Path(analysis["folder"]).resolve()
    deleted: list[str] = []
    errors: list[str] = []
    freed = 0

    for entry in analysis["files"]:
        if not entry["deletable"]:
            continue

        target = (base / entry["filename"]).resolve()
        # Never step outside the configured folder, and never touch non-demos.
        if target.parent != base or target.suffix != ".dem":
            errors.append(f"{entry['filename']}: refused, outside the demo folder")
            continue

        if dry_run:
            deleted.append(entry["filename"])
            freed += entry["bytes"]
            continue

        try:
            sidecar = Path(f"{target}.info")
            target.unlink()
            if sidecar.exists():
                sidecar.unlink()
            deleted.append(entry["filename"])
            freed += entry["bytes"]
        except OSError as exc:
            errors.append(f"{entry['filename']}: {exc}")

    return {
        "dry_run": dry_run,
        "deleted": deleted,
        "deleted_count": len(deleted),
        "freed_bytes": freed,
        "errors": errors,
    }
