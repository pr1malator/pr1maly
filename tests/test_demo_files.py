"""Tests for demo retention.

These rules decide which of the user's files get deleted, and a demo is roughly
280 MB that Valve will not hand back. The classification was previously buried
in api.py behind module globals and a raw SQL join, so none of it was tested;
it is a pure function now.

The rule that matters most: a demo whose replay frames never reached the
database must never be deleted, because the 2D viewer could never be populated
for that match afterwards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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

BASE = Path("/demos")


def _found(*names, size=280_000_000):
    return [
        {"filename": n, "bytes": size, "mtime": 1000 + i}
        for i, n in enumerate(names)
    ]


def _imported(name, *, steam_id="7656119", date="2026-08-01", replay_rounds=24):
    return {name: {"filename": name, "date": date,
                   "player_steam_id": steam_id, "replay_rounds": replay_rounds}}


def _classify(found, imported=None, **config):
    return classify(
        base=BASE,
        found=found,
        config={**STORAGE_DEFAULTS, **config},
        imported=imported or {},
        account_names={"7656119": "Main"},
    )


# ---------------------------------------------------------------------------
# What must never be deleted
# ---------------------------------------------------------------------------


def test_a_demo_without_stored_replay_frames_is_never_deletable():
    """Deleting it would make the 2D viewer permanently empty for that match."""
    name = f"{FETCHED_PREFIX}a.dem"
    result = _classify(_found(name), _imported(name, replay_rounds=0), keep_recent=0)
    entry = result["files"][0]
    assert entry["deletable"] is False
    assert entry["reason"] == "no replay data stored"


def test_an_unimported_demo_is_never_deletable():
    name = f"{FETCHED_PREFIX}b.dem"
    result = _classify(_found(name), keep_recent=0)
    assert result["files"][0]["deletable"] is False
    assert result["files"][0]["reason"] == "not imported yet"


def test_the_newest_are_protected_by_recency():
    names = [f"{FETCHED_PREFIX}{i}.dem" for i in range(5)]
    imported = {}
    for n in names:
        imported.update(_imported(n))
    result = _classify(_found(*names), imported, keep_recent=2)

    protected = [e for e in result["files"] if e["protected"]]
    assert len(protected) == 2
    assert result["protected_files"] == 2
    assert all("within the newest 2" in e["reason"] for e in protected)


def test_demos_this_app_did_not_download_are_left_alone_by_default():
    """fetched_only is on by default: the user's own recordings are not ours."""
    name = "my_own_recording.dem"
    result = _classify(_found(name), _imported(name), keep_recent=0)
    assert result["files"][0]["deletable"] is False
    assert result["files"][0]["reason"] == "not downloaded by this app"


def test_turning_off_fetched_only_makes_them_deletable():
    name = "my_own_recording.dem"
    result = _classify(_found(name), _imported(name), keep_recent=0, fetched_only=False)
    assert result["files"][0]["deletable"] is True


def test_an_imported_fetched_demo_with_replays_is_deletable():
    name = f"{FETCHED_PREFIX}old.dem"
    result = _classify(_found(name), _imported(name), keep_recent=0)
    assert result["files"][0]["deletable"] is True
    assert result["files"][0]["reason"] == "safe to delete"
    assert result["deletable_files"] == 1


# ---------------------------------------------------------------------------
# Per-account retention
# ---------------------------------------------------------------------------


def test_per_account_retention_protects_each_account_separately():
    """Otherwise a heavily-played account pushes every other account's demos
    out of the window."""
    found = _found(*[f"{FETCHED_PREFIX}{i}.dem" for i in range(4)])
    imported = {}
    for i, entry in enumerate(found):
        imported.update(_imported(entry["filename"],
                                  steam_id="A" if i < 2 else "B",
                                  date=f"2026-08-0{4 - i}"))
    result = classify(
        base=BASE, found=found,
        config={**STORAGE_DEFAULTS, "keep_recent": 1, "per_account": True},
        imported=imported, account_names={"A": "Main", "B": "Smurf"},
    )

    protected = {e["owner_steam_id"] for e in result["files"] if e["protected"]}
    assert protected == {"A", "B"}  # one each


def test_overall_retention_counts_across_all_accounts():
    found = _found(*[f"{FETCHED_PREFIX}{i}.dem" for i in range(4)])
    imported = {}
    for i, entry in enumerate(found):
        imported.update(_imported(entry["filename"],
                                  steam_id="A" if i < 2 else "B",
                                  date=f"2026-08-0{4 - i}"))
    result = classify(
        base=BASE, found=found,
        config={**STORAGE_DEFAULTS, "keep_recent": 1, "per_account": False},
        imported=imported, account_names={},
    )
    assert result["protected_files"] == 1


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------


def test_totals_add_up():
    names = [f"{FETCHED_PREFIX}{i}.dem" for i in range(3)]
    imported = {}
    for n in names:
        imported.update(_imported(n))
    result = _classify(_found(*names, size=100), imported, keep_recent=1)

    assert result["total_files"] == 3
    assert result["total_bytes"] == 300
    assert result["protected_bytes"] == 100
    assert result["deletable_bytes"] == result["deletable_files"] * 100


def test_preview_flag_is_reported():
    assert _classify(_found())["preview"] is False
    result = classify(base=BASE, found=[], config=STORAGE_DEFAULTS,
                      imported={}, account_names={}, preview=True)
    assert result["preview"] is True


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def test_an_unset_folder_is_reported_not_guessed():
    with pytest.raises(DemoFolderUnusable, match="No sync folder"):
        scan_folder(None)


def test_a_missing_folder_is_reported():
    with pytest.raises(DemoFolderUnusable, match="does not exist"):
        scan_folder("/definitely/not/here")


def test_scan_includes_the_sidecar_in_the_size(tmp_path):
    (tmp_path / "a.dem").write_bytes(b"x" * 100)
    (tmp_path / "a.dem.info").write_bytes(b"y" * 20)
    _base, found = scan_folder(str(tmp_path))
    assert found[0]["bytes"] == 120


def test_scan_ignores_non_demos(tmp_path):
    (tmp_path / "a.dem").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    _base, found = scan_folder(str(tmp_path))
    assert [f["filename"] for f in found] == ["a.dem"]


# ---------------------------------------------------------------------------
# Locating a demo — this one is a path-traversal guard
# ---------------------------------------------------------------------------


def test_resolve_accepts_a_bare_filename(tmp_path):
    (tmp_path / "match.dem").write_bytes(b"x")
    assert resolve_demo("match.dem", [tmp_path]) == (tmp_path / "match.dem").resolve()


@pytest.mark.parametrize("hostile", [
    "../../../../etc/passwd.dem",
    "sub/dir/match.dem",
    "..\\..\\windows\\system32\\config.dem",
])
def test_resolve_refuses_anything_with_a_path_in_it(hostile, tmp_path):
    """The name comes from the database but originated as user input."""
    assert resolve_demo(hostile, [tmp_path]) is None


def test_resolve_refuses_non_demo_extensions(tmp_path):
    (tmp_path / "secrets.json").write_bytes(b"x")
    assert resolve_demo("secrets.json", [tmp_path]) is None


def test_resolve_returns_none_when_absent(tmp_path):
    assert resolve_demo("nope.dem", [tmp_path]) is None
    assert resolve_demo(None, [tmp_path]) is None


def test_available_names_globs_every_directory(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir(), two.mkdir()
    (one / "a.dem").write_bytes(b"x")
    (two / "b.dem").write_bytes(b"x")
    (two / "c.txt").write_bytes(b"x")
    assert available_demo_names([one, two]) == {"a.dem", "b.dem"}


def test_search_dirs_skips_paths_that_do_not_exist(tmp_path, monkeypatch):
    """sync_config.json holds the container path under Docker."""
    monkeypatch.delenv("DEMO_DIR", raising=False)
    dirs = search_dirs("/demos-inside-a-container", tmp_path)
    assert Path("/demos-inside-a-container") not in dirs
    assert tmp_path in dirs


def test_search_dirs_does_not_repeat_a_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_DIR", str(tmp_path))
    dirs = search_dirs(str(tmp_path), tmp_path)
    assert len(dirs) == len({d.resolve() for d in dirs})


# ---------------------------------------------------------------------------
# Deleting
# ---------------------------------------------------------------------------


def test_delete_removes_the_demo_and_its_sidecar(tmp_path):
    (tmp_path / "a.dem").write_bytes(b"x" * 10)
    (tmp_path / "a.dem.info").write_bytes(b"y")
    analysis = {"folder": str(tmp_path), "files": [
        {"filename": "a.dem", "bytes": 11, "deletable": True},
    ]}

    result = delete_deletable(analysis)

    assert result["deleted"] == ["a.dem"]
    assert result["freed_bytes"] == 11
    assert not (tmp_path / "a.dem").exists()
    assert not (tmp_path / "a.dem.info").exists()


def test_dry_run_deletes_nothing_but_reports_what_it_would(tmp_path):
    (tmp_path / "a.dem").write_bytes(b"x")
    analysis = {"folder": str(tmp_path), "files": [
        {"filename": "a.dem", "bytes": 1, "deletable": True},
    ]}

    result = delete_deletable(analysis, dry_run=True)

    assert result["dry_run"] is True
    assert result["deleted"] == ["a.dem"]
    assert (tmp_path / "a.dem").exists()


def test_delete_refuses_to_step_outside_the_folder(tmp_path):
    outside = tmp_path.parent / "outside.dem"
    outside.write_bytes(b"x")
    analysis = {"folder": str(tmp_path), "files": [
        {"filename": "../outside.dem", "bytes": 1, "deletable": True},
    ]}

    result = delete_deletable(analysis)

    assert result["deleted"] == []
    assert "refused" in result["errors"][0]
    assert outside.exists()


def test_delete_skips_entries_not_marked_deletable(tmp_path):
    (tmp_path / "keep.dem").write_bytes(b"x")
    analysis = {"folder": str(tmp_path), "files": [
        {"filename": "keep.dem", "bytes": 1, "deletable": False},
    ]}
    assert delete_deletable(analysis)["deleted"] == []
    assert (tmp_path / "keep.dem").exists()


# ---------------------------------------------------------------------------
# Picking the next import
# ---------------------------------------------------------------------------


def test_next_unimported_takes_the_newest_fetched_one():
    """Newest first, so a long backlog produces useful matches straight away."""
    analysis = {"files": [
        {"filename": "old.dem", "fetched": True, "imported": False, "mtime": 1},
        {"filename": "new.dem", "fetched": True, "imported": False, "mtime": 9},
    ]}
    assert next_unimported(analysis, set()) == "new.dem"


def test_next_unimported_ignores_demos_the_app_did_not_download():
    analysis = {"files": [
        {"filename": "theirs.dem", "fetched": False, "imported": False, "mtime": 9},
    ]}
    assert next_unimported(analysis, set()) is None


def test_next_unimported_skips_the_set_aside():
    analysis = {"files": [
        {"filename": "corrupt.dem", "fetched": True, "imported": False, "mtime": 9},
        {"filename": "fine.dem", "fetched": True, "imported": False, "mtime": 1},
    ]}
    assert next_unimported(analysis, {"corrupt.dem"}) == "fine.dem"


def test_next_unimported_is_none_when_everything_is_in():
    analysis = {"files": [
        {"filename": "a.dem", "fetched": True, "imported": True, "mtime": 9},
    ]}
    assert next_unimported(analysis, set()) is None
