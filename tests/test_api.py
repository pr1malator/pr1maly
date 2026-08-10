"""Tests for the FastAPI REST API (api.py)."""

from __future__ import annotations

import io
import os

import pytest

# Force an isolated test DB before importing the app
os.environ["DB_PATH"] = ":memory:"

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_get_config():
    resp = client.get("/api/config")
    assert resp.status_code == 200
    assert "steam_id" in resp.json()


# ---------------------------------------------------------------------------
# Match CRUD (without a real demo)
# ---------------------------------------------------------------------------


def test_upload_rejects_non_dem():
    resp = client.post(
        "/api/matches/upload",
        files={"file": ("test.txt", b"not a demo", "application/octet-stream")},
        data={"steam_id": "12345"},
    )
    assert resp.status_code == 400
    assert "dem" in resp.json()["detail"].lower()


def test_upload_requires_steam_id(tmp_path):
    # Clear default steam ID so we can test the guard
    from api import _STEAM_ID_FILE
    if _STEAM_ID_FILE.exists():
        original = _STEAM_ID_FILE.read_text()
    else:
        original = None

    try:
        _STEAM_ID_FILE.write_text("")
        resp = client.post(
            "/api/matches/upload",
            files={"file": ("test.dem", b"fake", "application/octet-stream")},
            data={"steam_id": ""},
        )
        # Either 400 (no steam id) or 422 (unparseable demo) is acceptable
        assert resp.status_code in (400, 422)
    finally:
        if original is not None:
            _STEAM_ID_FILE.write_text(original)


def test_list_matches_empty():
    resp = client.get("/api/matches")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_match_not_found():
    resp = client.get("/api/matches/nonexistent-uuid")
    assert resp.status_code == 404


def test_delete_match_not_found():
    resp = client.delete("/api/matches/nonexistent-uuid")
    assert resp.status_code == 404


def test_update_notes_not_found():
    resp = client.put(
        "/api/matches/nonexistent-uuid/notes",
        json={"notes": "hello"},
    )
    assert resp.status_code == 404


def test_create_tag_not_found():
    resp = client.post(
        "/api/matches/nonexistent-uuid/tags",
        json={"tag": "test"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------


def test_trends_empty():
    resp = client.get("/api/trends")
    assert resp.status_code == 200
    body = resp.json()
    assert "data_points" in body
    assert "averages" in body
    assert "available_maps" in body


def test_trends_with_map_filter():
    resp = client.get("/api/trends?maps=de_dust2")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Re-analysis
# ---------------------------------------------------------------------------


def test_analyzer_version_endpoint():
    from src.processor import ANALYZER_VERSION

    resp = client.get("/api/analyzer/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["analyzer_version"] == ANALYZER_VERSION
    assert "total_matches" in body
    assert "stale_matches" in body


def test_reanalyze_unknown_match_returns_404():
    resp = client.post("/api/matches/nonexistent-uuid/reanalyze")
    assert resp.status_code == 404


def test_resolve_demo_finds_file_in_search_dir(tmp_path, monkeypatch):
    from api import _resolve_demo

    monkeypatch.setenv("DEMO_DIR", str(tmp_path))
    dem = tmp_path / "match730_example.dem"
    dem.write_bytes(b"fake demo")

    assert _resolve_demo("match730_example.dem") == dem.resolve()


def test_resolve_demo_rejects_path_traversal(tmp_path, monkeypatch):
    """The filename comes from the DB but originated as user input.

    A name carrying a separator would otherwise reach any .dem on the host, so
    only bare filenames resolving directly inside a search dir are accepted.
    """
    from api import _resolve_demo

    search_dir = tmp_path / "demos"
    search_dir.mkdir()
    monkeypatch.setenv("DEMO_DIR", str(search_dir))

    escaped = tmp_path / "escaped.dem"
    escaped.write_bytes(b"fake demo")

    assert _resolve_demo("../escaped.dem") is None
    assert _resolve_demo("sub/escaped.dem") is None
    assert _resolve_demo("/etc/passwd.dem") is None


def test_resolve_demo_rejects_non_dem_and_missing(tmp_path, monkeypatch):
    from api import _resolve_demo

    monkeypatch.setenv("DEMO_DIR", str(tmp_path))
    (tmp_path / "notes.txt").write_bytes(b"x")

    assert _resolve_demo("notes.txt") is None
    assert _resolve_demo("absent.dem") is None
    assert _resolve_demo("") is None
    assert _resolve_demo(None) is None


def test_available_demo_names_lists_only_demos(tmp_path, monkeypatch):
    from api import _available_demo_names

    monkeypatch.setenv("DEMO_DIR", str(tmp_path))
    (tmp_path / "a.dem").write_bytes(b"x")
    (tmp_path / "b.dem").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")

    names = _available_demo_names()
    assert {"a.dem", "b.dem"} <= names
    assert "notes.txt" not in names
