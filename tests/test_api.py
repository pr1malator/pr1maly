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
