"""Tests for the FastAPI REST API (api.py)."""

from __future__ import annotations

import os

import pytest

# Force an isolated test DB before importing the app
os.environ["DB_PATH"] = ":memory:"

from fastapi.testclient import TestClient

from api import app
from src.api import deps
from src.config import store as store_module
from src.services.ai_context import OVERALL_KEY

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
    from src.api.deps import steam_id_path
    if steam_id_path().exists():
        original = steam_id_path().read_text()
    else:
        original = None

    try:
        steam_id_path().write_text("")
        resp = client.post(
            "/api/matches/upload",
            files={"file": ("test.dem", b"fake", "application/octet-stream")},
            data={"steam_id": ""},
        )
        # Either 400 (no steam id) or 422 (unparseable demo) is acceptable
        assert resp.status_code in (400, 422)
    finally:
        if original is not None:
            steam_id_path().write_text(original)


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
    from src.api.deps import _resolve_demo

    monkeypatch.setenv("DEMO_DIR", str(tmp_path))
    dem = tmp_path / "match730_example.dem"
    dem.write_bytes(b"fake demo")

    assert _resolve_demo("match730_example.dem") == dem.resolve()


def test_resolve_demo_rejects_path_traversal(tmp_path, monkeypatch):
    """The filename comes from the DB but originated as user input.

    A name carrying a separator would otherwise reach any .dem on the host, so
    only bare filenames resolving directly inside a search dir are accepted.
    """
    from src.api.deps import _resolve_demo

    search_dir = tmp_path / "demos"
    search_dir.mkdir()
    monkeypatch.setenv("DEMO_DIR", str(search_dir))

    escaped = tmp_path / "escaped.dem"
    escaped.write_bytes(b"fake demo")

    assert _resolve_demo("../escaped.dem") is None
    assert _resolve_demo("sub/escaped.dem") is None
    assert _resolve_demo("/etc/passwd.dem") is None


def test_resolve_demo_rejects_non_dem_and_missing(tmp_path, monkeypatch):
    from src.api.deps import _resolve_demo

    monkeypatch.setenv("DEMO_DIR", str(tmp_path))
    (tmp_path / "notes.txt").write_bytes(b"x")

    assert _resolve_demo("notes.txt") is None
    assert _resolve_demo("absent.dem") is None
    assert _resolve_demo("") is None
    assert _resolve_demo(None) is None


# ---------------------------------------------------------------------------
# AI assessment (roles + patterns, one call)
# ---------------------------------------------------------------------------


def test_ai_assessment_without_a_map_means_the_whole_career(monkeypatch):
    """No map filter is not an error — it is the career assessment.

    It still needs matches to assess, so on an empty database it fails on the
    data rather than on the missing filter.
    """

    monkeypatch.setattr(deps, "load_ai_config", lambda: {
        "active_provider": "openai", "active_model": "gpt-4.1-mini",
        "providers": {"openai": {"api_key": "k"}},
    })
    resp = client.post("/api/performance/ai-assessment")
    assert resp.status_code == 404
    assert "match" in resp.json()["detail"].lower()


def test_ai_assessment_persisted_lookup_is_empty_without_one(monkeypatch, tmp_path):
    """Both lookups answer empty when nothing has been assessed.

    The stores are redirected at a temp directory: reading the real data/
    folder made this pass or fail depending on whether the developer had run an
    assessment on that map, which it did the first time someone used the
    feature for real.
    """
    monkeypatch.setattr(store_module, "DATA_DIR", tmp_path)

    for query in ("?maps=de_nuke", ""):
        resp = client.get("/api/performance/ai-assessment" + query)
        assert resp.status_code == 200
        assert resp.json() == {}




def test_career_assessment_is_stored_apart_from_the_maps(monkeypatch, tmp_path):
    """The career entry shares the file but never collides with a map.

    No map is called "__overall__", so one store holds both without the career
    assessment overwriting a map's or being returned in its place.
    """
    import json

    (tmp_path / "ai_assessment.json").write_text(json.dumps({
        OVERALL_KEY: {"aim": {"name": "Career Aim"}, "scope": "overall"},
        "de_mirage": {"aim": {"name": "Mirage Aim"}, "scope": "map"},
    }))
    monkeypatch.setattr(store_module, "DATA_DIR", tmp_path)

    assert client.get("/api/performance/ai-assessment").json()["aim"]["name"] == "Career Aim"
    assert client.get(
        "/api/performance/ai-assessment?maps=de_mirage"
    ).json()["aim"]["name"] == "Mirage Aim"
    assert OVERALL_KEY not in {"de_mirage", "de_dust2", "de_nuke", "de_inferno"}


def test_ai_assessment_falls_back_to_the_files_it_replaced(monkeypatch, tmp_path):
    """An assessment made before roles and patterns merged must still show.

    Splitting the store would otherwise leave existing users staring at an
    empty card and paying for a re-run to get back what they already had.
    """
    import json

    # ai_assessment.json is deliberately absent, so the lookup has to fall back.
    (tmp_path / "ai_roles.json").write_text(json.dumps({
        "de_mirage": {"ct_role": {"name": "Pit Anchor", "description": "holds pit"},
                      "t_role": {"name": "Ramp Entry", "description": "entries ramp"},
                      "model": "old-model"},
    }))
    (tmp_path / "ai_patterns.json").write_text(json.dumps({
        "de_mirage": {"aim": {"name": "Slow To Stop"}, "headline": "stop earlier",
                      "matches": 4, "rounds": 90, "model": "old-model"},
    }))
    monkeypatch.setattr(store_module, "DATA_DIR", tmp_path)

    body = client.get("/api/performance/ai-assessment?maps=de_mirage").json()

    assert body["ct_role"]["name"] == "Pit Anchor"
    assert body["t_role"]["name"] == "Ramp Entry"
    assert body["aim"]["name"] == "Slow To Stop"
    assert body["headline"] == "stop earlier"
    assert body["matches"] == 4 and body["rounds"] == 90


def test_ai_target_rejects_an_unknown_provider(monkeypatch):
    """The trends selector names a provider; it must not reach the dispatcher unchecked."""
    from fastapi import HTTPException


    monkeypatch.setattr(deps, "load_ai_config", lambda: {
        "active_provider": "openai", "active_model": "gpt-4.1-mini",
        "providers": {"openai": {"api_key": "k"}},
    })

    with pytest.raises(HTTPException) as exc:
        deps._resolve_ai_target("not-a-provider", "some-model")
    assert exc.value.status_code == 400

    # A named provider that does exist is honoured over the configured one,
    # but the key still comes from the stored config, never from the request.
    monkeypatch.setattr(deps, "load_ai_config", lambda: {
        "active_provider": "openai", "active_model": "gpt-4.1-mini",
        "providers": {"openai": {"api_key": "k"}, "anthropic": {"api_key": "k2"}},
    })
    assert deps._resolve_ai_target("anthropic", "claude-opus-5") == (
        "anthropic", "claude-opus-5", "k2",
    )
    assert deps._resolve_ai_target("", "") == ("openai", "gpt-4.1-mini", "k")


def test_ai_target_requires_a_key_for_the_chosen_provider(monkeypatch):
    from fastapi import HTTPException


    monkeypatch.setattr(deps, "load_ai_config", lambda: {
        "active_provider": "openai", "active_model": "gpt-4.1-mini",
        "providers": {"openai": {"api_key": ""}},
    })
    with pytest.raises(HTTPException) as exc:
        deps._resolve_ai_target()
    assert exc.value.status_code == 400
    assert "key" in exc.value.detail.lower()










def test_available_demo_names_lists_only_demos(tmp_path, monkeypatch):
    from src.api.deps import _available_demo_names

    monkeypatch.setenv("DEMO_DIR", str(tmp_path))
    (tmp_path / "a.dem").write_bytes(b"x")
    (tmp_path / "b.dem").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")

    names = _available_demo_names()
    assert {"a.dem", "b.dem"} <= names
    assert "notes.txt" not in names
