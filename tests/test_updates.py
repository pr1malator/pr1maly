"""The update check, and the promise it has to keep.

The app says no external services are required. This is the one thing in it
that talks to a server the user did not configure, so the tests that matter
most are the ones about when it does not: off by default, silent while off,
and sending nothing but the request when on.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from src.services import updates
from src.version import APP_VERSION, is_newer, parse


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Point the stores at a temp directory: these tests write config files."""
    monkeypatch.setattr("src.config.store.DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if anything reaches for the network."""
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("url"))
        raise AssertionError(f"the update check contacted {calls[-1]} when it should not")

    monkeypatch.setattr(httpx, "get", forbidden)
    return calls


def _published(version="9.9.9", released="2030-01-01", notes="https://example.invalid/notes"):
    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"version": version, "released": released, "notes": notes}

    return _Response()


# ---------------------------------------------------------------------------
# Off by default, and silent while off
# ---------------------------------------------------------------------------


def test_the_check_is_off_until_someone_turns_it_on():
    assert updates.config_store.read()["enabled"] is False


def test_nothing_is_contacted_while_it_is_off(no_network):
    """The whole promise in one assertion."""
    status = updates.check()
    assert status.enabled is False
    assert status.latest is None
    assert status.update_available is False


def test_turning_it_off_forgets_what_was_learned_while_it_was_on(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _published())
    updates.set_enabled(True)
    assert updates.check().latest == "9.9.9"

    updates.set_enabled(False)
    assert updates.cache_store.read()["version"] == ""
    assert updates.check().latest is None


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------


def test_the_button_works_even_with_the_setting_off(monkeypatch):
    """Pressing "check now" is itself the permission, so it does not need the
    setting; what it must not do is turn the setting on behind the user."""
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _published())
    status = updates.check(force=True)
    assert status.latest == "9.9.9"
    assert status.enabled is False
    assert updates.config_store.read()["enabled"] is False


def test_a_newer_published_version_is_reported_as_available(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _published(version="9.9.9"))
    status = updates.check(force=True)
    assert status.update_available is True
    assert status.current == APP_VERSION
    assert status.notes == "https://example.invalid/notes"


def test_the_same_version_is_not_an_update(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _published(version=APP_VERSION))
    assert updates.check(force=True).update_available is False


def test_an_older_published_version_is_not_an_update(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _published(version="0.0.1"))
    assert updates.check(force=True).update_available is False


def test_the_request_carries_nothing_about_this_install(monkeypatch):
    """No version, no id, no counter — the point of fetching a static file."""
    seen = {}

    def capture(url, **kwargs):
        seen["url"] = url
        seen["kwargs"] = kwargs
        return _published()

    monkeypatch.setattr(httpx, "get", capture)
    updates.check(force=True)

    assert "?" not in seen["url"], f"the request carries a query string: {seen['url']}"
    assert APP_VERSION not in seen["url"]
    assert "params" not in seen["kwargs"]
    assert "data" not in seen["kwargs"] and "json" not in seen["kwargs"]


# ---------------------------------------------------------------------------
# Being offline is normal, not an error
# ---------------------------------------------------------------------------


def test_a_failed_check_is_reported_and_not_raised(monkeypatch):
    def offline(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "get", offline)
    status = updates.check(force=True)
    assert status.error and "no route to host" in status.error
    assert status.update_available is False


def test_a_published_file_without_a_version_is_rejected(monkeypatch):
    class _Empty:
        def raise_for_status(self):
            return None

        def json(self):
            return {"note": "nothing here"}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Empty())
    status = updates.check(force=True)
    assert status.error
    assert status.latest is None


# ---------------------------------------------------------------------------
# Caching, so leaving it on costs one request a day
# ---------------------------------------------------------------------------


def test_a_fresh_answer_is_not_asked_for_again(monkeypatch, no_network):
    updates.config_store.write({"enabled": True, "url": updates.LATEST_URL})
    updates.cache_store.write({
        "checked_at": datetime.now(UTC).isoformat(),
        "version": "9.9.9",
        "released": "2030-01-01",
        "notes": "",
    })
    # no_network would raise if this reached out.
    assert updates.check().latest == "9.9.9"


def test_a_stale_answer_is_refreshed(monkeypatch):
    updates.config_store.write({"enabled": True, "url": updates.LATEST_URL})
    updates.cache_store.write({
        "checked_at": (datetime.now(UTC) - timedelta(days=3)).isoformat(),
        "version": "1.0.0",
        "released": "",
        "notes": "",
    })
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _published(version="9.9.9"))
    assert updates.check().latest == "9.9.9"


def test_an_unreadable_timestamp_counts_as_stale(monkeypatch):
    updates.config_store.write({"enabled": True, "url": updates.LATEST_URL})
    updates.cache_store.write({"checked_at": "not a date", "version": "1.0.0"})
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _published(version="9.9.9"))
    assert updates.check().latest == "9.9.9"


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


def test_the_app_version_matches_pyproject():
    """The number the check compares against must be the one that was released.

    It used to be written twice, and only pyproject.toml was verified against
    the git tag.
    """
    import tomllib
    from pathlib import Path

    from src.config.settings import PROJECT_ROOT

    declared = tomllib.loads(
        (Path(PROJECT_ROOT) / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    assert APP_VERSION == declared


def test_the_app_reports_the_same_version_over_http():
    import os

    os.environ.setdefault("DB_PATH", ":memory:")
    from fastapi.testclient import TestClient

    from api import app

    body = TestClient(app).get("/api/health").json()
    assert body["version"] == APP_VERSION


@pytest.mark.parametrize("newer,older", [
    ("1.0.1", "1.0.0"),
    ("1.1.0", "1.0.9"),
    ("2.0.0", "1.9.9"),
    ("1.0.10", "1.0.9"),
])
def test_version_ordering(newer, older):
    assert is_newer(newer, older)
    assert not is_newer(older, newer)


def test_an_unparseable_version_sorts_lowest_rather_than_raising():
    assert parse("not-a-version") == (0,)
    assert not is_newer("not-a-version", "1.0.0")
