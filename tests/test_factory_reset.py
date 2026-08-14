"""Factory reset has to actually delete everything.

It is offered as "delete all user data", and someone uses it before handing
over a machine, publishing a container image, or filing a bug report with their
data directory attached. Leaving a credential behind is the failure that
matters, and it is invisible: the endpoint returns {"status": "ok"} either way.

For a long time it left four files: the Steam refresh tokens, which are
password-equivalent, the Steam Web API key and match ledger, and the Auto-Sync
and retention settings.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("DB_PATH", ":memory:")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from src.config import store as store_module  # noqa: E402

client = TestClient(api.app)

# Everything the application writes under data/. A new store added without a
# line here is caught by test_every_store_the_app_writes_is_covered below.
CREDENTIALS = ("steam_tokens.json", "steam_sharecodes.json")
SETTINGS = ("auto_sync.json", "storage_config.json", "sync_config.json",
            "onboarding.json", "accounts.json", "friends.json",
            # Permission to reach the network is a setting, and a reset must
            # not leave the next user of this machine having granted it.
            "update_config.json", "update_cache.json")
ASSESSMENTS = ("ai_assessment.json", "ai_roles.json", "ai_patterns.json")


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A populated data directory, standing in for a used installation."""
    monkeypatch.setattr(store_module, "DATA_DIR", tmp_path)

    (tmp_path / "steam_tokens.json").write_text(json.dumps({
        "Main": {"refreshToken": "eyJ-a-real-looking-refresh-token",
                 "steamId": "76561198012345678"}
    }), encoding="utf-8")
    (tmp_path / "steam_sharecodes.json").write_text(json.dumps({
        "apiKey": "0123456789ABCDEF0123456789ABCDEF",
        "accounts": {"Main": {"authCode": "ABCD-12345-EFGH", "matches": {}}},
    }), encoding="utf-8")
    (tmp_path / "accounts.json").write_text(json.dumps(
        {"accounts": [{"name": "Main", "steam_id": "76561198012345678",
                       "active": True}]}), encoding="utf-8")
    (tmp_path / "friends.json").write_text(
        json.dumps({"friends": [{"steam_id": "76561198000000001"}]}), encoding="utf-8")
    (tmp_path / "sync_config.json").write_text(
        json.dumps({"folder": "C:/demos"}), encoding="utf-8")
    (tmp_path / "onboarding.json").write_text(
        json.dumps({"completed": True}), encoding="utf-8")
    (tmp_path / "auto_sync.json").write_text(
        json.dumps({"enabled": True, "interval_minutes": 5}), encoding="utf-8")
    (tmp_path / "storage_config.json").write_text(
        json.dumps({"keep_recent": 5}), encoding="utf-8")
    for name in ASSESSMENTS:
        (tmp_path / name).write_text(json.dumps({"de_mirage": {"aim": {}}}),
                                     encoding="utf-8")
    (tmp_path / "ai_config.json").write_text(json.dumps({
        "providers": {"openai": {"api_key": "sk-a-real-looking-key",
                                 "default_model": "gpt-5.4-mini"}},
        "active_provider": "openai",
        "system_instructions": "be blunt",
        "prompts": [{"name": "Overview", "prompt": "how did I do"}],
    }), encoding="utf-8")
    (tmp_path / "steamID").write_text("76561198012345678", encoding="utf-8")
    return tmp_path


def _reset() -> dict:
    body = client.post("/api/factory-reset").json()
    assert body["status"] in ("ok", "partial"), body
    return body


# ---------------------------------------------------------------------------
# The part that matters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CREDENTIALS)
def test_credentials_are_gone(data_dir, name):
    """Steam refresh tokens are password-equivalent and the Web API key is a
    credential. Neither may survive a reset."""
    assert (data_dir / name).exists(), "fixture did not write the file"
    _reset()
    assert not (data_dir / name).exists(), f"{name} survived the reset"


def test_no_secret_string_survives_anywhere_under_data(data_dir):
    """Stronger than checking filenames: nothing that looks like a secret may
    remain in any file, whatever it is called."""
    secrets = [
        "eyJ-a-real-looking-refresh-token",
        "0123456789ABCDEF0123456789ABCDEF",
        "ABCD-12345-EFGH",
        "sk-a-real-looking-key",
    ]
    _reset()

    survivors = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        survivors += [f"{secret!r} in {path.name}" for secret in secrets if secret in text]
    assert not survivors, f"secrets survived the reset: {survivors}"


def test_the_steam_id_file_does_not_keep_the_account(data_dir):
    _reset()
    leftover = data_dir / "steamID"
    assert not leftover.exists() or not leftover.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Settings and stored analysis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ASSESSMENTS)
def test_stored_assessments_are_removed(data_dir, name):
    _reset()
    assert not (data_dir / name).exists()


@pytest.mark.parametrize("name", ("auto_sync.json", "storage_config.json"))
def test_settings_return_to_their_defaults(data_dir, name):
    """These were left untouched, so a reset kept the previous owner's
    Auto-Sync interval and retention window."""
    _reset()
    assert not (data_dir / name).exists(), f"{name} survived with old values"


def test_accounts_and_friends_are_emptied(data_dir):
    _reset()
    assert json.loads((data_dir / "accounts.json").read_text())["accounts"] == []
    assert json.loads((data_dir / "friends.json").read_text())["friends"] == []


def test_onboarding_runs_again(data_dir):
    _reset()
    assert json.loads((data_dir / "onboarding.json").read_text())["completed"] is False


def test_ai_keys_are_cleared_but_the_prompts_survive(data_dir):
    """The provider list and prompts are the user's configuration, not their
    secrets; only the key and instructions go."""
    _reset()
    cfg = json.loads((data_dir / "ai_config.json").read_text())
    assert cfg["providers"]["openai"]["api_key"] == ""
    assert cfg["system_instructions"] == ""
    assert cfg["providers"]["openai"]["default_model"] == "gpt-5.4-mini"
    assert cfg["prompts"][0]["name"] == "Overview"


# ---------------------------------------------------------------------------
# Behaviour around the reset
# ---------------------------------------------------------------------------


def test_auto_sync_is_stopped_before_the_wipe(data_dir, monkeypatch):
    """Left running it would keep downloading with the tokens being deleted,
    and write a fresh ledger straight back."""
    stopped = []
    monkeypatch.setattr(
        api_settings := __import__(
            "src.api.routers.settings", fromlist=["_stop_auto_sync"]
        ),
        "_stop_auto_sync",
        lambda: stopped.append(True),
    )
    assert api_settings is not None
    _reset()
    assert stopped == [True], "Auto-Sync was not stopped"


def test_a_reset_on_a_fresh_install_is_not_an_error(tmp_path, monkeypatch):
    """Nothing to delete is a success, not a partial failure."""
    monkeypatch.setattr(store_module, "DATA_DIR", tmp_path)
    assert client.post("/api/factory-reset").json()["status"] == "ok"


def test_resetting_twice_is_safe(data_dir):
    assert _reset()["status"] == "ok"
    assert _reset()["status"] == "ok"


def test_every_store_the_app_writes_is_covered_by_this_test(data_dir):
    """A new data file added later must be considered here.

    The reset is only as complete as the list of things it knows about, and
    the failure mode is a credential quietly outliving it.
    """
    known = set(CREDENTIALS) | set(SETTINGS) | set(ASSESSMENTS) | {"ai_config.json"}
    _reset()
    # Whatever the app defines a store for must either be gone or accounted for.
    for name in _store_names():
        assert name in known, (
            f"{name} is written by the app but this test says nothing about "
            f"whether factory reset clears it"
        )


def _store_names() -> set[str]:
    import re
    from pathlib import Path

    source = "\n".join(
        p.read_text(encoding="utf-8") for p in Path("src").rglob("*.py")
    )
    return set(re.findall(r'JsonStore\(\s*"([^"]+)"', source))


def test_permission_to_check_for_updates_does_not_survive(data_dir):
    """A reset is what someone runs before handing the machine on or attaching
    data/ to a bug report. Consent to contact the network is theirs, not the
    next person's, so it goes with everything else."""
    from src.services import updates

    updates.config_store.write({"enabled": True, "url": updates.LATEST_URL})
    updates.cache_store.write({"checked_at": "2030-01-01T00:00:00+00:00", "version": "9.9.9"})

    _reset()

    assert not (data_dir / "update_config.json").exists()
    assert not (data_dir / "update_cache.json").exists()
    assert updates.config_store.read()["enabled"] is False
