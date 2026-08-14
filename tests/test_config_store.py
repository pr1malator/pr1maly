"""Tests for the JSON config store.

These files are user-owned and hold credentials, so the failure modes that
matter are: losing one to a crash, leaving one world-readable, and refusing to
read one a text editor has touched.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from src.config import store as store_module
from src.config.store import JsonStore


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Point every store at a scratch directory, never the developer's data/."""
    monkeypatch.setattr(store_module, "DATA_DIR", tmp_path)
    return tmp_path


def test_missing_file_reads_as_defaults():
    s = JsonStore("nope.json", defaults={"keep_recent": 30})
    assert s.read() == {"keep_recent": 30}
    assert not s.exists()


def test_defaults_fill_in_keys_the_file_lacks(data_dir):
    """A config gains new settings without the user editing their file."""
    (data_dir / "storage.json").write_text('{"keep_recent": 5}', encoding="utf-8")
    s = JsonStore("storage.json", defaults={"keep_recent": 30, "auto_cleanup": False})
    assert s.read() == {"keep_recent": 5, "auto_cleanup": False}


def test_file_wins_over_defaults(data_dir):
    (data_dir / "s.json").write_text('{"a": "file"}', encoding="utf-8")
    assert JsonStore("s.json", defaults={"a": "default"}).read()["a"] == "file"


def test_corrupt_file_reads_as_defaults(data_dir):
    (data_dir / "broken.json").write_text("{not json at all", encoding="utf-8")
    assert JsonStore("broken.json", defaults={"ok": True}).read() == {"ok": True}


def test_non_dict_json_reads_as_defaults(data_dir):
    """A file holding a bare list must not be spliced into a dict."""
    (data_dir / "list.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert JsonStore("list.json", defaults={"ok": True}).read() == {"ok": True}


def test_byte_order_mark_is_tolerated(data_dir):
    """Notepad and PowerShell both write one; json.loads rejects it."""
    (data_dir / "bom.json").write_bytes(
        b"\xef\xbb\xbf" + json.dumps({"steam_id": "123"}).encode("utf-8")
    )
    assert JsonStore("bom.json").read() == {"steam_id": "123"}


def test_round_trip(data_dir):
    s = JsonStore("rt.json")
    s.write({"b": 2, "a": [1, {"deep": True}]})
    assert s.read() == {"b": 2, "a": [1, {"deep": True}]}


def test_write_creates_the_data_directory(tmp_path, monkeypatch):
    nested = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(store_module, "DATA_DIR", nested)
    JsonStore("new.json").write({"created": True})
    assert (nested / "new.json").is_file()


def test_written_file_is_bom_free_and_newline_terminated(data_dir):
    JsonStore("out.json").write({"a": 1})
    raw = (data_dir / "out.json").read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf"
    assert raw.endswith(b"\n")
    json.loads(raw.decode("utf-8"))  # parses the way the app reads it


def test_write_leaves_no_temporary_file_behind(data_dir):
    JsonStore("clean.json").write({"a": 1})
    assert sorted(p.name for p in data_dir.iterdir()) == ["clean.json"]


def test_failed_write_keeps_the_previous_contents(data_dir):
    """The reason writes are not done in place: a ledger is not regenerable."""
    s = JsonStore("ledger.json")
    s.write({"apiKey": "keep-me"})

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        s.write({"apiKey": Unserialisable()})

    assert s.read() == {"apiKey": "keep-me"}
    assert not list(data_dir.glob(".*.tmp"))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
def test_private_files_are_owner_only(data_dir):
    JsonStore("secret.json", private=True).write({"apiKey": "x"})
    assert os.stat(data_dir / "secret.json").st_mode & 0o077 == 0


def test_delete_is_idempotent(data_dir):
    s = JsonStore("gone.json")
    s.write({"a": 1})
    s.delete()
    s.delete()
    assert not s.exists()


# -- list-shaped files ------------------------------------------------------


def test_read_list_of_missing_file_is_empty():
    assert JsonStore("accounts.json").read_list("accounts") == []


def test_list_round_trip(data_dir):
    s = JsonStore("accounts.json")
    s.write_list("accounts", [{"name": "Main", "active": True}])
    assert s.read_list("accounts") == [{"name": "Main", "active": True}]


def test_read_list_of_wrong_shape_is_empty(data_dir):
    (data_dir / "accounts.json").write_text('{"accounts": "oops"}', encoding="utf-8")
    assert JsonStore("accounts.json").read_list("accounts") == []
