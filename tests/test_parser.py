"""Tests for src/parser.py compatibility behavior."""

from __future__ import annotations

import sys
import types

import pytest

from src import parser as parser_module


class _FakeParserUnsupportedNewPatch:
    def __init__(self, _demo_path: str):
        pass

    def parse_header(self):
        return {"map_name": "de_anubis", "patch_version": 14152}

    def parse_event(self, _name: str, player=None):
        raise RuntimeError("EntityNotFound")


class _FakeParserUnsupportedOldPatch:
    def __init__(self, _demo_path: str):
        pass

    def parse_header(self):
        return {"map_name": "de_mirage", "patch_version": 14141}

    def parse_event(self, _name: str, player=None):
        raise RuntimeError("EntityNotFound")


def test_parse_demo_fallbacks_for_new_patch_entity_break(monkeypatch, tmp_path):
    dem = tmp_path / "new_patch.dem"
    dem.write_bytes(b"demo")

    fake_module = types.SimpleNamespace(DemoParser=_FakeParserUnsupportedNewPatch)
    monkeypatch.setitem(sys.modules, "demoparser2", fake_module)

    parsed = parser_module.parse_demo(dem)

    assert parsed["header"]["map_name"] == "de_anubis"
    assert parsed["header"]["parse_mode"] == "header_only_fallback"
    assert "parse_warning" in parsed["header"]
    assert parsed["player_death"].empty
    assert parsed["player_hurt"].empty
    assert parsed["round_end"].empty


def test_parse_demo_still_raises_for_old_patch_entity_errors(monkeypatch, tmp_path):
    dem = tmp_path / "old_patch.dem"
    dem.write_bytes(b"demo")

    fake_module = types.SimpleNamespace(DemoParser=_FakeParserUnsupportedOldPatch)
    monkeypatch.setitem(sys.modules, "demoparser2", fake_module)

    with pytest.raises(RuntimeError, match="EntityNotFound"):
        parser_module.parse_demo(dem)
