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


# ---------------------------------------------------------------------------
# Warmup exclusion
# ---------------------------------------------------------------------------


def test_warmup_events_are_not_assigned_to_round_one():
    """Warmup shares the demo with the match.

    Players hold ~$16k and buy freely during it, and every warmup event fell
    into round 1 because the first round_end is thousands of ticks away. That
    is what put a five-figure balance and an unaffordable buy in a pistol
    round.
    """
    import pandas as pd

    from src.parser import _assign_rounds

    events = pd.DataFrame({"tick": [1, 40, 65, 600, 6000]})
    round_ends = pd.DataFrame({"round": [1, 2], "tick": [5471, 11607]})

    kept = _assign_rounds(events, round_ends, match_start_tick=65)

    assert list(kept["tick"]) == [65, 600, 6000]
    assert list(kept["round"]) == [1, 1, 2]


def test_round_assignment_unchanged_without_a_match_start():
    """Demos with no round_start fall back to keeping everything."""
    import pandas as pd

    from src.parser import _assign_rounds

    events = pd.DataFrame({"tick": [1, 100]})
    round_ends = pd.DataFrame({"round": [1], "tick": [500]})

    kept = _assign_rounds(events, round_ends)
    assert len(kept) == 2
