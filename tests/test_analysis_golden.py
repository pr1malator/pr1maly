"""End-to-end golden for the analysis pipeline.

The four JSON blob columns (aim_stats, role_data, utility_data, impact_stats)
plus enriched_json and replay_json are opaque TEXT. Nothing in the database
constrains their shape, no response model describes them, and once the demo has
been deleted by the retention feature they cannot be recomputed. That makes
them the highest-risk surface in the codebase and the one with the least
mechanical protection.

This file pins the whole of calculate_match_stats against a synthetic match, so
that moving a function between modules has to produce identical numbers. It is
the pass condition for the metrics refactor: byte-identical blobs.

The fixture is deliberately rich — it exercises kills, deaths, damage-only
encounters, all four grenade types, purchases, bomb plants, sampled positions
for role classification, velocities for movement analysis, and high-frequency
replay frames. test_golden_is_substantive below enforces that it stays rich; a
golden that quietly degrades to empty dicts would pass forever while protecting
nothing.

Regenerate only when a metric is meant to change:

    UPDATE_SNAPSHOTS=1 python -m pytest tests/test_analysis_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.processor import ANALYZER_VERSION, calculate_match_stats

_SNAPSHOT = Path(__file__).parent / "snapshots" / "analysis_golden.json"
_UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"

STEAM_ID = "76561198012345678"
ENEMIES = ["76561198000000001", "76561198000000002", "76561198000000003"]
TEAMMATE = "76561198000000011"
ROUNDS = 4
TICKS_PER_ROUND = 3000

# Blob keys that carry the persisted columns. Each must stay populated.
_BLOB_KEYS = ("aim_stats", "role_data", "utility_data", "impact_stats",
              "enriched_rounds", "replay_data", "benchmarks", "all_players")


def _build_parsed_match() -> dict[str, Any]:
    """A synthetic 4-round de_mirage match: 2 rounds CT, 2 rounds T.

    Column names follow what src/parser.py actually produces — notably
    ``item_name`` on purchases and ``tick_offset`` on replay positions, both of
    which the consumers look for by name.
    """
    deaths: list[dict] = []
    hurts: list[dict] = []
    fires: list[dict] = []
    blinds: list[dict] = []
    purchases: list[dict] = []
    positions: list[dict] = []
    velocities: list[dict] = []
    round_positions: list[dict] = []
    replay_positions: list[dict] = []
    economy: list[dict] = []
    smokes: list[dict] = []
    molotovs: list[dict] = []
    hes: list[dict] = []
    flashes: list[dict] = []
    plants: list[dict] = []

    for rnd in range(1, ROUNDS + 1):
        base = rnd * TICKS_PER_ROUND
        side = "CT" if rnd <= 2 else "T"
        team_num = 3 if side == "CT" else 2
        enemy_team = 2 if side == "CT" else 3

        # Two kills for the player: one headshot rifle kill, one AWP kill.
        for i, victim in enumerate(ENEMIES[:2]):
            tick = base + 500 + i * 200
            deaths.append({
                "tick": tick, "round": rnd,
                "attacker_steamid": STEAM_ID, "attacker_name": "TestPlayer",
                "attacker_team_num": team_num,
                "user_steamid": victim, "user_name": f"Enemy{i}",
                "user_team_num": enemy_team,
                "assister_steamid": None, "assister_name": None,
                "weapon": "ak47" if i == 0 else "awp",
                "headshot": bool(i == 0), "distance": 12.5 + i * 20,
                "attacker_blind": False, "noscope": False, "thrusmoke": False,
            })
            hurts.append({
                "tick": tick - 10, "round": rnd,
                "attacker_steamid": STEAM_ID, "attacker_name": "TestPlayer",
                "user_steamid": victim, "user_name": f"Enemy{i}",
                "dmg_health": 100, "health": 0, "weapon": "ak47",
                "hitgroup": "head" if i == 0 else "chest",
            })
            fires.append({"tick": tick - 12, "round": rnd,
                          "user_steamid": STEAM_ID, "user_name": "TestPlayer",
                          "weapon": "ak47"})

        # The player then dies.
        death_tick = base + 1500
        deaths.append({
            "tick": death_tick, "round": rnd,
            "attacker_steamid": ENEMIES[2], "attacker_name": "Enemy2",
            "attacker_team_num": enemy_team,
            "user_steamid": STEAM_ID, "user_name": "TestPlayer",
            "user_team_num": team_num,
            "assister_steamid": None, "assister_name": None,
            "weapon": "m4a1", "headshot": False, "distance": 30.0,
            "attacker_blind": False, "noscope": False, "thrusmoke": False,
        })
        hurts.append({
            "tick": death_tick - 5, "round": rnd,
            "attacker_steamid": ENEMIES[2], "attacker_name": "Enemy2",
            "user_steamid": STEAM_ID, "user_name": "TestPlayer",
            "dmg_health": 100, "health": 0, "weapon": "m4a1",
            "hitgroup": "chest",
        })

        # Utility: one flash landing on an enemy, plus every grenade type.
        blinds.append({"tick": base + 300, "round": rnd,
                       "user_steamid": ENEMIES[0], "user_name": "Enemy0",
                       "attacker_steamid": STEAM_ID, "attacker_name": "TestPlayer",
                       "blind_duration": 2.4})
        for weapon, bucket in (("smokegrenade", smokes), ("molotov", molotovs),
                               ("hegrenade", hes), ("flashbang", flashes)):
            bucket.append({"tick": base + 250, "round": rnd,
                           "user_steamid": STEAM_ID, "user_name": "TestPlayer",
                           "x": 100.0, "y": 200.0, "z": 0.0})
            fires.append({"tick": base + 240, "round": rnd,
                          "user_steamid": STEAM_ID, "user_name": "TestPlayer",
                          "weapon": weapon})

        for item, cost in (("ak47", 2700), ("kevlar", 650), ("smokegrenade", 300)):
            purchases.append({"tick": base + 50, "round": rnd,
                              "user_steamid": STEAM_ID, "user_name": "TestPlayer",
                              "item_name": item, "cost": cost})

        economy.append({"steamid": STEAM_ID, "round": rnd,
                        "start_balance": 4500, "end_balance": 1200})

        if side == "T":
            plants.append({"tick": base + 1200, "round": rnd,
                           "user_steamid": STEAM_ID, "user_name": "TestPlayer",
                           "site": "A"})

        # XYZ at every event tick, and a velocity window before each.
        event_ticks = (
            {d["tick"] for d in deaths if d["round"] == rnd}
            | {h["tick"] for h in hurts if h["round"] == rnd}
            | {f["tick"] for f in fires if f["round"] == rnd}
        )
        for tick in sorted(event_ticks):
            for pid, (x, y) in [
                (STEAM_ID, (-1500.0, 500.0)),
                (ENEMIES[0], (-1400.0, 600.0)),
                (ENEMIES[1], (-1300.0, 700.0)),
                (ENEMIES[2], (-1200.0, 800.0)),
            ]:
                positions.append({"tick": tick, "steamid": pid, "name": "p",
                                  "X": x, "Y": y, "Z": 0.0})
            for pid in (STEAM_ID, ENEMIES[2]):
                for offset in range(9):
                    velocities.append({
                        "steamid": pid, "tick": tick - offset,
                        "velocity_X": 5.0, "velocity_Y": 3.0,
                        "X": -1500.0, "Y": 500.0, "Z": 0.0,
                        "yaw": 90.0, "pitch": 0.0, "ducked": False,
                    })

        # Sampled positions drive role classification.
        for k in range(6):
            round_positions.append({
                "tick": base + 200 + k * 128, "round": rnd, "steamid": STEAM_ID,
                "name": "TestPlayer", "X": -1200.0, "Y": 700.0, "Z": 0.0,
            })

        # High-frequency frames drive the 2D replay viewer.
        for k in range(10):
            for pid, name, tnum in [
                (STEAM_ID, "TestPlayer", team_num),
                (TEAMMATE, "Mate0", team_num),
                (ENEMIES[0], "Enemy0", enemy_team),
            ]:
                replay_positions.append({
                    "tick": base + 150 + k * 32, "tick_offset": k * 32,
                    "round": rnd, "steamid": pid, "name": name,
                    "X": -1200.0 + k * 10, "Y": 700.0 - k * 5, "Z": 0.0,
                    "yaw": 90.0, "health": 100, "team_num": tnum,
                    "is_alive": True,
                })

    round_end = pd.DataFrame({
        "round": list(range(1, ROUNDS + 1)),
        "tick": [r * TICKS_PER_ROUND + 2500 for r in range(1, ROUNDS + 1)],
        "winner": ["CT", "T", "T", "CT"],
        "reason": ["ct_win_elimination", "t_win_elimination",
                   "target_bombed", "ct_win_defuse"],
    })
    freeze_end = pd.DataFrame({
        "round": list(range(1, ROUNDS + 1)),
        "tick": [r * TICKS_PER_ROUND + 100 for r in range(1, ROUNDS + 1)],
    })

    return {
        "player_death": pd.DataFrame(deaths),
        "player_hurt": pd.DataFrame(hurts),
        "round_end": round_end,
        "round_freeze_end": freeze_end,
        "item_purchase": pd.DataFrame(purchases),
        "player_blind": pd.DataFrame(blinds),
        "bomb_planted": pd.DataFrame(plants),
        "bomb_defused": pd.DataFrame(),
        "bomb_exploded": pd.DataFrame(),
        "weapon_fire": pd.DataFrame(fires),
        "flash_detonate": pd.DataFrame(flashes),
        "he_detonate": pd.DataFrame(hes),
        "smoke_detonate": pd.DataFrame(smokes),
        "molotov_detonate": pd.DataFrame(molotovs),
        "positions": pd.DataFrame(positions),
        "velocities": pd.DataFrame(velocities),
        "round_positions": pd.DataFrame(round_positions),
        "replay_positions": pd.DataFrame(replay_positions),
        "economy": pd.DataFrame(economy),
        "spotted": pd.DataFrame(),
        "ranks": pd.DataFrame(),
        "rank_update": pd.DataFrame(),
        "end_stats": pd.DataFrame(),
        "header": {"map_name": "de_mirage", "patch_version": 14000},
    }


def _jsonable(value: Any) -> Any:
    """Normalise to plain JSON types.

    numpy scalars leak out of pandas operations, and dict keys are ints in
    replay_data but strings after a JSON round-trip. Floats are rounded to 6
    places so last-bit differences between platforms do not read as drift.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bool):
        return value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (int, str)) or value is None:
        return value
    return str(value)


@pytest.fixture(scope="module")
def live() -> dict[str, Any]:
    return _jsonable(calculate_match_stats(_build_parsed_match(), STEAM_ID))


@pytest.fixture(scope="module")
def stored(live) -> dict[str, Any]:
    if _UPDATE or not _SNAPSHOT.exists():
        _SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT.write_text(
            json.dumps(live, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))


def _leaf_count(value: Any) -> int:
    """Count leaves that carry actual information."""
    if isinstance(value, dict):
        return sum(_leaf_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(_leaf_count(v) for v in value)
    return 0 if value in (None, 0, 0.0, "", False) else 1


def test_golden_is_substantive(live):
    """The fixture must keep reaching the parts of the pipeline it claims to.

    Without this, a change that makes the processor return empty dicts would
    regenerate a snapshot of nothing and every other test here would still pass.
    """
    minimums = {
        "aim_stats": 50, "role_data": 10, "utility_data": 30,
        "impact_stats": 10, "enriched_rounds": 100, "replay_data": 100,
        "benchmarks": 5, "all_players": 10,
    }
    for key, floor in minimums.items():
        count = _leaf_count(live.get(key))
        assert count >= floor, (
            f"{key} only has {count} populated leaves (expected >= {floor}). "
            "The fixture has stopped exercising this metric — fix the fixture "
            "rather than lowering the floor."
        )


def test_headline_metrics(live):
    """Readable assertions on the numbers, independent of the snapshot file."""
    assert live["map_name"] == "de_mirage"
    assert live["total_rounds"] == ROUNDS
    assert live["kills"] == ROUNDS * 2
    assert live["deaths"] == ROUNDS
    assert live["player_name"] == "TestPlayer"
    assert live["analyzer_version"] == ANALYZER_VERSION
    assert live["rounds_2k"] == ROUNDS  # two kills every round


@pytest.mark.parametrize("key", _BLOB_KEYS)
def test_blob_unchanged(key, live, stored):
    """Each persisted blob keeps its exact shape and values."""
    assert key in stored, f"{key} missing from snapshot; regenerate deliberately"
    assert live[key] == stored[key], (
        f"{key} drifted. If this is an intended metric change, bump the metric "
        f"and regenerate with UPDATE_SNAPSHOTS=1."
    )


def test_scalar_columns_unchanged(live, stored):
    """The non-blob columns that land directly in the matches table."""
    scalars = {
        k: v for k, v in live.items()
        if not isinstance(v, (dict, list))
    }
    for key, value in scalars.items():
        assert key in stored, f"new scalar {key!r}; regenerate deliberately"
        assert value == stored[key], f"{key} changed: {stored[key]!r} -> {value!r}"


def test_top_level_keys_unchanged(live, stored):
    """save_match reads these by name; a rename silently drops data."""
    assert set(live) == set(stored)
