"""Behavioural axes and role classification for a set of rounds on one side.

This was two functions in api.py — `_compute_side_axes` and
`_classify_side_role` — that walked the same rounds accumulating the same
counters and computed the same five axes from the same formulas. Verified
before merging: on identical input both produced identical axes, so there was
one calculation here written down twice.

They had drifted in one respect that mattered. `_compute_side_axes` read
utility counters as ``u.get("enemies_flashed", 0) or 0``; `_classify_side_role`
read them as ``u.get("enemies_flashed", 0)`` and raised TypeError on a round
whose utility block carried an explicit null. The tolerant form is the one kept.

Three notions of "role" met in these functions and are kept apart deliberately:

  axes       five 0-100 scores describing how someone plays (aggression,
             trading, isolation, survival, sniper)
  archetype  a single name for that shape — AWPer, Entry Fragger, Lurker,
             Support, Anchor, Flex — from thresholds on the same inputs
  success    win rate in the rounds where each axis dominated

A fourth, the map-positional role, lives in src/processor.py and is a different
thing entirely: it answers "where on Inferno did you play", not "how".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

AXES = ("aggression", "trading", "isolation", "survival", "sniper")

# 30 units in the kill record. The comment it came from read "distance >= 30m
# (3000 units ≈ 30m in CS2)", so the field is already in metres.
_LONG_RANGE_DISTANCE = 30


def empty_axes() -> dict[str, int]:
    return dict.fromkeys(AXES, 0)


def empty_side_role() -> dict[str, Any]:
    """The placeholder a side with no rounds shows in the UI."""
    return {
        "name": "Unknown", "icon": "help", "description": "No data.",
        "kills": 0, "deaths": 0, "rounds": 0, "adr": 0,
        "opening_kills": 0, "opening_deaths": 0, "survival_pct": 0,
        "axes": empty_axes(),
    }


@dataclass
class SideTotals:
    """Everything both calculations need, gathered in one pass over the rounds."""

    rounds: int = 0
    kills: int = 0
    deaths: int = 0
    survived: int = 0
    damage: int = 0
    opening_kills: int = 0
    opening_deaths: int = 0
    opening_duel_involved: int = 0
    weapon_kills: dict[str, int] = field(default_factory=dict)
    awp_kills: int = 0
    long_range_kills: int = 0
    enemies_flashed: int = 0
    flash_assists: int = 0
    util_damage: float = 0.0
    traded_deaths: int = 0
    rounds_with_deaths: int = 0
    # Per round: which axis dominated, and whether the round was won.
    round_outcomes: list[dict[str, Any]] = field(default_factory=list)


def _number(value: Any) -> float:
    """Treat a missing or null counter as zero.

    Utility blocks are read back out of stored JSON written by older analyzer
    versions, so a key can be absent or explicitly null.
    """
    return value or 0


def accumulate(side_rounds: list[dict[str, Any]]) -> SideTotals:
    """Walk the rounds once, gathering every counter the axes and role need."""
    totals = SideTotals(rounds=len(side_rounds))

    for entry in side_rounds:
        enriched = entry.get("enriched", {})
        kills_detail = enriched.get("kills_detail", [])
        round_kills = len(kills_detail)
        totals.kills += round_kills

        round_awp = 0
        round_long_range = 0
        for kill in kills_detail:
            weapon = kill.get("weapon", "Unknown")
            totals.weapon_kills[weapon] = totals.weapon_kills.get(weapon, 0) + 1
            if weapon == "AWP":
                totals.awp_kills += 1
                round_awp += 1
            if _number(kill.get("distance")) >= _LONG_RANGE_DISTANCE:
                totals.long_range_kills += 1
                round_long_range += 1

        died = bool(enriched.get("death_detail"))
        if died:
            totals.deaths += 1
        else:
            totals.survived += 1

        duel = enriched.get("opening_duel")
        took_opening_duel = bool(duel)
        won_opening_duel = False
        if duel:
            totals.opening_duel_involved += 1
            if duel.get("role") == "opening_kill":
                totals.opening_kills += 1
                won_opening_duel = True
            elif duel.get("role") == "opening_death":
                totals.opening_deaths += 1

        utility = enriched.get("utility", {})
        round_flashed = _number(utility.get("enemies_flashed"))
        round_flash_assists = _number(utility.get("flash_assists"))
        totals.enemies_flashed += round_flashed
        totals.flash_assists += round_flash_assists
        round_util_damage = _number(utility.get("he_damage")) + sum(
            _number(m.get("damage")) for m in (utility.get("molotov_damage") or [])
        )
        totals.util_damage += round_util_damage

        traded = bool(entry.get("traded"))
        if traded:
            totals.traded_deaths += 1
        if entry.get("deaths", 0) > 0:
            totals.rounds_with_deaths += 1
        totals.damage += entry.get("damage", 0)

        totals.round_outcomes.append({
            "dominant": _dominant_axis(
                took_opening_duel=took_opening_duel,
                won_opening_duel=won_opening_duel,
                traded=traded,
                died=died,
                round_kills=round_kills,
                flash_assists=round_flash_assists,
                flashed=round_flashed,
                util_damage=round_util_damage,
                awp_kills=round_awp,
                long_range_kills=round_long_range,
            ),
            "won": enriched.get("side", "") == enriched.get("round_winner", ""),
        })

    return totals


def _dominant_axis(
    *,
    took_opening_duel: bool,
    won_opening_duel: bool,
    traded: bool,
    died: bool,
    round_kills: int,
    flash_assists: float,
    flashed: float,
    util_damage: float,
    awp_kills: int,
    long_range_kills: int,
) -> str:
    """Which axis best describes a single round. A heuristic, used only to
    bucket rounds for the per-axis win rate."""
    scores = {
        "aggression": (1.0 if took_opening_duel else 0) + (0.5 if won_opening_duel else 0),
        "trading": (
            (0.5 if traded else 0)
            + min(flash_assists * 0.5, 1.0)
            + min(flashed * 0.3, 0.6)
        ),
        "isolation": (
            (0.8 if not died and not took_opening_duel else 0)
            + min(round_kills * 0.3, 0.6)
        ),
        "survival": (0.7 if not died else 0) + min(util_damage * 0.02, 0.3),
        "sniper": min(awp_kills * 0.8, 1.5) + min(long_range_kills * 0.3, 0.5),
    }
    return max(scores, key=scores.get)


def axes_from(totals: SideTotals) -> dict[str, int]:
    """The five 0-100 scores. Formulas are documented in the README."""
    n = totals.rounds
    if n == 0:
        return empty_axes()

    survival_pct = totals.survived / n * 100
    opening_duels = totals.opening_kills + totals.opening_deaths
    opening_kill_pct = (totals.opening_kills / opening_duels * 100) if opening_duels else 0
    involvement_rate = totals.opening_duel_involved / n * 100

    aggression = min(100, round(involvement_rate * 0.5 + opening_kill_pct * 0.5))

    trade_pct = (
        totals.traded_deaths / totals.rounds_with_deaths * 100
        if totals.rounds_with_deaths else 0
    )
    trading = min(100, round(
        trade_pct * 0.4
        + min(totals.flash_assists / n * 50, 30)
        + min(totals.enemies_flashed / n * 25, 30)
    ))

    isolation = min(100, round(
        survival_pct * 0.4
        + (100 - involvement_rate) * 0.3
        + min(totals.kills / n * 40, 30)
    ))

    death_rate = totals.deaths / n * 100
    survival = min(100, round(
        survival_pct * 0.5
        + min(totals.util_damage / n * 3, 25)
        + max(0, 100 - death_rate) * 0.25
    ))

    awp_ratio = (totals.awp_kills / totals.kills * 100) if totals.kills else 0
    long_range_ratio = (totals.long_range_kills / totals.kills * 100) if totals.kills else 0
    sniper = min(100, round(awp_ratio * 0.7 + long_range_ratio * 0.3))

    return {
        "aggression": aggression,
        "trading": trading,
        "isolation": isolation,
        "survival": survival,
        "sniper": sniper,
    }


def success_from(totals: SideTotals, axes: dict[str, int]) -> dict[str, dict]:
    """Win rate in the rounds where each axis was the dominant behaviour."""
    success: dict[str, dict] = {}
    for axis in axes:
        matching = [r for r in totals.round_outcomes if r["dominant"] == axis]
        if matching:
            wins = sum(1 for r in matching if r["won"])
            success[axis] = {
                "rounds": len(matching),
                "wins": wins,
                "win_pct": round(wins / len(matching) * 100, 0),
            }
    return success


def side_axes(side_rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Axes plus per-axis win rate for one side."""
    if not side_rounds:
        return {"axes": empty_axes(), "success": {}}
    totals = accumulate(side_rounds)
    axes = axes_from(totals)
    return {"axes": axes, "success": success_from(totals, axes)}


def classify_archetype(
    *,
    opening_kill_pct: float,
    survival_pct: float,
    util_per_round: float,
    trade_pct: float,
    weapon_kills: dict[str, int],
    total_kills: int,
) -> dict[str, str]:
    """Name the playstyle. Thresholds are heuristic and ordered — the first
    that matches wins, so AWP usage outranks everything else."""
    awp_ratio = (weapon_kills.get("AWP", 0) / total_kills * 100) if total_kills else 0

    if awp_ratio >= 30:
        return {
            "name": "AWPer",
            "icon": "precision_manufacturing",
            "description": "Primary AWP player. High-impact picks from long-range angles, "
            "controlling sightlines and creating openings for the team.",
        }
    if opening_kill_pct >= 55:
        return {
            "name": "Entry Fragger",
            "icon": "bolt",
            "description": "Aggressive entry style. Frequently takes the first duel of the "
            "round, creating space for teammates to trade and execute.",
        }
    if survival_pct >= 55 and opening_kill_pct < 35:
        return {
            "name": "Lurker",
            "icon": "visibility",
            "description": "Patient rotator who picks off distracted enemies. High survival "
            "rate indicates good timing and map awareness.",
        }
    if util_per_round >= 1.5 and trade_pct >= 30:
        return {
            "name": "Support",
            "icon": "shield_with_heart",
            "description": "Team-oriented playstyle with strong utility usage and trade discipline. "
            "Enables teammates through flashes, trades, and info plays.",
        }
    if survival_pct >= 50:
        return {
            "name": "Anchor",
            "icon": "anchor",
            "description": "Reliable site holder with strong survival instincts. Holds positions "
            "patiently and trades effectively during retakes.",
        }
    return {
        "name": "Flex",
        "icon": "sync_alt",
        "description": "Versatile player who adapts role based on the round situation. "
        "Balanced across entry, utility, and trading metrics.",
    }


def side_role(side_rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Archetype, headline counters and axes for one side."""
    if not side_rounds:
        return empty_side_role()

    totals = accumulate(side_rounds)
    n = totals.rounds

    survival_pct = round(totals.survived / n * 100, 1)
    opening_duels = totals.opening_kills + totals.opening_deaths
    opening_kill_pct = (
        round(totals.opening_kills / opening_duels * 100, 1) if opening_duels else 0
    )
    util_per_round = round((totals.enemies_flashed + totals.flash_assists) / n, 2)
    trade_pct = (
        round(totals.traded_deaths / totals.rounds_with_deaths * 100, 1)
        if totals.rounds_with_deaths else 0
    )

    role = classify_archetype(
        opening_kill_pct=opening_kill_pct,
        survival_pct=survival_pct,
        util_per_round=util_per_round,
        trade_pct=trade_pct,
        weapon_kills=totals.weapon_kills,
        total_kills=totals.kills,
    )
    role.update({
        "kills": totals.kills,
        "deaths": totals.deaths,
        "rounds": n,
        "adr": round(totals.damage / n, 1),
        "opening_kills": totals.opening_kills,
        "opening_deaths": totals.opening_deaths,
        "survival_pct": survival_pct,
        "axes": axes_from(totals),
    })
    return role


def match_behavioral_axes(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-match axes for both sides.

    Deserialises ``enriched_json`` into ``enriched`` in place, which callers
    rely on afterwards.
    """
    for entry in rounds:
        raw = entry.get("enriched_json")
        if raw and isinstance(raw, str):
            try:
                entry["enriched"] = json.loads(raw)
            except ValueError:
                entry["enriched"] = {}
        elif not entry.get("enriched"):
            entry["enriched"] = {}

    return {
        "ct": side_axes([r for r in rounds if r.get("enriched", {}).get("side") == "CT"]),
        "t": side_axes([r for r in rounds if r.get("enriched", {}).get("side") == "T"]),
    }
