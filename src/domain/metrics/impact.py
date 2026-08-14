"""How much each kill and death moved the round.

Weighted by the measured win-probability table in
src/domain/calibration/winprob.py, so a recalibration changes every figure
here — which is why that table has invariant tests.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.domain.calibration import win_probability
from src.domain.metrics._shared import (
    _build_round_team_map,
    _confidence,
    _median,
)
from src.domain.metrics.context import MetricContext
from src.domain.metrics.registry import metric


@metric(
    id="impact.stats",
    label="Round Impact",
    group="impact",
    version=1,
    requires={"parsed:player_death", "parsed:bomb_planted", "parsed:round_end"},
    output_key="impact_stats",
    description="Win-probability swing attributed to each kill and death.",
)
def impact_stats(ctx: MetricContext) -> dict[str, Any]:
    death_df = ctx.parsed.get("player_death", pd.DataFrame())
    round_end_df = ctx.parsed.get("round_end", pd.DataFrame())
    return _calculate_impact_stats(
        ctx.parsed,
        ctx.steam_id,
        ctx.total_rounds,
        _build_round_team_map(death_df, ctx.steam_id, round_end_df),
    )


def _calculate_impact_stats(
    parsed_data: dict[str, Any],
    steam_id: str,
    total_rounds: int,
    round_team_map: dict[int, str],
) -> dict[str, Any]:
    """Score every kill and death by how much it moved the round.

    Walks each round's deaths in order, tracking who is alive and whether the
    bomb is down, and prices each event as the change in the player's team's
    chance of winning.  A kill that turns 5v5 into 5v4 is worth far more than
    one that turns 4v1 into 4v0, and the same kill is worth more after the
    bomb is planted — differences that counting kills cannot express.

    Credit here goes entirely to whoever landed the kill.  Splitting it with
    the players who did the damage, traded, or threw the flash is the next
    step, and needs attribution this function does not yet do.
    """
    death_df = parsed_data.get("player_death", pd.DataFrame())
    bomb_df = parsed_data.get("bomb_planted", pd.DataFrame())
    sid = str(steam_id)

    required = {"round", "tick", "user_team_num", "user_steamid"}
    if death_df.empty or not required.issubset(death_df.columns):
        # Without a tick to order deaths by, or a team to attribute them to,
        # the round state cannot be reconstructed and no swing is measurable.
        return {}

    plant_tick_by_round: dict[int, int] = {}
    if not bomb_df.empty and "round" in bomb_df.columns and "tick" in bomb_df.columns:
        for _, row in bomb_df.iterrows():
            try:
                rnd = int(row["round"])
                tick = int(row["tick"])
            except (TypeError, ValueError):
                continue
            plant_tick_by_round.setdefault(rnd, tick)

    kill_swings: list[float] = []
    death_swings: list[float] = []
    per_round: list[dict[str, Any]] = []

    for rnd in range(1, total_rounds + 1):
        rows = death_df[death_df["round"] == rnd]
        if rows.empty:
            continue
        rows = rows.sort_values("tick")

        side = round_team_map.get(rnd)
        if side not in ("CT", "T"):
            continue
        player_is_ct = side == "CT"

        ct_alive, t_alive = 5, 5
        plant_tick = plant_tick_by_round.get(rnd)
        round_kill_swing = 0.0
        round_death_swing = 0.0

        for _, row in rows.iterrows():
            try:
                tick = int(row["tick"])
                victim_team = int(row["user_team_num"])
            except (TypeError, ValueError):
                continue
            if victim_team not in (2, 3):
                continue

            planted = plant_tick is not None and tick >= plant_tick
            before_ct = win_probability(ct_alive, t_alive, planted)

            if victim_team == 3:
                ct_after, t_after = ct_alive - 1, t_alive
            else:
                ct_after, t_after = ct_alive, t_alive - 1
            after_ct = win_probability(ct_after, t_after, planted)

            # Expressed from the player's own side.
            before = before_ct if player_is_ct else 1.0 - before_ct
            after = after_ct if player_is_ct else 1.0 - after_ct
            swing = after - before

            attacker = str(row.get("attacker_steamid", ""))
            victim = str(row.get("user_steamid", ""))
            if attacker == sid and victim != sid:
                kill_swings.append(swing)
                round_kill_swing += swing
            if victim == sid:
                death_swings.append(swing)
                round_death_swing += swing

            ct_alive, t_alive = ct_after, t_after
            if ct_alive <= 0 or t_alive <= 0:
                break

        if round_kill_swing or round_death_swing:
            per_round.append({
                "round": rnd,
                "side": side,
                "kill_swing": round(round_kill_swing, 4),
                "death_swing": round(round_death_swing, 4),
                "net_swing": round(round_kill_swing + round_death_swing, 4),
            })

    if not kill_swings and not death_swings:
        return {}

    total_kill = sum(kill_swings)
    total_death = sum(death_swings)
    net = total_kill + total_death
    n_events = len(kill_swings) + len(death_swings)

    return {
        "n": n_events,
        "confidence": _confidence(n_events),
        "kills_scored": len(kill_swings),
        "deaths_scored": len(death_swings),
        "kill_swing_total": round(total_kill, 3),
        "death_swing_total": round(total_death, 3),
        "net_swing_total": round(net, 3),
        "net_swing_per_round": round(net / total_rounds, 4) if total_rounds else 0.0,
        "best_kill_swing": round(max(kill_swings), 4) if kill_swings else None,
        "median_kill_swing": round(_median(kill_swings), 4) if kill_swings else None,
        "per_round": per_round,
    }
