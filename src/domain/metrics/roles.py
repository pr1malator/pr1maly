"""Where on the map the player actually plays.

Distinct from the behavioural axes in src/metrics/behavior.py, which describe
*how* someone plays. This one answers "which position on Inferno", and reads
the zone data in role_zones/.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.callouts import get_callout
from src.domain.metrics.context import MetricContext
from src.domain.metrics.registry import ENRICHED_ROUNDS, metric
from src.domain.metrics.role_zones import role_zones, roles_for


@metric(
    id="roles.positional",
    label="Map Role",
    group="roles",
    # v2: de_overpass CT gained "A Site". No CT role claimed the A bombsite, so
    # a player anchoring it scored nothing from their primary position.
    version=2,
    requires={ENRICHED_ROUNDS, "parsed:round_positions"},
    output_key="role_data",
    description="The position played on each side, from where the player fights and dies.",
)
def positional_role(ctx: MetricContext) -> dict[str, Any] | None:
    return _calculate_roles(
        ctx.enriched_rounds,
        ctx.map_name,
        ctx.parsed.get("round_positions", pd.DataFrame()),
        ctx.steam_id,
    )


def _classify_round_role(
    enriched_round: dict[str, Any],
    map_name: str,
    round_positions: list[tuple[str, int]] | None = None,
) -> dict[str, float]:
    """Classify the player's role for a single round based on positions.

    Uses sampled mid-round positions (time-weighted: first 30s = 3× weight)
    plus kill/death positions. Returns a dict of role_name → normalised
    score (0-1) for every role on this side.  Empty dict when no data.
    """
    side = enriched_round.get("side")
    if not side or map_name not in role_zones():
        return {}
    side_roles = roles_for(map_name, side)
    if not side_roles:
        return {}

    # Build weighted callout list from sampled positions + kill/death
    weighted: list[tuple[str, float]] = []

    # Sampled positions: (callout, tick_offset)
    _EARLY_CUTOFF = 1920  # 30s at 64-tick
    if round_positions:
        for callout, tick_offset in round_positions:
            if callout == "unknown":
                continue
            w = 3.0 if tick_offset <= _EARLY_CUTOFF else 1.0
            weighted.append((callout, w))

    # Kill positions (weight 4 — strongest signal of actual role)
    for k in enriched_round.get("kills_detail", []):
        p = k.get("attacker_position")
        if p and p != "unknown":
            weighted.append((p, 4.0))
    # Death position (weight 4)
    death = enriched_round.get("death_detail")
    if death:
        p = death.get("victim_position")
        if p and p != "unknown":
            weighted.append((p, 4.0))

    if not weighted:
        return {}

    # Score each role
    scores: dict[str, float] = {}
    for role_name, callouts in side_roles.items():
        callout_set = set(callouts)
        scores[role_name] = sum(w for c, w in weighted if c in callout_set)

    total = sum(scores.values())
    if total <= 0:
        return {}
    # Normalise to 0-1
    for k in scores:
        scores[k] = round(scores[k] / total, 3)
    return scores


def _calculate_roles(
    enriched_rounds: list[dict[str, Any]],
    map_name: str,
    round_positions_df: Any = None,
    steam_id: str = "",
) -> dict[str, Any] | None:
    """Classify roles for every round and produce a summary.

    Returns:
      { "map": str,
        "roles_ct": [role_name, ...],   -- ordered axes for spider chart
        "roles_t":  [role_name, ...],
        "rounds": [{"round": int, "side": str, "role": str|null,
                     "scores": {role: float}, ...}...],
        "ct_summary": {role: count},
        "t_summary":  {role: count},
        "ct_primary": str|null,
        "t_primary":  str|null }
    """
    if map_name not in role_zones():
        return None

    # Pre-index sampled positions per round for the target player
    round_pos_lookup: dict[int, list[tuple[str, int]]] = {}
    if (
        round_positions_df is not None
        and isinstance(round_positions_df, pd.DataFrame)
        and not round_positions_df.empty
        and steam_id
    ):
        sid = str(steam_id)
        mask = round_positions_df["steamid"] == sid
        player_pos = round_positions_df.loc[mask]
        for _, row in player_pos.iterrows():
            rnd = int(row.get("round", 0))
            x, y = row.get("X", 0), row.get("Y", 0)
            callout = get_callout(map_name, x, y)
            offset = int(row.get("tick_offset", 0))
            round_pos_lookup.setdefault(rnd, []).append((callout, offset))

    # Ordered role lists per side (stable axis order for spider chart)
    roles_ct = list(roles_for(map_name, "CT"))
    roles_t = list(roles_for(map_name, "T"))

    round_roles: list[dict[str, Any]] = []
    ct_counts: dict[str, int] = {}
    t_counts: dict[str, int] = {}

    for r in enriched_rounds:
        side = r.get("side")
        rnd = r["round"]
        sampled = round_pos_lookup.get(rnd)
        scores = _classify_round_role(r, map_name, sampled)

        role = max(scores, key=scores.get) if scores else None

        round_roles.append({
            "round": rnd,
            "side": side,
            "role": role,
            "scores": scores,
        })

        if role:
            if side == "CT":
                ct_counts[role] = ct_counts.get(role, 0) + 1
            elif side == "T":
                t_counts[role] = t_counts.get(role, 0) + 1

    ct_primary = max(ct_counts, key=ct_counts.get) if ct_counts else None
    t_primary = max(t_counts, key=t_counts.get) if t_counts else None

    return {
        "map": map_name,
        "roles_ct": roles_ct,
        "roles_t": roles_t,
        "rounds": round_roles,
        "ct_summary": ct_counts,
        "t_summary": t_counts,
        "ct_primary": ct_primary,
        "t_primary": t_primary,
    }
