"""Per-round frames for the 2D replay viewer.

By far the largest thing stored — roughly 40 KB a round, about 95% of the
database — and read by exactly one page.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.domain.metrics.context import MetricContext
from src.domain.metrics.registry import metric
from src.parser import _build_freeze_end_map


@metric(
    id="replay.frames",
    label="2D Replay",
    group="replay",
    version=1,
    requires={"parsed:replay_positions", "parsed:round_end", "parsed:player_death"},
    output_key="replay_data",
    description="Tick-sampled player positions per round, for the replay viewer.",
    # Keyed by round number; a version key would read as an extra round.
    version_in_blob=False,
)
def replay_frames(ctx: MetricContext) -> dict[int, dict]:
    return _build_replay_data(
        ctx.parsed,
        ctx.parsed.get("replay_positions", pd.DataFrame()),
        ctx.total_rounds,
    )


def _build_replay_data(
    parsed_data: dict[str, Any],
    replay_positions_df: Any,
    total_rounds: int,
) -> dict[int, dict]:
    """Convert replay position DataFrame into per-round frame structures.

    Returns ``{round_num: {"players": {...}, "frames": [...], "events": [...]}}``.
    """
    if (
        replay_positions_df is None
        or not isinstance(replay_positions_df, pd.DataFrame)
        or replay_positions_df.empty
    ):
        return {}

    # Build name mapping from death events
    death_df = parsed_data.get("player_death", pd.DataFrame())
    name_map: dict[str, str] = {}
    if not death_df.empty:
        for col_sid, col_name in [
            ("attacker_steamid", "attacker_name"),
            ("user_steamid", "user_name"),
        ]:
            if col_sid in death_df.columns and col_name in death_df.columns:
                for _, row in (
                    death_df[[col_sid, col_name]].drop_duplicates().iterrows()
                ):
                    sid = str(row.get(col_sid, ""))
                    nm = row.get(col_name, "")
                    if sid and nm:
                        name_map[sid] = nm

    # Build per-round kill event timeline (with tick offsets matching replay frames)
    round_end_df = parsed_data.get("round_end", pd.DataFrame())
    round_freeze_end_df = parsed_data.get("round_freeze_end", pd.DataFrame())
    round_start_ticks: dict[int, int] = {}
    if not round_end_df.empty and "tick" in round_end_df.columns:
        re_sorted = round_end_df.sort_values("round")
        end_ticks = re_sorted["tick"].values.tolist()
        round_nums = re_sorted["round"].values.tolist()
        # Use freeze-end ticks when available (same reference as replay frames)
        freeze_map = _build_freeze_end_map(round_freeze_end_df, end_ticks)
        for i, rnd in enumerate(round_nums):
            round_start_ticks[int(rnd)] = freeze_map.get(
                i, int(end_ticks[i - 1]) if i > 0 else 0
            )

    kill_events_by_round: dict[int, list] = {}
    # Build authoritative per-round team map from death AND hurt events.
    # Event-level team_num is always correct (recorded at event time),
    # unlike tick-sampled team_num which can be stale around halftime.
    event_team_map: dict[int, dict[str, int]] = {}  # {round: {steamid: team}}

    def _record_event_teams(df: pd.DataFrame, sid_team_pairs: list[tuple[str, str]]) -> None:
        """Extract team assignments from event DataFrame rows."""
        if df.empty or "round" not in df.columns:
            return
        for _, row in df.iterrows():
            rnd = int(row.get("round", 0))
            if rnd < 1:
                continue
            for sid_col, team_col in sid_team_pairs:
                if sid_col not in row.index or team_col not in row.index:
                    continue
                sid = str(row.get(sid_col, ""))
                if not sid:
                    continue
                try:
                    t = int(row[team_col])
                    if t in (2, 3):
                        event_team_map.setdefault(rnd, {})[sid] = t
                except (ValueError, TypeError):
                    pass

    # Gather teams from hurt events first (most numerous — covers almost everyone)
    hurt_df = parsed_data.get("player_hurt", pd.DataFrame())
    _record_event_teams(hurt_df, [
        ("attacker_steamid", "attacker_team_num"),
        ("user_steamid", "user_team_num"),
    ])

    # Then from death events (also builds kill timeline)
    if not death_df.empty and "tick" in death_df.columns and "round" in death_df.columns:
        for _, row in death_df.iterrows():
            rnd = int(row.get("round", 0))
            tick = int(row.get("tick", 0))
            start = round_start_ticks.get(rnd, 0)
            kill_events_by_round.setdefault(rnd, []).append({
                "t": tick - start,
                "type": "kill",
                "attacker": str(row.get("attacker_steamid", "")),
                "victim": str(row.get("user_steamid", "")),
                "weapon": row.get("weapon", ""),
                "headshot": bool(row.get("headshot", False)),
            })
            for sid_col, team_col in [
                ("attacker_steamid", "attacker_team_num"),
                ("user_steamid", "user_team_num"),
            ]:
                sid = str(row.get(sid_col, ""))
                if sid and team_col in row.index:
                    try:
                        t = int(row[team_col])
                        if t in (2, 3):
                            event_team_map.setdefault(rnd, {})[sid] = t
                    except (ValueError, TypeError):
                        pass

    # Build per-round grenade event timeline (flash/he/smoke/molotov with positions)
    grenade_events_by_round: dict[int, list] = {}
    _grenade_sources = [
        ("flash_detonate", "flash"),
        ("he_detonate", "he"),
        ("smoke_detonate", "smoke"),
        ("molotov_detonate", "molotov"),
    ]
    for data_key, nade_type in _grenade_sources:
        nade_df = parsed_data.get(data_key, pd.DataFrame())
        if nade_df.empty or "tick" not in nade_df.columns or "round" not in nade_df.columns:
            continue
        for _, row in nade_df.iterrows():
            rnd = int(row.get("round", 0))
            tick = int(row.get("tick", 0))
            start = round_start_ticks.get(rnd, 0)
            if rnd < 1:
                continue
            ev: dict[str, Any] = {
                "t": tick - start,
                "type": "grenade",
                "grenade": nade_type,
            }
            # Position (game coordinates — converted to pixel in API)
            for coord in ("x", "y"):
                if coord in row.index:
                    try:
                        ev[coord] = float(row[coord])
                    except (ValueError, TypeError):
                        pass
            # Thrower info (not available for inferno_startburn)
            sid_col = "user_steamid" if "user_steamid" in row.index else None
            if sid_col:
                ev["thrower"] = str(row.get(sid_col, ""))
            grenade_events_by_round.setdefault(rnd, []).append(ev)

    # Detect halftime round (standard MR12 = round 12, but could differ).
    # Find the round where a player's event-team flips compared to the previous
    # round — that boundary marks the side swap.
    halftime_round = 12  # default
    for check_rnd in sorted(event_team_map.keys()):
        prev_rnd = check_rnd - 1
        if prev_rnd not in event_team_map:
            continue
        overlap = set(event_team_map[check_rnd]) & set(event_team_map[prev_rnd])
        flipped = sum(
            1 for s in overlap
            if event_team_map[check_rnd][s] != event_team_map[prev_rnd][s]
        )
        if flipped >= 3:  # majority of overlapping players swapped
            halftime_round = prev_rnd
            break

    result: dict[int, dict] = {}
    for rnd in range(1, total_rounds + 1):
        round_df = replay_positions_df[replay_positions_df["round"] == rnd]
        if round_df.empty:
            continue

        # Build player roster with team assignment.
        # Priority: (1) event team_num for this round (authoritative),
        #           (2) event team from nearest round in SAME HALF,
        #           (3) tick-sampled mode (fallback).
        round_event_teams = event_team_map.get(rnd, {})
        # Determine valid range for nearby-round search (stay in same half)
        if rnd <= halftime_round:
            search_lo, search_hi = 1, halftime_round
        else:
            search_lo, search_hi = halftime_round + 1, total_rounds

        players: dict[str, dict] = {}
        for sid in round_df["steamid"].unique():
            team = round_event_teams.get(sid, 0)
            if team == 0:
                # Search ALL rounds within the same half (closest first)
                for delta in range(1, search_hi - search_lo + 1):
                    for nearby in (rnd - delta, rnd + delta):
                        if search_lo <= nearby <= search_hi:
                            if nearby in event_team_map and sid in event_team_map[nearby]:
                                team = event_team_map[nearby][sid]
                                break
                    if team:
                        break
            if team == 0:
                # Try the OTHER half and flip the team (2↔3) since sides swapped
                if rnd <= halftime_round:
                    alt_lo, alt_hi = halftime_round + 1, total_rounds
                else:
                    alt_lo, alt_hi = 1, halftime_round
                for alt_rnd in range(alt_lo, alt_hi + 1):
                    if alt_rnd in event_team_map and sid in event_team_map[alt_rnd]:
                        other = event_team_map[alt_rnd][sid]
                        team = 3 if other == 2 else 2
                        break
            if team == 0:
                # Final fallback: tick-sampled mode
                sid_rows = round_df[round_df["steamid"] == sid]
                if "team_num" in sid_rows.columns:
                    try:
                        teams = sid_rows["team_num"].dropna()
                        if not teams.empty:
                            team = int(teams.mode().iloc[0])
                    except (ValueError, TypeError):
                        team = 0
            players[sid] = {
                "name": name_map.get(sid, sid[:8]),
                "team": team,
            }

        # Team composition inference: if one side has 4 and the other has
        # 5+, any team=0 players belong to the short side.
        team_counts = {2: 0, 3: 0}
        unknowns = []
        for sid, info in players.items():
            if info["team"] in (2, 3):
                team_counts[info["team"]] += 1
            else:
                unknowns.append(sid)
        if unknowns:
            short_team = 2 if team_counts[2] < team_counts[3] else 3
            for sid in unknowns:
                players[sid]["team"] = short_team

        # Build frames (sorted by tick offset)
        frames: list[list] = []
        for tick_offset in sorted(round_df["tick_offset"].unique()):
            tick_df = round_df[round_df["tick_offset"] == tick_offset]
            positions: dict[str, list] = {}
            for _, row in tick_df.iterrows():
                sid = row["steamid"]
                try:
                    hp = int(row["health"]) if "health" in row.index else 100
                except (ValueError, TypeError):
                    hp = 0
                try:
                    x = float(row["X"])
                    y = float(row["Y"])
                except (ValueError, TypeError):
                    continue
                if x != x or y != y:  # NaN check
                    continue
                positions[sid] = [round(x, 1), round(y, 1), hp]
            frames.append([int(tick_offset), positions])

        result[rnd] = {
            "players": players,
            "frames": frames,
            "events": kill_events_by_round.get(rnd, [])
                + grenade_events_by_round.get(rnd, []),
        }

    return result
