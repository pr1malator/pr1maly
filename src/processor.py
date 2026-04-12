"""
Layer 2: Metrics Processor
Filters raw event DataFrames for a specific Steam ID and calculates advanced
CS2 statistics: KPR, DPR, ADR, KAST, Impact, and an approximated HLTV 2.0
Rating.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.callouts import get_callout, is_map_supported
from src.parser import _build_freeze_end_map


# ---------------------------------------------------------------------------
# HLTV 2.0 Rating formula coefficients (publicly documented approximation).
# Source: https://www.hltv.org/news/20695/introducing-rating-20
# ---------------------------------------------------------------------------
_HLTV_COEFFICIENTS = {
    "kast_weight": 0.0073,
    "kpr_weight": 0.3591,
    "dpr_weight": -0.5329,
    "impact_weight": 0.2372,
    "adr_weight": 0.0032,
    "intercept": 0.1587,
}


def calculate_match_stats(
    parsed_data: dict[str, Any],
    steam_id: str,
) -> dict[str, Any]:
    """
    Calculate full match statistics for a player identified by *steam_id*.

    Args:
        parsed_data: Output dict from :func:`src.parser.parse_demo`.
        steam_id: The player's 64-bit Steam ID (as a string).

    Returns:
        A dict with keys:
            - ``player_name``: Detected player name.
            - ``map_name``: Map played.
            - ``total_rounds``: Number of rounds played.
            - ``kills``, ``deaths``, ``assists``: Integer counts.
            - ``kpr``, ``dpr``, ``adr``, ``kast``, ``impact``: Per-round /
              percentage metrics.
            - ``hltv_rating``: Approximated HLTV 2.0 rating.
            - ``round_stats``: List of per-round stat dicts.
    """
    death_df: pd.DataFrame = parsed_data.get("player_death", pd.DataFrame())
    hurt_df: pd.DataFrame = parsed_data.get("player_hurt", pd.DataFrame())
    round_end_df: pd.DataFrame = parsed_data.get("round_end", pd.DataFrame())
    header: dict[str, Any] = parsed_data.get("header", {})

    map_name: str = str(header.get("map_name", "unknown"))
    total_rounds: int = _count_total_rounds(round_end_df)

    if total_rounds == 0:
        total_rounds = 1  # avoid division by zero for malformed demos

    # ------------------------------------------------------------------ #
    # Kills and assists                                                    #
    # ------------------------------------------------------------------ #
    player_kills_df = _filter_attacker(death_df, steam_id)
    player_assists_df = _filter_assister(death_df, steam_id)
    player_deaths_df = _filter_victim(death_df, steam_id)

    kills: int = len(player_kills_df)
    deaths: int = len(player_deaths_df)
    assists: int = _count_valid_assists(player_assists_df, hurt_df, steam_id)

    player_name: str = _detect_player_name(death_df, steam_id)

    # ------------------------------------------------------------------ #
    # Damage (ADR)                                                         #
    # ------------------------------------------------------------------ #
    total_damage: int = _calculate_damage(hurt_df, steam_id)

    # ------------------------------------------------------------------ #
    # Per-round aggregates                                                 #
    # ------------------------------------------------------------------ #
    round_stats: list[dict[str, Any]] = _build_round_stats(
        death_df, hurt_df, steam_id, total_rounds
    )

    # ------------------------------------------------------------------ #
    # KAST (Kill, Assist, Survived or Traded in the round)                #
    # ------------------------------------------------------------------ #
    kast_rounds: int = _calculate_kast_rounds(round_stats)

    # ------------------------------------------------------------------ #
    # Derived metrics                                                      #
    # ------------------------------------------------------------------ #
    kpr: float = round(kills / total_rounds, 4)
    dpr: float = round(deaths / total_rounds, 4)
    adr: float = round(total_damage / total_rounds, 4)
    kast: float = round(kast_rounds / total_rounds * 100, 2)
    impact: float = round(
        2.13 * kpr + 0.42 * (assists / total_rounds) - 0.41, 4
    )
    hltv_rating: float = _compute_hltv_rating(kast, kpr, dpr, impact, adr)

    # ------------------------------------------------------------------ #
    # K/D ratio                                                            #
    # ------------------------------------------------------------------ #
    kd_ratio: float = round(kills / deaths, 2) if deaths > 0 else float(kills)

    # ------------------------------------------------------------------ #
    # Multi-kill rounds (2K, 3K, 4K, 5K)                                  #
    # ------------------------------------------------------------------ #
    multikills = _count_multikill_rounds(round_stats)

    # ------------------------------------------------------------------ #
    # Match score & result                                                 #
    # ------------------------------------------------------------------ #
    score = _calculate_match_score(round_end_df, death_df, steam_id)

    enriched_rounds = build_enriched_rounds(parsed_data, steam_id, total_rounds)

    # ------------------------------------------------------------------ #
    # Aim & Movement aggregate stats                                       #
    # ------------------------------------------------------------------ #
    aim_stats = _calculate_aim_stats(enriched_rounds)

    # ------------------------------------------------------------------ #
    # Utility & Economics                                                  #
    # ------------------------------------------------------------------ #
    utility_data = _calculate_utility_stats(
        enriched_rounds, parsed_data, steam_id, total_rounds, map_name,
    )

    # ------------------------------------------------------------------ #
    # Role classification                                                  #
    # ------------------------------------------------------------------ #
    round_positions_df = parsed_data.get("round_positions", pd.DataFrame())
    role_data = _calculate_roles(enriched_rounds, map_name, round_positions_df, steam_id)

    # ------------------------------------------------------------------ #
    # Benchmark tier classifications                                       #
    # ------------------------------------------------------------------ #
    benchmarks = compute_benchmarks(aim_stats, utility_data, total_rounds, map_name)

    # ------------------------------------------------------------------ #
    # 2D replay data (per-round player positions for replay viewer)        #
    # ------------------------------------------------------------------ #
    replay_positions_df = parsed_data.get("replay_positions", pd.DataFrame())
    replay_data = _build_replay_data(
        parsed_data, replay_positions_df, total_rounds,
    )

    return {
        "player_name": player_name,
        "map_name": map_name,
        "total_rounds": total_rounds,
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "kpr": kpr,
        "dpr": dpr,
        "adr": adr,
        "kast": kast,
        "impact": impact,
        "hltv_rating": hltv_rating,
        "kd_ratio": kd_ratio,
        "rounds_2k": multikills[2],
        "rounds_3k": multikills[3],
        "rounds_4k": multikills[4],
        "rounds_5k": multikills[5],
        "team_score": score["team_score"],
        "enemy_score": score["enemy_score"],
        "match_result": score["result"],
        "round_stats": round_stats,
        "enriched_rounds": enriched_rounds,
        "aim_stats": aim_stats,
        "role_data": role_data,
        "utility_data": utility_data,
        "benchmarks": benchmarks,
        "replay_data": replay_data,
        "all_players": calculate_all_players_stats(
            parsed_data, steam_id, total_rounds
        ),
    }


# ---------------------------------------------------------------------------
# 2D Replay data builder
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Aim & Movement aggregate stats
# ---------------------------------------------------------------------------


# Weapons with low movement-inaccuracy penalty — running kills with these are
# expected and should not count against the player's movement discipline score.
_LOW_PENALTY_WEAPONS: set[str] = {
    # SMGs
    "MAC-10", "MP9", "MP7", "MP5-SD", "UMP-45", "P90", "PP-Bizon",
    # Shotguns
    "MAG-7", "Sawed-Off", "Nova", "XM1014",
    # Pistols
    "Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9",
    "CZ75-Auto", "Dual Berettas", "Desert Eagle", "R8 Revolver",
    # Machine guns
    "M249", "Negev",
    # Melee / utility (not penalised)
    "Knife",
}


def _calculate_aim_stats(enriched_rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-kill movement, pre-aim, and TTK data into match-level stats.

    Returns a dict with:
      - movement: {speeds: [...], weapons: [...], avg, min, max, standing_pct,
                   counterstrafed_pct, running_pct, low_penalty_weapons: [...]}
      - preaim: {errors: [...], avg, min, max, excellent_pct, good_pct, moderate_pct, poor_pct}
      - ttk: {values: [...], avg, min, max}  (seconds)
      - aim_rating: 0-100 approximate overall aim quality
    """
    shot_speeds: list[float] = []
    kill_weapons: list[str] = []
    movement_qualities: list[str] = []
    preaim_errors: list[float] = []
    preaim_qualities: list[str] = []
    ttk_values: list[float] = []
    ttk_shots: list[int] = []
    ttk_hits: list[int] = []
    reaction_values: list[float] = []
    reaction_categories: list[str] = []

    # Accuracy per-encounter
    accuracy_values: list[float] = []    # hit_pct per engagement
    first_bullet_hits: list[bool] = []
    hitgroup_head: int = 0
    hitgroup_upper: int = 0
    hitgroup_lower: int = 0
    hitgroup_total: int = 0

    # Per-data-point encounter outcomes: "kill" | "death" | "damage"
    outcomes_mov: list[str] = []
    outcomes_preaim: list[str] = []
    outcomes_ttk: list[str] = []
    outcomes_rxn: list[str] = []
    outcomes_acc: list[str] = []

    # Per-encounter objects for the 2D scatter plot (each has whichever KPIs are available)
    encounters: list[dict[str, Any]] = []

    for r in enriched_rounds:
        died_this_round = r.get("death_detail") is not None

        for k in r.get("kills_detail", []):
            outcome = "death" if died_this_round else "kill"
            weapon = k.get("weapon", "")
            enc: dict[str, Any] = {"outcome": outcome}

            mv = k.get("movement")
            if mv:
                shot_speeds.append(mv["shot_speed"])
                movement_qualities.append(mv["movement_quality"])
                kill_weapons.append(weapon)
                outcomes_mov.append(outcome)
                enc["movement"] = mv["shot_speed"]

            pa = k.get("preaim")
            if pa:
                preaim_errors.append(pa["crosshair_error"])
                preaim_qualities.append(pa["preaim_quality"])
                outcomes_preaim.append(outcome)
                enc["preaim"] = pa["crosshair_error"]

            ttd = k.get("ttd")
            if ttd and ttd.get("ttk_seconds", 0) > 0:
                ttk_values.append(ttd["ttk_seconds"])
                ttk_shots.append(ttd.get("shots_fired", ttd.get("hits", 1)))
                ttk_hits.append(ttd.get("hits", 1))
                outcomes_ttk.append(outcome)
                enc["ttk"] = ttd["ttk_seconds"]

            rxn = k.get("reaction")
            if rxn and rxn.get("reaction_ms") is not None:
                reaction_values.append(rxn["reaction_ms"])
                reaction_categories.append(rxn["category"])
                outcomes_rxn.append(outcome)
                enc["reaction"] = rxn["reaction_ms"]

            # Accuracy (from ttd sub-dict)
            acc = ttd.get("accuracy") if ttd else None
            if acc and acc.get("hit_pct") is not None:
                accuracy_values.append(acc["hit_pct"])
                first_bullet_hits.append(acc["first_bullet_hit"])
                hitgroup_head += acc.get("head", 0)
                hitgroup_upper += acc.get("upper", 0)
                hitgroup_lower += acc.get("lower", 0)
                hitgroup_total += acc.get("head", 0) + acc.get("upper", 0) + acc.get("lower", 0)
                outcomes_acc.append(outcome)
                enc["accuracy"] = acc["hit_pct"]

            encounters.append(enc)

        # Damage-only encounters (hurt enemy but did not kill)
        for d in r.get("damage_encounters", []):
            enc = {"outcome": "damage"}

            mv = d.get("movement")
            if mv:
                shot_speeds.append(mv["shot_speed"])
                movement_qualities.append(mv["movement_quality"])
                kill_weapons.append(d.get("weapon", ""))
                outcomes_mov.append("damage")
                enc["movement"] = mv["shot_speed"]

            pa = d.get("preaim")
            if pa:
                preaim_errors.append(pa["crosshair_error"])
                preaim_qualities.append(pa["preaim_quality"])
                outcomes_preaim.append("damage")
                enc["preaim"] = pa["crosshair_error"]

            encounters.append(enc)

    n_mov = len(movement_qualities)
    n_aim = len(preaim_qualities)

    movement = {}
    if shot_speeds:
        standing = sum(1 for q in movement_qualities if q == "standing")
        cs = sum(1 for q in movement_qualities if q == "counter-strafed")
        running = sum(1 for q in movement_qualities if q == "running")
        # Identify which kills used low-penalty weapons
        low_penalty_flags = [
            w in _LOW_PENALTY_WEAPONS for w in kill_weapons
        ]
        running_low = sum(
            1 for q, lp in zip(movement_qualities, low_penalty_flags)
            if q == "running" and lp
        )
        movement = {
            "speeds": [round(s, 1) for s in shot_speeds],
            "weapons": kill_weapons,
            "low_penalty": low_penalty_flags,
            "outcomes": outcomes_mov,
            "avg": round(sum(shot_speeds) / len(shot_speeds), 1),
            "min": round(min(shot_speeds), 1),
            "max": round(max(shot_speeds), 1),
            "standing_pct": round(standing / n_mov * 100, 1) if n_mov else 0,
            "counterstrafed_pct": round(cs / n_mov * 100, 1) if n_mov else 0,
            "running_pct": round(running / n_mov * 100, 1) if n_mov else 0,
            "running_total": running,
            "running_low_penalty": running_low,
        }

    preaim = {}
    if preaim_errors:
        exc = sum(1 for q in preaim_qualities if q == "excellent")
        good = sum(1 for q in preaim_qualities if q == "good")
        mod = sum(1 for q in preaim_qualities if q == "moderate")
        poor = sum(1 for q in preaim_qualities if q == "poor")
        preaim = {
            "errors": [round(e, 1) for e in preaim_errors],
            "outcomes": outcomes_preaim,
            "avg": round(sum(preaim_errors) / len(preaim_errors), 1),
            "min": round(min(preaim_errors), 1),
            "max": round(max(preaim_errors), 1),
            "excellent_pct": round(exc / n_aim * 100, 1) if n_aim else 0,
            "good_pct": round(good / n_aim * 100, 1) if n_aim else 0,
            "moderate_pct": round(mod / n_aim * 100, 1) if n_aim else 0,
            "poor_pct": round(poor / n_aim * 100, 1) if n_aim else 0,
        }

    ttk = {}
    if ttk_values:
        total_shots = sum(ttk_shots)
        total_hits = sum(ttk_hits)
        ttk = {
            "values": [round(v, 3) for v in ttk_values],
            "outcomes": outcomes_ttk,
            "avg": round(sum(ttk_values) / len(ttk_values), 3),
            "min": round(min(ttk_values), 3),
            "max": round(max(ttk_values), 3),
            "total_shots": total_shots,
            "total_hits": total_hits,
            "accuracy_pct": round(total_hits / total_shots * 100, 1) if total_shots else 0,
        }

    reaction = {}
    if reaction_values:
        n_rxn = len(reaction_values)
        lightning = sum(1 for c in reaction_categories if c == "lightning")
        fast = sum(1 for c in reaction_categories if c == "fast")
        average = sum(1 for c in reaction_categories if c == "average")
        slow = sum(1 for c in reaction_categories if c == "slow")
        reaction = {
            "values": [round(v, 1) for v in reaction_values],
            "outcomes": outcomes_rxn,
            "avg": round(sum(reaction_values) / n_rxn, 1),
            "min": round(min(reaction_values), 1),
            "max": round(max(reaction_values), 1),
            "lightning_pct": round(lightning / n_rxn * 100, 1),
            "fast_pct": round(fast / n_rxn * 100, 1),
            "average_pct": round(average / n_rxn * 100, 1),
            "slow_pct": round(slow / n_rxn * 100, 1),
        }

    accuracy = {}
    if accuracy_values:
        n_acc = len(accuracy_values)
        fb_hit = sum(1 for fb in first_bullet_hits if fb)
        accuracy = {
            "values": [round(v, 1) for v in accuracy_values],
            "outcomes": outcomes_acc,
            "avg": round(sum(accuracy_values) / n_acc, 1),
            "min": round(min(accuracy_values), 1),
            "max": round(max(accuracy_values), 1),
            "first_bullet_pct": round(fb_hit / n_acc * 100, 1) if n_acc else 0,
            "head_pct": round(hitgroup_head / hitgroup_total * 100, 1) if hitgroup_total else 0,
            "upper_pct": round(hitgroup_upper / hitgroup_total * 100, 1) if hitgroup_total else 0,
            "lower_pct": round(hitgroup_lower / hitgroup_total * 100, 1) if hitgroup_total else 0,
            "head_count": hitgroup_head,
            "upper_count": hitgroup_upper,
            "lower_count": hitgroup_lower,
        }

    # Approximate aim rating (0-100)
    # Weighted: 30% crosshair placement, 25% movement, 25% TTK, 20% reaction
    aim_rating = 50.0  # baseline
    if preaim_errors or movement_qualities or ttk_values or reaction_values:
        aim_rating = 0.0
        # 30% crosshair placement: 0° = 100, 20°+ = 0
        if preaim_errors:
            avg_err = sum(preaim_errors) / len(preaim_errors)
            cp_score = max(0.0, min(100.0, 100.0 - avg_err * 5.0))
        else:
            cp_score = 50.0
        aim_rating += cp_score * 0.30
        # 25% movement discipline: non-running %
        if n_mov:
            good_mov = sum(1 for q in movement_qualities if q != "running")
            mv_score = (good_mov / n_mov) * 100.0
        else:
            mv_score = 50.0
        aim_rating += mv_score * 0.25
        # 25% TTK efficiency: 0.15s = 100, 0.8s+ = 0
        if ttk_values:
            avg_ttk = sum(ttk_values) / len(ttk_values)
            ttk_score = max(0.0, min(100.0, (0.8 - avg_ttk) / 0.65 * 100.0))
        else:
            ttk_score = 50.0
        aim_rating += ttk_score * 0.25
        # 20% reaction time: 150ms = 100, 500ms+ = 0
        if reaction_values:
            avg_rxn = sum(reaction_values) / len(reaction_values)
            rxn_score = max(0.0, min(100.0, (500.0 - avg_rxn) / 350.0 * 100.0))
        else:
            rxn_score = 50.0
        aim_rating += rxn_score * 0.20
    aim_rating = round(min(100, max(0, aim_rating)), 1)

    return {
        "movement": movement,
        "preaim": preaim,
        "ttk": ttk,
        "reaction": reaction,
        "accuracy": accuracy,
        "aim_rating": aim_rating,
        "encounters": encounters,
    }


# ---------------------------------------------------------------------------
# Benchmarks — tier classification for match metrics
# ---------------------------------------------------------------------------

# Each benchmark defines thresholds for 4 tiers.  For "lower is better" metrics
# the tiers are ordered high→low (first threshold is the *best* ceiling).
# For "higher is better" metrics the tiers are low→high (first threshold is
# the *best* floor).
#
# Tier labels: "pro", "high_amateur", "average", "below_average"

# Map-specific enemies-flashed benchmarks (24-round base).
_FLASH_BENCHMARKS: dict[str, tuple[int, int, int]] = {
    # (high_amateur_floor, average_floor, below_average_ceiling)
    # Pro ≥ high_amateur_floor+x depending on map, but we use ranges:
    # Pro: >= t1, High Amateur: >= t2, Average: >= t3, Below Average: < t3
    "de_dust2":   (14, 8, 3),   # Pro 14-22, HA 8-14, Avg 3-7, BA <3
    "de_inferno": (9, 6, 3),    # Pro 9-15, HA 6-9, Avg 2-5, BA <3
    # Default for maps not specifically listed
    "_default":   (10, 6, 3),
}


def _classify_tier_lower_better(value: float, pro_max: float, ha_max: float, avg_max: float) -> str:
    """Classify where lower values are better (speed, offset, times)."""
    if value <= pro_max:
        return "pro"
    if value <= ha_max:
        return "high_amateur"
    if value <= avg_max:
        return "average"
    return "below_average"


def _classify_tier_higher_better(value: float, pro_min: float, ha_min: float, avg_min: float) -> str:
    """Classify where higher values are better (flash count, damage)."""
    if value >= pro_min:
        return "pro"
    if value >= ha_min:
        return "high_amateur"
    if value >= avg_min:
        return "average"
    return "below_average"


def compute_benchmarks(
    aim_stats: dict[str, Any],
    utility_data: dict[str, Any],
    total_rounds: int,
    map_name: str,
) -> dict[str, Any]:
    """Compute benchmark tier labels for key metrics.

    Returns a dict of metric_key → {value, tier, label, unit} where tier is
    one of "pro", "high_amateur", "average", "below_average".
    """
    benchmarks: dict[str, Any] = {}
    # Normalisation factor: benchmarks assume a 24-round (MR12) map
    norm = 24 / max(total_rounds, 1)

    # --- Utility benchmarks ---
    if utility_data:
        # Enemies Flashed / Map (normalised to 24 rounds)
        fl = utility_data.get("flash", {})
        enemies_flashed = fl.get("enemies_flashed", 0)
        flashed_norm = round(enemies_flashed * norm, 1)
        thresholds = _FLASH_BENCHMARKS.get(map_name, _FLASH_BENCHMARKS["_default"])
        benchmarks["enemies_flashed"] = {
            "value": flashed_norm,
            "raw": enemies_flashed,
            "tier": _classify_tier_higher_better(flashed_norm, thresholds[0], thresholds[1], thresholds[2]),
            "unit": "per map",
        }

        # $ Wasted on Utility (% of total utility spend)
        eco = utility_data.get("economics", {})
        total_spent = eco.get("total_spent", 0)
        total_wasted = eco.get("total_wasted", 0)
        waste_pct = round(total_wasted / total_spent * 100, 1) if total_spent > 0 else 0
        benchmarks["utility_waste_pct"] = {
            "value": waste_pct,
            "tier": _classify_tier_lower_better(waste_pct, 12, 22, 40),
            "unit": "%",
        }

        # Utility Damage / Map (HE + Molotov, normalised to 24 rounds)
        he_dmg = utility_data.get("he", {}).get("total_damage", 0)
        molly_dmg = utility_data.get("molotov", {}).get("total_damage", 0)
        util_dmg = he_dmg + molly_dmg
        util_dmg_norm = round(util_dmg * norm, 1)
        benchmarks["utility_damage"] = {
            "value": util_dmg_norm,
            "raw": util_dmg,
            "tier": _classify_tier_higher_better(util_dmg_norm, 150, 80, 25),
            "unit": "HP per map",
        }

    # --- Aim benchmarks ---
    if aim_stats:
        # Speed When Shooting (u/s) — lower is better
        mv = aim_stats.get("movement", {})
        if mv.get("avg") is not None:
            benchmarks["shot_speed"] = {
                "value": mv["avg"],
                "tier": _classify_tier_lower_better(mv["avg"], 15, 40, 100),
                "unit": "u/s",
            }

        # Counterstrafe Quality — using avg shot speed as proxy (lower = better stop)
        # Thresholds: Pro < 50, High Amateur 50–100, Average 100–200, Below Avg > 200
        if mv.get("avg") is not None:
            benchmarks["counterstrafe"] = {
                "value": mv["avg"],
                "tier": _classify_tier_lower_better(mv["avg"], 50, 100, 200),
                "unit": "u/s",
            }

        # Pre-Aim Offset (degrees) — lower is better
        pa = aim_stats.get("preaim", {})
        if pa.get("avg") is not None:
            benchmarks["preaim_offset"] = {
                "value": pa["avg"],
                "tier": _classify_tier_lower_better(pa["avg"], 3, 10, 25),
                "unit": "°",
            }

        # Reaction Time (ms) — lower is better
        rxn = aim_stats.get("reaction", {})
        if rxn.get("avg") is not None:
            benchmarks["reaction_time"] = {
                "value": rxn["avg"],
                "tier": _classify_tier_lower_better(rxn["avg"], 180, 230, 320),
                "unit": "ms",
            }

        # Engagement Time to Kill (ms) — lower is better
        ttk = aim_stats.get("ttk", {})
        if ttk.get("avg") is not None:
            ttk_ms = round(ttk["avg"] * 1000, 0)
            benchmarks["engagement_ttk"] = {
                "value": ttk_ms,
                "tier": _classify_tier_lower_better(ttk_ms, 400, 650, 1100),
                "unit": "ms",
            }

    return benchmarks


# ---------------------------------------------------------------------------
# Utility & Economics
# ---------------------------------------------------------------------------

# Internal grenade key → cost in CS2
_GRENADE_ITEMS: dict[str, int] = {
    "flashbang": 200,
    "smokegrenade": 300,
    "hegrenade": 300,
    "molotov": 400,
    "incgrenade": 400,
    "decoy": 50,
}

_GRENADE_DISPLAY: dict[str, str] = {
    "flashbang": "Flash",
    "smokegrenade": "Smoke",
    "hegrenade": "HE",
    "molotov": "Molotov",
    "incgrenade": "Incendiary",
    "decoy": "Decoy",
}

# demoparser2 item_purchase "item_name" display values → internal key
_PURCHASE_NAME_MAP: dict[str, str] = {
    "flashbang": "flashbang",
    "smoke grenade": "smokegrenade",
    "high explosive grenade": "hegrenade",
    "molotov": "molotov",
    "incendiary grenade": "incgrenade",
    "decoy grenade": "decoy",
}

# demoparser2 weapon_fire "weapon" values → internal key
_WEAPON_NAME_MAP: dict[str, str] = {
    "weapon_flashbang": "flashbang",
    "weapon_smokegrenade": "smokegrenade",
    "weapon_hegrenade": "hegrenade",
    "weapon_molotov": "molotov",
    "weapon_incgrenade": "incgrenade",
    "weapon_decoy": "decoy",
}

# Weapon slot classification for teamplayer drop detection.
# item_purchase "item_name" (lowered) → slot name.
# If a player buys more than 1 item in the same slot per round,
# the extras were dropped for teammates.
_WEAPON_SLOT: dict[str, str] = {
    # Primaries
    "ak-47": "primary", "ak47": "primary",
    "m4a1-s": "primary", "m4a1": "primary", "m4a1_silencer": "primary",
    "m4a4": "primary",
    "awp": "primary",
    "galil ar": "primary", "galilar": "primary",
    "famas": "primary",
    "sg 553": "primary", "sg556": "primary",
    "aug": "primary",
    "ssg 08": "primary", "ssg08": "primary",
    "scar-20": "primary", "scar20": "primary",
    "g3sg1": "primary",
    "mac-10": "primary", "mac10": "primary",
    "mp9": "primary",
    "mp7": "primary",
    "mp5-sd": "primary", "mp5sd": "primary",
    "ump-45": "primary", "ump45": "primary",
    "p90": "primary",
    "pp-bizon": "primary", "bizon": "primary",
    "nova": "primary",
    "xm1014": "primary",
    "mag-7": "primary", "mag7": "primary",
    "sawed-off": "primary", "sawedoff": "primary",
    "m249": "primary",
    "negev": "primary",
    # Pistols
    "glock-18": "secondary", "glock": "secondary",
    "usp-s": "secondary", "usp_silencer": "secondary",
    "p2000": "secondary", "hkp2000": "secondary",
    "p250": "secondary",
    "five-seven": "secondary", "fiveseven": "secondary",
    "tec-9": "secondary", "tec9": "secondary",
    "cz75-auto": "secondary", "cz75a": "secondary",
    "dual berettas": "secondary", "elite": "secondary",
    "desert eagle": "secondary", "deagle": "secondary",
    "r8 revolver": "secondary", "revolver": "secondary",
}


def _calculate_utility_stats(
    enriched_rounds: list[dict[str, Any]],
    parsed_data: dict[str, Any],
    steam_id: str,
    total_rounds: int,
    map_name: str,
) -> dict[str, Any]:
    """Aggregate utility economics and efficiency across all rounds.

    Returns a dict with:
      - economics: purchase/use/waste tracking per grenade type
      - efficiency: flash, HE, molotov impact metrics
      - smokes: spatial impact assessment (zone coverage)
      - per_round: per-round utility breakdown
      - utility_rating: 0-100 overall utility score
    """
    sid = str(steam_id)
    blind_df = parsed_data.get("player_blind", pd.DataFrame())
    hurt_df = parsed_data.get("player_hurt", pd.DataFrame())
    purchase_df = parsed_data.get("item_purchase", pd.DataFrame())
    weapon_fire_df = parsed_data.get("weapon_fire", pd.DataFrame())
    smoke_det_df = parsed_data.get("smoke_detonate", pd.DataFrame())
    molotov_det_df = parsed_data.get("molotov_detonate", pd.DataFrame())

    # ------------------------------------------------------------------ #
    # T1: Economics — bought vs. thrown vs. wasted                         #
    # ------------------------------------------------------------------ #
    bought: dict[str, int] = {g: 0 for g in _GRENADE_ITEMS}
    thrown: dict[str, int] = {g: 0 for g in _GRENADE_ITEMS}

    # CS2 per-round grenade carry limits
    _MAX_PER_ROUND: dict[str, int] = {
        "flashbang": 2, "smokegrenade": 1, "hegrenade": 1,
        "molotov": 1, "incgrenade": 1, "decoy": 1,
    }

    # Count purchases (deduplicated: cap at carry limit per round)
    if not purchase_df.empty:
        id_col = _find_id_col(purchase_df, ("steamid", "attacker_steamid", "user_steamid"))
        if id_col:
            name_col = "item_name" if "item_name" in purchase_df.columns else (
                "weapon" if "weapon" in purchase_df.columns else None
            )
            if name_col and "round" in purchase_df.columns:
                player_buys = purchase_df[purchase_df[id_col].astype(str) == sid]
                for rnd_num in player_buys["round"].unique():
                    rnd_buys = player_buys[player_buys["round"] == rnd_num]
                    rnd_counts: dict[str, int] = {}
                    for _, row in rnd_buys.iterrows():
                        raw = str(row[name_col]).lower()
                        key = _PURCHASE_NAME_MAP.get(raw)
                        if key is None:
                            key = raw.replace("weapon_", "")
                        if key in _GRENADE_ITEMS:
                            rnd_counts[key] = rnd_counts.get(key, 0) + 1
                    for key, cnt in rnd_counts.items():
                        bought[key] += min(cnt, _MAX_PER_ROUND.get(key, 1))

    # Count throws from weapon_fire
    if not weapon_fire_df.empty:
        id_col = _find_id_col(weapon_fire_df, ("user_steamid", "steamid", "attacker_steamid"))
        if id_col:
            wep_col = "weapon" if "weapon" in weapon_fire_df.columns else None
            if wep_col:
                fires = weapon_fire_df[weapon_fire_df[id_col].astype(str) == sid]
                for _, row in fires.iterrows():
                    raw = str(row[wep_col]).lower()
                    key = _WEAPON_NAME_MAP.get(raw)
                    if key is None:
                        key = raw.replace("weapon_", "")
                    if key in _GRENADE_ITEMS:
                        thrown[key] += 1

    total_bought = sum(bought.values())
    total_thrown = sum(thrown.values())
    total_spent = sum(bought[g] * _GRENADE_ITEMS[g] for g in _GRENADE_ITEMS)
    total_wasted_value = sum(
        max(0, bought[g] - thrown[g]) * _GRENADE_ITEMS[g] for g in _GRENADE_ITEMS
    )
    use_rate = round(total_thrown / total_bought * 100, 1) if total_bought > 0 else 0.0

    economics: dict[str, Any] = {
        "total_spent": total_spent,
        "total_wasted": total_wasted_value,
        "use_rate": use_rate,
        "per_type": {},
    }
    for g in _GRENADE_ITEMS:
        if bought[g] > 0 or thrown[g] > 0:
            economics["per_type"][_GRENADE_DISPLAY.get(g, g)] = {
                "bought": bought[g],
                "thrown": thrown[g],
                "wasted": max(0, bought[g] - thrown[g]),
                "cost": bought[g] * _GRENADE_ITEMS[g],
                "wasted_value": max(0, bought[g] - thrown[g]) * _GRENADE_ITEMS[g],
            }

    # ------------------------------------------------------------------ #
    # T2: Efficiency — direct impact of utility                            #
    # ------------------------------------------------------------------ #

    # --- Flashbangs ---
    enemy_flashes = 0
    team_flashes = 0
    total_enemy_blind_duration = 0.0
    total_team_blind_duration = 0.0
    flash_assists = 0

    if not blind_df.empty:
        id_col = _find_id_col(blind_df, ("attacker_steamid", "user_steamid", "steamid"))
        if id_col:
            player_blinds = blind_df[blind_df[id_col].astype(str) == sid]
            if not player_blinds.empty and "blind_duration" in player_blinds.columns:
                # Determine victim's team vs. attacker's team
                atk_team_col = "attacker_team_num" if "attacker_team_num" in player_blinds.columns else None
                vic_team_col = "user_team_num" if "user_team_num" in player_blinds.columns else None

                for _, brow in player_blinds.iterrows():
                    dur = float(brow.get("blind_duration", 0))
                    is_team = False
                    if atk_team_col and vic_team_col:
                        try:
                            is_team = int(brow[atk_team_col]) == int(brow[vic_team_col])
                        except (ValueError, TypeError):
                            pass
                    if is_team:
                        team_flashes += 1
                        total_team_blind_duration += dur
                    else:
                        enemy_flashes += 1
                        total_enemy_blind_duration += dur

    # Flash assists from death events
    death_df = parsed_data.get("player_death", pd.DataFrame())
    if not death_df.empty and "assistedflash" in death_df.columns:
        fa = death_df[
            (death_df.get("assister_steamid", pd.Series(dtype=str)).astype(str) == sid)
            & (death_df["assistedflash"] == True)  # noqa: E712
        ]
        flash_assists = len(fa)

    flash_efficiency: dict[str, Any] = {
        "thrown": thrown.get("flashbang", 0),
        "enemies_flashed": enemy_flashes,
        "team_flashed": team_flashes,
        "avg_enemy_blind_duration": round(
            total_enemy_blind_duration / enemy_flashes, 1
        ) if enemy_flashes > 0 else 0.0,
        "total_enemy_blind_duration": round(total_enemy_blind_duration, 1),
        "flash_assists": flash_assists,
        "enemies_per_flash": round(
            enemy_flashes / thrown["flashbang"], 2
        ) if thrown.get("flashbang", 0) > 0 else 0.0,
    }

    # --- HE Grenades ---
    total_he_damage = 0
    he_hits = 0
    if not hurt_df.empty and "weapon" in hurt_df.columns:
        id_col = _find_id_col(hurt_df, ("attacker_steamid",))
        if id_col:
            he_dmg = hurt_df[
                (hurt_df[id_col].astype(str) == sid)
                & (hurt_df["weapon"].astype(str).str.contains("hegrenade", case=False, na=False))
            ]
            if not he_dmg.empty and "dmg_health" in he_dmg.columns:
                total_he_damage = int(he_dmg["dmg_health"].sum())
                he_hits = len(he_dmg)

    he_efficiency: dict[str, Any] = {
        "thrown": thrown.get("hegrenade", 0),
        "total_damage": total_he_damage,
        "hits": he_hits,
        "avg_damage_per_throw": round(
            total_he_damage / thrown["hegrenade"], 1
        ) if thrown.get("hegrenade", 0) > 0 else 0.0,
    }

    # --- Molotovs / Incendiaries ---
    total_molly_damage = 0
    molly_hits = 0
    if not hurt_df.empty and "weapon" in hurt_df.columns:
        id_col = _find_id_col(hurt_df, ("attacker_steamid",))
        if id_col:
            molly_dmg = hurt_df[
                (hurt_df[id_col].astype(str) == sid)
                & (hurt_df["weapon"].astype(str).str.contains(
                    "inferno|molotov", case=False, na=False
                ))
            ]
            if not molly_dmg.empty and "dmg_health" in molly_dmg.columns:
                total_molly_damage = int(molly_dmg["dmg_health"].sum())
                molly_hits = len(molly_dmg)

    molly_efficiency: dict[str, Any] = {
        "thrown": thrown.get("molotov", 0) + thrown.get("incgrenade", 0),
        "total_damage": total_molly_damage,
        "hits": molly_hits,
        "avg_damage_per_throw": round(
            total_molly_damage / max(1, thrown.get("molotov", 0) + thrown.get("incgrenade", 0)),
            1,
        ) if (thrown.get("molotov", 0) + thrown.get("incgrenade", 0)) > 0 else 0.0,
    }

    # ------------------------------------------------------------------ #
    # T3: Smokes — spatial enablement (zone coverage)                      #
    # ------------------------------------------------------------------ #
    smoke_count = thrown.get("smokegrenade", 0)
    smoke_locations: list[dict[str, Any]] = []

    if not smoke_det_df.empty:
        id_col = _find_id_col(smoke_det_df, ("user_steamid", "steamid", "attacker_steamid"))
        if id_col:
            player_smokes = smoke_det_df[smoke_det_df[id_col].astype(str) == sid]
            for _, srow in player_smokes.iterrows():
                sx = float(srow.get("x", 0))
                sy = float(srow.get("y", 0))
                rnd = int(srow.get("round", 0)) if "round" in srow.index else 0
                callout = "unknown"
                if is_map_supported(map_name):
                    callout = get_callout(map_name, sx, sy)
                smoke_locations.append({
                    "round": rnd,
                    "location": callout,
                    "x": round(sx, 1),
                    "y": round(sy, 1),
                })

    # Check if smokes extinguished enemy molotovs (within ~300 unit radius)
    molly_extinguishes = 0
    if smoke_locations and not molotov_det_df.empty and "x" in molotov_det_df.columns:
        for sm in smoke_locations:
            for _, mrow in molotov_det_df.iterrows():
                mx = float(mrow.get("x", 0))
                my = float(mrow.get("y", 0))
                m_rnd = int(mrow.get("round", 0)) if "round" in mrow.index else 0
                if m_rnd == sm["round"]:
                    dist = ((sm["x"] - mx) ** 2 + (sm["y"] - my) ** 2) ** 0.5
                    if dist < 300:
                        molly_extinguishes += 1
                        break  # one extinguish per smoke max

    # Summarise smoke zone coverage
    zone_counts: dict[str, int] = {}
    for sl in smoke_locations:
        loc = sl["location"]
        if loc != "unknown":
            zone_counts[loc] = zone_counts.get(loc, 0) + 1
    top_zones = sorted(zone_counts.items(), key=lambda x: -x[1])[:5]

    smoke_efficiency: dict[str, Any] = {
        "thrown": smoke_count,
        "locations": smoke_locations,
        "top_zones": [{"zone": z, "count": c} for z, c in top_zones],
        "molotov_extinguishes": molly_extinguishes,
    }

    # ------------------------------------------------------------------ #
    # Per-round utility breakdown                                          #
    # ------------------------------------------------------------------ #
    per_round: list[dict[str, Any]] = []
    for er in enriched_rounds:
        rnd = er.get("round", 0)
        eco = er.get("economy", {})
        util = er.get("utility", {})
        items = eco.get("items", [])
        nade_items = [
            i for i in items if _PURCHASE_NAME_MAP.get(i.lower()) is not None
        ]
        nade_spend = sum(
            _GRENADE_ITEMS.get(_PURCHASE_NAME_MAP.get(i.lower(), ""), 0)
            for i in nade_items
        )
        per_round.append({
            "round": rnd,
            "side": er.get("side", "?"),
            "nades_bought": len(nade_items),
            "nade_spend": nade_spend,
            "enemies_flashed": util.get("enemies_flashed", 0),
            "enemy_blind_duration": round(
                sum(f.get("duration", 0) for f in util.get("flash_instances", []) if not f.get("is_friendly")),
                1,
            ),
            "flash_assists": util.get("flash_assists", 0),
            "he_damage": util.get("he_damage", 0),
            "molotov_damage": sum(
                d.get("damage", 0) for d in util.get("molotov_damage", [])
            ),
        })

    # ------------------------------------------------------------------ #
    # Utility rating (0-100)                                               #
    # ------------------------------------------------------------------ #
    # Weighted: 30% use rate, 30% flash efficiency, 20% damage utility,
    #           20% smoke coverage
    rating = 50.0

    # Use rate component (0-100, 100% = perfect)
    use_score = min(100, use_rate) if total_bought > 0 else 50
    # Flash component: enemies per flash (0.5+ is good, 1+ is great)
    flash_score = 0
    if thrown.get("flashbang", 0) > 0:
        epf = enemy_flashes / thrown["flashbang"]
        flash_score = min(100, epf * 100)  # 1.0 enemies/flash = 100
    elif total_bought == 0:
        flash_score = 50
    # Damage component: avg damage per HE/molly (30+ is good)
    total_dmg_nades = thrown.get("hegrenade", 0) + thrown.get("molotov", 0) + thrown.get("incgrenade", 0)
    dmg_score = 0
    if total_dmg_nades > 0:
        avg_dmg = (total_he_damage + total_molly_damage) / total_dmg_nades
        dmg_score = min(100, avg_dmg * 2.5)  # 40 dmg/nade = 100
    elif total_bought == 0:
        dmg_score = 50
    # Smoke coverage score
    smoke_score = 0
    if smoke_count > 0:
        used_zones = len([s for s in smoke_locations if s["location"] != "unknown"])
        smoke_score = min(100, (used_zones / smoke_count) * 100)
    elif total_bought == 0:
        smoke_score = 50

    rating = (use_score * 0.30 + flash_score * 0.30
              + dmg_score * 0.20 + smoke_score * 0.20)
    # Teamplayer penalty — minimal: minor incidents are normal in CS2.
    # Team flash: only penalise beyond 3 flashes (−0.5 each)
    if team_flashes > 3:
        rating -= (team_flashes - 3) * 0.5
    rating = round(min(100, max(0, rating)), 1)

    # ------------------------------------------------------------------ #
    # Teamplayer — per-round: teammate attacks, drops, team flashes        #
    # ------------------------------------------------------------------ #
    team_attacks_total = 0
    team_attack_damage_total = 0
    drops_total = 0
    team_flashes_total = 0
    teamplayer_rounds: list[dict[str, Any]] = []

    # Pre-filter once: player's team-hits (hurt where same team, not self)
    _team_hit_rows = pd.DataFrame()
    if not hurt_df.empty:
        id_col = _find_id_col(hurt_df, ("attacker_steamid",))
        if id_col and "attacker_team_num" in hurt_df.columns and "user_team_num" in hurt_df.columns:
            player_attacks = hurt_df[hurt_df[id_col].astype(str) == sid]
            if not player_attacks.empty:
                same_team = player_attacks[
                    player_attacks["attacker_team_num"] == player_attacks["user_team_num"]
                ]
                vic_id_col = "user_steamid" if "user_steamid" in same_team.columns else None
                if vic_id_col:
                    _team_hit_rows = same_team[same_team[vic_id_col].astype(str) != sid]
                else:
                    _team_hit_rows = same_team

    # Pre-filter once: player's team-flashes from blind_df
    _team_flash_rows = pd.DataFrame()
    if not blind_df.empty:
        id_col = _find_id_col(blind_df, ("attacker_steamid", "user_steamid", "steamid"))
        if id_col:
            player_blinds = blind_df[blind_df[id_col].astype(str) == sid]
            if not player_blinds.empty:
                atk_t = "attacker_team_num" if "attacker_team_num" in player_blinds.columns else None
                vic_t = "user_team_num" if "user_team_num" in player_blinds.columns else None
                if atk_t and vic_t:
                    _team_flash_rows = player_blinds[
                        player_blinds[atk_t] == player_blinds[vic_t]
                    ]

    # Pre-compute per-round weapon drops
    _drop_rounds: dict[int, list[str]] = {}
    if not purchase_df.empty:
        id_col = _find_id_col(purchase_df, ("steamid", "attacker_steamid", "user_steamid"))
        if id_col:
            name_col = "item_name" if "item_name" in purchase_df.columns else (
                "weapon" if "weapon" in purchase_df.columns else None
            )
            if name_col and "round" in purchase_df.columns:
                player_buys = purchase_df[purchase_df[id_col].astype(str) == sid]
                for rnd_num in player_buys["round"].unique():
                    rnd_buys = player_buys[player_buys["round"] == rnd_num]
                    slot_items: dict[str, list[str]] = {}
                    for _, row in rnd_buys.iterrows():
                        raw = str(row[name_col]).lower()
                        slot = _WEAPON_SLOT.get(raw)
                        if slot:
                            slot_items.setdefault(slot, []).append(raw)
                    dropped: list[str] = []
                    for slot, items in slot_items.items():
                        if len(items) > 1:
                            # First item kept, rest dropped
                            dropped.extend(items[1:])
                    if dropped:
                        _drop_rounds[int(rnd_num)] = dropped

    # Build per-round teamplayer breakdown
    all_rounds = sorted(set(
        [int(r) for r in _team_hit_rows["round"].unique()] if "round" in _team_hit_rows.columns and not _team_hit_rows.empty else []
    ) | set(
        [int(r) for r in _team_flash_rows["round"].unique()] if "round" in _team_flash_rows.columns and not _team_flash_rows.empty else []
    ) | set(_drop_rounds.keys()))

    for rnd in sorted(all_rounds):
        rnd_entry: dict[str, Any] = {"round": rnd}

        # --- Teammate attacks this round ---
        attacks: list[dict[str, Any]] = []
        if not _team_hit_rows.empty and "round" in _team_hit_rows.columns:
            rnd_hits = _team_hit_rows[_team_hit_rows["round"] == rnd]
            for _, hrow in rnd_hits.iterrows():
                victim = str(hrow.get("user_name", "?")) if "user_name" in rnd_hits.columns else "?"
                dmg = int(hrow.get("dmg_health", 0)) if "dmg_health" in rnd_hits.columns else 0
                weapon = str(hrow.get("weapon", "?")) if "weapon" in rnd_hits.columns else "?"
                attacks.append({"victim": victim, "damage": dmg, "weapon": weapon})
                team_attacks_total += 1
                team_attack_damage_total += dmg
        rnd_entry["attacks"] = attacks

        # --- Team flashes this round ---
        flashes: list[dict[str, Any]] = []
        if not _team_flash_rows.empty and "round" in _team_flash_rows.columns:
            rnd_flashes = _team_flash_rows[_team_flash_rows["round"] == rnd]
            vic_name_col = "user_name" if "user_name" in rnd_flashes.columns else None
            for _, frow in rnd_flashes.iterrows():
                victim = str(frow.get(vic_name_col, "?")) if vic_name_col else "?"
                dur = round(float(frow.get("blind_duration", 0)), 1) if "blind_duration" in rnd_flashes.columns else 0
                flashes.append({"victim": victim, "duration": dur})
                team_flashes_total += 1
        rnd_entry["team_flashes"] = flashes

        # --- Drops this round ---
        rnd_drops = _drop_rounds.get(rnd, [])
        rnd_entry["drops"] = rnd_drops
        drops_total += len(rnd_drops)

        teamplayer_rounds.append(rnd_entry)

    teamplayer: dict[str, Any] = {
        "team_attacks": team_attacks_total,
        "team_attack_damage": team_attack_damage_total,
        "team_flashes": team_flashes_total,
        "drops_for_teammates": drops_total,
        "per_round": teamplayer_rounds,
    }

    return {
        "economics": economics,
        "flash": flash_efficiency,
        "he": he_efficiency,
        "molotov": molly_efficiency,
        "smoke": smoke_efficiency,
        "per_round": per_round,
        "utility_rating": rating,
        "teamplayer": teamplayer,
    }


def _find_id_col(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Return the first column from *candidates* that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Role classification per round
# ---------------------------------------------------------------------------

# Callout → role mapping per map per side.
# Each role lists the callout names that indicate that role.
# Order matters: first match wins when a player has multiple positions.

_ROLE_ZONES: dict[str, dict[str, dict[str, list[str]]]] = {
    "de_mirage": {
        "CT": {
            "A Anchor": [
                "A Site", "A Default", "Ticket", "Firebox", "A Ramp",
            ],
            "Connector / Jungle": [
                "Jungle", "Stairs", "Connector", "A Main",
            ],
            "Window (AWP)": [
                "Window", "Mid Window", "Snipers Nest", "Ladder Room",
            ],
            "Short / Catwalk": [
                "Short", "Catwalk", "Underpass", "Mid", "Mid Area",
                "Top Mid",
            ],
            "B Anchor": [
                "B Site", "Bench", "B Van", "Market", "Market Door",
                "Kitchen", "B Apartments", "B Apartments Entrance",
                "B Short",
            ],
        },
        "T": {
            "Mid Pack": [
                "Top Mid", "Mid", "Mid Area", "Short", "Catwalk",
                "Underpass", "Connector", "Window", "Mid Window",
                "Snipers Nest",
            ],
            "A Lurk / Palace": [
                "A Ramp", "A Palace", "A Site", "A Default", "Ticket",
                "Firebox", "Jungle", "Stairs", "Tetris", "A Main",
                "A Side", "Chair",
            ],
            "B Apps": [
                "B Apartments", "B Apartments Entrance", "B Short",
                "B Site", "Bench", "B Van", "Market", "Market Door",
                "Kitchen", "B Side",
            ],
        },
    },
    "de_inferno": {
        "CT": {
            "Pit / A Anchor": [
                "Pit", "A Site", "Truck", "Graveyard", "Balcony",
            ],
            "A Short / Boiler": [
                "Top Mid", "Boiler", "Second Mid",
            ],
            "Arch / Speedway": [
                "Arch", "Library", "Mid", "Alt Mid", "Underpass",
            ],
            "B Rotator (CT/Spools)": [
                "CT", "New Box", "Construction",
            ],
            "B Anchor (Banana)": [
                "B Site", "Banana", "Oranges", "Car", "Dark", "Coffins",
            ],
        },
        "T": {
            "Banana Pack": [
                "Banana", "Car", "Oranges", "B Site", "Dark", "Coffins",
                "Construction", "New Box", "CT", "B Side",
            ],
            "Apps Control": [
                "T Apartments", "Apartments", "Second Mid", "Boiler",
                "Balcony",
            ],
            "Mid / Arch Lurk": [
                "Mid", "Alt Mid", "Underpass", "Top Mid", "Arch",
                "Library", "A Site", "Pit", "Truck", "Graveyard",
                "A Side",
            ],
        },
    },
    "de_nuke": {
        "CT": {
            "A Anchor (Hut/Mini)": [
                "A Site", "Hut", "Squeaky", "Main",
            ],
            "Heaven (Rotator)": [
                "Heaven", "Hell",
            ],
            "Outside": [
                "Outside",
            ],
            "Secret / Lower": [
                "Secret", "B Site",
            ],
            "Ramp Anchor": [
                "Ramp", "Lobby",
            ],
        },
        "T": {
            "Outside Pack": [
                "Outside", "Secret",
            ],
            "Lobby / Hut Control": [
                "Lobby", "Main", "Hut", "Squeaky", "A Site", "Heaven",
                "Hell",
            ],
            "Ramp Entry / Lurk": [
                "Ramp", "B Site",
            ],
        },
    },
    "de_ancient": {
        "CT": {
            "B Anchor": [
                "B Site", "B Pillar",
            ],
            "Cave / Short B": [
                "B Connector", "B Main",
            ],
            "Mid": [
                "Mid", "Top Mid",
            ],
            "Donut (Rotator)": [
                "A Connector",
            ],
            "A Anchor": [
                "A Site", "A Main", "A Bridge",
            ],
        },
        "T": {
            "Mid Control": [
                "Mid", "Top Mid", "A Connector", "B Connector",
            ],
            "A Main Lurk": [
                "A Main", "A Bridge", "A Site",
            ],
            "B Ramp / Cave Pack": [
                "B Main", "B Site", "B Pillar",
            ],
        },
    },
    "de_anubis": {
        "CT": {
            "B Anchor": [
                "B Site", "B Pillar",
            ],
            "Palace / B Connector": [
                "B Connector", "B Main",
            ],
            "Mid (AWP)": [
                "Mid", "Top Mid",
            ],
            "A Connector / Water": [
                "A Connector",
            ],
            "A Anchor": [
                "A Site", "A Main", "A Bridge",
            ],
        },
        "T": {
            "Water / Canals Pack": [
                "A Connector", "B Connector",
            ],
            "Bridge / Mid": [
                "Mid", "Top Mid",
            ],
            "A Main Lurk": [
                "A Main", "A Bridge", "A Site",
            ],
            "B Main": [
                "B Main", "B Site", "B Pillar",
            ],
        },
    },
    "de_overpass": {
        "CT": {
            "A Long Anchor": [
                "A Long",
            ],
            "Toilets / Mid": [
                "Toilets", "Mid", "Connector", "Party", "Balloons",
            ],
            "Heaven / Sniper": [
                "Heaven", "Sniper",
            ],
            "B Short / Water": [
                "B Short", "Water", "Sandbags",
            ],
            "B Anchor (Monster)": [
                "B Site", "Monster", "Pit", "Pillar", "Barrels",
            ],
        },
        "T": {
            "Lower / Water Pack": [
                "B Short", "Water", "Connector", "B Site", "Monster",
            ],
            "Toilets / Mid Pack": [
                "Toilets", "Mid", "Party", "Balloons",
            ],
            "Extremity Lurks": [
                "A Long", "A Site",
            ],
        },
    },
    "de_dust2": {
        "CT": {
            "B Anchor": [
                "B Site", "B Window", "B Back Site", "B Car", "B Doors",
            ],
            "Mid (AWP)": [
                "Mid", "Lower Tunnels", "Mid Doors", "Xbox",
            ],
            "Short / Catwalk": [
                "Catwalk", "A Short (Cat)",
            ],
            "Long A (Rotator)": [
                "A Long", "A Long Doors",
            ],
            "A Anchor": [
                "A Site", "Goose", "A Ramp", "A Car", "A Pit",
            ],
        },
        "T": {
            "Long A Pack": [
                "A Long Doors", "A Long", "A Site", "A Pit", "A Car",
            ],
            "Mid / Catwalk Control": [
                "Mid", "Mid Doors", "Catwalk", "A Short (Cat)", "Xbox",
                "Lower Tunnels",
            ],
            "Upper Tunnels Lurk": [
                "Upper Tunnels", "B Tunnels", "B Site", "B Doors",
                "B Window", "B Back Site", "B Car",
            ],
        },
    },
}


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
    if not side or map_name not in _ROLE_ZONES:
        return {}
    side_roles = _ROLE_ZONES[map_name].get(side)
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

    # Kill positions (weight 2)
    for k in enriched_round.get("kills_detail", []):
        p = k.get("attacker_position")
        if p and p != "unknown":
            weighted.append((p, 2.0))
    # Death position (weight 2)
    death = enriched_round.get("death_detail")
    if death:
        p = death.get("victim_position")
        if p and p != "unknown":
            weighted.append((p, 2.0))

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
    if map_name not in _ROLE_ZONES:
        return None

    import pandas as pd

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
    roles_ct = list(_ROLE_ZONES[map_name].get("CT", {}).keys())
    roles_t = list(_ROLE_ZONES[map_name].get("T", {}).keys())

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


# ---------------------------------------------------------------------------
# All-players scoreboard
# ---------------------------------------------------------------------------


def calculate_all_players_stats(
    parsed_data: dict[str, Any],
    user_steam_id: str,
    total_rounds: int,
) -> list[dict[str, Any]]:
    """Calculate stats for every player in the match.

    Returns a list of dicts (one per player) with keys: ``steam_id``,
    ``name``, ``team``, ``is_user``, ``kills``, ``deaths``, ``assists``,
    ``kd_ratio``, ``adr``, ``kast``, ``hltv_rating``, ``rounds_2k``,
    ``rounds_3k``, ``rounds_4k``, ``rounds_5k``.
    """
    death_df: pd.DataFrame = parsed_data.get("player_death", pd.DataFrame())
    hurt_df: pd.DataFrame = parsed_data.get("player_hurt", pd.DataFrame())
    ranks_df: pd.DataFrame = parsed_data.get("ranks", pd.DataFrame())
    rank_update_df: pd.DataFrame = parsed_data.get("rank_update", pd.DataFrame())
    end_stats_df: pd.DataFrame = parsed_data.get("end_stats", pd.DataFrame())

    # Build rank lookup: steam_id -> rank int
    rank_lookup: dict[str, int] = {}
    if not ranks_df.empty and "steamid" in ranks_df.columns and "rank" in ranks_df.columns:
        for _, rr in ranks_df.iterrows():
            rank_lookup[str(rr["steamid"])] = int(rr["rank"])

    # Build rank_type lookup from tick data: steam_id -> comp_rank_type
    rank_type_lookup: dict[str, int] = {}
    if not ranks_df.empty and "steamid" in ranks_df.columns and "comp_rank_type" in ranks_df.columns:
        for _, rr in ranks_df.iterrows():
            val = int(rr.get("comp_rank_type", 0))
            if val > 0:
                rank_type_lookup[str(rr["steamid"])] = val

    # Build rank_update lookup: steam_id -> {rank_old, rank_new, rank_change, num_wins, rank_type_id}
    rank_update_lookup: dict[str, dict] = {}
    if not rank_update_df.empty and "user_steamid" in rank_update_df.columns:
        for _, rr in rank_update_df.iterrows():
            rank_update_lookup[str(rr["user_steamid"])] = {
                "rank_old": int(rr.get("rank_old", 0)),
                "rank_new": int(rr.get("rank_new", 0)),
                "rank_change": float(rr.get("rank_change", 0)),
                "num_wins": int(rr.get("num_wins", 0)),
                "rank_type_id": int(rr.get("rank_type_id", 0)),
            }

    # Build end-of-match stats lookup: steam_id -> {comp_wins, mvps}
    end_stats_lookup: dict[str, dict] = {}
    if not end_stats_df.empty and "steamid" in end_stats_df.columns:
        for _, rr in end_stats_df.iterrows():
            end_stats_lookup[str(rr["steamid"])] = {
                "comp_wins": int(rr.get("comp_wins", 0)),
                "mvps": int(rr.get("mvps", 0)),
            }

    steam_ids = _collect_all_steam_ids(death_df)
    if not steam_ids:
        return []

    if total_rounds == 0:
        total_rounds = 1

    players: list[dict[str, Any]] = []
    for sid in steam_ids:
        name = _detect_player_name(death_df, sid)
        team = _detect_player_team(death_df, sid)

        kills = len(_filter_attacker(death_df, sid))
        deaths = len(_filter_victim(death_df, sid))
        assists = _count_valid_assists(_filter_assister(death_df, sid), hurt_df, sid)
        total_damage = _calculate_damage(hurt_df, sid)

        kpr = round(kills / total_rounds, 4)
        dpr = round(deaths / total_rounds, 4)
        adr = round(total_damage / total_rounds, 4)

        round_stats = _build_round_stats(death_df, hurt_df, sid, total_rounds)
        kast_rounds = _calculate_kast_rounds(round_stats)
        kast = round(kast_rounds / total_rounds * 100, 2)

        impact = round(
            2.13 * kpr + 0.42 * (assists / total_rounds) - 0.41, 4
        )
        hltv_rating = _compute_hltv_rating(kast, kpr, dpr, impact, adr)
        kd_ratio = round(kills / deaths, 2) if deaths > 0 else float(kills)
        multikills = _count_multikill_rounds(round_stats)

        ru = rank_update_lookup.get(str(sid), {})
        es = end_stats_lookup.get(str(sid), {})

        # Resolve rank_type_id: prefer rank_update event, fall back to tick data
        resolved_rank_type = ru.get("rank_type_id", 0) or rank_type_lookup.get(str(sid), 0)

        players.append({
            "steam_id": sid,
            "name": name,
            "team": team,
            "is_user": str(sid) == str(user_steam_id),
            "kills": kills,
            "deaths": deaths,
            "assists": assists,
            "kd_ratio": kd_ratio,
            "adr": adr,
            "kast": kast,
            "hltv_rating": hltv_rating,
            "rank": ru.get("rank_new", 0) or rank_lookup.get(str(sid), 0),
            "rank_old": ru.get("rank_old", 0),
            "rank_change": ru.get("rank_change", 0.0),
            "comp_wins": ru.get("num_wins", 0) or es.get("comp_wins", 0),
            "mvps": es.get("mvps", 0),
            "rank_type_id": resolved_rank_type,
            "rounds_2k": multikills[2],
            "rounds_3k": multikills[3],
            "rounds_4k": multikills[4],
            "rounds_5k": multikills[5],
        })

    # Sort: user's team first, then by kills descending
    user_team = None
    for p in players:
        if p["is_user"]:
            user_team = p["team"]
            break

    def _sort_key(p: dict) -> tuple:
        team_priority = 0 if p["team"] == user_team else 1
        return (team_priority, -p["kills"])

    players.sort(key=_sort_key)
    return players


def _collect_all_steam_ids(death_df: pd.DataFrame) -> list[str]:
    """Return a deduplicated list of all player Steam IDs from kill events."""
    if death_df.empty:
        return []
    ids: set[str] = set()
    for col in ("attacker_steamid", "user_steamid", "assister_steamid"):
        if col in death_df.columns:
            ids.update(
                death_df[col].dropna().astype(str).unique()
            )
    # Filter out obvious non-player IDs (e.g. "0", empty, "None")
    return [s for s in ids if s and s not in ("0", "None", "nan")]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_total_rounds(round_end_df: pd.DataFrame) -> int:
    """Return the number of completed rounds from the round_end event table."""
    if round_end_df.empty:
        return 0
    if "round" in round_end_df.columns:
        return int(round_end_df["round"].max())
    return len(round_end_df)


def _filter_attacker(df: pd.DataFrame, steam_id: str) -> pd.DataFrame:
    """Return rows where the attacker matches *steam_id*."""
    if df.empty or "attacker_steamid" not in df.columns:
        return pd.DataFrame()
    return df[df["attacker_steamid"].astype(str) == str(steam_id)]


def _filter_victim(df: pd.DataFrame, steam_id: str) -> pd.DataFrame:
    """Return rows where the victim matches *steam_id*."""
    if df.empty or "user_steamid" not in df.columns:
        return pd.DataFrame()
    return df[df["user_steamid"].astype(str) == str(steam_id)]


def _filter_assister(df: pd.DataFrame, steam_id: str) -> pd.DataFrame:
    """Return rows where the assister matches *steam_id*."""
    if df.empty or "assister_steamid" not in df.columns:
        return pd.DataFrame()
    return df[
        df["assister_steamid"].astype(str) == str(steam_id)
    ]


def _count_valid_assists(
    assist_df: pd.DataFrame,
    hurt_df: pd.DataFrame,
    steam_id: str,
) -> int:
    """Count assists where the player dealt damage to the victim in the same round.

    The demo engine sometimes credits assists for damage dealt in previous
    rounds.  Platforms like Refrag/Leetify only count assists where the
    assister damaged the victim in the round the kill happened.
    """
    if assist_df.empty:
        return 0
    if hurt_df.empty or "round" not in assist_df.columns or "round" not in hurt_df.columns:
        return len(assist_df)

    sid = str(steam_id)
    count = 0
    for _, row in assist_df.iterrows():
        rnd = row["round"]
        victim_id = str(row["user_steamid"])
        dealt = hurt_df[
            (hurt_df["round"] == rnd)
            & (hurt_df["attacker_steamid"].astype(str) == sid)
            & (hurt_df["user_steamid"].astype(str) == victim_id)
        ]
        if not dealt.empty:
            count += 1
    return count


def _detect_player_name(death_df: pd.DataFrame, steam_id: str) -> str:
    """Best-effort detection of the player's in-game name from event rows."""
    if death_df.empty:
        return "Unknown"
    sid = str(steam_id)
    for col in ("attacker_steamid", "user_steamid", "assister_steamid"):
        name_col = col.replace("steamid", "name")
        if col in death_df.columns and name_col in death_df.columns:
            mask = death_df[col].astype(str) == sid
            names = death_df.loc[mask, name_col].dropna()
            if not names.empty:
                return str(names.iloc[0])
    return "Unknown"


def _sum_capped_damage(hurt_rows: pd.DataFrame) -> int:
    """Sum damage capped at 100 per victim per round (standard ADR rule).

    This is a fallback used when the full ``hurt_df`` (all attackers) is
    not available for HP-tracking.  See :func:`_calculate_actual_damage`
    for the preferred, accurate method.
    """
    if hurt_rows.empty or "dmg_health" not in hurt_rows.columns:
        return 0
    group_cols = []
    if "round" in hurt_rows.columns:
        group_cols.append("round")
    if "user_steamid" in hurt_rows.columns:
        group_cols.append("user_steamid")
    if not group_cols:
        return min(int(hurt_rows["dmg_health"].sum()), 100)
    capped = hurt_rows.groupby(group_cols)["dmg_health"].sum().clip(upper=100)
    return int(capped.sum())


def _exclude_team_damage(hurt_rows: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where attacker and victim are on the same team."""
    if (
        hurt_rows.empty
        or "attacker_team_num" not in hurt_rows.columns
        or "user_team_num" not in hurt_rows.columns
    ):
        return hurt_rows
    return hurt_rows[hurt_rows["attacker_team_num"] != hurt_rows["user_team_num"]]


def _calculate_damage(hurt_df: pd.DataFrame, steam_id: str) -> int:
    """Sum total damage dealt by the player using HP-tracking.

    Processes ALL enemy hurt events per round to track each victim's
    remaining HP, then attributes only the actual HP lost per hit to the
    attacking player.  This avoids double-counting overkill damage that
    ``dmg_health`` can include.

    Falls back to the 100-per-victim cap when the ``health`` column is
    missing.
    """
    if hurt_df.empty or "attacker_steamid" not in hurt_df.columns:
        return 0

    enemy_hurt = _exclude_team_damage(hurt_df)
    if enemy_hurt.empty or "dmg_health" not in enemy_hurt.columns:
        return 0

    sid = str(steam_id)

    # Fast path: if no health column, fall back to capped sum
    if "health" not in enemy_hurt.columns or "round" not in enemy_hurt.columns:
        mask = enemy_hurt["attacker_steamid"].astype(str) == sid
        return _sum_capped_damage(enemy_hurt[mask])

    total = 0
    for rnd, grp in enemy_hurt.groupby("round"):
        victim_hp: dict[str, int] = {}
        for _, row in grp.sort_values("tick").iterrows():
            vid = str(row["user_steamid"])
            aid = str(row["attacker_steamid"])

            if vid not in victim_hp:
                victim_hp[vid] = 100

            actual = min(int(row["dmg_health"]), victim_hp[vid])
            # Use engine's reported remaining health as ground truth
            victim_hp[vid] = int(row["health"])

            if aid == sid:
                total += actual
    return total


def _build_round_stats(
    death_df: pd.DataFrame,
    hurt_df: pd.DataFrame,
    steam_id: str,
    total_rounds: int,
) -> list[dict[str, Any]]:
    """Build a per-round stats list for the player."""
    stats: list[dict[str, Any]] = []
    sid = str(steam_id)

    # Pre-filter enemy hurt events for HP-tracking
    enemy_hurt = _exclude_team_damage(hurt_df) if not hurt_df.empty else hurt_df
    has_hp_tracking = (
        not enemy_hurt.empty
        and "health" in enemy_hurt.columns
        and "round" in enemy_hurt.columns
    )

    for r in range(1, total_rounds + 1):
        round_kills = 0
        round_deaths = 0
        round_assists = 0
        round_damage = 0
        survived = True

        if not death_df.empty and "round" in death_df.columns:
            round_deaths_df = death_df[
                (death_df["round"] == r)
                & (death_df["user_steamid"].astype(str) == sid)
            ]
            round_deaths = len(round_deaths_df)
            survived = round_deaths == 0

            if "attacker_steamid" in death_df.columns:
                round_kills = len(
                    death_df[
                        (death_df["round"] == r)
                        & (death_df["attacker_steamid"].astype(str) == sid)
                    ]
                )

            if "assister_steamid" in death_df.columns:
                raw_assists = death_df[
                    (death_df["round"] == r)
                    & (death_df["assister_steamid"].astype(str) == sid)
                ]
                # Only count assists where damage was dealt this round
                round_assists = _count_valid_assists(
                    raw_assists, hurt_df, sid
                )

        # Damage: use HP-tracking when possible, else capped sum
        if has_hp_tracking:
            round_hurt = enemy_hurt[enemy_hurt["round"] == r]
            if not round_hurt.empty:
                victim_hp: dict[str, int] = {}
                for _, row in round_hurt.sort_values("tick").iterrows():
                    vid = str(row["user_steamid"])
                    aid = str(row["attacker_steamid"])
                    if vid not in victim_hp:
                        victim_hp[vid] = 100
                    actual = min(int(row["dmg_health"]), victim_hp[vid])
                    victim_hp[vid] = int(row["health"])
                    if aid == sid:
                        round_damage += actual
        elif not hurt_df.empty and "round" in hurt_df.columns:
            if (
                "attacker_steamid" in hurt_df.columns
                and "dmg_health" in hurt_df.columns
            ):
                round_hurt = hurt_df[
                    (hurt_df["round"] == r)
                    & (hurt_df["attacker_steamid"].astype(str) == sid)
                ]
                round_hurt = _exclude_team_damage(round_hurt)
                round_damage = _sum_capped_damage(round_hurt)

        # Traded: player died but their killer was killed by a teammate
        # within 5 seconds (~320 ticks at 64-tick).
        traded = False
        if round_deaths > 0 and not death_df.empty and "tick" in death_df.columns:
            my_death = death_df[
                (death_df["round"] == r)
                & (death_df["user_steamid"].astype(str) == sid)
            ].iloc[0]
            killer_id = str(my_death["attacker_steamid"])
            death_tick = int(my_death["tick"])
            # Did anyone kill the killer within 320 ticks?
            killer_died = death_df[
                (death_df["round"] == r)
                & (death_df["user_steamid"].astype(str) == killer_id)
                & (death_df["tick"] > death_tick)
                & (death_df["tick"] <= death_tick + 320)
            ]
            traded = not killer_died.empty

        stats.append(
            {
                "round": r,
                "kills": round_kills,
                "deaths": round_deaths,
                "assists": round_assists,
                "damage": round_damage,
                "survived": int(survived),
                "traded": int(traded),
            }
        )

    return stats


def _calculate_kast_rounds(round_stats: list[dict[str, Any]]) -> int:
    """
    Count rounds where the player had a Kill, Assist, Survived, or was Traded.
    """
    count = 0
    for rs in round_stats:
        if (
            rs["kills"] > 0
            or rs["assists"] > 0
            or rs["survived"]
            or rs["traded"]
        ):
            count += 1
    return count


def _count_multikill_rounds(
    round_stats: list[dict[str, Any]],
) -> dict[int, int]:
    """Count rounds where the player got exactly 2, 3, 4, or 5+ kills."""
    counts = {2: 0, 3: 0, 4: 0, 5: 0}
    for rs in round_stats:
        k = rs["kills"]
        if k >= 5:
            counts[5] += 1
        elif k in counts:
            counts[k] += 1
    return counts


def _calculate_match_score(
    round_end_df: pd.DataFrame,
    death_df: pd.DataFrame,
    steam_id: str,
) -> dict[str, Any]:
    """Derive team/enemy round scores and match result.

    Determines the player's team for each round by inspecting their
    actual ``team_num`` in kill/death events rather than relying on
    hardcoded half lengths.  This correctly handles arbitrary overtime
    formats (MR3, MR5, etc.) and non-standard round numbering.
    """
    default = {"team_score": 0, "enemy_score": 0, "result": "unknown"}

    if round_end_df.empty or "winner" not in round_end_df.columns:
        return default

    round_team = _build_round_team_map(death_df, steam_id, round_end_df)
    if not round_team:
        return default

    team_wins = 0
    for _, row in round_end_df.iterrows():
        rnd = int(row.get("round", 0))
        winner = str(row["winner"])
        player_side = round_team.get(rnd)
        if player_side and winner == player_side:
            team_wins += 1

    total = len(round_end_df)
    enemy_wins = total - team_wins

    if team_wins > enemy_wins:
        result = "win"
    elif team_wins < enemy_wins:
        result = "loss"
    else:
        result = "draw"

    return {"team_score": team_wins, "enemy_score": enemy_wins, "result": result}


def _build_round_team_map(
    death_df: pd.DataFrame,
    steam_id: str,
    round_end_df: pd.DataFrame,
) -> dict[int, str]:
    """Build a mapping of round number → player team ("CT" or "T").

    Inspects kill/death events to find the player's ``team_num`` for each
    round they appear in.  For rounds with no player events (survived
    without kills), the team is carried forward from the last known round.
    """
    _TEAM_MAP = {2: "T", 3: "CT"}
    if death_df.empty or "round" not in death_df.columns:
        return {}

    sid = str(steam_id)
    round_team: dict[int, str] = {}

    # Collect team observations from kill events (as attacker) and death
    # events (as victim).  Prefer earlier ticks for each round.
    for id_col, team_col in [
        ("attacker_steamid", "attacker_team_num"),
        ("user_steamid", "user_team_num"),
    ]:
        if id_col not in death_df.columns or team_col not in death_df.columns:
            continue
        mask = death_df[id_col].astype(str) == sid
        subset = death_df.loc[mask, ["round", team_col]].dropna()
        for _, row in subset.iterrows():
            rnd = int(row["round"])
            if rnd in round_team:
                continue  # already have data for this round
            team = _TEAM_MAP.get(int(row[team_col]))
            if team:
                round_team[rnd] = team

    if not round_team:
        return {}

    # Fill gaps: for rounds where the player had no events, carry forward
    # from the last known round (team only changes at halftime boundaries).
    all_rounds = sorted(int(r) for r in round_end_df["round"].dropna().unique())
    known_sorted = sorted(round_team.keys())
    last_known: str | None = None
    for rnd in all_rounds:
        if rnd in round_team:
            last_known = round_team[rnd]
        elif last_known is not None:
            round_team[rnd] = last_known

    # Back-fill: if the first few rounds were missing, fill backward from
    # the first known round.
    if all_rounds and all_rounds[0] not in round_team:
        first_known = round_team[known_sorted[0]]
        for rnd in all_rounds:
            if rnd in round_team:
                break
            round_team[rnd] = first_known

    return round_team


def _detect_player_team(
    death_df: pd.DataFrame, steam_id: str
) -> str | None:
    """Return the team label (``'CT'`` or ``'T'``) for the player.

    Uses the player's ``team_num`` from their **earliest round** so that
    the halftime side-swap does not cause inconsistent labels between
    teammates.
    """
    _TEAM_MAP = {2: "T", 3: "CT"}
    if death_df.empty:
        return None
    sid = str(steam_id)
    for id_col, team_col in [
        ("attacker_steamid", "attacker_team_num"),
        ("user_steamid", "user_team_num"),
    ]:
        if id_col not in death_df.columns or team_col not in death_df.columns:
            continue
        mask = death_df[id_col].astype(str) == sid
        subset = death_df.loc[mask, [team_col]].copy()
        if "round" in death_df.columns:
            subset["round"] = death_df.loc[mask, "round"]
        subset = subset.dropna(subset=[team_col])
        if subset.empty:
            continue
        # Pick team_num from earliest round (before halftime swap)
        if "round" in subset.columns:
            earliest = subset.sort_values("round").iloc[0]
        else:
            earliest = subset.iloc[0]
        num = int(earliest[team_col])
        return _TEAM_MAP.get(num)
    return None


def _compute_hltv_rating(
    kast: float,
    kpr: float,
    dpr: float,
    impact: float,
    adr: float,
) -> float:
    """Apply the HLTV 2.0 rating formula and return a rounded result."""
    c = _HLTV_COEFFICIENTS
    rating = (
        c["kast_weight"] * kast
        + c["kpr_weight"] * kpr
        + c["dpr_weight"] * dpr
        + c["impact_weight"] * impact
        + c["adr_weight"] * adr
        + c["intercept"]
    )
    return round(rating, 4)


# ---------------------------------------------------------------------------
# Enriched round data for AI context
# ---------------------------------------------------------------------------

_TEAM_MAP_INV = {2: "T", 3: "CT"}

# Weapon display names
_WEAPON_NAMES: dict[str, str] = {
    "ak47": "AK-47", "m4a1": "M4A4", "m4a1_silencer": "M4A1-S",
    "m4a1_silencer_off": "M4A1-S", "awp": "AWP", "deagle": "Desert Eagle",
    "usp_silencer": "USP-S", "usp_silencer_off": "USP-S",
    "glock": "Glock-18", "p250": "P250",
    "fiveseven": "Five-SeveN", "tec9": "Tec-9", "cz75_auto": "CZ75-Auto",
    "elite": "Dual Berettas", "revolver": "R8 Revolver",
    "ssg08": "Scout", "scar20": "SCAR-20", "g3sg1": "G3SG1",
    "famas": "FAMAS", "galilar": "Galil AR",
    "aug": "AUG", "sg556": "SG 553",
    "mac10": "MAC-10", "mp9": "MP9", "mp7": "MP7", "mp5sd": "MP5-SD",
    "ump45": "UMP-45", "p90": "P90", "bizon": "PP-Bizon",
    "mag7": "MAG-7", "sawedoff": "Sawed-Off", "nova": "Nova",
    "xm1014": "XM1014", "m249": "M249", "negev": "Negev",
    "hkp2000": "P2000", "knife": "Knife", "knife_t": "Knife",
    "hegrenade": "HE Grenade", "inferno": "Molotov/Incendiary",
    "world": "World (fall/bomb)",
}


def _weapon_display(weapon: str) -> str:
    """Human-readable weapon name."""
    if not weapon:
        return "Unknown"
    clean = str(weapon).replace("weapon_", "")
    return _WEAPON_NAMES.get(clean, clean.upper())


def _classify_buy(player_spend: int, is_pistol_round: bool) -> str:
    """Classify a player's buy based on their spending."""
    if is_pistol_round:
        return "PISTOL"
    if player_spend >= 4000:
        return "FULL BUY"
    if player_spend >= 2500:
        return "HALF BUY"
    if player_spend >= 1000:
        return "FORCE BUY"
    return "ECO"


def build_enriched_rounds(
    parsed_data: dict[str, Any],
    steam_id: str,
    total_rounds: int,
) -> list[dict[str, Any]]:
    """Build enriched per-round data for AI context.

    Returns a list of dicts (one per round) containing economy, kill details,
    death details, utility usage, bomb events, and opening duel info.
    """
    death_df = parsed_data.get("player_death", pd.DataFrame())
    hurt_df = parsed_data.get("player_hurt", pd.DataFrame())
    round_end_df = parsed_data.get("round_end", pd.DataFrame())
    purchase_df = parsed_data.get("item_purchase", pd.DataFrame())
    blind_df = parsed_data.get("player_blind", pd.DataFrame())
    bomb_planted_df = parsed_data.get("bomb_planted", pd.DataFrame())
    bomb_defused_df = parsed_data.get("bomb_defused", pd.DataFrame())
    bomb_exploded_df = parsed_data.get("bomb_exploded", pd.DataFrame())
    positions_df = parsed_data.get("positions", pd.DataFrame())
    velocities_df = parsed_data.get("velocities", pd.DataFrame())
    weapon_fire_df = parsed_data.get("weapon_fire", pd.DataFrame())
    flash_det_df = parsed_data.get("flash_detonate", pd.DataFrame())
    he_det_df = parsed_data.get("he_detonate", pd.DataFrame())
    smoke_det_df = parsed_data.get("smoke_detonate", pd.DataFrame())
    molotov_det_df = parsed_data.get("molotov_detonate", pd.DataFrame())
    economy_df = parsed_data.get("economy", pd.DataFrame())

    header = parsed_data.get("header", {})
    map_name = str(header.get("map_name", "unknown"))

    sid = str(steam_id)
    player_team = _detect_player_team(death_df, sid)
    round_team_map = _build_round_team_map(death_df, sid, round_end_df)

    enriched: list[dict[str, Any]] = []

    for r in range(1, total_rounds + 1):
        round_data: dict[str, Any] = {"round": r}

        # --- Side ---
        round_data["side"] = round_team_map.get(r, _get_round_side(death_df, sid, r, player_team, total_rounds))

        # --- Economy ---
        round_data["economy"] = _get_round_economy(purchase_df, sid, r, total_rounds)

        # --- Money balances from tick snapshots ---
        if not economy_df.empty:
            eco_row = economy_df[
                (economy_df["round"] == r)
                & (economy_df["steamid"] == sid)
            ]
            if not eco_row.empty:
                row = eco_row.iloc[0]
                round_data["economy"]["start_money"] = int(row["start_balance"]) if pd.notna(row.get("start_balance")) else None
                round_data["economy"]["end_money"] = int(row["end_balance"]) if pd.notna(row.get("end_balance")) else None

        # --- Kill details ---
        round_data["kills_detail"] = _get_round_kills(
            death_df, sid, r, positions_df, map_name, velocities_df, hurt_df,
            weapon_fire_df,
        )

        # --- Death detail ---
        round_data["death_detail"] = _get_round_death(death_df, sid, r, positions_df, map_name)

        # --- Damage-only encounters ---
        round_data["damage_encounters"] = _get_round_damage_encounters(
            hurt_df, death_df, sid, r, velocities_df,
        )

        # --- Opening duel ---
        round_data["opening_duel"] = _get_opening_duel(death_df, sid, r)

        # --- Utility usage ---
        round_data["utility"] = _get_round_utility(
            death_df, hurt_df, blind_df, sid, r,
            weapon_fire_df=weapon_fire_df,
            flash_det_df=flash_det_df,
            he_det_df=he_det_df,
            smoke_det_df=smoke_det_df,
            molotov_det_df=molotov_det_df,
            positions_df=positions_df,
            map_name=map_name,
        )

        # --- Bomb events ---
        round_data["bomb"] = _get_round_bomb(
            bomb_planted_df, bomb_defused_df, bomb_exploded_df, sid, r
        )

        # --- Round outcome ---
        round_data["round_winner"] = _get_round_winner(round_end_df, r)
        round_data["round_reason"] = _get_round_reason(round_end_df, r)

        # --- Clutch detection ---
        round_data["clutch"] = _detect_clutch(death_df, sid, r, player_team)

        # --- Teamplayer incidents (team damage, team flashes) ---
        round_data["teamplayer"] = _get_round_teamplayer(
            hurt_df, blind_df, sid, r,
        )

        enriched.append(round_data)

    return enriched


def _get_round_side(
    death_df: pd.DataFrame, sid: str, rnd: int,
    first_half_team: str | None, total_rounds: int,
) -> str:
    """Determine if player is CT or T this round."""
    if not first_half_team:
        return "?"
    second_half_team = "T" if first_half_team == "CT" else "CT"
    # CS2 MR12: halftime is always after round 12
    if rnd <= 12:
        return first_half_team
    if rnd <= 24:
        return second_half_team
    # Overtime MR3: sides alternate every 3 rounds starting at round 25
    ot_half = (rnd - 25) // 3  # which OT half (0, 1, 2, ...)
    if ot_half % 2 == 0:
        return second_half_team
    return first_half_team


def _get_round_economy(
    purchase_df: pd.DataFrame, sid: str, rnd: int, total_rounds: int,
) -> dict[str, Any]:
    """Get economy info for this round."""
    result: dict[str, Any] = {
        "player_spend": 0,
        "buy_type": "ECO",
        "items": [],
    }
    if purchase_df.empty or "round" not in purchase_df.columns:
        return result

    id_col = None
    for col in ("steamid", "attacker_steamid", "user_steamid"):
        if col in purchase_df.columns:
            id_col = col
            break
    if not id_col:
        return result

    round_buys = purchase_df[
        (purchase_df["round"] == rnd)
        & (purchase_df[id_col].astype(str) == sid)
    ]
    if round_buys.empty:
        return result

    # Deduplicate: item_purchase fires on equip too; take unique items with costs
    items = []
    total_cost = 0
    if "item_name" in round_buys.columns:
        item_names = round_buys["item_name"].tolist()
    elif "weapon" in round_buys.columns:
        item_names = round_buys["weapon"].tolist()
    else:
        item_names = []

    costs = round_buys["cost"].tolist() if "cost" in round_buys.columns else [0] * len(item_names)

    # Filter out $0 items (default equipment) and deduplicate
    seen: dict[str, int] = {}
    for name, cost in zip(item_names, costs):
        name = str(name)
        cost = int(cost) if cost else 0
        if cost > 0:
            key = name
            if key not in seen or seen[key] < cost:
                seen[key] = cost

    items = list(seen.keys())
    total_cost = sum(seen.values())

    is_pistol = rnd in (1, 13) or (total_rounds > 24 and rnd == 25)
    result["player_spend"] = total_cost
    result["buy_type"] = _classify_buy(total_cost, is_pistol)
    result["items"] = items
    return result


def _lookup_position(
    positions_df: pd.DataFrame, steam_id: str, tick: int,
) -> tuple[float, float] | None:
    """Look up (X, Y) for a player at a specific tick from the positions DF."""
    if positions_df.empty:
        return None
    match = positions_df[
        (positions_df["steamid"] == steam_id)
        & (positions_df["tick"] == tick)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    x = row.get("X")
    y = row.get("Y")
    if x is None or y is None:
        return None
    return (float(x), float(y))


def _analyze_movement(
    velocities_df: pd.DataFrame, attacker_steamid: int, tick: int,
    window: int = 16,
) -> dict[str, Any] | None:
    """Compute movement metrics for the attacker at kill time.

    Returns a dict with:
      - shot_speed: speed at the kill tick (units/s)
      - pre_speed: peak speed in the window before the kill
      - movement_quality: 'standing' | 'counter-strafed' | 'running'
      - movement_direction: 'still' | 'forward' | 'backward' | 'left' | 'right'
    """
    import math

    if velocities_df.empty:
        return None

    atk = velocities_df[velocities_df["steamid"] == attacker_steamid]
    if atk.empty:
        return None

    # Get ticks in the window [tick-window .. tick]
    window_ticks = atk[
        (atk["tick"] >= tick - window) & (atk["tick"] <= tick)
    ].sort_values("tick")
    if window_ticks.empty:
        return None

    def _speed(row: Any) -> float:
        vx = row.get("velocity_X", 0)
        vy = row.get("velocity_Y", 0)
        if vx != vx or vy != vy:  # NaN check
            return 0.0
        vx = vx or 0
        vy = vy or 0
        return math.sqrt(vx ** 2 + vy ** 2)

    # Speed at shot tick (or closest)
    shot_row = window_ticks.iloc[-1]
    shot_speed = _speed(shot_row)

    # Peak speed in the window (ignoring NaN ticks)
    speeds = [_speed(r) for _, r in window_ticks.iterrows()]
    pre_speed = max(speeds) if speeds else 0.0

    # Movement quality classification
    # CS2 rifles max speed ~250, accurate below ~34% = ~85 u/s
    if shot_speed < 10:
        quality = "standing"
    elif shot_speed < 85:
        quality = "counter-strafed"
    else:
        quality = "running"

    # Movement direction relative to facing at shot tick
    vx = shot_row.get("velocity_X", 0)
    vy = shot_row.get("velocity_Y", 0)
    yaw = shot_row.get("yaw", 0)
    if vx != vx:
        vx = 0
    if vy != vy:
        vy = 0
    if yaw != yaw:
        yaw = 0
    vx = vx or 0
    vy = vy or 0
    yaw = yaw or 0

    if shot_speed < 10:
        direction = "still"
    else:
        move_angle = math.degrees(math.atan2(float(vy), float(vx)))
        relative = (move_angle - float(yaw) + 180) % 360 - 180
        if abs(relative) < 45:
            direction = "forward"
        elif abs(relative) > 135:
            direction = "backward"
        elif relative > 0:
            direction = "left"
        else:
            direction = "right"

    return {
        "shot_speed": round(shot_speed, 1),
        "pre_speed": round(pre_speed, 1),
        "movement_quality": quality,
        "movement_direction": direction,
    }


def _analyze_preaim(
    velocities_df: pd.DataFrame,
    attacker_steamid: int,
    victim_steamid: int,
    tick: int,
    offset: int = 32,
) -> dict[str, Any] | None:
    """Measure crosshair placement accuracy before the kill.

    Computes the angular distance between where the attacker was looking
    and where the victim actually was, at ``offset`` ticks before the kill
    (default 32 ticks ≈ 0.5s — before the engagement starts).

    Returns a dict with:
      - crosshair_error: angular offset in degrees (lower = better)
      - preaim_quality: 'excellent' | 'good' | 'moderate' | 'poor'
    """
    import math

    if velocities_df.empty:
        return None

    sample_tick = tick - offset

    # Get attacker state at sample tick
    atk = velocities_df[
        (velocities_df["steamid"] == attacker_steamid)
        & (velocities_df["tick"] == sample_tick)
    ]
    if atk.empty:
        return None
    atk_row = atk.iloc[0]

    # Get victim position at sample tick
    vic = velocities_df[
        (velocities_df["steamid"] == victim_steamid)
        & (velocities_df["tick"] == sample_tick)
    ]
    if vic.empty:
        return None
    vic_row = vic.iloc[0]

    # Extract values
    ax = atk_row.get("X", None)
    ay = atk_row.get("Y", None)
    az = atk_row.get("Z", None)
    a_yaw = atk_row.get("yaw", None)
    a_pitch = atk_row.get("pitch", None)
    vx = vic_row.get("X", None)
    vy = vic_row.get("Y", None)
    vz = vic_row.get("Z", None)

    # NaN guard
    vals = [ax, ay, az, a_yaw, a_pitch, vx, vy, vz]
    if any(v is None or v != v for v in vals):
        return None

    ax, ay, az = float(ax), float(ay), float(az)
    vx, vy, vz = float(vx), float(vy), float(vz)
    a_yaw, a_pitch = float(a_yaw), float(a_pitch)

    dx = vx - ax
    dy = vy - ay
    dz = vz - az
    horiz_dist = math.sqrt(dx * dx + dy * dy)
    if horiz_dist < 1.0:
        return None  # Too close, meaningless

    # Ideal angle to victim
    ideal_yaw = math.degrees(math.atan2(dy, dx))
    ideal_pitch = -math.degrees(math.atan2(dz, horiz_dist))

    # Angular difference (shortest arc)
    yaw_err = (ideal_yaw - a_yaw + 180) % 360 - 180
    pitch_err = ideal_pitch - a_pitch
    crosshair_error = math.sqrt(yaw_err ** 2 + pitch_err ** 2)

    if crosshair_error < 5:
        quality = "excellent"
    elif crosshair_error < 10:
        quality = "good"
    elif crosshair_error < 20:
        quality = "moderate"
    else:
        quality = "poor"

    return {
        "crosshair_error": round(crosshair_error, 1),
        "preaim_quality": quality,
    }


# ---------------------------------------------------------------------------
# Reaction-time analysis
# ---------------------------------------------------------------------------

# Angular threshold (degrees) to consider the crosshair "on target".
_AIM_ON_TARGET_DEG = 8.0

# Ticks of the look-back window before the first shot for reaction analysis.
_REACTION_WINDOW = 64  # ≈ 1 s at 64-tick


def _analyze_reaction_time(
    velocities_df: pd.DataFrame,
    attacker_steamid: int,
    victim_steamid: int,
    first_shot_tick: int,
    weapon_fire_df: pd.DataFrame | None = None,
    attacker_sid_str: str = "",
    rnd: int = 0,
) -> dict[str, Any] | None:
    """Estimate reaction time: how fast the player fired after aiming at the enemy.

    Walks backward from ``first_shot_tick`` through the velocity/yaw data
    to find the first tick where the attacker's crosshair was NOT aimed at
    the victim (angle > threshold).  The tick immediately after that is the
    "aim acquisition" tick.  Reaction time = first_shot_tick - acquisition tick.

    Returns a dict with:
      - reaction_ticks: ticks from aim-on-target to first shot
      - reaction_ms: same in milliseconds (assuming 64-tick)
      - category: 'lightning' | 'fast' | 'average' | 'slow'
    Returns None if data is insufficient or the player was pre-aimed.
    """
    import math

    if velocities_df.empty:
        return None

    window_start = first_shot_tick - _REACTION_WINDOW

    # Get attacker data in the window
    atk = velocities_df[
        (velocities_df["steamid"] == attacker_steamid)
        & (velocities_df["tick"] >= window_start)
        & (velocities_df["tick"] <= first_shot_tick)
    ].sort_values("tick")
    if len(atk) < 3:
        return None

    # Get victim data in the same window
    vic = velocities_df[
        (velocities_df["steamid"] == victim_steamid)
        & (velocities_df["tick"] >= window_start)
        & (velocities_df["tick"] <= first_shot_tick)
    ].sort_values("tick")
    if vic.empty:
        return None

    # Build a dict of victim positions indexed by tick for fast lookup
    vic_pos: dict[int, tuple[float, float, float]] = {}
    for _, row in vic.iterrows():
        t = int(row["tick"])
        x, y, z = row.get("X"), row.get("Y"), row.get("Z")
        if x is not None and x == x and y is not None and y == y:
            vic_pos[t] = (float(x), float(y), float(z) if (z is not None and z == z) else 0.0)

    if not vic_pos:
        return None

    def _angle_to_target(atk_row: Any, vpos: tuple[float, float, float]) -> float | None:
        """Compute angular distance from attacker's aim to victim position."""
        ax = atk_row.get("X")
        ay = atk_row.get("Y")
        az = atk_row.get("Z")
        a_yaw = atk_row.get("yaw")
        a_pitch = atk_row.get("pitch")
        if any(v is None or v != v for v in (ax, ay, az, a_yaw, a_pitch)):
            return None
        ax, ay, az = float(ax), float(ay), float(az)
        a_yaw, a_pitch = float(a_yaw), float(a_pitch)
        dx = vpos[0] - ax
        dy = vpos[1] - ay
        dz = vpos[2] - az
        horiz = math.sqrt(dx * dx + dy * dy)
        if horiz < 1.0:
            return None
        ideal_yaw = math.degrees(math.atan2(dy, dx))
        ideal_pitch = -math.degrees(math.atan2(dz, horiz))
        yaw_err = (ideal_yaw - a_yaw + 180) % 360 - 180
        pitch_err = ideal_pitch - a_pitch
        return math.sqrt(yaw_err ** 2 + pitch_err ** 2)

    # Walk backward from the shot tick to find when aim was NOT on target
    atk_rows = list(atk.iterrows())
    atk_rows.reverse()  # newest first

    # Find the closest victim tick for each attacker tick
    vic_ticks_sorted = sorted(vic_pos.keys())

    def _closest_vic(t: int) -> tuple[float, float, float] | None:
        # Binary search for closest tick
        import bisect
        idx = bisect.bisect_left(vic_ticks_sorted, t)
        candidates = []
        if idx < len(vic_ticks_sorted):
            candidates.append(vic_ticks_sorted[idx])
        if idx > 0:
            candidates.append(vic_ticks_sorted[idx - 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda ct: abs(ct - t))
        if abs(best - t) > 8:  # too far apart
            return None
        return vic_pos[best]

    # Walk backward: find the first tick where aim diverges from target
    acquisition_tick = None
    for _, atk_row in atk_rows:
        t = int(atk_row["tick"])
        vp = _closest_vic(t)
        if vp is None:
            continue
        angle = _angle_to_target(atk_row, vp)
        if angle is None:
            continue
        if angle > _AIM_ON_TARGET_DEG:
            # Aim was OFF target at this tick — the next tick is acquisition
            break
        acquisition_tick = t

    if acquisition_tick is None:
        # Player was already aimed at the target for the entire window (pre-aimed)
        return None

    if acquisition_tick >= first_shot_tick:
        # No measurable gap
        return None

    reaction_ticks = first_shot_tick - acquisition_tick
    reaction_ms = round(reaction_ticks / 64 * 1000)

    # Filter out implausibly long values (>800ms usually isn't a "reaction")
    if reaction_ms > 800:
        return None

    if reaction_ms < 150:
        category = "lightning"
    elif reaction_ms < 200:
        category = "fast"
    elif reaction_ms < 300:
        category = "average"
    else:
        category = "slow"

    return {
        "reaction_ticks": reaction_ticks,
        "reaction_ms": reaction_ms,
        "category": category,
    }


# Maximum gap (in ticks) between consecutive damage events before we
# consider them separate encounters.  128 ticks ≈ 2 s at 64-tick.
_ENGAGEMENT_GAP = 128

# How far before the first hit we look for weapon_fire events that are
# likely misses aimed at the same target.  64 ticks ≈ 1 s.
_PRE_HIT_WINDOW = 64


def _analyze_time_to_damage(
    hurt_df: pd.DataFrame,
    attacker_sid: str,
    victim_sid: str,
    rnd: int,
    kill_tick: int,
    weapon_fire_df: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Compute engagement reaction time for an attacker→victim kill.

    Clusters damage events by gap to isolate the *final* continuous
    engagement leading to the kill (discards earlier poke damage from
    a prior encounter).  Then looks for weapon_fire events shortly
    before the first hit of that engagement to capture missed shots.

    Returns a dict with:
      - first_shot_tick: tick of first shot (or first hit if no fire data)
      - first_hit_tick: tick of first damage in the engagement
      - ttk_ticks: ticks from first shot to kill
      - ttk_seconds: same in seconds (assuming 64-tick)
      - hits: damage events in the engagement
      - shots_fired: total shots in the engagement window
    """
    if hurt_df.empty or "round" not in hurt_df.columns:
        return None

    pair_hits = hurt_df[
        (hurt_df["round"] == rnd)
        & (hurt_df["attacker_steamid"].astype(str) == attacker_sid)
        & (hurt_df["user_steamid"].astype(str) == victim_sid)
        & (hurt_df["tick"] <= kill_tick)
    ]
    if pair_hits.empty:
        return None

    pair_hits = pair_hits.sort_values("tick")
    ticks = pair_hits["tick"].tolist()

    # Walk backward from the kill and find the start of the final engagement
    cluster_start_idx = 0
    for i in range(len(ticks) - 1):
        if ticks[i + 1] - ticks[i] > _ENGAGEMENT_GAP:
            cluster_start_idx = i + 1

    engage_first_hit = int(ticks[cluster_start_idx])
    engage_hits = len(ticks) - cluster_start_idx

    # Hitgroup distribution for the engagement cluster
    engage_rows = pair_hits.iloc[cluster_start_idx:]
    hitgroups: list[str] = []
    if "hitgroup" in engage_rows.columns:
        hitgroups = engage_rows["hitgroup"].dropna().astype(str).str.lower().tolist()

    # Look for weapon_fire events (misses) shortly before the first hit
    first_shot_tick = engage_first_hit
    shots_fired = engage_hits  # at least as many as hits

    if (
        weapon_fire_df is not None
        and not weapon_fire_df.empty
        and "round" in weapon_fire_df.columns
    ):
        fires = weapon_fire_df[
            (weapon_fire_df["round"] == rnd)
            & (weapon_fire_df["user_steamid"].astype(str) == attacker_sid)
            & (weapon_fire_df["tick"] >= engage_first_hit - _PRE_HIT_WINDOW)
            & (weapon_fire_df["tick"] <= kill_tick)
        ].sort_values("tick")
        if not fires.empty:
            first_shot_tick = min(first_shot_tick, int(fires.iloc[0]["tick"]))
            shots_fired = len(fires)

    ttk_ticks = kill_tick - first_shot_tick

    result = {
        "first_shot_tick": first_shot_tick,
        "first_hit_tick": engage_first_hit,
        "ttk_ticks": ttk_ticks,
        "ttk_seconds": round(ttk_ticks / 64, 3),
        "hits": engage_hits,
        "shots_fired": shots_fired,
    }

    # Accuracy metrics for this engagement
    if shots_fired > 0:
        hit_pct = min(100.0, round(engage_hits / shots_fired * 100, 1))
        # First-bullet accuracy: did the first shot hit?
        first_bullet_hit = engage_first_hit <= first_shot_tick + 2  # small tick tolerance
        # Hitgroup breakdown: string labels from demoparser2
        head = sum(1 for h in hitgroups if h in ("head", "neck"))
        upper = sum(1 for h in hitgroups if h in ("chest", "stomach"))
        lower = sum(1 for h in hitgroups if h in ("left_arm", "right_arm", "left_leg", "right_leg"))
        result["accuracy"] = {
            "hit_pct": hit_pct,
            "first_bullet_hit": first_bullet_hit,
            "hitgroups": hitgroups,
            "head": head,
            "upper": upper,
            "lower": lower,
        }

    return result


def _get_round_kills(
    death_df: pd.DataFrame, sid: str, rnd: int,
    positions_df: pd.DataFrame | None = None, map_name: str = "",
    velocities_df: pd.DataFrame | None = None,
    hurt_df: pd.DataFrame | None = None,
    weapon_fire_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Get detailed kill info for each kill the player got this round."""
    if death_df.empty or "round" not in death_df.columns:
        return []

    kills = death_df[
        (death_df["round"] == rnd)
        & (death_df["attacker_steamid"].astype(str) == sid)
    ]
    if kills.empty:
        return []

    use_callouts = (
        positions_df is not None
        and not positions_df.empty
        and is_map_supported(map_name)
    )
    has_positions = positions_df is not None and not positions_df.empty

    result = []
    for _, row in kills.iterrows():
        kill_info: dict[str, Any] = {
            "victim": str(row.get("user_name", "?")),
        }
        if "weapon" in death_df.columns:
            kill_info["weapon"] = _weapon_display(str(row.get("weapon", "")))
        if "headshot" in death_df.columns:
            kill_info["headshot"] = bool(row.get("headshot", False))
        if "distance" in death_df.columns:
            kill_info["distance"] = round(float(row.get("distance", 0)), 1)
        # Special conditions
        specials = []
        if row.get("noscope"):
            specials.append("noscope")
        if row.get("thrusmoke"):
            specials.append("thru smoke")
        if row.get("penetrated") and int(row.get("penetrated", 0)) > 0:
            specials.append("wallbang")
        if row.get("attackerblind"):
            specials.append("while blind")
        if specials:
            kill_info["specials"] = specials

        # Position coords + callouts (coords need positions_df; callouts also need zone data)
        if has_positions and "tick" in death_df.columns:
            tick = int(row["tick"])
            attacker_sid = sid
            victim_sid = str(row.get("user_steamid", ""))
            attacker_pos = _lookup_position(positions_df, attacker_sid, tick)
            victim_pos = _lookup_position(positions_df, victim_sid, tick)
            if attacker_pos:
                kill_info["attacker_xy"] = [round(attacker_pos[0], 1), round(attacker_pos[1], 1)]
                if use_callouts:
                    kill_info["attacker_position"] = get_callout(map_name, attacker_pos[0], attacker_pos[1])
            if victim_pos:
                kill_info["victim_xy"] = [round(victim_pos[0], 1), round(victim_pos[1], 1)]
                if use_callouts:
                    kill_info["victim_position"] = get_callout(map_name, victim_pos[0], victim_pos[1])

        # Movement analysis, pre-aim, time-to-damage
        if "tick" in death_df.columns:
            tick = int(row["tick"])
            try:
                atk_steamid = int(sid)
            except (ValueError, TypeError):
                atk_steamid = None
            victim_sid_str = str(row.get("user_steamid", ""))

            if atk_steamid is not None and velocities_df is not None and not velocities_df.empty:
                movement = _analyze_movement(velocities_df, atk_steamid, tick)
                if movement:
                    kill_info["movement"] = movement

                try:
                    vic_steamid = int(victim_sid_str)
                except (ValueError, TypeError):
                    vic_steamid = None
                if vic_steamid is not None:
                    preaim = _analyze_preaim(velocities_df, atk_steamid, vic_steamid, tick)
                    if preaim:
                        kill_info["preaim"] = preaim

            if hurt_df is not None and not hurt_df.empty:
                ttd = _analyze_time_to_damage(
                    hurt_df, sid, victim_sid_str, rnd, tick, weapon_fire_df,
                )
                if ttd:
                    kill_info["ttd"] = ttd

            # Reaction time (yaw-snap approach)
            if (
                atk_steamid is not None
                and vic_steamid is not None
                and velocities_df is not None
                and not velocities_df.empty
            ):
                first_shot_tick = (
                    ttd["first_shot_tick"] if ttd and "first_shot_tick" in ttd else tick
                )
                rxn = _analyze_reaction_time(
                    velocities_df, atk_steamid, vic_steamid,
                    first_shot_tick, weapon_fire_df,
                    attacker_sid_str=sid, rnd=rnd,
                )
                if rxn:
                    kill_info["reaction"] = rxn

        result.append(kill_info)

    return result


def _get_round_death(
    death_df: pd.DataFrame, sid: str, rnd: int,
    positions_df: pd.DataFrame | None = None, map_name: str = "",
) -> dict[str, Any] | None:
    """Get how the player died this round (None if survived)."""
    if death_df.empty or "round" not in death_df.columns:
        return None

    deaths = death_df[
        (death_df["round"] == rnd)
        & (death_df["user_steamid"].astype(str) == sid)
    ]
    if deaths.empty:
        return None

    row = deaths.iloc[0]
    info: dict[str, Any] = {
        "killer": str(row.get("attacker_name", "?")),
    }
    if "weapon" in death_df.columns:
        info["weapon"] = _weapon_display(str(row.get("weapon", "")))
    if "headshot" in death_df.columns:
        info["headshot"] = bool(row.get("headshot", False))
    if "distance" in death_df.columns:
        info["distance"] = round(float(row.get("distance", 0)), 1)

    # Position coords + callouts
    has_positions = positions_df is not None and not positions_df.empty
    use_callouts = has_positions and is_map_supported(map_name)
    if has_positions and "tick" in death_df.columns:
        tick = int(row["tick"])
        killer_sid = str(row.get("attacker_steamid", ""))
        killer_pos = _lookup_position(positions_df, killer_sid, tick)
        victim_pos = _lookup_position(positions_df, sid, tick)
        if killer_pos:
            info["killer_xy"] = [round(killer_pos[0], 1), round(killer_pos[1], 1)]
            if use_callouts:
                info["killer_position"] = get_callout(map_name, killer_pos[0], killer_pos[1])
        if victim_pos:
            info["victim_xy"] = [round(victim_pos[0], 1), round(victim_pos[1], 1)]
            if use_callouts:
                info["victim_position"] = get_callout(map_name, victim_pos[0], victim_pos[1])

    return info


def _get_round_damage_encounters(
    hurt_df: pd.DataFrame,
    death_df: pd.DataFrame,
    sid: str,
    rnd: int,
    velocities_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Get damage-only encounters: enemies hurt but not killed by the player."""
    if hurt_df.empty or "round" not in hurt_df.columns:
        return []

    player_damage = hurt_df[
        (hurt_df["round"] == rnd)
        & (hurt_df["attacker_steamid"].astype(str) == sid)
    ]
    if player_damage.empty:
        return []

    # Find victims the player killed this round
    killed_victims: set[str] = set()
    if not death_df.empty and "round" in death_df.columns:
        kills = death_df[
            (death_df["round"] == rnd)
            & (death_df["attacker_steamid"].astype(str) == sid)
        ]
        killed_victims = set(kills["user_steamid"].astype(str))

    result: list[dict[str, Any]] = []
    for victim_sid, group in player_damage.groupby(
        player_damage["user_steamid"].astype(str)
    ):
        if victim_sid in killed_victims:
            continue  # Already counted as a kill

        first_hit = group.sort_values("tick").iloc[0]
        tick = int(first_hit["tick"])
        weapon = str(first_hit.get("weapon", ""))

        enc: dict[str, Any] = {
            "weapon": _weapon_display(weapon),
        }

        # Movement and preaim analysis at time of first hit
        if velocities_df is not None and not velocities_df.empty:
            try:
                atk_steamid = int(sid)
            except (ValueError, TypeError):
                atk_steamid = None

            if atk_steamid is not None:
                movement = _analyze_movement(velocities_df, atk_steamid, tick)
                if movement:
                    enc["movement"] = movement

                try:
                    vic_steamid_int = int(victim_sid)
                except (ValueError, TypeError):
                    vic_steamid_int = None
                if vic_steamid_int is not None:
                    preaim = _analyze_preaim(
                        velocities_df, atk_steamid, vic_steamid_int, tick,
                    )
                    if preaim:
                        enc["preaim"] = preaim

        result.append(enc)

    return result


def _get_opening_duel(
    death_df: pd.DataFrame, sid: str, rnd: int,
) -> dict[str, Any] | None:
    """Check if the player was involved in the first kill of the round."""
    if death_df.empty or "round" not in death_df.columns or "tick" not in death_df.columns:
        return None

    round_kills = death_df[death_df["round"] == rnd]
    if round_kills.empty:
        return None

    first_kill = round_kills.sort_values("tick").iloc[0]
    attacker = str(first_kill.get("attacker_steamid", ""))
    victim = str(first_kill.get("user_steamid", ""))

    if attacker == sid:
        return {
            "role": "opening_kill",
            "opponent": str(first_kill.get("user_name", "?")),
            "weapon": _weapon_display(str(first_kill.get("weapon", ""))) if "weapon" in death_df.columns else "?",
        }
    elif victim == sid:
        return {
            "role": "opening_death",
            "opponent": str(first_kill.get("attacker_name", "?")),
            "weapon": _weapon_display(str(first_kill.get("weapon", ""))) if "weapon" in death_df.columns else "?",
        }
    return None


def _get_round_utility(
    death_df: pd.DataFrame,
    hurt_df: pd.DataFrame,
    blind_df: pd.DataFrame,
    sid: str,
    rnd: int,
    *,
    weapon_fire_df: pd.DataFrame | None = None,
    flash_det_df: pd.DataFrame | None = None,
    he_det_df: pd.DataFrame | None = None,
    smoke_det_df: pd.DataFrame | None = None,
    molotov_det_df: pd.DataFrame | None = None,
    positions_df: pd.DataFrame | None = None,
    map_name: str = "",
) -> dict[str, Any]:
    """Get utility usage stats for the player this round.

    Includes per-grenade details with throw/land positions and per-instance
    flash blind data with enemy/friendly distinction.
    """
    use_callouts = (
        positions_df is not None
        and not positions_df.empty
        and is_map_supported(map_name)
    )

    util: dict[str, Any] = {
        "enemies_flashed": 0,
        "avg_blind_duration": 0.0,
        "flash_assists": 0,
        "he_damage": 0,
        "flash_victims": [],
        "molotov_damage": [],
        "grenades": [],       # per-grenade detail with positions
        "flash_instances": [], # per-instance flash blind (enemy & friendly separate)
    }

    # ── Per-grenade detail with throw → land positions ────────────────
    _wf = weapon_fire_df if weapon_fire_df is not None else pd.DataFrame()
    _grenade_weapons = {
        "weapon_flashbang": "flash",
        "weapon_smokegrenade": "smoke",
        "weapon_hegrenade": "he",
        "weapon_molotov": "molotov",
        "weapon_incgrenade": "molotov",
    }
    _det_dfs: dict[str, pd.DataFrame] = {
        "flash": flash_det_df if flash_det_df is not None else pd.DataFrame(),
        "he": he_det_df if he_det_df is not None else pd.DataFrame(),
        "smoke": smoke_det_df if smoke_det_df is not None else pd.DataFrame(),
        "molotov": molotov_det_df if molotov_det_df is not None else pd.DataFrame(),
    }

    # Collect grenade throws this round
    if not _wf.empty and "round" in _wf.columns and "weapon" in _wf.columns:
        id_col = _find_id_col(_wf, ("user_steamid", "steamid", "attacker_steamid"))
        if id_col:
            round_fires = _wf[
                (_wf["round"] == rnd)
                & (_wf[id_col].astype(str) == sid)
            ]
            # Track detonation index per type to match throws with detonations
            det_indices: dict[str, int] = {"flash": 0, "he": 0, "smoke": 0, "molotov": 0}

            for _, frow in round_fires.iterrows():
                wep = str(frow.get("weapon", "")).lower()
                nade_type = _grenade_weapons.get(wep)
                if not nade_type:
                    continue

                nade_info: dict[str, Any] = {"type": nade_type}

                # Throw position from positions_df at weapon_fire tick
                throw_tick = int(frow.get("tick", 0)) if "tick" in frow.index else 0
                if throw_tick and use_callouts:
                    throw_pos = _lookup_position(positions_df, sid, throw_tick)
                    if throw_pos:
                        nade_info["throw_xy"] = [round(throw_pos[0], 1), round(throw_pos[1], 1)]
                        nade_info["throw_callout"] = get_callout(map_name, throw_pos[0], throw_pos[1])

                # Land position from detonation DF
                det_df = _det_dfs.get(nade_type, pd.DataFrame())
                if not det_df.empty and "round" in det_df.columns:
                    det_id_col = _find_id_col(det_df, ("user_steamid", "steamid", "attacker_steamid"))
                    if det_id_col:
                        round_dets = det_df[
                            (det_df["round"] == rnd)
                            & (det_df[det_id_col].astype(str) == sid)
                        ]
                    else:
                        round_dets = det_df[det_df["round"] == rnd]
                    idx = det_indices[nade_type]
                    if idx < len(round_dets):
                        drow = round_dets.iloc[idx]
                        dx = float(drow.get("x", 0))
                        dy = float(drow.get("y", 0))
                        if dx != 0 or dy != 0:
                            nade_info["land_xy"] = [round(dx, 1), round(dy, 1)]
                            if is_map_supported(map_name):
                                nade_info["land_callout"] = get_callout(map_name, dx, dy)
                    det_indices[nade_type] = idx + 1

                util["grenades"].append(nade_info)

    # ── Flash effectiveness from player_blind ─────────────────────────
    if not blind_df.empty and "round" in blind_df.columns:
        id_col = None
        for col in ("attacker_steamid", "user_steamid", "steamid"):
            if col in blind_df.columns:
                id_col = col
                break
        if id_col:
            round_blinds = blind_df[
                (blind_df["round"] == rnd)
                & (blind_df[id_col].astype(str) == sid)
            ]
            if not round_blinds.empty and "blind_duration" in round_blinds.columns:
                # Determine victim's team vs. attacker's team
                atk_team_col = "attacker_team_num" if "attacker_team_num" in round_blinds.columns else None
                vic_team_col = "user_team_num" if "user_team_num" in round_blinds.columns else None
                victim_name_col = "user_name" if id_col == "attacker_steamid" else "attacker_name"

                enemies_flashed = 0
                total_enemy_dur = 0.0
                victims = []
                flash_instances = []

                for _, brow in round_blinds.iterrows():
                    dur = round(float(brow.get("blind_duration", 0)), 2)
                    vname = str(brow.get(victim_name_col, "?")) if victim_name_col in round_blinds.columns else "?"

                    is_team = False
                    if atk_team_col and vic_team_col:
                        try:
                            is_team = int(brow[atk_team_col]) == int(brow[vic_team_col])
                        except (ValueError, TypeError):
                            pass

                    # Look up victim position at the flash tick
                    victim_xy = None
                    if (
                        positions_df is not None
                        and not positions_df.empty
                        and "tick" in brow.index
                        and "user_steamid" in brow.index
                    ):
                        vtick = int(brow["tick"])
                        vsid = str(brow["user_steamid"])
                        vmatch = positions_df[
                            (positions_df["tick"] == vtick)
                            & (positions_df["steamid"] == vsid)
                        ]
                        if not vmatch.empty:
                            vr = vmatch.iloc[0]
                            victim_xy = [round(float(vr["X"]), 1), round(float(vr["Y"]), 1)]

                    inst = {
                        "name": vname,
                        "duration": dur,
                        "is_friendly": is_team,
                    }
                    if victim_xy:
                        inst["victim_xy"] = victim_xy
                    flash_instances.append(inst)

                    if not is_team:
                        enemies_flashed += 1
                        total_enemy_dur += dur
                        v_entry: dict[str, Any] = {"name": vname, "duration": dur}
                        if victim_xy:
                            v_entry["victim_xy"] = victim_xy
                        victims.append(v_entry)

                util["enemies_flashed"] = enemies_flashed
                util["avg_blind_duration"] = round(
                    total_enemy_dur / enemies_flashed, 1
                ) if enemies_flashed > 0 else 0.0
                util["flash_victims"] = victims
                util["flash_instances"] = flash_instances

    # Flash assists from death events
    if not death_df.empty and "round" in death_df.columns and "assistedflash" in death_df.columns:
        flash_assists = death_df[
            (death_df["round"] == rnd)
            & (death_df.get("assister_steamid", pd.Series(dtype=str)).astype(str) == sid)
            & (death_df["assistedflash"] == True)  # noqa: E712
        ]
        util["flash_assists"] = len(flash_assists)

    # HE damage from player_hurt
    if not hurt_df.empty and "round" in hurt_df.columns and "weapon" in hurt_df.columns:
        he_dmg = hurt_df[
            (hurt_df["round"] == rnd)
            & (hurt_df["attacker_steamid"].astype(str) == sid)
            & (hurt_df["weapon"].astype(str).str.contains("hegrenade", case=False, na=False))
        ]
        if not he_dmg.empty and "dmg_health" in he_dmg.columns:
            util["he_damage"] = int(he_dmg["dmg_health"].sum())
            # Aggregate HE damage per victim with positions
            he_victims_map: dict[str, dict[str, Any]] = {}
            for _, hrow in he_dmg.iterrows():
                vname = str(hrow.get("user_name", "?")) if "user_name" in he_dmg.columns else "?"
                if vname not in he_victims_map:
                    he_victims_map[vname] = {"name": vname, "damage": 0}
                he_victims_map[vname]["damage"] += int(hrow["dmg_health"])
                if "victim_xy" not in he_victims_map[vname] and positions_df is not None and not positions_df.empty:
                    htick = int(hrow["tick"])
                    hsid = str(hrow.get("user_steamid", ""))
                    hmatch = positions_df[
                        (positions_df["tick"] == htick)
                        & (positions_df["steamid"] == hsid)
                    ]
                    if not hmatch.empty:
                        hr = hmatch.iloc[0]
                        he_victims_map[vname]["victim_xy"] = [round(float(hr["X"]), 1), round(float(hr["Y"]), 1)]
            util["he_victims"] = list(he_victims_map.values())

    # Molotov/incendiary damage per victim
    if not hurt_df.empty and "round" in hurt_df.columns and "weapon" in hurt_df.columns:
        molly_dmg = hurt_df[
            (hurt_df["round"] == rnd)
            & (hurt_df["attacker_steamid"].astype(str) == sid)
            & (hurt_df["weapon"].astype(str).str.contains("inferno|molotov", case=False, na=False))
        ]
        if not molly_dmg.empty and "dmg_health" in molly_dmg.columns:
            victim_col = "user_name" if "user_name" in molly_dmg.columns else None
            if victim_col:
                molly_victims_map: dict[str, dict[str, Any]] = {}
                for _, mrow in molly_dmg.iterrows():
                    vname = str(mrow.get(victim_col, "?"))
                    if vname not in molly_victims_map:
                        molly_victims_map[vname] = {"victim": vname, "damage": 0}
                    molly_victims_map[vname]["damage"] += int(mrow["dmg_health"])
                    if "victim_xy" not in molly_victims_map[vname] and positions_df is not None and not positions_df.empty:
                        mtick = int(mrow["tick"])
                        msid = str(mrow.get("user_steamid", ""))
                        mmatch = positions_df[
                            (positions_df["tick"] == mtick)
                            & (positions_df["steamid"] == msid)
                        ]
                        if not mmatch.empty:
                            mr = mmatch.iloc[0]
                            molly_victims_map[vname]["victim_xy"] = [round(float(mr["X"]), 1), round(float(mr["Y"]), 1)]
                util["molotov_damage"] = list(molly_victims_map.values())

    return util


def _get_round_teamplayer(
    hurt_df: pd.DataFrame,
    blind_df: pd.DataFrame,
    sid: str,
    rnd: int,
) -> dict[str, Any]:
    """Get teamplayer incidents: team damage dealt and teammates flashed."""
    tp: dict[str, Any] = {"team_damage": [], "team_flashes": []}

    # Team damage (same-team hurt, not self)
    if not hurt_df.empty and "round" in hurt_df.columns:
        id_col = _find_id_col(hurt_df, ("attacker_steamid",))
        if (
            id_col
            and "attacker_team_num" in hurt_df.columns
            and "user_team_num" in hurt_df.columns
        ):
            attacks = hurt_df[
                (hurt_df["round"] == rnd)
                & (hurt_df[id_col].astype(str) == sid)
            ]
            if not attacks.empty:
                same = attacks[
                    attacks["attacker_team_num"] == attacks["user_team_num"]
                ]
                # Exclude self-damage
                vic_col = _find_id_col(same, ("user_steamid",))
                if vic_col:
                    same = same[same[vic_col].astype(str) != sid]
                for _, row in same.iterrows():
                    vname = str(row.get("user_name", "?")) if "user_name" in same.columns else "?"
                    dmg = int(row.get("dmg_health", 0)) if "dmg_health" in same.columns else 0
                    wpn = str(row.get("weapon", "?")) if "weapon" in same.columns else "?"
                    tp["team_damage"].append({"victim": vname, "damage": dmg, "weapon": wpn})

    # Team flashes (from flash_instances already in utility, but compute here for AI)
    if not blind_df.empty and "round" in blind_df.columns:
        id_col = _find_id_col(blind_df, ("attacker_steamid",))
        if (
            id_col
            and "attacker_team_num" in blind_df.columns
            and "user_team_num" in blind_df.columns
        ):
            blinds = blind_df[
                (blind_df["round"] == rnd)
                & (blind_df[id_col].astype(str) == sid)
            ]
            if not blinds.empty:
                same = blinds[
                    blinds["attacker_team_num"] == blinds["user_team_num"]
                ]
                for _, row in same.iterrows():
                    vname = str(row.get("user_name", "?")) if "user_name" in same.columns else "?"
                    dur = round(float(row.get("blind_duration", 0)), 2)
                    tp["team_flashes"].append({"victim": vname, "duration": dur})

    return tp


def _get_round_bomb(
    planted_df: pd.DataFrame,
    defused_df: pd.DataFrame,
    exploded_df: pd.DataFrame,
    sid: str,
    rnd: int,
) -> dict[str, Any] | None:
    """Get bomb event info for this round related to the player."""
    result: dict[str, Any] = {}

    # Check if player planted
    if not planted_df.empty and "round" in planted_df.columns:
        id_col = None
        for col in ("user_steamid", "attacker_steamid", "steamid"):
            if col in planted_df.columns:
                id_col = col
                break
        if id_col:
            plants = planted_df[
                (planted_df["round"] == rnd)
                & (planted_df[id_col].astype(str) == sid)
            ]
            if not plants.empty:
                site = str(plants.iloc[0].get("site", "?")) if "site" in planted_df.columns else "?"
                result["planted"] = site

    # Check if player defused
    if not defused_df.empty and "round" in defused_df.columns:
        id_col = None
        for col in ("user_steamid", "attacker_steamid", "steamid"):
            if col in defused_df.columns:
                id_col = col
                break
        if id_col:
            defuses = defused_df[
                (defused_df["round"] == rnd)
                & (defused_df[id_col].astype(str) == sid)
            ]
            if not defuses.empty:
                result["defused"] = True

    # Check if bomb exploded this round (not player-specific)
    if not exploded_df.empty and "round" in exploded_df.columns:
        if not exploded_df[exploded_df["round"] == rnd].empty:
            result["exploded"] = True

    return result if result else None


def _get_round_winner(round_end_df: pd.DataFrame, rnd: int) -> str | None:
    """Get which team won this round."""
    if round_end_df.empty or "round" not in round_end_df.columns:
        return None
    row = round_end_df[round_end_df["round"] == rnd]
    if row.empty or "winner" not in round_end_df.columns:
        return None
    return str(row.iloc[0]["winner"])


def _get_round_reason(round_end_df: pd.DataFrame, rnd: int) -> str | None:
    """Get the reason a round ended (e.g. t_killed, ct_killed, bomb_defused)."""
    if round_end_df.empty or "round" not in round_end_df.columns:
        return None
    row = round_end_df[round_end_df["round"] == rnd]
    if row.empty or "reason" not in round_end_df.columns:
        return None
    return str(row.iloc[0]["reason"])


def _detect_clutch(
    death_df: pd.DataFrame, sid: str, rnd: int,
    player_team: str | None,
) -> dict[str, Any] | None:
    """Detect if the player was in a clutch situation (1vN) this round."""
    if death_df.empty or "round" not in death_df.columns or not player_team:
        return None
    if "tick" not in death_df.columns:
        return None

    _team_num = {"CT": 3, "T": 2}
    player_team_num = _team_num.get(player_team)
    if not player_team_num:
        return None

    round_deaths = death_df[death_df["round"] == rnd].sort_values("tick")
    if round_deaths.empty:
        return None

    # Track teammates alive (5 start)
    teammates_alive = 5
    enemies_alive = 5
    clutch_started = False
    enemies_at_clutch = 0

    for _, row in round_deaths.iterrows():
        victim_team = int(row.get("user_team_num", 0))
        victim_sid = str(row.get("user_steamid", ""))

        if victim_team == player_team_num:
            if victim_sid != sid:
                teammates_alive -= 1
            else:
                # Player died — if already in clutch, they lost
                if clutch_started:
                    return {"vs": enemies_at_clutch, "won": False}
                return None
        else:
            enemies_alive -= 1

        # Check: is the player the last one alive on their team?
        if teammates_alive == 1 and not clutch_started and enemies_alive >= 2:
            # Player must still be alive (check they haven't died yet this round)
            player_died = death_df[
                (death_df["round"] == rnd)
                & (death_df["user_steamid"].astype(str) == sid)
            ]
            if player_died.empty:
                clutch_started = True
                enemies_at_clutch = enemies_alive

    if clutch_started:
        # Player survived = won the clutch
        return {"vs": enemies_at_clutch, "won": True}
    return None
