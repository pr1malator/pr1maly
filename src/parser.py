"""
Layer 1: Demo Parsing Engine
Reads a raw CS2 .dem file using demoparser2 and returns structured DataFrames
of in-game events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def parse_demo(demo_path: str | Path) -> dict[str, Any]:
    """
    Parse a CS2 .dem file and return a dictionary of event DataFrames.

    Args:
        demo_path: Path to the .dem file to parse.

    Returns:
        A dict with keys:
            - ``player_death``: DataFrame of kill/death events (enriched).
            - ``player_hurt``: DataFrame of damage events.
            - ``round_end``: DataFrame of round-end events.
            - ``item_purchase``: DataFrame of item purchase events.
            - ``player_blind``: DataFrame of flash blind events.
            - ``bomb_planted``: DataFrame of bomb plant events.
            - ``bomb_defused``: DataFrame of bomb defuse events.
            - ``bomb_exploded``: DataFrame of bomb explosion events.
            - ``header``: Dict of match metadata (map name, game type, etc.).
    """
    try:
        from demoparser2 import DemoParser  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "demoparser2 is required for demo parsing. "
            "Install it with: pip install demoparser2"
        ) from exc

    demo_path = Path(demo_path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    parser = DemoParser(str(demo_path))

    header: dict[str, Any] = parser.parse_header()

    # Enriched death events — weapon, headshot, distance, special conditions
    death_df: pd.DataFrame = parser.parse_event(
        "player_death",
        player=["steamid", "name", "team_num"],
    )

    hurt_df: pd.DataFrame = parser.parse_event(
        "player_hurt",
        player=["steamid", "name", "team_num"],
    )

    round_end_df: pd.DataFrame = parser.parse_event("round_end")

    # Round freeze-end events (marks when buy time ends and action starts)
    round_freeze_end_df = _safe_parse_event(parser, "round_freeze_end", [])

    # Economy: item purchases with costs
    item_purchase_df = _safe_parse_event(parser, "item_purchase", ["steamid", "name", "team_num"])

    # Flash blinds
    player_blind_df = _safe_parse_event(parser, "player_blind", ["steamid", "name", "team_num"])

    # Bomb events
    bomb_planted_df = _safe_parse_event(parser, "bomb_planted", ["steamid", "name", "team_num"])
    bomb_defused_df = _safe_parse_event(parser, "bomb_defused", ["steamid", "name", "team_num"])
    bomb_exploded_df = _safe_parse_event(parser, "bomb_exploded", [])

    # Weapon fire events (for reaction-time / first-shot analysis)
    weapon_fire_df = _safe_parse_event(parser, "weapon_fire", ["steamid", "name"])

    # Grenade detonation events (for positional utility tracking)
    flash_detonate_df = _safe_parse_event(parser, "flashbang_detonate", ["steamid", "name"])
    he_detonate_df = _safe_parse_event(parser, "hegrenade_detonate", ["steamid", "name"])
    smoke_detonate_df = _safe_parse_event(parser, "smokegrenade_detonate", ["steamid", "name"])
    molotov_detonate_df = _safe_parse_event(parser, "inferno_startburn", ["steamid", "name"])

    # Player positions at death ticks + grenade throw/detonation ticks
    positions_df = _extract_event_positions(
        parser, death_df, weapon_fire_df,
        flash_detonate_df, he_detonate_df, smoke_detonate_df, molotov_detonate_df,
        hurt_df=hurt_df,
    )

    # Sampled player positions throughout each round (for role classification)
    round_positions_df = _extract_round_positions(parser, round_end_df, round_freeze_end_df)

    # High-frequency player positions for 2D replay
    replay_positions_df = _extract_replay_positions(parser, round_end_df, round_freeze_end_df)

    # Player velocity data around kill ticks (for movement analysis)
    velocities_df = _extract_kill_velocities(parser, death_df)

    # Player ranks (competitive skill group)
    ranks_df = _extract_player_ranks(parser)

    # Rank update event (end-of-match rank changes)
    rank_update_df = _extract_rank_update(parser)

    # End-of-match player stats (comp_wins, mvps, score)
    end_stats_df = _extract_end_of_match_stats(parser)

    # Per-round economy snapshots (balance at round start/end)
    economy_df = _extract_round_economy(parser, round_end_df)

    # Assign round numbers to all event DataFrames
    death_df = _assign_rounds(death_df, round_end_df)
    hurt_df = _assign_rounds(hurt_df, round_end_df)
    item_purchase_df = _assign_rounds(item_purchase_df, round_end_df)
    player_blind_df = _assign_rounds(player_blind_df, round_end_df)
    bomb_planted_df = _assign_rounds(bomb_planted_df, round_end_df)
    bomb_defused_df = _assign_rounds(bomb_defused_df, round_end_df)
    bomb_exploded_df = _assign_rounds(bomb_exploded_df, round_end_df)
    flash_detonate_df = _assign_rounds(flash_detonate_df, round_end_df)
    he_detonate_df = _assign_rounds(he_detonate_df, round_end_df)
    smoke_detonate_df = _assign_rounds(smoke_detonate_df, round_end_df)
    molotov_detonate_df = _assign_rounds(molotov_detonate_df, round_end_df)
    weapon_fire_df = _assign_rounds(weapon_fire_df, round_end_df)

    return {
        "player_death": death_df,
        "player_hurt": hurt_df,
        "round_end": round_end_df,
        "item_purchase": item_purchase_df,
        "player_blind": player_blind_df,
        "bomb_planted": bomb_planted_df,
        "bomb_defused": bomb_defused_df,
        "bomb_exploded": bomb_exploded_df,
        "positions": positions_df,
        "ranks": ranks_df,
        "rank_update": rank_update_df,
        "end_stats": end_stats_df,
        "flash_detonate": flash_detonate_df,
        "he_detonate": he_detonate_df,
        "smoke_detonate": smoke_detonate_df,
        "molotov_detonate": molotov_detonate_df,
        "velocities": velocities_df,
        "weapon_fire": weapon_fire_df,
        "round_positions": round_positions_df,
        "replay_positions": replay_positions_df,
        "round_freeze_end": round_freeze_end_df,
        "economy": economy_df,
        "header": header,
    }


def _extract_death_positions(
    parser: Any, death_df: pd.DataFrame
) -> pd.DataFrame:
    """Fetch player XYZ positions at each death tick.

    Uses ``parse_ticks`` to get all 10 players' coordinates at each unique
    death tick.  Returns a DataFrame with columns: ``steamid`` (str),
    ``tick``, ``X``, ``Y``, ``Z``.
    """
    if death_df.empty or "tick" not in death_df.columns:
        return pd.DataFrame()
    ticks = death_df["tick"].dropna().unique().tolist()
    if not ticks:
        return pd.DataFrame()
    try:
        pos_df = parser.parse_ticks(["X", "Y", "Z"], ticks=ticks)
        if not pos_df.empty and "steamid" in pos_df.columns:
            pos_df["steamid"] = pos_df["steamid"].astype(str)
        return pos_df
    except Exception:
        return pd.DataFrame()


def _extract_event_positions(
    parser: Any,
    death_df: pd.DataFrame,
    weapon_fire_df: pd.DataFrame,
    flash_det_df: pd.DataFrame,
    he_det_df: pd.DataFrame,
    smoke_det_df: pd.DataFrame,
    molotov_det_df: pd.DataFrame,
    hurt_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch player XYZ positions at death ticks AND grenade-related ticks.

    Merges ticks from kills, grenade throws (weapon_fire), detonations,
    and player_hurt (for HE/molotov victim positions) into a single
    parse_ticks call for efficiency.
    """
    all_ticks: set[int] = set()

    dfs = [death_df, weapon_fire_df, flash_det_df, he_det_df,
           smoke_det_df, molotov_det_df]
    if hurt_df is not None:
        dfs.append(hurt_df)
    for df in dfs:
        if not df.empty and "tick" in df.columns:
            all_ticks.update(int(t) for t in df["tick"].dropna().unique())

    if not all_ticks:
        return pd.DataFrame()

    try:
        pos_df = parser.parse_ticks(["X", "Y", "Z"], ticks=sorted(all_ticks))
        if not pos_df.empty and "steamid" in pos_df.columns:
            pos_df["steamid"] = pos_df["steamid"].astype(str)
        return pos_df
    except Exception:
        return pd.DataFrame()


_SAMPLE_INTERVAL = 128  # ticks between samples (~2s at 64-tick)
_MAX_SAMPLES = 30       # cap samples per round

_REPLAY_SAMPLE_INTERVAL = 32  # ticks between replay samples (~0.5s at 64-tick)
_REPLAY_MAX_SAMPLES = 500     # generous cap per round (~250s)


def _build_freeze_end_map(
    round_freeze_end_df: pd.DataFrame | None,
    end_ticks: list,
) -> dict[int, int]:
    """Map round index → freeze-end tick (the moment buy time ends).

    Each freeze-end tick is matched to the round whose end tick comes
    *after* it. Falls back to an empty dict when the event isn't available.
    """
    if round_freeze_end_df is None or round_freeze_end_df.empty:
        return {}
    if "tick" not in round_freeze_end_df.columns:
        return {}

    fe_ticks = sorted(round_freeze_end_df["tick"].dropna().astype(int).tolist())
    mapping: dict[int, int] = {}
    fe_idx = 0
    for i, end_tick in enumerate(end_ticks):
        # Find the freeze-end tick that falls before this round's end tick
        # and after the previous round's end tick (or 0 for round 1).
        prev_end = end_ticks[i - 1] if i > 0 else 0
        while fe_idx < len(fe_ticks) and fe_ticks[fe_idx] <= int(prev_end):
            fe_idx += 1
        if fe_idx < len(fe_ticks) and fe_ticks[fe_idx] < int(end_tick):
            mapping[i] = fe_ticks[fe_idx]
    return mapping


def _extract_round_positions(
    parser: Any, round_end_df: pd.DataFrame,
    round_freeze_end_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Sample player XY positions at regular intervals throughout each round.

    Returns a DataFrame with columns: ``steamid``, ``tick``, ``X``, ``Y``,
    ``round``, ``tick_offset`` (ticks from round start).
    """
    if round_end_df.empty or "tick" not in round_end_df.columns:
        return pd.DataFrame()

    re_sorted = round_end_df.sort_values("round")
    end_ticks = re_sorted["tick"].values.tolist()
    round_nums = re_sorted["round"].values.tolist()

    # Use round_freeze_end ticks as round-start markers when available
    freeze_end_ticks = _build_freeze_end_map(round_freeze_end_df, end_ticks)

    # Build sample ticks for each round
    sample_ticks: list[int] = []
    tick_to_round: dict[int, int] = {}
    tick_to_offset: dict[int, int] = {}

    for i, (rnd, end_tick) in enumerate(zip(round_nums, end_ticks)):
        start_tick = freeze_end_ticks.get(i, end_ticks[i - 1] if i > 0 else 0)
        t = int(start_tick) + _SAMPLE_INTERVAL
        n = 0
        while t < int(end_tick) and n < _MAX_SAMPLES:
            sample_ticks.append(t)
            tick_to_round[t] = int(rnd)
            tick_to_offset[t] = t - int(start_tick)
            t += _SAMPLE_INTERVAL
            n += 1

    if not sample_ticks:
        return pd.DataFrame()

    try:
        df = parser.parse_ticks(["X", "Y"], ticks=sorted(sample_ticks))
        if df.empty:
            return pd.DataFrame()
        if "steamid" in df.columns:
            df["steamid"] = df["steamid"].astype(str)
        df["round"] = df["tick"].map(tick_to_round)
        df["tick_offset"] = df["tick"].map(tick_to_offset)
        return df
    except Exception:
        return pd.DataFrame()


def _extract_replay_positions(
    parser: Any, round_end_df: pd.DataFrame,
    round_freeze_end_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Sample ALL players' positions at high frequency for 2D replay.

    Returns a DataFrame with columns: ``steamid``, ``tick``, ``X``, ``Y``,
    ``health``, ``team_num``, ``round``, ``tick_offset``.
    """
    if round_end_df.empty or "tick" not in round_end_df.columns:
        return pd.DataFrame()

    re_sorted = round_end_df.sort_values("round")
    end_ticks = re_sorted["tick"].values.tolist()
    round_nums = re_sorted["round"].values.tolist()

    # Use round_freeze_end ticks as round-start markers when available
    freeze_end_ticks = _build_freeze_end_map(round_freeze_end_df, end_ticks)

    sample_ticks: list[int] = []
    tick_to_round: dict[int, int] = {}
    tick_to_offset: dict[int, int] = {}

    for i, (rnd, end_tick) in enumerate(zip(round_nums, end_ticks)):
        start_tick = freeze_end_ticks.get(i, end_ticks[i - 1] if i > 0 else 0)
        t = int(start_tick) + _REPLAY_SAMPLE_INTERVAL
        n = 0
        while t < int(end_tick) and n < _REPLAY_MAX_SAMPLES:
            sample_ticks.append(t)
            tick_to_round[t] = int(rnd)
            tick_to_offset[t] = t - int(start_tick)
            t += _REPLAY_SAMPLE_INTERVAL
            n += 1
        # Always include the round_end tick itself so the final kill frame
        # is captured even when it falls in the last sample-interval gap.
        et = int(end_tick)
        if et not in tick_to_round:
            sample_ticks.append(et)
            tick_to_round[et] = int(rnd)
            tick_to_offset[et] = et - int(start_tick)

    if not sample_ticks:
        return pd.DataFrame()

    try:
        df = parser.parse_ticks(
            ["X", "Y", "health", "team_num"],
            ticks=sorted(sample_ticks),
        )
        if df.empty:
            return pd.DataFrame()
        if "steamid" in df.columns:
            df["steamid"] = df["steamid"].astype(str)
        df["round"] = df["tick"].map(tick_to_round)
        df["tick_offset"] = df["tick"].map(tick_to_offset)
        return df
    except Exception:
        return pd.DataFrame()


def _extract_kill_velocities(
    parser: Any, death_df: pd.DataFrame, window: int = 64
) -> pd.DataFrame:
    """Fetch player velocity and yaw around each kill tick.

    For each unique death tick, extracts velocity_X, velocity_Y, and yaw
    for all players across a window of ticks before and including the kill
    tick (default 64 ticks ≈ 1s at 64-tick, enough for reaction-time analysis).

    Returns a DataFrame with columns: ``steamid`` (int), ``tick``,
    ``velocity_X``, ``velocity_Y``, ``yaw``.
    """
    if death_df.empty or "tick" not in death_df.columns:
        return pd.DataFrame()
    kill_ticks = death_df["tick"].dropna().unique().tolist()
    if not kill_ticks:
        return pd.DataFrame()
    # Build expanded tick list: [kill_tick - window .. kill_tick] for each kill
    all_ticks: set[int] = set()
    for t in kill_ticks:
        t = int(t)
        for offset in range(window + 1):
            all_ticks.add(t - offset)
    try:
        df = parser.parse_ticks(
            ["velocity_X", "velocity_Y", "X", "Y", "Z", "yaw", "pitch"],
            ticks=sorted(all_ticks),
        )
        return df
    except Exception:
        return pd.DataFrame()


def _extract_player_ranks(parser: Any) -> pd.DataFrame:
    """Extract player competitive ranks and rank type from tick 1.

    Returns a DataFrame with ``rank`` (skill group 1-18 for Competitive,
    or CS Rating integer for Premier) and ``comp_rank_type`` (11 = Premier,
    12 = Competitive).
    """
    try:
        df = parser.parse_ticks(["rank", "comp_rank_type"], ticks=[1])
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        if "steamid" in df.columns:
            df["steamid"] = df["steamid"].astype(str)
        return df
    except Exception:
        return pd.DataFrame()


def _extract_rank_update(parser: Any) -> pd.DataFrame:
    """Extract the rank_update event (end-of-match rank changes)."""
    try:
        df = parser.parse_event("rank_update")
        if isinstance(df, pd.DataFrame) and not df.empty:
            if "user_steamid" in df.columns:
                df["user_steamid"] = df["user_steamid"].astype(str)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _extract_end_of_match_stats(parser: Any) -> pd.DataFrame:
    """Extract comp_wins and mvps from the last tick."""
    try:
        df = parser.parse_ticks(["comp_wins", "mvps", "score"])
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        last_tick = df["tick"].max()
        end = df[df["tick"] == last_tick].copy()
        if "steamid" in end.columns:
            end["steamid"] = end["steamid"].astype(str)
        return end
    except Exception:
        return pd.DataFrame()


def _extract_round_economy(
    parser: Any, round_end_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract player balance at the start and end of each round.

    Returns a DataFrame with columns: ``steamid``, ``round``,
    ``start_balance``, ``end_balance``.
    """
    if round_end_df.empty or "tick" not in round_end_df.columns:
        return pd.DataFrame()

    re_sorted = round_end_df.sort_values("tick")
    end_ticks = re_sorted["tick"].values.tolist()
    # Candidate start ticks: a small window after each round end to find
    # the nearest valid tick recorded in the demo.
    start_candidates: list[list[int]] = [[1]]
    for t in end_ticks[:-1]:
        base = int(t)
        start_candidates.append([base + off for off in range(1, 200, 2)])

    all_ticks: set[int] = set()
    for cands in start_candidates:
        all_ticks.update(cands)
    all_ticks.update(int(t) for t in end_ticks)

    if not all_ticks:
        return pd.DataFrame()

    try:
        df = parser.parse_ticks(["balance"], ticks=sorted(all_ticks))
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if "steamid" in df.columns:
        df["steamid"] = df["steamid"].astype(str)

    available_ticks = set(df["tick"].unique())

    # Resolve actual start tick for each round (first candidate that exists)
    resolved_start: list[int | None] = []
    for cands in start_candidates:
        found = None
        for c in cands:
            if c in available_ticks:
                found = c
                break
        resolved_start.append(found)

    # Build round-indexed start/end balance rows
    rows: list[dict[str, Any]] = []
    for i, et in enumerate(end_ticks):
        rnd = i + 1
        st = resolved_start[i]
        et_int = int(et)
        et_rows = df[df["tick"] == et_int]

        for _, er in et_rows.iterrows():
            sid = er["steamid"]
            end_bal = int(er["balance"])
            start_bal = None
            if st is not None:
                sr = df[(df["tick"] == st) & (df["steamid"] == sid)]
                if not sr.empty:
                    start_bal = int(sr.iloc[0]["balance"])
            rows.append({
                "steamid": sid,
                "round": rnd,
                "start_balance": start_bal,
                "end_balance": end_bal,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def parse_info_file(info_bytes: bytes) -> dict[str, Any]:
    """Parse a .dem.info protobuf sidecar file.

    Extracts match timestamp (field 2) and player account IDs (field 5.2).
    Returns a dict with ``match_date`` (ISO string or None) and
    ``account_ids`` (list of steamid64 strings).
    """
    import datetime

    def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
        result = 0
        shift = 0
        while pos < len(buf):
            b = buf[pos]
            result |= (b & 0x7F) << shift
            pos += 1
            if not (b & 0x80):
                break
            shift += 7
        return result, pos

    result: dict[str, Any] = {"match_date": None, "account_ids": []}
    pos = 0
    while pos < len(info_bytes):
        try:
            tag, pos = _read_varint(info_bytes, pos)
        except Exception:
            break
        field = tag >> 3
        wtype = tag & 0x07
        if wtype == 0:  # varint
            val, pos = _read_varint(info_bytes, pos)
            if field == 2 and 1_000_000_000 < val < 2_000_000_000:
                dt = datetime.datetime.fromtimestamp(val, tz=datetime.timezone.utc)
                result["match_date"] = dt.date().isoformat()
        elif wtype == 2:  # length-delimited
            length, pos = _read_varint(info_bytes, pos)
            payload = info_bytes[pos:pos + length]
            pos += length
            if field == 5:  # game_info submessage — extract account IDs
                sub_pos = 0
                while sub_pos < len(payload):
                    try:
                        stag, sub_pos = _read_varint(payload, sub_pos)
                    except Exception:
                        break
                    sfield = stag >> 3
                    swtype = stag & 0x07
                    if swtype == 0:
                        sval, sub_pos = _read_varint(payload, sub_pos)
                        if sfield == 2:  # repeated field 2 = account IDs
                            sub_sub_pos = 0
                            # field 2 is a packed/submessage of repeated varints
                            pass
                    elif swtype == 2:
                        slen, sub_pos = _read_varint(payload, sub_pos)
                        sub_payload = payload[sub_pos:sub_pos + slen]
                        sub_pos += slen
                        if sfield == 2:  # account IDs submessage
                            aid_pos = 0
                            base = 76561197960265728
                            while aid_pos < len(sub_payload):
                                try:
                                    aid_tag, aid_pos = _read_varint(sub_payload, aid_pos)
                                    aid_field = aid_tag >> 3
                                    aid_wtype = aid_tag & 0x07
                                    if aid_wtype == 0:
                                        aid_val, aid_pos = _read_varint(sub_payload, aid_pos)
                                        if aid_field == 1 and aid_val > 0:
                                            result["account_ids"].append(
                                                str(base + aid_val)
                                            )
                                    else:
                                        break
                                except Exception:
                                    break
                    elif swtype == 5:
                        sub_pos += 4
                    elif swtype == 1:
                        sub_pos += 8
                    else:
                        break
        elif wtype == 5:  # fixed32
            pos += 4
        elif wtype == 1:  # fixed64
            pos += 8
        else:
            break
    # Deduplicate account IDs while preserving order
    seen: set[str] = set()
    unique_ids: list[str] = []
    for aid in result["account_ids"]:
        if aid not in seen:
            seen.add(aid)
            unique_ids.append(aid)
    result["account_ids"] = unique_ids
    return result


def _safe_parse_event(
    parser: Any,
    event_name: str,
    player_fields: list[str],
) -> pd.DataFrame:
    """Parse an event, returning an empty DataFrame on failure."""
    try:
        if player_fields:
            result = parser.parse_event(event_name, player=player_fields)
        else:
            result = parser.parse_event(event_name)
        if isinstance(result, pd.DataFrame):
            return result
        return pd.DataFrame(result) if result else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _assign_rounds(
    event_df: pd.DataFrame, round_end_df: pd.DataFrame
) -> pd.DataFrame:
    """Add a ``round`` column to *event_df* based on tick boundaries.

    Each event is placed in the round whose ``round_end`` tick is the first
    one >= the event tick.  Events after the last round keep the last round
    number.
    """
    if event_df.empty or round_end_df.empty:
        if not event_df.empty and "round" not in event_df.columns:
            event_df = event_df.copy()
            event_df["round"] = 1
        return event_df
    if "tick" not in event_df.columns or "tick" not in round_end_df.columns:
        return event_df

    import numpy as np

    # Sorted round-end ticks; use searchsorted for fast mapping
    re_sorted = round_end_df.sort_values("round")
    end_ticks = re_sorted["tick"].values
    round_nums = re_sorted["round"].values

    indices = np.searchsorted(end_ticks, event_df["tick"].values, side="left")
    indices = np.clip(indices, 0, len(round_nums) - 1)

    event_df = event_df.copy()
    event_df["round"] = round_nums[indices]
    return event_df
