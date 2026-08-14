"""
Layer 1: Demo Parsing Engine
Reads a raw CS2 .dem file using demoparser2 and returns structured DataFrames
of in-game events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_CS2_ENTITY_SCHEMA_BREAK_PATCH = 14152


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

    try:
        parser = DemoParser(str(demo_path))
        header: dict[str, Any] = parser.parse_header()
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(f"Failed to parse demo header: {exc}") from exc

    try:
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
    except Exception as exc:
        # April 2026 CS2 patch changed entity schema in a way older demoparser2
        # builds cannot decode. Preserve import flow with a metadata-only fallback.
        if _should_use_header_only_fallback(header, exc):
            fallback_header = dict(header)
            fallback_header["parse_mode"] = "header_only_fallback"
            fallback_header[
                "parse_warning"
            ] = (
                "This demo uses a newer CS2 entity schema that is not fully "
                "supported by the installed demoparser2 build. Imported in "
                "metadata-only mode; advanced stats were set to 0."
            )
            return _empty_parsed_result(fallback_header)
        raise

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
    velocities_df = _extract_kill_velocities(parser, death_df, hurt_df)

    # Visibility at trigger-pull time, for separating duels from spray
    spotted_df = _extract_spotted(parser, weapon_fire_df)

    # Player ranks (competitive skill group)
    ranks_df = _extract_player_ranks(parser)

    # Rank update event (end-of-match rank changes)
    rank_update_df = _extract_rank_update(parser)

    # End-of-match player stats (comp_wins, mvps, score)
    end_stats_df = _extract_end_of_match_stats(parser)

    # Warmup shares the demo with the match; everything before this tick is
    # pre-match and must not be attributed to round 1.
    match_start_tick = _find_match_start_tick(parser)

    # Per-round economy snapshots (balance at round start/end)
    economy_df = _extract_round_economy(parser, round_end_df, match_start_tick)

    # Assign round numbers to all event DataFrames
    death_df = _assign_rounds(death_df, round_end_df, match_start_tick)
    hurt_df = _assign_rounds(hurt_df, round_end_df, match_start_tick)
    item_purchase_df = _assign_rounds(item_purchase_df, round_end_df, match_start_tick)
    player_blind_df = _assign_rounds(player_blind_df, round_end_df, match_start_tick)
    bomb_planted_df = _assign_rounds(bomb_planted_df, round_end_df, match_start_tick)
    bomb_defused_df = _assign_rounds(bomb_defused_df, round_end_df, match_start_tick)
    bomb_exploded_df = _assign_rounds(bomb_exploded_df, round_end_df, match_start_tick)
    flash_detonate_df = _assign_rounds(flash_detonate_df, round_end_df, match_start_tick)
    he_detonate_df = _assign_rounds(he_detonate_df, round_end_df, match_start_tick)
    smoke_detonate_df = _assign_rounds(smoke_detonate_df, round_end_df, match_start_tick)
    molotov_detonate_df = _assign_rounds(molotov_detonate_df, round_end_df, match_start_tick)
    weapon_fire_df = _assign_rounds(weapon_fire_df, round_end_df, match_start_tick)

    return {
        "player_death": death_df,
        "player_hurt": hurt_df,
        "round_end": round_end_df,
        "item_purchase": item_purchase_df,
        "player_blind": player_blind_df,
        "spotted": spotted_df,
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


# Ticks of player velocity/view-angle history sampled before each kill.
#
# The mechanical metrics in processor.py are anchored to the tick the
# engagement *started* on, not the kill tick, so the window has to cover
# both the engagement itself and the lookback taken from its first shot:
#
#   engagement length (up to ~1 s)  +  reaction lookback (64 ticks)  = 128
#
# Sampling only 64 ticks left the reaction lookback running off the end of
# the data on any engagement longer than an instant.
_VELOCITY_WINDOW = 128


def _extract_kill_velocities(
    parser: Any,
    death_df: pd.DataFrame,
    hurt_df: pd.DataFrame | None = None,
    window: int = _VELOCITY_WINDOW,
) -> pd.DataFrame:
    """Fetch player velocity and yaw around every engagement.

    Sampling was originally anchored on kills alone, which left roughly 40% of
    a player's duels — the ones where they damaged an enemy without finishing
    them — with no movement, crosshair or reaction data at all.  Damage ticks
    are included as well so those engagements can be measured; the windows
    overlap heavily, so the extra cost is far below the extra event count.

    See ``_VELOCITY_WINDOW`` for why the window is sized the way it is.

    Returns a DataFrame with columns: ``steamid`` (int), ``tick``,
    ``velocity_X``, ``velocity_Y``, ``yaw``.
    """
    anchors: list[int] = []
    if not death_df.empty and "tick" in death_df.columns:
        anchors.extend(int(t) for t in death_df["tick"].dropna().unique())
    if hurt_df is not None and not hurt_df.empty and "tick" in hurt_df.columns:
        anchors.extend(int(t) for t in hurt_df["tick"].dropna().unique())
    if not anchors:
        return pd.DataFrame()

    all_ticks: set[int] = set()
    for t in anchors:
        for offset in range(window + 1):
            all_ticks.add(t - offset)
    try:
        df = parser.parse_ticks(
            # ``ducked`` rides along on the same call: crouching caps speed
            # below the accuracy threshold without any counter-strafe, so the
            # movement classifier has to be able to tell the two apart.
            ["velocity_X", "velocity_Y", "X", "Y", "Z", "yaw", "pitch", "ducked"],
            ticks=sorted(all_ticks),
        )
        return df
    except Exception:
        return pd.DataFrame()



def _extract_spotted(parser: Any, weapon_fire_df: pd.DataFrame) -> pd.DataFrame:
    """Per-player visibility at every tick somebody pulled a trigger.

    Needed to tell a duel from spray: a burst fired with no enemy visible is
    smoke spam, a wallbang or a pre-fire, and counting it against aim accuracy
    would punish deliberate utility usage. Cheap — one prop over the fire ticks
    costs a fraction of a second even on a long match.

    Returns a DataFrame with ``tick``, ``steamid``, ``spotted``, ``team_num``,
    or an empty frame when there is nothing to sample.
    """
    if weapon_fire_df.empty or "tick" not in weapon_fire_df.columns:
        return pd.DataFrame()
    ticks = sorted({int(t) for t in weapon_fire_df["tick"].dropna()})
    if not ticks:
        return pd.DataFrame()
    try:
        return parser.parse_ticks(["spotted", "team_num"], ticks=ticks)
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
    parser: Any, round_end_df: pd.DataFrame, match_start_tick: int = 1,
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
    # Round 1 starts where the match does, not at tick 1: the balance only
    # resets from the warmup figure to $800 on that tick.
    start_candidates: list[list[int]] = [
        [match_start_tick + off for off in range(0, 200, 2)]
    ]
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


def extract_player_names(demo_path: str | Path) -> dict[str, str]:
    """Quickly extract a mapping of steamid64 → in-game name from a .dem file.

    Parses only a single tick, making this far cheaper than a full parse.
    Returns a dict like ``{"<steamid64>": "<in-game name>", ...}``.
    """
    try:
        from demoparser2 import DemoParser  # type: ignore[import-untyped]
    except ImportError:
        return {}

    demo_path = Path(demo_path)
    if not demo_path.exists():
        return {}

    try:
        parser = DemoParser(str(demo_path))
        df = parser.parse_ticks(["name"], ticks=[1])
        name_map: dict[str, str] = {}
        for _, row in df.iterrows():
            sid = str(row.get("steamid", ""))
            name = str(row.get("name", ""))
            if sid and name and sid not in name_map:
                name_map[sid] = name
        return name_map
    except Exception:
        return {}


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

    result: dict[str, Any] = {"match_date": None, "map_name": None, "account_ids": []}
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
                dt = datetime.datetime.fromtimestamp(val, tz=datetime.UTC)
                result["match_date"] = dt.date().isoformat()
        elif wtype == 2:  # length-delimited
            length, pos = _read_varint(info_bytes, pos)
            payload = info_bytes[pos:pos + length]
            pos += length
            if field == 3:  # watchablematchinfo submessage — extract game_map (field 2)
                sub_pos = 0
                while sub_pos < len(payload):
                    try:
                        stag, sub_pos = _read_varint(payload, sub_pos)
                    except Exception:
                        break
                    sfield = stag >> 3
                    swtype = stag & 0x07
                    if swtype == 2:
                        slen, sub_pos = _read_varint(payload, sub_pos)
                        sval = payload[sub_pos:sub_pos + slen]
                        sub_pos += slen
                        if sfield == 2:  # game_map string
                            try:
                                result["map_name"] = sval.decode("utf-8").strip()
                            except Exception:
                                pass
                    elif swtype == 0:
                        _, sub_pos = _read_varint(payload, sub_pos)
                    elif swtype == 5:
                        sub_pos += 4
                    elif swtype == 1:
                        sub_pos += 8
                    else:
                        break
            elif field == 5:  # game_info submessage — extract account IDs
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
                        if sfield == 2:
                            # Account IDs arrive as a length-delimited submessage,
                            # decoded in the swtype == 2 branch below. A bare varint
                            # here carries no account ID, so there is nothing to read.
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


def read_demo_map(demo_path: str | Path) -> str | None:
    """Return the map name from a demo file's header, or None on failure.

    Uses demoparser2's parse_header() which only reads the first portion of
    the file, so it is fast enough to call during a folder scan.
    """
    try:
        from demoparser2 import DemoParser  # type: ignore[import-untyped]
        parser = DemoParser(str(demo_path))
        return parser.parse_header().get("map_name") or None
    except Exception:
        return None


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


def _is_entity_not_found_error(exc: Exception) -> bool:
    """Return True when demoparser2 failed with an EntityNotFound-style error."""
    text = str(exc)
    return "EntityNotFound" in text or "entity not found" in text.lower()


def _should_use_header_only_fallback(header: dict[str, Any], exc: Exception) -> bool:
    """Gate fallback mode to known post-update demos only."""
    if not _is_entity_not_found_error(exc):
        return False
    patch_version = header.get("patch_version")
    try:
        return int(patch_version) >= _CS2_ENTITY_SCHEMA_BREAK_PATCH
    except (TypeError, ValueError):
        return False


def _empty_parsed_result(header: dict[str, Any]) -> dict[str, Any]:
    """Build a parse result payload with empty event DataFrames."""
    empty = pd.DataFrame()
    return {
        "player_death": empty.copy(),
        "player_hurt": empty.copy(),
        "round_end": empty.copy(),
        "item_purchase": empty.copy(),
        "player_blind": empty.copy(),
        "spotted": empty.copy(),
        "bomb_planted": empty.copy(),
        "bomb_defused": empty.copy(),
        "bomb_exploded": empty.copy(),
        "positions": empty.copy(),
        "ranks": empty.copy(),
        "rank_update": empty.copy(),
        "end_stats": empty.copy(),
        "flash_detonate": empty.copy(),
        "he_detonate": empty.copy(),
        "smoke_detonate": empty.copy(),
        "molotov_detonate": empty.copy(),
        "velocities": empty.copy(),
        "weapon_fire": empty.copy(),
        "round_positions": empty.copy(),
        "replay_positions": empty.copy(),
        "round_freeze_end": empty.copy(),
        "economy": empty.copy(),
        "header": header,
    }



def _find_match_start_tick(parser: Any) -> int:
    """First tick that belongs to the match rather than to warmup.

    Warmup shares the demo with the match: players hold $16,000, buy freely,
    and kill each other, and the first round_end is far enough away that all of
    it fell into round 1. That is where a pistol round with a $14,300 balance
    and an unaffordable buy came from.

    ``round_start`` marks it precisely — the balance resets to $800 on that
    exact tick.  ``begin_new_match`` is the fallback for demos without it.
    """
    for event, reducer in (("round_start", min), ("begin_new_match", min)):
        try:
            df = parser.parse_event(event)
        except Exception:
            continue
        if hasattr(df, "empty") and not df.empty and "tick" in df.columns:
            ticks = [int(t) for t in df["tick"].dropna()]
            if ticks:
                return reducer(ticks)
    return 1


def _assign_rounds(
    event_df: pd.DataFrame, round_end_df: pd.DataFrame,
    match_start_tick: int = 1,
) -> pd.DataFrame:
    """Add a ``round`` column to *event_df* based on tick boundaries.

    Each event is placed in the round whose ``round_end`` tick is the first
    one >= the event tick.  Events after the last round keep the last round
    number.  Anything before *match_start_tick* is warmup and is dropped —
    left in, it all landed in round 1.
    """
    if not event_df.empty and "tick" in event_df.columns and match_start_tick > 1:
        event_df = event_df[event_df["tick"] >= match_start_tick]
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
