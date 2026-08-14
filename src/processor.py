"""
Layer 2: Metrics Processor
Filters raw event DataFrames for a specific Steam ID and calculates advanced
CS2 statistics: KPR, DPR, ADR, KAST, Impact, and an approximated HLTV 2.0
Rating.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.callouts import get_callout, is_map_supported

# Aliased: `hltv_rating` is also the name of the computed value in
# calculate_match_stats, and the local would shadow the function.
from src.domain.calibration import (
    hltv_rating as compute_hltv_rating,
)
from src.domain.metrics._shared import (
    _aim_bounds,
    _build_round_team_map,
    _find_id_col,
)
from src.domain.metrics.aim import _calculate_aim_stats
from src.domain.metrics.benchmarks import compute_benchmarks
from src.domain.metrics.impact import _calculate_impact_stats
from src.domain.metrics.replay import _build_replay_data
from src.domain.metrics.roles import _calculate_roles
from src.domain.metrics.utility import _calculate_utility_stats


def metric_versions() -> dict[str, int]:
    """The registry's per-metric stamp.

    Imported inside the function on purpose: the metric modules currently adapt
    functions defined in this one, so importing the registry at module scope
    would be a cycle.
    """
    from src.domain.metrics import REGISTRY

    return REGISTRY.versions()


def stamp_metric_versions(stats: dict[str, Any]) -> dict[str, Any]:
    """Record which version of each metric produced the value under its key.

    Written *inside* the metric's own blob, which is already opaque TEXT, so
    there is no schema change and no new column. A reader that does not know
    about it sees one extra key.

    This is what makes staleness per-metric. A match stamped with
    ``{"aim.stats": 3}`` and read by code at version 4 is stale on aim and
    nothing else, instead of being marked stale wholesale by ANALYZER_VERSION
    and dragging the user's entire library into a re-parse.
    """
    from src.domain.metrics import REGISTRY

    for spec in REGISTRY:
        if not spec.version_in_blob:
            continue
        value = stats.get(spec.output_key)
        if isinstance(value, dict):
            value["_metric_version"] = spec.version
    return stats


def stored_metric_versions(stats: dict[str, Any]) -> dict[str, int]:
    """Read the stamps back off a stored match.

    Missing entirely for anything imported before this existed, which
    ``MetricRegistry.stale_ids`` reads as "stale on everything".
    """
    from src.domain.metrics import REGISTRY

    found: dict[str, int] = {}
    for spec in REGISTRY:
        value = stats.get(spec.output_key)
        if isinstance(value, dict) and "_metric_version" in value:
            found[spec.id] = int(value["_metric_version"])
    return found


# ---------------------------------------------------------------------------
# Analyzer version
#
# Bump this whenever a change here alters the numbers a stored match would
# produce.  Every saved match records the version that produced it, so the
# matches list can flag rows that predate the current analysis and offer to
# re-run them against the demo on disk.  Matches saved before versioning
# existed read back as 0 and therefore count as stale.
#
# History:
#   1  Baseline — the first version to be stamped.
#   2  Aim metrics anchored to the first shot of the engagement rather than
#      the kill tick; counter-strafes separated from standing still and from
#      coasting to a halt; reaction time no longer invented on a truncated
#      velocity sample.
#   3  Standing/stopped split moved onto the accuracy threshold instead of a
#      separate constant, and round-restart teleport artifacts (all ten players
#      reading tens of thousands of u/s on the same tick) excluded from the
#      movement window.
#   4  Aim metrics summarised by median rather than mean, engagements over 1 s
#      excluded from the time-to-kill aggregate, aim rating weighted by sample
#      size instead of substituting 50 for anything unmeasured, and reaction
#      time demoted to a diagnostic that no longer feeds the rating.
#   5  Crouch state read from the demo, and the counter-strafe rate restricted
#      to rifle engagements that were not crouched — the technique only
#      decides the shot on weapons with the full movement penalty.
#   6  Benchmarks carry their sample size and a "heuristic" calibration marker,
#      and the tier bands stopped claiming to be pro/amateur comparisons.
#      No metric changed value; this is a labelling and metadata release.
#   7  Utility: smoke "coverage" dropped from the rating (it scored callout-map
#      completeness, not smoke quality), flashes scored on blind seconds
#      delivered rather than heads counted, own molotovs no longer counted as
#      smoke extinguishes, and the rating weighted by evidence instead of
#      substituting 50 for anything unmeasured.
#   8  Impact: kills and deaths priced by how far they moved the round, against
#      a win-probability table measured from the local demo corpus.
#  19  Peek speed reports its full distribution across the regions of its own
#      axis, not just the two outer percentages, so the chart legend can name
#      every band drawn on it.  Version 18 matches carry peek data but no such
#      breakdown and render the legend empty until re-run.  Rounds also record
#      the damage the player absorbed, split health vs armour, which is what
#      separates a vest that stopped bullets from $650 spent on nothing.
#  18  Peek speed reported as a metric of its own.  The speed carried into
#      each duel was already measured per shot and then discarded; it is now
#      aggregated, plottable per encounter, and the counter-strafe rate is
#      additionally split by how fast the peek was.  No existing number
#      changed value — stored matches simply carry no peek data at all until
#      they are re-run.
#  17  Self-flashes excluded from the per-round flash instances too, not
#      just the totals — they were putting the player on their own
#      friendly-flash chart and inflating per-round enemy blind time.
#  16  Warmup excluded from the match. Players hold ~$16k and buy freely
#      during it, and every warmup event fell into round 1 — which is how
#      a pistol round showed a five-figure balance and a buy nobody could
#      afford. Round 1's balance is now read where the match starts.
#  15  Utility: flashing yourself no longer counts as a team flash, and the
#      weapon-drop detector stopped reading the game's periodic inventory
#      re-emissions (and refunds) as second purchases.
#  14  Grenade and molotov damage excluded from aim accuracy. player_hurt
#      reports them with a 'generic' hitgroup, and counting them as hits
#      credited fire damage to the player's gun while the grenade was never
#      counted as a shot. They remain in ADR and the utility metrics.
#  13  One-tap kills restored to engagement time: requiring a non-zero
#      duration silently dropped the fastest kills in every match.
#  12  Bursts that hit nothing now count against accuracy, gated on an enemy
#      actually being visible so smoke spray and wallbangs are not charged
#      to aim. Shots are counted over the whole burst rather than stopping
#      at the last bullet that connected.
#  11  Accuracy pooled across every bullet instead of averaging per-engagement
#      percentages, which let one-bullet exchanges outvote full sprays.
#  10  Damage-only engagements now measured, not discarded: hits against one
#      enemy are split into separate fights on a silence gap, and accuracy
#      and reaction are computed for duels that did not end in a kill —
#      roughly 40% of a player's shooting that previously went unmeasured.
#      Velocity sampling extended to damage ticks so those engagements have
#      movement and crosshair data at all.
#   9  One set of bands per aim metric, shared by the per-kill buckets, the
#      benchmark tiers and the scatter plot, which had drifted onto three
#      different sets. Scatter data now excludes the same outliers the
#      aggregates do, and carries stop_ticks so counter-strafe is plottable.
#   20 Fixed five map zones that could never match, found by cross-checking the
#      zone data once it moved to JSON. de_dust2 "Upper Tunnels" had inverted y
#      bounds so nothing ever fell inside it; "Xbox" sat inside "Mid",
#      de_inferno "Boiler" inside "Arch" and de_anubis "B Pillar" inside
#      "B Site", each listed after the zone enclosing it and therefore dead
#      — first match wins. Positions that should have read as those callouts
#      were recorded as the enclosing area instead, in stored enriched rounds
#      and in role classification. Also de_overpass CT gained "A Site": no CT
#      role claimed the A bombsite, so an A anchor scored nothing from the
#      position they actually held.
#
#      Still outstanding: on de_nuke the A and B bombsites share a 2D footprint
#      because the map is stacked, so B Site reads as A Site. Separating them
#      needs a Z coordinate the zone model does not carry.
# ---------------------------------------------------------------------------
ANALYZER_VERSION = 20




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
    hltv_rating: float = compute_hltv_rating(kast, kpr, dpr, impact, adr)

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
    # Impact — how much each kill and death moved the round                #
    # ------------------------------------------------------------------ #
    impact_stats = _calculate_impact_stats(
        parsed_data, steam_id, total_rounds,
        _build_round_team_map(death_df, steam_id, round_end_df),
    )

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

    return stamp_metric_versions({
        "analyzer_version": ANALYZER_VERSION,
        "impact_stats": impact_stats,
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
    })


# ---------------------------------------------------------------------------
# 2D Replay data builder
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Aim & Movement aggregate stats
# ---------------------------------------------------------------------------

































# ---------------------------------------------------------------------------
# Benchmarks — tier classification for match metrics
# ---------------------------------------------------------------------------

# Each benchmark defines thresholds for 4 tiers.  For "lower is better" metrics
# the tiers are ordered high→low (first threshold is the *best* ceiling).
# For "higher is better" metrics the tiers are low→high (first threshold is
# the *best* floor).
#
# Tier keys are "pro", "high_amateur", "average", "below_average" for storage
# compatibility, but nothing behind them is calibrated: every threshold in this
# file is hand-set, not derived from a population of real players.  They are
# presented as plain bands (Excellent/Strong/Fair/Needs Work) rather than as a
# claim about how the player compares to pros, and each benchmark carries
# ``calibration: "heuristic"`` so a consumer can tell.  Replacing these with
# percentiles needs a corpus far larger than one player's match history.









# ---------------------------------------------------------------------------
# Utility & Economics
# ---------------------------------------------------------------------------


















# ---------------------------------------------------------------------------
# Role classification per round
# ---------------------------------------------------------------------------

# Callout → role mapping per map per side.
# Each role lists the callout names that indicate that role.
# Order matters: first match wins when a player has multiple positions.

# Callout → role mapping lives in src/domain/metrics/role_zones/*.json, one
# file per map. The callout names there have to match the labels in
# src/domain/callouts/zones/, and tests/test_role_zones.py checks that they do:
# a name matching nothing contributes no score and the role quietly stops being
# detected, which is not a failure anything else would notice.






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
        hltv_rating = compute_hltv_rating(kast, kpr, dpr, impact, adr)
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
    #
    # Sorted, because iterating the set directly hands back a different order in
    # every process — Python randomises string hashing per run. That made the
    # all_players scoreboard, and so the match_players insert order, differ
    # between two imports of the same demo. get_match_players re-sorts by team
    # and kills on read, which hid it in the UI for everyone except tied rows.
    return sorted(s for s in ids if s and s not in ("0", "None", "nan"))


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
    rounds.  Requiring the damage to have happened in the same round as the
    kill is what makes the number mean what the word does.
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
    for _, grp in enemy_hurt.groupby("round"):
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
            hurt_df, death_df, sid, r, velocities_df, weapon_fire_df,
        )

        # --- Bursts that hit nothing (accuracy denominator) ---
        _side = round_data.get("side")
        round_data["whiffed_engagements"] = _get_round_whiffed_engagements(
            weapon_fire_df, hurt_df, parsed_data.get("spotted"), sid, r,
            3 if _side == "CT" else 2 if _side == "T" else None,
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

        # --- Damage absorbed (tells a used vest from a wasted one) ---
        round_data["damage_taken"] = _get_round_damage_taken(hurt_df, sid, r)

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


def _get_round_damage_taken(
    hurt_df: pd.DataFrame, sid: str, rnd: int,
) -> dict[str, Any]:
    """Damage the player absorbed this round, split into health and armour.

    The armour figure is what makes a kevlar purchase readable as spent or
    wasted: a vest that never stopped a bullet cost $650 for nothing, and the
    economy timeline has no other way to tell those two rounds apart.

    ``dmg_armor`` is a standard ``player_hurt`` field but not every demo build
    carries it.  When it is missing the key is reported as None rather than 0,
    so a consumer can tell "the vest absorbed nothing" from "this demo cannot
    say" instead of drawing an unused vest either way.
    """
    result: dict[str, Any] = {"health": 0, "armor": None}
    if hurt_df is None or hurt_df.empty or "round" not in hurt_df.columns:
        return result

    victim_col = _find_id_col(hurt_df, ("user_steamid", "steamid"))
    if not victim_col:
        return result

    taken = hurt_df[
        (hurt_df["round"] == rnd) & (hurt_df[victim_col].astype(str) == sid)
    ]
    if taken.empty:
        if "dmg_armor" in hurt_df.columns:
            result["armor"] = 0
        return result

    if "dmg_health" in taken.columns:
        result["health"] = int(taken["dmg_health"].fillna(0).sum())
    if "dmg_armor" in taken.columns:
        result["armor"] = int(taken["dmg_armor"].fillna(0).sum())
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


# Speed (u/s) below which a player counts as effectively stationary.
_STATIC_SPEED = 10.0

# CS2 rifles are accurate below ~34% of the 250 u/s max speed.
_ACCURATE_SPEED = 85.0

# Speeds above this are not movement.  Round restarts teleport all ten players
# at once and the engine reports the jump as velocity, so a handful of ticks in
# every match carry speeds in the tens of thousands for everybody
# simultaneously.  Nothing on foot in CS2 approaches 400 u/s (the cap is ~250),
# and measured samples cluster either below ~300 or above ~1000, so the
# boundary is not delicate.
_MAX_PLAUSIBLE_SPEED = 400.0

# Ticks from the last above-_ACCURATE_SPEED sample to the shot.  Counter-
# strafing cancels ~250 u/s in roughly 4 ticks; letting go and leaving it to
# friction takes ~13 (sv_friction 5.2 decays speed ~8%/tick at 64-tick).
# 7 sits between the two with margin on both sides.
_COUNTERSTRAFE_MAX_TICKS = 7


def _analyze_movement(
    velocities_df: pd.DataFrame, attacker_steamid: int, tick: int,
    window: int = 32,
) -> dict[str, Any] | None:
    """Compute movement metrics for the attacker at the moment of the shot.

    Returns a dict with:
      - shot_speed: speed at the shot tick (units/s)
      - pre_speed: peak speed in the window before the shot — the speed the
        player carried into the duel, reported as "peek speed"
      - stop_ticks: ticks from the last running-speed sample to the shot,
        or None if the player never exceeded the accuracy threshold
      - window_span: ticks of history actually available (< window when the
        sample runs off the start of the parsed data)
      - movement_quality: 'standing' | 'counter-strafed' | 'stopped' | 'running'
      - movement_direction: 'still' | 'forward' | 'backward' | 'left' | 'right'

    Speed at the shot alone cannot tell a counter-strafe from standing still —
    a clean counter-strafe reads ~0 u/s, exactly like never having moved.  The
    two are separated by how quickly the speed collapsed: an active
    counter-strafe is several times faster than coasting to a halt on friction,
    which is what ``stopped`` means.  ``stopped`` is still accurate fire, just
    slow and telegraphed.
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
    if shot_speed > _MAX_PLAUSIBLE_SPEED:
        # The sample at the moment of the shot is a teleport artifact; the
        # player's real speed here is unknown, so report nothing rather than
        # guess from an earlier tick.
        return None

    # Speed history across the window, minus any teleport artifacts.  These are
    # dropped rather than clamped: one of them left in would set pre_speed for
    # the whole window and mask whatever the player actually did.
    samples = [
        (int(r["tick"]), _speed(r))
        for _, r in window_ticks.iterrows()
    ]
    samples = [(t, s) for t, s in samples if s <= _MAX_PLAUSIBLE_SPEED]
    if not samples:
        return None

    speeds = [s for _, s in samples]
    sample_ticks = [t for t, _ in samples]
    shot_tick = sample_ticks[-1]
    pre_speed = max(speeds)
    window_span = shot_tick - sample_ticks[0]

    # How long ago the player was last moving too fast to shoot accurately.
    # Ascending iteration leaves this holding the *last* such sample.
    stop_ticks: int | None = None
    for t, s in samples:
        if s > _ACCURATE_SPEED:
            stop_ticks = shot_tick - t

    # Crouching caps speed under the accuracy threshold on its own, so a
    # crouched shot is accurate without any counter-strafe having happened.
    # Recorded rather than reclassified: it still belongs in the speed
    # distribution, it just must not count as evidence of a good stop.
    ducked_raw = shot_row.get("ducked")
    crouched = bool(ducked_raw) if ducked_raw == ducked_raw and ducked_raw is not None else False

    # Movement quality classification.  The standing/stopped boundary is the
    # accuracy threshold itself rather than a separate constant: a player whose
    # peak speed never crossed it never had to stop to shoot straight, which is
    # exactly what "standing" should mean.  stop_ticks is None in precisely
    # that case, so the same measurement decides both questions.
    if shot_speed >= _ACCURATE_SPEED:
        quality = "running"
    elif stop_ticks is None:
        quality = "standing"
    elif stop_ticks <= _COUNTERSTRAFE_MAX_TICKS:
        quality = "counter-strafed"
    else:
        # Was moving fast enough to need a stop, but bled it off slowly.
        quality = "stopped"

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

    if shot_speed < _STATIC_SPEED:
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
        "stop_ticks": stop_ticks,
        "window_span": window_span,
        "crouched": crouched,
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
    """Measure crosshair placement accuracy before the engagement.

    Computes the angular distance between where the attacker was looking and
    where the victim actually was, at ``offset`` ticks before ``tick`` (default
    32 ticks ≈ 0.5 s).  ``tick`` must be the *first shot* of the engagement —
    measuring back from the kill instead would land mid-duel on any drawn-out
    fight, when the crosshair is already on target, and report that as
    excellent placement.

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

    exc, good, moderate = _aim_bounds("preaim")
    if crosshair_error < exc:
        quality = "excellent"
    elif crosshair_error < good:
        quality = "good"
    elif crosshair_error < moderate:
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
    found_off_target = False
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
            found_off_target = True
            break
        acquisition_tick = t

    if not found_off_target:
        # The crosshair never left the target inside the data we have.  Either
        # the player was pre-aimed, or the velocity sample does not reach far
        # enough back to contain the moment they acquired.  Both are
        # unmeasurable: falling through here would report the oldest sampled
        # tick as the acquisition and invent a "reaction time" that is really
        # just the size of the remaining window — which is longer the longer
        # the engagement ran, so it would file clean pre-aims as "slow".
        return None

    if acquisition_tick is None:
        # Aim was off target at the shot itself (spray transfer, or a miss).
        return None

    if acquisition_tick >= first_shot_tick:
        # No measurable gap
        return None

    reaction_ticks = first_shot_tick - acquisition_tick
    reaction_ms = round(reaction_ticks / 64 * 1000)

    # Filter out implausibly long values (>800ms usually isn't a "reaction")
    if reaction_ms > 800:
        return None

    lightning, fast, average = _aim_bounds("reaction")
    if reaction_ms < lightning:
        category = "lightning"
    elif reaction_ms < fast:
        category = "fast"
    elif reaction_ms < average:
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


def _engagement_accuracy(
    hit_ticks: list[int],
    hitgroups: list[str],
    shots_fired: int,
    first_shot_tick: int,
) -> dict[str, Any]:
    """Accuracy for one continuous engagement.

    Split out of the kill path so damage-only engagements can be measured the
    same way: nothing about hit rate or hitgroup distribution depends on
    whether the duel ended in a kill.
    """
    hits = len(hit_ticks)
    hit_pct = min(100.0, round(hits / shots_fired * 100, 1)) if shots_fired else 0.0
    return {
        "hit_pct": hit_pct,
        # Small tolerance: the first hit registers a tick or two after the shot.
        "first_bullet_hit": bool(hit_ticks) and hit_ticks[0] <= first_shot_tick + 2,
        "hitgroups": hitgroups,
        "head": sum(1 for h in hitgroups if h in ("head", "neck")),
        "upper": sum(1 for h in hitgroups if h in ("chest", "stomach")),
        "lower": sum(
            1 for h in hitgroups
            if h in ("left_arm", "right_arm", "left_leg", "right_leg")
        ),
    }


def _cluster_ticks(ticks: list[int], gap: int = _ENGAGEMENT_GAP) -> list[list[int]]:
    """Split ordered hit ticks into separate fights on a silence gap.

    Two exchanges with the same opponent twenty seconds apart are two duels,
    not one.  Without this they collapsed into a single encounter measured at
    the first shot of the first fight.
    """
    if not ticks:
        return []
    clusters: list[list[int]] = []
    current = [ticks[0]]
    for t in ticks[1:]:
        if t - current[-1] > gap:
            clusters.append(current)
            current = [t]
        else:
            current.append(t)
    clusters.append(current)
    return clusters


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

    pair_hits = _gun_damage(hurt_df[
        (hurt_df["round"] == rnd)
        & (hurt_df["attacker_steamid"].astype(str) == attacker_sid)
        & (hurt_df["user_steamid"].astype(str) == victim_sid)
        & (hurt_df["tick"] <= kill_tick)
    ])
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

    # Accuracy metrics for this engagement, via the same helper the damage-only
    # path uses so the two populations are measured identically.
    if shots_fired > 0:
        result["accuracy"] = _engagement_accuracy(
            [int(t) for t in ticks[cluster_start_idx:]],
            hitgroups,
            shots_fired,
            first_shot_tick,
        )

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
            try:
                vic_steamid = int(victim_sid_str)
            except (ValueError, TypeError):
                vic_steamid = None

            # Time-to-damage runs first because it resolves the tick the
            # engagement actually opened on.  Everything mechanical below has
            # to be anchored there: the death tick is the *last* bullet of the
            # spray, often hundreds of ms after the stop and the crosshair
            # placement we are trying to measure.
            ttd = None
            if hurt_df is not None and not hurt_df.empty:
                ttd = _analyze_time_to_damage(
                    hurt_df, sid, victim_sid_str, rnd, tick, weapon_fire_df,
                )
                if ttd:
                    kill_info["ttd"] = ttd

            first_shot_tick = (
                ttd["first_shot_tick"] if ttd and "first_shot_tick" in ttd else tick
            )

            if atk_steamid is not None and velocities_df is not None and not velocities_df.empty:
                movement = _analyze_movement(velocities_df, atk_steamid, first_shot_tick)
                if movement:
                    kill_info["movement"] = movement

                if vic_steamid is not None:
                    preaim = _analyze_preaim(
                        velocities_df, atk_steamid, vic_steamid, first_shot_tick,
                    )
                    if preaim:
                        kill_info["preaim"] = preaim

                    # Reaction time (yaw-snap approach)
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
        "killer_steamid": str(row.get("attacker_steamid", "")),
    }
    if "tick" in death_df.columns and pd.notna(row.get("tick")):
        info["tick"] = int(row["tick"])
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


def _first_shot_tick(
    weapon_fire_df: pd.DataFrame | None, sid: str, rnd: int, hit_tick: int,
) -> int:
    """Earliest shot by *sid* within ``_PRE_HIT_WINDOW`` ticks before a hit.

    Falls back to ``hit_tick`` when there is no weapon_fire data to refine it.
    """
    if (
        weapon_fire_df is None
        or weapon_fire_df.empty
        or "round" not in weapon_fire_df.columns
        or "tick" not in weapon_fire_df.columns
    ):
        return hit_tick
    id_col = _find_id_col(weapon_fire_df, ("user_steamid", "steamid", "attacker_steamid"))
    if id_col is None:
        return hit_tick
    fires = weapon_fire_df[
        (weapon_fire_df["round"] == rnd)
        & (weapon_fire_df[id_col].astype(str) == sid)
        & (weapon_fire_df["tick"] >= hit_tick - _PRE_HIT_WINDOW)
        & (weapon_fire_df["tick"] <= hit_tick)
    ]
    if fires.empty:
        return hit_tick
    return int(fires["tick"].min())


# Damage that did not come out of a gun barrel. player_hurt carries molotov
# ticks and grenade blasts alongside bullet hits, and CS reports those with a
# "generic" hitgroup because there is no hitbox involved. Counting them as hits
# credited a burning enemy to the player's gun accuracy while the molotov was
# never counted as a shot — which is how engagements ended up reading 100%.
# These stay in ADR and in the utility metrics, where they belong.
_NON_BULLET_DAMAGE = {
    "inferno", "molotov", "incgrenade", "hegrenade", "flashbang", "decoy",
    "world", "knife", "knife_t", "bomb", "c4", "taser",
}


def _gun_damage(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a player_hurt frame to bullet hits."""
    if df.empty or "weapon" not in df.columns:
        return df
    weapons = df["weapon"].astype(str).str.replace("weapon_", "", regex=False).str.lower()
    return df[~weapons.isin(_NON_BULLET_DAMAGE)]


# Weapons whose fire events are shots at a player rather than utility.
_NON_AIM_FIRE = {
    "flashbang", "smokegrenade", "hegrenade", "molotov", "incgrenade",
    "decoy", "knife", "knife_t", "bomb", "c4", "taser",
}

# A hit lands a tick or two after the shot, and the tail of a spray keeps
# arriving after it.  Damage inside this window belongs to the burst.
_FIRE_HIT_LEAD = 8
_FIRE_HIT_TAIL = 32


def _player_fire_clusters(
    weapon_fire_df: pd.DataFrame | None, sid: str, rnd: int,
) -> list[list[int]]:
    """The player's bursts this round, as lists of tick numbers.

    Grenades and knives are excluded — they are not aim.
    """
    if (
        weapon_fire_df is None
        or weapon_fire_df.empty
        or "round" not in weapon_fire_df.columns
        or "tick" not in weapon_fire_df.columns
    ):
        return []
    id_col = _find_id_col(weapon_fire_df, ("user_steamid", "steamid", "attacker_steamid"))
    if id_col is None:
        return []
    fires = weapon_fire_df[
        (weapon_fire_df["round"] == rnd)
        & (weapon_fire_df[id_col].astype(str) == sid)
    ]
    if fires.empty:
        return []
    if "weapon" in fires.columns:
        weapons = fires["weapon"].astype(str).str.replace("weapon_", "", regex=False)
        fires = fires[~weapons.isin(_NON_AIM_FIRE)]
    if fires.empty:
        return []
    return _cluster_ticks(sorted(int(t) for t in fires["tick"]))


def _get_round_whiffed_engagements(
    weapon_fire_df: pd.DataFrame | None,
    hurt_df: pd.DataFrame,
    spotted_df: pd.DataFrame | None,
    sid: str,
    rnd: int,
    player_team: int | None,
) -> list[dict[str, Any]]:
    """Bursts fired at a visible enemy that landed nothing.

    Engagements were previously anchored on damage, so a duel the player lost
    without connecting simply did not exist — which biased accuracy upward by
    removing their worst engagements from the denominator.

    The visibility gate matters as much as the engagements do: around 60% of
    bursts that hit nothing are fired with no enemy on screen at all — smoke
    spray, wallbangs, pre-fires — and charging those to aim would penalise
    deliberate utility use.  Only bursts opened while an enemy was spotted
    count.
    """
    if player_team not in (2, 3):
        return []
    clusters = _player_fire_clusters(weapon_fire_df, sid, rnd)
    if not clusters:
        return []

    hit_ticks: list[int] = []
    if (
        not hurt_df.empty
        and "round" in hurt_df.columns
        and "tick" in hurt_df.columns
        and "attacker_steamid" in hurt_df.columns
    ):
        mine = _gun_damage(hurt_df[
            (hurt_df["round"] == rnd)
            & (hurt_df["attacker_steamid"].astype(str) == sid)
        ])
        hit_ticks = sorted(int(t) for t in mine["tick"])

    # Enemies visible at each tick somebody fired.
    visible_at: dict[int, bool] = {}
    if (
        spotted_df is not None
        and not spotted_df.empty
        and {"tick", "spotted", "team_num"}.issubset(spotted_df.columns)
    ):
        enemies = spotted_df[spotted_df["team_num"] != player_team]
        for tick, group in enemies.groupby("tick"):
            visible_at[int(tick)] = bool(group["spotted"].any())

    result: list[dict[str, Any]] = []
    for cluster in clusters:
        lo = cluster[0] - _FIRE_HIT_LEAD
        hi = cluster[-1] + _FIRE_HIT_TAIL
        if any(lo <= h <= hi for h in hit_ticks):
            continue  # landed something — the damage/kill path owns it
        if not visible_at.get(cluster[0], False):
            continue  # nothing to shoot at; not an aim duel
        result.append({
            "shots_fired": len(cluster),
            "hits": 0,
            "first_tick": cluster[0],
            "last_tick": cluster[-1],
            "accuracy": _engagement_accuracy([], [], len(cluster), cluster[0]),
        })
    return result


def _get_round_damage_encounters(
    hurt_df: pd.DataFrame,
    death_df: pd.DataFrame,
    sid: str,
    rnd: int,
    velocities_df: pd.DataFrame | None = None,
    weapon_fire_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Get damage-only encounters: enemies hurt but not killed by the player."""
    if hurt_df.empty or "round" not in hurt_df.columns:
        return []
    if "user_steamid" not in hurt_df.columns or "tick" not in hurt_df.columns:
        # Without a victim to group by (or a tick to order on) there are no
        # encounters to reconstruct.
        return []

    player_damage = _gun_damage(hurt_df[
        (hurt_df["round"] == rnd)
        & (hurt_df["attacker_steamid"].astype(str) == sid)
    ])
    if player_damage.empty:
        return []

    # Find victims the player killed this round
    killed_victims: set[str] = set()
    if (
        not death_df.empty
        and "round" in death_df.columns
        and "user_steamid" in death_df.columns
    ):
        kills = death_df[
            (death_df["round"] == rnd)
            & (death_df["attacker_steamid"].astype(str) == sid)
        ]
        killed_victims = set(kills["user_steamid"].astype(str))

    try:
        atk_steamid: int | None = int(sid)
    except (ValueError, TypeError):
        atk_steamid = None

    fire_clusters = _player_fire_clusters(weapon_fire_df, sid, rnd)

    result: list[dict[str, Any]] = []
    for victim_sid, group in player_damage.groupby(
        player_damage["user_steamid"].astype(str)
    ):
        sorted_group = group.sort_values("tick")
        tick_list = [int(t) for t in sorted_group["tick"].tolist()]

        # Two exchanges with the same opponent minutes apart are two duels.
        # Grouping only by victim merged them into one, measured at the first
        # shot of the first fight and counted once.
        clusters = _cluster_ticks(tick_list)
        was_killed = victim_sid in killed_victims

        try:
            vic_steamid_int: int | None = int(victim_sid)
        except (ValueError, TypeError):
            vic_steamid_int = None

        for idx, cluster in enumerate(clusters):
            # Only the last exchange can be the one that produced the kill;
            # everything before it is a genuine damage-only engagement.
            if was_killed and idx == len(clusters) - 1:
                continue

            first_tick, last_tick = cluster[0], cluster[-1]
            rows = sorted_group[
                (sorted_group["tick"] >= first_tick) & (sorted_group["tick"] <= last_tick)
            ]
            weapon = str(rows.iloc[0].get("weapon", "")) if len(rows) else ""

            enc: dict[str, Any] = {
                "weapon": _weapon_display(weapon),
                "victim_sid": victim_sid,
                "last_tick": last_tick,
            }

            # Anchor to the first shot of the engagement rather than the first
            # hit, so these land on the same point of the duel as the kill path
            # — both distributions get pooled in _calculate_aim_stats.
            shot_tick = _first_shot_tick(weapon_fire_df, sid, rnd, first_tick)

            # Accuracy: nothing about hit rate depends on the duel being won,
            # and these engagements are ~40% of all the shooting a player does.
            # Count the whole burst, not just up to the last bullet that
            # connected: the tail of a spray that missed is exactly the part
            # that should count against accuracy.
            shots_fired = len(cluster)
            for burst in fire_clusters:
                if burst[0] - _FIRE_HIT_TAIL <= first_tick <= burst[-1] + _FIRE_HIT_TAIL:
                    shots_fired = max(shots_fired, len(burst))
                    break

            hitgroups: list[str] = []
            if "hitgroup" in rows.columns:
                hitgroups = rows["hitgroup"].dropna().astype(str).str.lower().tolist()
            enc["accuracy"] = _engagement_accuracy(
                cluster, hitgroups, shots_fired, shot_tick,
            )
            enc["shots_fired"] = shots_fired
            enc["hits"] = len(cluster)

            if velocities_df is not None and not velocities_df.empty and atk_steamid is not None:
                movement = _analyze_movement(velocities_df, atk_steamid, shot_tick)
                if movement:
                    enc["movement"] = movement

                if vic_steamid_int is not None:
                    preaim = _analyze_preaim(
                        velocities_df, atk_steamid, vic_steamid_int, shot_tick,
                    )
                    if preaim:
                        enc["preaim"] = preaim

                    # Reaction is about how fast the crosshair got there and
                    # the trigger followed; whether the duel ended in a kill
                    # has no bearing on it.
                    rxn = _analyze_reaction_time(
                        velocities_df, atk_steamid, vic_steamid_int, shot_tick,
                    )
                    if rxn:
                        enc["reaction"] = rxn

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

                    # Blinding yourself is not blinding a teammate. The team
                    # check alone says otherwise, since you share your own
                    # team — which put the player on their own friendly-fire
                    # chart. Self-flashes are tagged and left out of both the
                    # enemy and friendly sides.
                    is_self = str(brow.get("user_steamid", "")) == sid
                    is_team = False
                    if not is_self and atk_team_col and vic_team_col:
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
                        "is_self": is_self,
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
