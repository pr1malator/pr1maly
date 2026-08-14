"""Aggregates across matches: trends, career performance, AI assessment."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.ai_service import (
    chat_completion,
)
from src.api.deps import (
    _db,
    _load_ai_assessments,
    _load_ai_patterns,
    _load_ai_roles,
    _resolve_ai_target,
    _save_ai_assessments,
)
from src.database import (
    get_all_matches,
    get_analyzer_versions,
    get_players_for_matches,
    get_rounds_for_matches,
)
from src.domain.blobs import stored_value
from src.domain.metrics import REGISTRY, catalogue
from src.metrics.behavior import (
    classify_archetype,
    empty_side_role,
    side_role,
)
from src.processor import ANALYZER_VERSION
from src.services.ai_context import (
    ASSESSMENT_SYSTEM_PROMPT,
    OVERALL_KEY,
    OVERALL_KEYS,
    OVERALL_SYSTEM_PROMPT,
    build_overall_context,
    build_patterns_context,
    build_role_context,
    matches_with_recorded_play,
    strip_json_fences,
)

router = APIRouter()

# The sections one assessment is expected to fill in.
_ASSESSMENT_KEYS = ("ct_role", "t_role", "aim", "utility", "behaviour")


@router.get("/api/analyzer/version")
def get_analyzer_version():
    """Current analyzer version, and how many stored matches predate it.

    ``metrics`` reports the same question per measurement. A version bump used
    to mean "re-parse everything"; with the registry it means "these matches
    are missing this one measurement", and for a metric that reads nothing but
    the stored rounds, refreshing it does not need the demo back at all.
    """
    conn = _db()
    try:
        versions = get_analyzer_versions(conn)
    finally:
        conn.close()
    stale = sum(1 for v in versions if v < ANALYZER_VERSION)
    return {
        "analyzer_version": ANALYZER_VERSION,
        "total_matches": len(versions),
        "stale_matches": stale,
        "metrics": [spec.as_catalog_entry() for spec in REGISTRY],
    }


@router.get("/api/metrics")
def list_metrics():
    """Every measurement the analyzer produces, by name.

    Exists so the UI and the prompt builder can read a catalogue rather than
    hardcode blob key names. ``needs_demo`` is the useful field: metrics
    without it can be recomputed from the database for matches whose .dem the
    retention feature has already deleted.

    ``fields`` is the human-facing half: what each figure on screen measures,
    how it is derived, and where any tier shown against it came from. The
    interface reads it rather than carrying its own copy of the wording, and
    METRICS.md is generated from the same table.
    """
    groups = REGISTRY.groups()
    return {
        "count": len(REGISTRY),
        "groups": {
            name: [spec.as_catalog_entry() for spec in specs]
            for name, specs in groups.items()
        },
        "recomputable_without_demo": [
            spec.id for spec in REGISTRY.select(without_demo=True)
        ],
        "fields": catalogue.as_dicts(),
        "provenance": catalogue.PROVENANCE,
        "statistics": catalogue.STATISTICS,
    }


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------
@router.get("/api/trends")
def get_trends(maps: str = "", steam_ids: str = ""):
    """Return trend data for charts (rating, ADR, KAST, K/D/A over time).

    Optional ``maps`` query param: comma-separated map filter.
    Optional ``steam_ids`` query param: comma-separated player filter.
    """
    conn = _db()
    matches = get_all_matches(conn)
    conn.close()

    # Filter by maps if provided
    if maps.strip():
        allowed = {m.strip().lower() for m in maps.split(",")}
        matches = [m for m in matches if m.get("map_name", "").lower() in allowed]

    # Filter by player if provided
    if steam_ids.strip():
        allowed_sids = {s.strip() for s in steam_ids.split(",") if s.strip()}
        matches = [m for m in matches if m.get("player_steam_id", "") in allowed_sids]

    # Sort chronologically (oldest first for charts)
    matches.sort(key=lambda m: m.get("date", ""))

    data_points = []
    for m in matches:
        # Extract aim_rating from stored JSON
        aim_rating = stored_value(m, "aim_stats", "aim_rating")
        # Extract utility_rating from stored JSON
        utility_rating = stored_value(m, "utility_data", "utility_rating")
        data_points.append({
            "match_id": m["match_id"],
            "date": m.get("date"),
            "map_name": m.get("map_name"),
            "hltv_rating": m.get("hltv_rating"),
            "adr": m.get("adr"),
            "kast": m.get("kast"),
            "kills": m.get("kills"),
            "deaths": m.get("deaths"),
            "assists": m.get("assists"),
            "match_result": m.get("match_result"),
            "team_score": m.get("team_score"),
            "enemy_score": m.get("enemy_score"),
            "aim_rating": aim_rating,
            "utility_rating": utility_rating,
        })

    # Career averages
    if data_points:
        n = len(data_points)
        aim_vals = [d["aim_rating"] for d in data_points if d["aim_rating"] is not None]
        util_vals = [d["utility_rating"] for d in data_points if d["utility_rating"] is not None]
        averages = {
            "avg_rating": round(sum(d["hltv_rating"] or 0 for d in data_points) / n, 4),
            "avg_adr": round(sum(d["adr"] or 0 for d in data_points) / n, 2),
            "avg_kast": round(sum(d["kast"] or 0 for d in data_points) / n, 2),
            "avg_kills": round(sum(d["kills"] or 0 for d in data_points) / n, 1),
            "avg_deaths": round(sum(d["deaths"] or 0 for d in data_points) / n, 1),
            "avg_aim_rating": round(sum(aim_vals) / len(aim_vals), 1) if aim_vals else None,
            "avg_utility_rating": round(sum(util_vals) / len(util_vals), 1) if util_vals else None,
            "total_matches": n,
        }
    else:
        averages = {
            "avg_rating": 0, "avg_adr": 0, "avg_kast": 0,
            "avg_kills": 0, "avg_deaths": 0,
            "avg_aim_rating": None, "avg_utility_rating": None,
            "total_matches": 0,
        }

    return {
        "data_points": data_points,
        "averages": averages,
        "available_maps": sorted({m.get("map_name", "") for m in matches}),
    }

# ---------------------------------------------------------------------------
# Performance analytics (powers the breakdown / stats page)
# ---------------------------------------------------------------------------
@router.get("/api/performance")
def get_performance(maps: str = "", steam_ids: str = ""):
    """Aggregate enriched round data into role / mechanic / phase stats."""

    conn = _db()
    matches = get_all_matches(conn)

    if maps.strip():
        allowed = {m.strip().lower() for m in maps.split(",")}
        matches = [m for m in matches if m.get("map_name", "").lower() in allowed]

    if steam_ids.strip():
        allowed_sids = {s.strip() for s in steam_ids.split(",") if s.strip()}
        matches = [m for m in matches if m.get("player_steam_id", "") in allowed_sids]

    if not matches:
        conn.close()
        return _empty_performance()

    match_ids = [m["match_id"] for m in matches]

    # Two queries for the whole history rather than two per match.
    all_rounds = get_rounds_for_matches(conn, match_ids)
    user_players = get_players_for_matches(conn, match_ids, user_only=True)
    conn.close()

    total_rounds = len(all_rounds)
    if total_rounds == 0:
        return _empty_performance()

    # --- Headshot % ---
    total_kills_hs = 0
    total_kills_count = 0
    weapon_kills: dict[str, int] = {}
    for r in all_rounds:
        kd = r["enriched"].get("kills_detail", [])
        for k in kd:
            total_kills_count += 1
            if k.get("headshot"):
                total_kills_hs += 1
            wep = k.get("weapon", "Unknown")
            weapon_kills[wep] = weapon_kills.get(wep, 0) + 1
    hs_pct = round((total_kills_hs / total_kills_count * 100) if total_kills_count else 0, 1)

    # --- Side stats ---
    ct_rounds = [r for r in all_rounds if r["enriched"].get("side") == "CT"]
    t_rounds = [r for r in all_rounds if r["enriched"].get("side") == "T"]
    ct_wins = sum(1 for r in ct_rounds if r["enriched"].get("side") == r["enriched"].get("round_winner"))
    t_wins = sum(1 for r in t_rounds if r["enriched"].get("side") == r["enriched"].get("round_winner"))
    ct_win_pct = round((ct_wins / len(ct_rounds) * 100) if ct_rounds else 0, 1)
    t_win_pct = round((t_wins / len(t_rounds) * 100) if t_rounds else 0, 1)

    # Pistol rounds (round 1 and first round of second half)
    half_start = 13  # MR12 → round 13 is second pistol
    ct_pistol = [r for r in ct_rounds if r.get("round_number") in (1, half_start)]
    t_pistol = [r for r in t_rounds if r.get("round_number") in (1, half_start)]
    ct_pistol_wins = sum(1 for r in ct_pistol if r["enriched"].get("side") == r["enriched"].get("round_winner"))
    t_pistol_wins = sum(1 for r in t_pistol if r["enriched"].get("side") == r["enriched"].get("round_winner"))
    ct_pistol_pct = round((ct_pistol_wins / len(ct_pistol) * 100) if ct_pistol else 0, 0)
    t_pistol_pct = round((t_pistol_wins / len(t_pistol) * 100) if t_pistol else 0, 0)

    # --- Opening duels ---
    opening_kills = 0
    opening_deaths = 0
    for r in all_rounds:
        od = r["enriched"].get("opening_duel")
        if od:
            if od.get("role") == "opening_kill":
                opening_kills += 1
            elif od.get("role") == "opening_death":
                opening_deaths += 1
    opening_total = opening_kills + opening_deaths
    opening_kill_pct = round((opening_kills / opening_total * 100) if opening_total else 0, 1)

    # --- Survival rate ---
    survived = sum(1 for r in all_rounds if r.get("survived"))
    survival_pct = round((survived / total_rounds * 100), 1)

    # --- Clutch stats ---
    clutch_attempts = 0
    clutch_wins = 0
    for r in all_rounds:
        c = r["enriched"].get("clutch")
        if c:
            clutch_attempts += 1
            if c.get("won"):
                clutch_wins += 1
    clutch_win_pct = round((clutch_wins / clutch_attempts * 100) if clutch_attempts else 0, 0)

    # --- Utility stats ---
    total_flashed = 0
    total_flash_assists = 0
    total_he_dmg = 0
    total_molly_dmg = 0
    for r in all_rounds:
        u = r["enriched"].get("utility", {})
        total_flashed += u.get("enemies_flashed", 0)
        total_flash_assists += u.get("flash_assists", 0)
        total_he_dmg += u.get("he_damage", 0)
        total_molly_dmg += sum(m.get("damage", 0) for m in u.get("molotov_damage", []))
    util_per_round = round((total_flashed + total_flash_assists) / total_rounds, 2) if total_rounds else 0
    util_dmg_per_round = round((total_he_dmg + total_molly_dmg) / total_rounds, 1) if total_rounds else 0

    # --- Trade stats ---
    traded_deaths = sum(1 for r in all_rounds if r.get("traded"))
    total_deaths = sum(1 for r in all_rounds if r.get("deaths", 0) > 0)
    trade_pct = round((traded_deaths / total_deaths * 100) if total_deaths else 0, 1)

    # --- Multi-kill stats ---
    total_2k = sum(p.get("rounds_2k", 0) for p in user_players)
    total_3k = sum(p.get("rounds_3k", 0) for p in user_players)
    total_4k = sum(p.get("rounds_4k", 0) for p in user_players)
    total_5k = sum(p.get("rounds_5k", 0) for p in user_players)
    multikill_rounds = total_2k + total_3k + total_4k + total_5k
    multikill_pct = round((multikill_rounds / total_rounds * 100) if total_rounds else 0, 1)

    # --- Role classification ---
    role = classify_archetype(
        opening_kill_pct=opening_kill_pct,
        survival_pct=survival_pct,
        util_per_round=util_per_round,
        trade_pct=trade_pct,
        weapon_kills=weapon_kills,
        total_kills=total_kills_count,
    )

    # --- Per-side role classification ---
    ct_role = side_role(ct_rounds)
    t_role = side_role(t_rounds)

    # --- Win/Loss streak (last 10) ---
    matches.sort(key=lambda m: m.get("date", ""), reverse=True)
    recent_results = [m.get("match_result", "")[:1] for m in matches[:10]]

    # Top weapon
    top_weapon = max(weapon_kills, key=weapon_kills.get) if weapon_kills else "Unknown"

    return {
        "total_rounds": total_rounds,
        "total_matches": len(matches),
        "hs_pct": hs_pct,
        "ct_win_pct": ct_win_pct,
        "t_win_pct": t_win_pct,
        "ct_pistol_pct": ct_pistol_pct,
        "t_pistol_pct": t_pistol_pct,
        "opening_kill_pct": opening_kill_pct,
        "opening_kills": opening_kills,
        "opening_deaths": opening_deaths,
        "survival_pct": survival_pct,
        "clutch_attempts": clutch_attempts,
        "clutch_wins": clutch_wins,
        "clutch_win_pct": clutch_win_pct,
        "util_per_round": util_per_round,
        "util_dmg_per_round": util_dmg_per_round,
        "trade_pct": trade_pct,
        "multikill_pct": multikill_pct,
        "multikill_rounds": multikill_rounds,
        "rounds_2k": total_2k,
        "rounds_3k": total_3k,
        "rounds_4k": total_4k,
        "rounds_5k": total_5k,
        "role": role,
        "ct_role": ct_role,
        "t_role": t_role,
        "recent_results": recent_results,
        "top_weapon": top_weapon,
        "weapon_kills": dict(sorted(weapon_kills.items(), key=lambda x: -x[1])[:5]),
    }

def _empty_performance() -> dict:
    return {
        "total_rounds": 0, "total_matches": 0, "hs_pct": 0,
        "ct_win_pct": 0, "t_win_pct": 0, "ct_pistol_pct": 0, "t_pistol_pct": 0,
        "opening_kill_pct": 0, "opening_kills": 0, "opening_deaths": 0,
        "survival_pct": 0, "clutch_attempts": 0, "clutch_wins": 0, "clutch_win_pct": 0,
        "util_per_round": 0, "util_dmg_per_round": 0, "trade_pct": 0,
        "multikill_pct": 0, "multikill_rounds": 0,
        "rounds_2k": 0, "rounds_3k": 0, "rounds_4k": 0, "rounds_5k": 0,
        "role": {"name": "Unknown", "icon": "help",
                 "description": "Upload demos to analyze your playstyle."},
        "ct_role": empty_side_role(),
        "t_role": empty_side_role(),
        "recent_results": [], "top_weapon": "Unknown", "weapon_kills": {},
    }

def _map_scoped_rounds(maps: str) -> tuple[list[dict], list[dict]]:
    """Matches on the filtered map, and every enriched round belonging to them."""
    conn = _db()
    try:
        allowed = {m.strip().lower() for m in maps.split(",")}
        matches = matches_with_recorded_play([
            m for m in get_all_matches(conn)
            if m.get("map_name", "").lower() in allowed
        ])
        if not matches:
            raise HTTPException(status_code=404, detail="No matches found for this map")

        rounds = get_rounds_for_matches(conn, [m["match_id"] for m in matches])
    finally:
        conn.close()

    if not rounds:
        raise HTTPException(status_code=404, detail="No enriched round data for this map")
    return matches, rounds

def _all_rounds_for(matches: list[dict]) -> list[dict]:
    """Every enriched round belonging to *matches*."""
    conn = _db()
    try:
        return get_rounds_for_matches(conn, [m["match_id"] for m in matches])
    finally:
        conn.close()

@router.post("/api/performance/ai-assessment")
async def ai_assessment(maps: str = "", provider: str = "", model: str = ""):
    """Assess playing patterns in a single model call, for one map or for all.

    Roles and patterns used to be two buttons hitting two endpoints, which cost
    two requests and split what is really one question — where the player plays
    and how — across two prompts that could not see each other.  One call also
    lets the model tie the halves together, which is where most of the insight
    is.

    Without a map filter this becomes the career assessment: the same pattern
    reading over every match, with a map comparison in place of the roles.
    """
    import json as _json

    provider, model, api_key = _resolve_ai_target(provider, model)
    overall = not maps.strip()

    if overall:
        conn = _db()
        try:
            matches = matches_with_recorded_play(get_all_matches(conn))
        finally:
            conn.close()
        if not matches:
            raise HTTPException(status_code=404, detail="No matches analysed yet")
        all_rounds = _all_rounds_for(matches)
        if not all_rounds:
            raise HTTPException(status_code=404, detail="No enriched round data")
        context = build_overall_context(matches, all_rounds)
        system_prompt = OVERALL_SYSTEM_PROMPT
        keys = OVERALL_KEYS
        store_key = OVERALL_KEY
    else:
        matches, all_rounds = _map_scoped_rounds(maps)
        ct_rounds = [r for r in all_rounds if (r.get("enriched") or {}).get("side") == "CT"]
        t_rounds = [r for r in all_rounds if (r.get("enriched") or {}).get("side") == "T"]
        map_display = maps.strip().split(",")[0]
        context = "\n\n".join([
            build_patterns_context(map_display, matches, all_rounds),
            "=== ROUND-BY-ROUND POSITIONS ===",
            build_role_context(map_display, ct_rounds, t_rounds),
        ])
        system_prompt = ASSESSMENT_SYSTEM_PROMPT
        keys = _ASSESSMENT_KEYS
        store_key = map_display.lower()

    messages = [{"role": "user", "content": context}]
    try:
        response = await chat_completion(
            provider, model, api_key, messages, system_prompt
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI request failed: {exc}")

    try:
        result = _json.loads(strip_json_fences(response))
    except (_json.JSONDecodeError, IndexError):
        # Show the model's prose rather than nothing, so a provider that
        # ignores the format is still of some use.
        result = {
            "headline": response[:600],
            "aim": {"name": "AI Analysis", "icon": "smart_toy", "description": response[:600]},
        }

    entry = {key: result.get(key, {}) for key in keys}
    entry.update({
        "headline": result.get("headline", ""),
        "scope": "overall" if overall else "map",
        "matches": len(matches),
        "rounds": len(all_rounds),
        "provider": provider,
        "model": model,
    })

    stored = _load_ai_assessments()
    stored[store_key] = entry
    _save_ai_assessments(stored)

    return entry

@router.get("/api/performance/ai-assessment")
def get_persisted_ai_assessment(maps: str = ""):
    """Return the persisted assessment for a map, or the career one without.

    Falls back to the separate role and pattern files this endpoint replaced,
    so assessments produced before the two were merged still appear instead of
    silently disappearing behind an empty card.  Only per-map assessments have
    such a history; the career one is new, so it has nothing to fall back to.
    """
    if not maps.strip():
        return _load_ai_assessments().get(OVERALL_KEY) or {}
    map_key = maps.strip().split(",")[0].lower()

    entry = _load_ai_assessments().get(map_key)
    if entry:
        return entry

    legacy: dict[str, Any] = {}
    old_roles = _load_ai_roles().get(map_key) or {}
    old_patterns = _load_ai_patterns().get(map_key) or {}
    for key in _ASSESSMENT_KEYS:
        value = old_roles.get(key) or old_patterns.get(key)
        if value:
            legacy[key] = value
    if not legacy:
        return {}
    legacy["headline"] = old_patterns.get("headline", "")
    legacy["model"] = old_patterns.get("model") or old_roles.get("model", "")
    legacy["matches"] = old_patterns.get("matches")
    legacy["rounds"] = old_patterns.get("rounds")
    return legacy
