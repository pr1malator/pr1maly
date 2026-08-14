"""Reading, annotating and re-analysing stored matches."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.ai_service import (
    build_match_context,
    chat_completion,
    load_config as load_ai_config,
)
from src.api.deps import (
    _available_demo_names,
    _db,
    _load_friends,
    _match_summary,
    _resolve_demo,
)
from src.api.schemas import (
    ChatMessageIn,
    NotesUpdate,
    TagCreate,
)
from src.callouts import (
    game_to_pixel,
    get_radar_config,
)
from src.database import (
    add_tag,
    clear_chat_history,
    delete_match,
    get_all_matches,
    get_chat_history,
    get_match,
    get_match_players,
    get_round_stats,
    get_tags,
    get_tags_for_all_matches,
    move_chat_history,
    save_chat_message,
    save_match,
    update_context_notes,
)
from src.domain.blobs import decode_dict, decode_match_blobs
from src.metrics.behavior import (
    match_behavioral_axes,
)
from src.parser import parse_info_file
from src.processor import ANALYZER_VERSION, compute_benchmarks
from src.services.import_service import (
    analyse_demo,
)

router = APIRouter()

# Heavy JSON blobs that only the single-match view needs. The list endpoint
# strips them so a career page does not ship megabytes it will not render.
_MATCH_DETAIL_ONLY_FIELDS = ("utility_data", "aim_stats", "role_data", "impact_stats")


@router.get("/api/matches")
def list_matches(player_steam_id: str = None):
    """Return all matches ordered by date descending, optionally filtered by player.

    Detail-only blobs are stripped: this feeds match lists, and every consumer
    fetches /api/matches/{id} for anything beyond the summary row.
    """
    conn = _db()
    matches = get_all_matches(conn, player_steam_id=player_steam_id)

    # One query for every tag rather than one per match.
    tags_by_match = get_tags_for_all_matches(conn)
    conn.close()

    # One directory listing for the whole page rather than a stat per row.
    on_disk = _available_demo_names()

    for m in matches:
        m["tags"] = tags_by_match.get(m["match_id"], [])
        version = int(m.get("analyzer_version") or 0)
        m["analyzer_version"] = version
        m["analysis_stale"] = version < ANALYZER_VERSION
        m["demo_available"] = bool(m.get("filename")) and m["filename"] in on_disk
        for field in _MATCH_DETAIL_ONLY_FIELDS:
            m.pop(field, None)
    return matches

# ---------------------------------------------------------------------------
# Career averages for match-level KPI comparison
# ---------------------------------------------------------------------------
@router.get("/api/matches/career-averages")
def get_career_averages():
    """Compute per-match KPI averages across all matches for trend comparison.

    Returns averages for: HS%, K/D, KAST, enemies flashed, avg blind duration,
    HE damage, Molotov damage, clutch win %, trade %, opening kill rate,
    multi-kill rounds, aim score, utility use rate, utility rating.
    """

    conn = _db()
    matches = get_all_matches(conn)
    if not matches:
        conn.close()
        return {"total_matches": 0, "averages": {}}

    # Per-match accumulators
    hs_pcts: list[float] = []
    kd_ratios: list[float] = []
    kast_vals: list[float] = []
    flash_counts: list[int] = []
    blind_durs: list[float] = []
    he_dmgs: list[int] = []
    molly_dmgs: list[int] = []
    clutch_pcts: list[float] = []
    trade_pcts: list[float] = []
    opening_rates: list[float] = []
    multikill_counts: list[int] = []
    aim_ratings: list[float] = []
    movement_avgs: list[float] = []
    ttk_avgs: list[float] = []
    preaim_avgs: list[float] = []
    reaction_avgs: list[float] = []
    use_rates: list[float] = []
    utility_ratings: list[float] = []

    for m in matches:
        # Top-level match stats
        kd_ratios.append(m.get("kd_ratio") or 0)
        kast_vals.append(m.get("kast") or 0)
        multikill_counts.append(
            (m.get("rounds_2k") or 0) + (m.get("rounds_3k") or 0) +
            (m.get("rounds_4k") or 0) + (m.get("rounds_5k") or 0)
        )

        # Aim rating from stored JSON
        aim_raw = m.get("aim_stats")
        if aim_raw:
            try:
                aim_obj = decode_dict(aim_raw)
                ar = aim_obj.get("aim_rating")
                if ar is not None:
                    aim_ratings.append(ar)
                # Prefer the median, which is what the match page shows and
                # what the benchmarks grade.  Matches analysed before medians
                # existed only carry "avg", so fall back rather than drop them
                # out of the career line entirely.
                def _headline(block: dict) -> float | None:
                    v = block.get("median")
                    return v if v is not None else block.get("avg")

                for block_key, sink in (
                    ("movement", movement_avgs),
                    ("ttk", ttk_avgs),
                    ("preaim", preaim_avgs),
                    ("reaction", reaction_avgs),
                ):
                    value = _headline(aim_obj.get(block_key, {}))
                    if value is not None:
                        sink.append(value)
            except Exception:
                pass

        # Utility data from stored JSON
        util_raw = m.get("utility_data")
        if util_raw:
            try:
                util_obj = decode_dict(util_raw)
                ur = util_obj.get("utility_rating")
                if ur is not None:
                    utility_ratings.append(ur)
                eco = util_obj.get("economics") or {}
                use_r = eco.get("use_rate")
                if use_r is not None:
                    use_rates.append(use_r)
            except Exception:
                pass

        # Round-level stats
        rounds = get_round_stats(conn, m["match_id"])
        total_kills = 0
        hs_kills = 0
        m_flashed = 0
        m_blind_dur = 0.0
        m_blind_hits = 0
        m_he = 0
        m_molly = 0
        clutch_won = 0
        clutch_total = 0
        deaths = 0
        traded = 0
        open_kills = 0
        open_deaths = 0

        for r in rounds:
            ej = r.get("enriched_json")
            if not ej:
                continue
            try:
                e = decode_dict(ej)
            except Exception:
                continue

            # HS%
            for k in e.get("kills_detail", []):
                total_kills += 1
                if k.get("headshot"):
                    hs_kills += 1

            # Utility
            u = e.get("utility", {})
            ef = u.get("enemies_flashed", 0) or 0
            m_flashed += ef
            abd = u.get("avg_blind_duration", 0) or 0
            if ef > 0 and abd > 0:
                m_blind_dur += abd * ef
                m_blind_hits += ef
            m_he += u.get("he_damage", 0) or 0
            for md in u.get("molotov_damage", []):
                m_molly += md.get("damage", 0) or 0

            # Clutch
            c = e.get("clutch")
            if c:
                clutch_total += 1
                if c.get("won"):
                    clutch_won += 1

            # Trade
            if r.get("deaths", 0) > 0:
                deaths += 1
                if r.get("traded"):
                    traded += 1

            # Opening duels
            od = e.get("opening_duel")
            if od:
                if od.get("role") == "opening_kill":
                    open_kills += 1
                elif od.get("role") == "opening_death":
                    open_deaths += 1

        hs_pcts.append((hs_kills / total_kills * 100) if total_kills else 0)
        flash_counts.append(m_flashed)
        blind_durs.append((m_blind_dur / m_blind_hits) if m_blind_hits else 0)
        he_dmgs.append(m_he)
        molly_dmgs.append(m_molly)
        clutch_pcts.append((clutch_won / clutch_total * 100) if clutch_total else 0)
        trade_pcts.append((traded / deaths * 100) if deaths else 0)
        od_total = open_kills + open_deaths
        opening_rates.append((open_kills / od_total * 100) if od_total else 0)

    conn.close()

    def _avg(vals: list) -> float | None:
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "total_matches": len(matches),
        "averages": {
            "hs_pct": _avg(hs_pcts),
            "kd_ratio": _avg(kd_ratios),
            "kast": _avg(kast_vals),
            "enemies_flashed": _avg(flash_counts),
            "avg_blind_duration": _avg(blind_durs),
            "he_damage": _avg(he_dmgs),
            "molotov_damage": _avg(molly_dmgs),
            "clutch_win_pct": _avg(clutch_pcts),
            "trade_pct": _avg(trade_pcts),
            "opening_kill_rate": _avg(opening_rates),
            "multikill_rounds": _avg(multikill_counts),
            "aim_rating": _avg(aim_ratings),
            "movement_avg": _avg(movement_avgs),
            "ttk_avg": _avg(ttk_avgs),
            "preaim_avg": _avg(preaim_avgs),
            "reaction_avg": _avg(reaction_avgs),
            "use_rate": _avg(use_rates),
            "utility_rating": _avg(utility_ratings),
        },
    }

# ---------------------------------------------------------------------------
# 2D Replay data
# ---------------------------------------------------------------------------
@router.get("/api/matches/{match_id}/replay")
def get_replay_data(match_id: str, round_number: int = 0):
    """Return 2D replay frames for a match.

    If *round_number* is 0, returns metadata only (round list with frame
    counts). If a specific round is given, returns full frame data with
    pixel-space coordinates.
    """
    import json as _json

    conn = _db()
    match = get_match(conn, match_id)
    if not match:
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")

    map_name = match.get("map_name", "")
    radar = get_radar_config(map_name)
    if not radar:
        conn.close()
        raise HTTPException(status_code=400, detail=f"No radar data for {map_name}")

    # The one caller that genuinely wants the frames.
    rounds = get_round_stats(conn, match_id, include_replay=True)
    conn.close()

    # Check if replay data exists at all
    has_replay = any(r.get("replay_json") for r in rounds)
    if not has_replay:
        return {
            "match_id": match_id,
            "map_name": map_name,
            "has_replay": False,
            "message": "No replay data available. Re-upload the demo to generate replay data.",
        }

    # Determine radar image URL
    clean = map_name.replace("de_", "").replace("cs_", "")
    import time as _time
    radar_image = f"/frontend/img/radar/{clean}.png?v={int(_time.time())}"

    if round_number == 0:
        # Return round list overview (no frames)
        round_list = []
        for r in rounds:
            rj = r.get("replay_json")
            if not rj:
                round_list.append({"round": r["round_number"], "frames": 0})
                continue
            # An unreadable blob decodes to {}, which counts as zero frames —
            # the same answer the try/except gave.
            rd = decode_dict(rj)
            round_list.append({
                "round": r["round_number"],
                "frames": len(rd.get("frames", [])),
            })
        return {
            "match_id": match_id,
            "map_name": map_name,
            "has_replay": True,
            "radar_image": radar_image,
            "radar": {"pos_x": radar["pos_x"], "pos_y": radar["pos_y"], "scale": radar["scale"]},
            "total_rounds": len(rounds),
            "rounds": round_list,
        }

    # Return full frame data for a specific round
    target = None
    for r in rounds:
        if r["round_number"] == round_number:
            target = r
            break
    if not target or not target.get("replay_json"):
        raise HTTPException(status_code=404, detail=f"No replay data for round {round_number}")

    # Deliberately not decode_dict: the viewer has nothing to show without
    # these frames, so an unreadable blob is an error rather than an empty round.
    try:
        rd = _json.loads(target["replay_json"])
    except ValueError:
        raise HTTPException(status_code=500, detail="Corrupt replay data") from None

    # Convert game coordinates to pixel coordinates
    players = rd.get("players", {})
    raw_frames = rd.get("frames", [])
    events = rd.get("events", [])

    pixel_frames = []
    for frame in raw_frames:
        tick_offset = frame[0]
        positions = frame[1]
        px_positions = {}
        for sid, coords in positions.items():
            gx, gy, hp = coords[0], coords[1], coords[2]
            pix = game_to_pixel(map_name, gx, gy)
            if pix:
                px_positions[sid] = [round(pix[0], 1), round(pix[1], 1), hp]
            else:
                px_positions[sid] = [0, 0, hp]
        pixel_frames.append([tick_offset, px_positions])

    # Convert kill event positions to pixel coords (attacker/victim from frames)
    # Convert grenade event game coordinates to pixel coordinates
    for ev in events:
        if ev.get("type") == "grenade" and "x" in ev and "y" in ev:
            pix = game_to_pixel(map_name, ev["x"], ev["y"])
            if pix:
                ev["px"] = round(pix[0], 1)
                ev["py"] = round(pix[1], 1)
            del ev["x"]
            del ev["y"]

    return {
        "match_id": match_id,
        "map_name": map_name,
        "has_replay": True,
        "radar_image": radar_image,
        "radar": {"pos_x": radar["pos_x"], "pos_y": radar["pos_y"], "scale": radar["scale"]},
        "round_number": round_number,
        "players": players,
        "frames": pixel_frames,
        "events": events,
        "sample_interval": 32,
        "tick_rate": 64,
    }

# ---------------------------------------------------------------------------
# Single match detail
# ---------------------------------------------------------------------------
@router.get("/api/matches/{match_id}")
def get_match_detail(match_id: str):
    """Return full match detail including players, rounds, and tags."""
    conn = _db()
    match = get_match(conn, match_id)
    if not match:
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")

    players = get_match_players(conn, match_id)
    rounds = get_round_stats(conn, match_id)
    tags = get_tags(conn, match_id)
    conn.close()

    # Split players by team
    user_team = None
    for p in players:
        if p.get("is_user"):
            user_team = p.get("team")
            break

    my_team = [p for p in players if p.get("team") == user_team]
    enemy_team = [p for p in players if p.get("team") != user_team]

    # Annotate friends
    friend_ids = {f["steam_id"] for f in _load_friends()}
    for p in my_team + enemy_team:
        p["is_friend"] = p.get("steam_id", "") in friend_ids

    # All four blob columns, decoded the same tolerant way. A match analysed
    # before one of these existed simply has None there.
    decode_match_blobs(match)
    aim_stats = match.get("aim_stats")
    role_data = match.get("role_data")
    utility_data = match.get("utility_data")

    # Compute benchmark tier labels from stored stats
    benchmarks = compute_benchmarks(
        aim_stats or {},
        utility_data or {},
        match.get("total_rounds", 0),
        match.get("map_name", ""),
    )

    return {
        **match,
        "aim_stats": aim_stats,
        "role_data": role_data,
        "utility_data": utility_data,
        "benchmarks": benchmarks,
        "behavioral_axes": match_behavioral_axes(rounds),
        "tags": tags,
        "round_stats": rounds,
        "my_team": my_team,
        "enemy_team": enemy_team,
    }

# ---------------------------------------------------------------------------
# Match mutations
# ---------------------------------------------------------------------------
@router.put("/api/matches/{match_id}/notes")
def update_notes(match_id: str, body: NotesUpdate):
    """Update context notes for a match."""
    conn = _db()
    if not get_match(conn, match_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")
    update_context_notes(conn, match_id, body.notes)
    conn.close()
    return {"match_id": match_id, "notes": body.notes}

@router.post("/api/matches/{match_id}/tags")
def create_tag(match_id: str, body: TagCreate):
    """Add a tag to a match."""
    conn = _db()
    if not get_match(conn, match_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")
    add_tag(conn, match_id, body.tag)
    conn.close()
    return {"match_id": match_id, "tag": body.tag}

@router.delete("/api/matches/{match_id}")
def remove_match(match_id: str):
    """Delete a match and all associated data."""
    conn = _db()
    if not get_match(conn, match_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")
    delete_match(conn, match_id)
    conn.close()
    return {"deleted": match_id}

def _replace_match_from_demo(
    conn,
    existing: dict[str, Any],
    dem_path: str,
    sid: str,
    *,
    filename: str,
    match_date: str | None,
) -> tuple[str, dict[str, Any]]:
    """Re-parse *dem_path* and swap it in for an existing match row.

    Tags, context notes, the match date and the AI chat history all carry over;
    the old row is deleted only once the new one is safely written.  Shared by
    /reimport (which parses an uploaded file) and /reanalyze (which parses one
    already on disk).

    The chat history has to be re-pointed *before* the delete, because
    delete_match drops a match's ai_chats along with it — without this a bulk
    re-analysis would quietly wipe every conversation the user had about their
    matches.
    """
    old_match_id = existing["match_id"]
    old_tags = get_tags(conn, old_match_id)
    old_notes = existing.get("context_notes", "") or ""

    stats = analyse_demo(dem_path, sid)

    new_match_id = save_match(
        conn,
        stats,
        filename=filename,
        steam_id=sid,
        context_notes=old_notes,
        match_date=match_date,
    )
    for tag in old_tags:
        add_tag(conn, new_match_id, tag)
    move_chat_history(conn, old_match_id, new_match_id)
    delete_match(conn, old_match_id)
    return new_match_id, stats

@router.post("/api/matches/{match_id}/reanalyze")
def reanalyze_match(match_id: str):
    """Re-run the current analyzer over a match, reading the demo from disk.

    Unlike /reimport this takes no upload: the demo is found by the filename
    recorded on the match.  Returns 409 when the file is gone, which is the
    normal outcome for matches that were originally drag-and-dropped.
    """
    conn = _db()
    existing = get_match(conn, match_id)
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")

    filename = str(existing.get("filename") or "")
    dem_path = _resolve_demo(filename)
    if dem_path is None:
        conn.close()
        raise HTTPException(
            status_code=409,
            detail=(
                f"No demo on disk for '{filename or 'this match'}'. Uploaded "
                "demos are not kept, so this match can only be refreshed by "
                "re-importing the file."
            ),
        )

    sid = str(existing.get("player_steam_id") or "")
    if not sid:
        conn.close()
        raise HTTPException(status_code=400, detail="Match has no player Steam ID")

    try:
        new_match_id, stats = _replace_match_from_demo(
            conn, existing, str(dem_path), sid,
            filename=filename, match_date=existing.get("date"),
        )
    except Exception as exc:
        conn.close()
        logging.getLogger("uvicorn.error").exception("Re-analysis failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse demo: {type(exc).__name__}: {exc}",
        )
    conn.close()

    response: dict[str, Any] = {
        "reanalyzed_from": match_id,
        "match_id": new_match_id,
        "analyzer_version": stats.get("analyzer_version"),
        "stats": _match_summary(stats),
    }
    if stats.get("partial_import"):
        response["partial_import"] = True
    if stats.get("parse_warning"):
        response["parse_warning"] = stats["parse_warning"]
    return response

@router.post("/api/matches/{match_id}/reimport")
def reimport_match(
    match_id: str,
    file: UploadFile = File(...),
    info_file: UploadFile | None = File(default=None),
    steam_id: str = Form(default=""),
):
    """Replace an existing match by parsing a newly provided .dem file."""
    if not file.filename or not file.filename.endswith(".dem"):
        raise HTTPException(status_code=400, detail="File must be a .dem demo file")

    conn = _db()
    existing = get_match(conn, match_id)
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")

    sid = steam_id.strip() or str(existing.get("player_steam_id") or "")
    if not sid:
        conn.close()
        raise HTTPException(status_code=400, detail="Steam ID is required")

    info_date: str | None = None
    if info_file and info_file.filename:
        try:
            info_data = parse_info_file(info_file.file.read())
            info_date = info_data.get("match_date")
        except Exception:
            pass
    resolved_date = info_date or existing.get("date")

    with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        new_match_id, stats = _replace_match_from_demo(
            conn, existing, tmp_path, sid,
            filename=file.filename or str(existing.get("filename") or "reimport.dem"),
            match_date=resolved_date,
        )
    except Exception as exc:
        conn.close()
        logging.getLogger("uvicorn.error").exception("Demo parse failed: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse demo: {type(exc).__name__}: {exc}",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    conn.close()

    response: dict[str, Any] = {
        "reimported_from": match_id,
        "match_id": new_match_id,
        "stats": _match_summary(stats),
    }
    if stats.get("partial_import"):
        response["partial_import"] = True
    if stats.get("parse_warning"):
        response["parse_warning"] = stats["parse_warning"]
    return response

# ---------------------------------------------------------------------------
# Match Chat
# ---------------------------------------------------------------------------
@router.get("/api/matches/{match_id}/chat")
def get_match_chat(match_id: str):
    """Return chat history for a match."""
    conn = _db()
    if not get_match(conn, match_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")
    history = get_chat_history(conn, match_id)
    conn.close()
    return {"match_id": match_id, "messages": history}

@router.post("/api/matches/{match_id}/chat")
async def send_match_chat(match_id: str, body: ChatMessageIn):
    """Send a message, get AI response, persist both."""
    conn = _db()
    match = get_match(conn, match_id)
    if not match:
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")

    # Resolve provider/model
    ai_config = load_ai_config()
    provider = body.provider or ai_config.get("active_provider", "")
    model = body.model or ai_config.get("active_model", "")
    if not provider or not model:
        conn.close()
        raise HTTPException(status_code=400, detail="No AI provider/model configured")

    prov_config = ai_config.get("providers", {}).get(provider, {})
    api_key = prov_config.get("api_key", "")
    if not api_key:
        conn.close()
        raise HTTPException(status_code=400, detail=f"No API key set for {provider}")

    # Build context
    players = get_match_players(conn, match_id)
    rounds = get_round_stats(conn, match_id)
    system_prompt = build_match_context(
        match, players, rounds,
        custom_instructions=ai_config.get("system_instructions", ""),
    )

    # Load existing history
    history = get_chat_history(conn, match_id)
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": body.message})

    # Call AI
    try:
        response = await chat_completion(provider, model, api_key, messages, system_prompt)
    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=502, detail=f"AI request failed: {exc}")

    # Persist both messages
    save_chat_message(conn, match_id, "user", body.message, provider, model)
    save_chat_message(conn, match_id, "assistant", response, provider, model)
    conn.close()

    return {"role": "assistant", "content": response, "provider": provider, "model": model}

@router.delete("/api/matches/{match_id}/chat")
def clear_match_chat(match_id: str):
    """Clear all chat history for a match."""
    conn = _db()
    if not get_match(conn, match_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")
    clear_chat_history(conn, match_id)
    conn.close()
    return {"match_id": match_id, "cleared": True}

# ---------------------------------------------------------------------------
# Minimap
# ---------------------------------------------------------------------------
@router.get("/api/matches/{match_id}/minimap")
def get_minimap_data(match_id: str, round_number: int = 0):
    """Return position data for minimap rendering.

    If *round_number* is 0, returns all rounds. Otherwise, returns data
    for the specified round only.
    """

    conn = _db()
    match = get_match(conn, match_id)
    if not match:
        conn.close()
        raise HTTPException(status_code=404, detail="Match not found")

    map_name = match.get("map_name", "")
    radar = get_radar_config(map_name)
    if not radar:
        conn.close()
        raise HTTPException(status_code=400, detail=f"No radar data for {map_name}")

    rounds = get_round_stats(conn, match_id)
    conn.close()

    result_rounds = []
    for r in rounds:
        rn = r.get("round_number", 0)
        if round_number and rn != round_number:
            continue

        enriched = decode_dict(r.get("enriched_json"))

        events = []
        # Kill events (attacker + victim positions)
        for k in enriched.get("kills_detail", []):
            if k.get("attacker_xy"):
                px = game_to_pixel(map_name, k["attacker_xy"][0], k["attacker_xy"][1])
                if px:
                    events.append({
                        "type": "kill",
                        "role": "attacker",
                        "px": px[0], "py": px[1],
                        "victim": k.get("victim", "?"),
                        "weapon": k.get("weapon", "?"),
                        "headshot": k.get("headshot", False),
                    })
            if k.get("victim_xy"):
                px = game_to_pixel(map_name, k["victim_xy"][0], k["victim_xy"][1])
                if px:
                    events.append({
                        "type": "death",
                        "role": "victim_of_kill",
                        "px": px[0], "py": px[1],
                        "name": k.get("victim", "?"),
                    })

        # Player death (user died)
        dd = enriched.get("death_detail")
        if dd and dd.get("victim_xy"):
            px = game_to_pixel(map_name, dd["victim_xy"][0], dd["victim_xy"][1])
            if px:
                events.append({
                    "type": "player_death",
                    "role": "user_died",
                    "px": px[0], "py": px[1],
                    "killer": dd.get("killer", "?"),
                    "weapon": dd.get("weapon", "?"),
                })
        if dd and dd.get("killer_xy"):
            px = game_to_pixel(map_name, dd["killer_xy"][0], dd["killer_xy"][1])
            if px:
                events.append({
                    "type": "killer_pos",
                    "role": "killed_user",
                    "px": px[0], "py": px[1],
                    "name": dd.get("killer", "?"),
                })

        # Grenade events (throw → land positions)
        for g in enriched.get("utility", {}).get("grenades", []):
            throw_px = None
            land_px = None
            if g.get("throw_xy"):
                throw_px = game_to_pixel(map_name, g["throw_xy"][0], g["throw_xy"][1])
            if g.get("land_xy"):
                land_px = game_to_pixel(map_name, g["land_xy"][0], g["land_xy"][1])
            if throw_px or land_px:
                ev = {
                    "type": "grenade",
                    "nade_type": g.get("type", "?"),
                    "throw_callout": g.get("throw_callout", ""),
                    "land_callout": g.get("land_callout", ""),
                }
                if throw_px:
                    ev["throw_px"] = throw_px[0]
                    ev["throw_py"] = throw_px[1]
                if land_px:
                    ev["land_px"] = land_px[0]
                    ev["land_py"] = land_px[1]
                events.append(ev)

        # Flash victim positions
        for fv in enriched.get("utility", {}).get("flash_instances", []):
            if fv.get("victim_xy") and not fv.get("is_friendly", False):
                px = game_to_pixel(map_name, fv["victim_xy"][0], fv["victim_xy"][1])
                if px:
                    events.append({
                        "type": "flash_victim",
                        "px": px[0], "py": px[1],
                        "name": fv.get("name", "?"),
                        "duration": fv.get("duration", 0),
                    })

        # HE victim positions
        for hv in enriched.get("utility", {}).get("he_victims", []):
            if hv.get("victim_xy"):
                px = game_to_pixel(map_name, hv["victim_xy"][0], hv["victim_xy"][1])
                if px:
                    events.append({
                        "type": "he_victim",
                        "px": px[0], "py": px[1],
                        "name": hv.get("name", "?"),
                        "damage": hv.get("damage", 0),
                    })

        # Molotov/incendiary victim positions
        for mv in enriched.get("utility", {}).get("molotov_damage", []):
            if mv.get("victim_xy"):
                px = game_to_pixel(map_name, mv["victim_xy"][0], mv["victim_xy"][1])
                if px:
                    events.append({
                        "type": "molotov_victim",
                        "px": px[0], "py": px[1],
                        "name": mv.get("victim", "?"),
                        "damage": mv.get("damage", 0),
                    })

        result_rounds.append({
            "round": rn,
            "side": enriched.get("side", "?"),
            "won": enriched.get("side") == enriched.get("round_winner"),
            "events": events,
        })

    return {
        "map_name": map_name,
        "radar": radar,
        "radar_image": f"/frontend/img/radar/{map_name.removeprefix('de_').removeprefix('cs_')}.png?v={int(__import__('time').time())}",
        "rounds": result_rounds,
    }
