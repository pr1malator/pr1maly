"""
AI Service — Unified interface for chat completions across providers.

Supports OpenAI, Anthropic, and Google Gemini.
Configuration stored in data/ai_config.json (gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

_CONFIG_PATH = Path(__file__).parent.parent / "data" / "ai_config.json"

_DEFAULT_CONFIG: dict[str, Any] = {
    "providers": {},
    "active_provider": "",
    "active_model": "",
    "system_instructions": "",
    "prompts": [
        {
            "name": "Match Overview",
            "prompt": "Give me a detailed analysis of my performance. Focus on economy decisions, opening duels, and the key rounds that decided the match.",
        },
        {
            "name": "Round-by-Round",
            "prompt": "Walk me through my worst rounds. What went wrong — bad buys, losing opening duels, poor utility? Be specific.",
        },
        {
            "name": "Economy & Buys",
            "prompt": "Analyze my economy decisions across the match. Did I force at the wrong times? Did I eco when I should have forced? Were my buys optimal for my side?",
        },
        {
            "name": "Opening Duels",
            "prompt": "Analyze my opening duel performance. When I got opening kills, what happened? When I was the opening death, what can I learn? Am I taking too many risks or not enough?",
        },
        {
            "name": "Clutch & Impact",
            "prompt": "Analyze my impact rounds — multi-kills, clutches, and high-damage rounds. Where did I make a difference and where did I fail to convert advantages?",
        },
    ],
}

# Available providers with their known models
PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "models": [
            "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
            "o4-mini", "o3",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "gpt-4o", "gpt-4o-mini",
        ],
        "default_model": "gpt-5.4-mini",
    },
    "anthropic": {
        "label": "Anthropic",
        "models": [
            "claude-opus-4.6", "claude-opus-4-20250514",
            "claude-sonnet-4.6", "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251015",
            "claude-3-7-sonnet-20250219",
        ],
        "default_model": "claude-sonnet-4.6",
    },
    "google": {
        "label": "Google Gemini",
        "models": [
            "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        ],
        "default_model": "gemini-2.5-flash",
    },
    "mistral": {
        "label": "Mistral AI",
        "models": [
            "mistral-large-2512", "mistral-medium-2508", "mistral-small-2603",
            "codestral-2501",
        ],
        "default_model": "mistral-small-2603",
    },
}


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return json.loads(json.dumps(_DEFAULT_CONFIG))


def save_config(config: dict[str, Any]) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def mask_key(key: str) -> str:
    if not key or len(key) <= 8:
        return "****" if key else ""
    return key[:4] + "****" + key[-4:]


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_match_context(
    match: dict[str, Any],
    players: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    custom_instructions: str = "",
) -> str:
    """Build a system prompt from match data for AI context.

    Uses enriched round data (stored as JSON) to build detailed narratives
    the AI can reason over — economy, kill/death details, opening duels,
    utility effectiveness, bomb events, and clutch situations.
    """
    lines = [
        "You are PULSE_AI, an expert CS2 coach and analyst. You have DETAILED round-by-round "
        "data from a competitive match including economy, weapon kills, opening duels, utility "
        "usage, bomb events, clutch situations, and EXACT MAP POSITIONS (callouts) for every "
        "kill and death.\n\n"
        "CRITICAL RULES:\n"
        "- ALWAYS reference the exact callout positions provided in the data (e.g. 'you died at "
        "B Apartments from Window' not just 'you died in a vulnerable position').\n"
        "- When analyzing positioning, explain WHY a position was good or bad using CS2 map "
        "knowledge (sight lines, common angles, rotation paths).\n"
        "- For trading analysis, check if a teammate got a kill within ~5 seconds of the player's "
        "death (TRADED tag). If the player died without being traded ([DIED] not [DIED - TRADED]), "
        "explain what positioning would have allowed a trade.\n"
        "- Reference specific callout names from the data — never say generic things like "
        "'reconsider your positioning'. Say WHERE they should position instead.\n"
        "- Identify repeated death positions across rounds (e.g. dying at the same spot multiple "
        "rounds = predictable pattern the enemy is reading).\n"
        "- Don't just read stats back — identify patterns, mistakes, and opportunities. "
        "Be direct, specific, and reference the EXACT round numbers and positions.",
    ]

    if custom_instructions:
        lines.extend(["", f"Additional instructions: {custom_instructions}"])

    lines.extend([
        "",
        "=== MATCH OVERVIEW ===",
        f"Map: {match.get('map_name', 'unknown')}",
        f"Score: {match.get('team_score', 0)}-{match.get('enemy_score', 0)} ({match.get('match_result', 'unknown')})",
        f"Total Rounds: {match.get('total_rounds', 0)}",
        "",
        f"=== PLAYER STATS ({match.get('player_name', 'Unknown')}) ===",
        f"HLTV Rating: {float(match.get('hltv_rating') or 0):.2f}",
        f"ADR: {float(match.get('adr') or 0):.1f}",
        f"KAST: {float(match.get('kast') or 0):.1f}%",
        f"K/D/A: {match.get('kills', 0)}/{match.get('deaths', 0)}/{match.get('assists', 0)}",
        f"K/D Ratio: {float(match.get('kd_ratio') or 0):.2f}",
        f"Impact: {float(match.get('impact') or 0):.2f}",
        f"Multi-kills: 2K={match.get('rounds_2k', 0)} 3K={match.get('rounds_3k', 0)} "
        f"4K={match.get('rounds_4k', 0)} 5K={match.get('rounds_5k', 0)}",
    ])

    # --- Enriched round-by-round narratives ---
    if rounds:
        lines.extend(["", "=== ROUND-BY-ROUND DETAIL ==="])
        for r in rounds:
            lines.append(_format_round_narrative(r))

    # --- Team scoreboard ---
    if players:
        user_team = None
        for p in players:
            if p.get("is_user"):
                user_team = p.get("team")
                break

        my_team = [p for p in players if p.get("team") == user_team]
        enemy_team = [p for p in players if p.get("team") != user_team]

        lines.extend(["", "=== MY TEAM ==="])
        for p in my_team:
            marker = " (YOU)" if p.get("is_user") else ""
            lines.append(
                f"  {p.get('name', '?')}{marker}: "
                f"{p.get('kills', 0)}/{p.get('deaths', 0)}/{p.get('assists', 0)} "
                f"Rating:{float(p.get('hltv_rating') or 0):.2f} "
                f"ADR:{float(p.get('adr') or 0):.1f} "
                f"KAST:{float(p.get('kast') or 0):.1f}%"
            )

        lines.extend(["", "=== ENEMY TEAM ==="])
        for p in enemy_team:
            lines.append(
                f"  {p.get('name', '?')}: "
                f"{p.get('kills', 0)}/{p.get('deaths', 0)}/{p.get('assists', 0)} "
                f"Rating:{float(p.get('hltv_rating') or 0):.2f} "
                f"ADR:{float(p.get('adr') or 0):.1f} "
                f"KAST:{float(p.get('kast') or 0):.1f}%"
            )

    return "\n".join(lines)


def _format_round_narrative(r: dict[str, Any]) -> str:
    """Format a single round into a rich text narrative for the AI."""
    rnum = r.get("round_number", "?")
    kills = r.get("kills", 0)
    deaths = r.get("deaths", 0)
    assists = r.get("assists", 0)
    damage = r.get("damage", 0)
    survived = r.get("survived")
    traded = r.get("traded")

    # Parse enriched data
    enriched_str = r.get("enriched_json", "")
    enriched: dict[str, Any] = {}
    if enriched_str:
        try:
            enriched = json.loads(enriched_str)
        except (json.JSONDecodeError, TypeError):
            pass

    # Build narrative parts
    parts: list[str] = []

    # Side + Economy
    side = enriched.get("side", "?")
    econ = enriched.get("economy", {})
    buy_type = econ.get("buy_type", "?")
    spend = econ.get("player_spend", 0)
    items = econ.get("items", [])

    header = f"  Round {rnum} [{side}] [{buy_type} ${spend}]:"
    parts.append(f"{kills}K {deaths}D {assists}A {damage}dmg")

    # Survival status
    if survived:
        parts.append("[SURVIVED]")
    elif traded:
        parts.append("[DIED - TRADED]")
    else:
        parts.append("[DIED]")

    # Opening duel
    opening = enriched.get("opening_duel")
    if opening:
        role = opening.get("role", "")
        opp = opening.get("opponent", "?")
        wpn = opening.get("weapon", "?")
        if role == "opening_kill":
            parts.append(f"OPENING KILL on {opp} ({wpn})")
        elif role == "opening_death":
            parts.append(f"OPENING DEATH by {opp} ({wpn})")

    line = f"{header} {' | '.join(parts)}"

    # Kill details
    kills_detail = enriched.get("kills_detail", [])
    for k in kills_detail:
        victim = k.get("victim", "?")
        weapon = k.get("weapon", "?")
        hs = " HS" if k.get("headshot") else ""
        dist = f" {k['distance']}m" if k.get("distance") else ""
        specials = f" [{', '.join(k['specials'])}]" if k.get("specials") else ""
        # Position info
        pos_parts = []
        if k.get("victim_position") and k["victim_position"] != "unknown":
            pos_parts.append(f"at {k['victim_position']}")
        if k.get("attacker_position") and k["attacker_position"] != "unknown":
            pos_parts.append(f"from {k['attacker_position']}")
        pos_str = f" ({' '.join(pos_parts)})" if pos_parts else ""
        line += f"\n    → Killed {victim}{pos_str} with {weapon}{hs}{dist}{specials}"

    # Death detail
    death = enriched.get("death_detail")
    if death:
        killer = death.get("killer", "?")
        weapon = death.get("weapon", "?")
        hs = " HS" if death.get("headshot") else ""
        dist = f" {death['distance']}m" if death.get("distance") else ""
        # Position info
        pos_parts = []
        if death.get("victim_position") and death["victim_position"] != "unknown":
            pos_parts.append(f"at {death['victim_position']}")
        if death.get("killer_position") and death["killer_position"] != "unknown":
            pos_parts.append(f"from {death['killer_position']}")
        pos_str = f" ({' '.join(pos_parts)})" if pos_parts else ""
        line += f"\n    ✗ Killed by {killer}{pos_str} with {weapon}{hs}{dist}"

    # Utility
    util = enriched.get("utility", {})
    util_parts = []
    flash_victims = util.get("flash_victims", [])
    if flash_victims:
        victim_strs = [f"{v['name']} {v['duration']}s" for v in flash_victims]
        util_parts.append(f"Flashed: {', '.join(victim_strs)}")
    elif util.get("enemies_flashed"):
        dur = util.get("avg_blind_duration", 0)
        util_parts.append(f"Flashed {util['enemies_flashed']} (avg {dur}s)")
    if util.get("flash_assists"):
        util_parts.append(f"Flash assist ×{util['flash_assists']}")
    if util.get("he_damage"):
        util_parts.append(f"HE dmg: {util['he_damage']}")
    molly_dmg = util.get("molotov_damage", [])
    if molly_dmg:
        molly_strs = [f"{m['victim']} {m['damage']}hp" for m in molly_dmg]
        util_parts.append(f"Molotov: {', '.join(molly_strs)}")
    if util_parts:
        line += f"\n    ⚡ Utility: {' | '.join(util_parts)}"

    # Teamplayer incidents (team damage & team flashes)
    tp = enriched.get("teamplayer", {})
    tp_parts: list[str] = []
    team_dmg = tp.get("team_damage", [])
    if team_dmg:
        dmg_strs = [f"{d['victim']} {d['damage']}hp ({d['weapon']})" for d in team_dmg]
        tp_parts.append(f"Team damage: {', '.join(dmg_strs)}")
    team_fl = tp.get("team_flashes", [])
    if team_fl:
        fl_strs = [f"{f['victim']} {f['duration']}s" for f in team_fl]
        tp_parts.append(f"Team flashed: {', '.join(fl_strs)}")
    if tp_parts:
        line += f"\n    ⚠ Teamplay: {' | '.join(tp_parts)}"

    # Bomb
    bomb = enriched.get("bomb")
    if bomb:
        if bomb.get("planted"):
            line += f"\n    💣 Planted bomb ({bomb['planted']})"
        if bomb.get("defused"):
            line += "\n    💣 Defused bomb"

    # Clutch
    clutch = enriched.get("clutch")
    if clutch:
        vs = clutch.get("vs", "?")
        won = "WON" if clutch.get("won") else "LOST"
        line += f"\n    🏆 1v{vs} CLUTCH — {won}"

    # Equipment purchased
    if items:
        line += f"\n    🛒 Bought: {', '.join(items)}"

    return line


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

async def chat_completion(
    provider: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    system_prompt: str,
) -> str:
    """Send messages to an AI provider and return the response text."""
    if provider == "openai":
        return await _openai_chat(api_key, model, system_prompt, messages)
    elif provider == "anthropic":
        return await _anthropic_chat(api_key, model, system_prompt, messages)
    elif provider == "google":
        return await _google_chat(api_key, model, system_prompt, messages)
    elif provider == "mistral":
        return await _mistral_chat(api_key, model, system_prompt, messages)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def _openai_chat(
    api_key: str, model: str, system_prompt: str, messages: list[dict[str, str]]
) -> str:
    all_messages = [{"role": "system", "content": system_prompt}]
    all_messages.extend(messages)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": all_messages, "temperature": 0.7},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def _anthropic_chat(
    api_key: str, model: str, system_prompt: str, messages: list[dict[str, str]]
) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


async def _google_chat(
    api_key: str, model: str, system_prompt: str, messages: list[dict[str, str]]
) -> str:
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": contents,
                "systemInstruction": {"parts": [{"text": system_prompt}]},
            },
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


async def _mistral_chat(
    api_key: str, model: str, system_prompt: str, messages: list[dict[str, str]]
) -> str:
    all_messages = [{"role": "system", "content": system_prompt}]
    all_messages.extend(messages)
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": all_messages, "temperature": 0.7},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
