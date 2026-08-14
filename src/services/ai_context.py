"""Turning stored match data into the text an LLM is asked to assess.

These builders lived in api.py, roughly five hundred lines of string assembly
sitting between two route handlers. None of them touches HTTP or the database:
they take matches and rounds that have already been loaded and return a prompt.

The guiding rule, which is why the functions are long: every figure carries the
sample it came from. A tendency claimed off three engagements is not a tendency,
and the model has no way to know which numbers are thin unless the counts travel
with them.

Transport and provider configuration stay in src/ai_service.py. This module only
builds text.
"""

from __future__ import annotations

import json

from src.ai_service import format_round_narrative


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def _pct(part: int, whole: int) -> float | None:
    return round(part / whole * 100, 1) if whole else None


def build_patterns_context(
    map_name: str, matches: list[dict], rounds: list[dict],
) -> str:
    """Summarise how the player performs on *map_name* for the model.

    Every figure carries the sample it came from.  A tendency claimed off three
    engagements is not a tendency, and the model has no way to know which
    numbers are thin unless the counts travel with them.
    """
    lines: list[str] = [
        f"Map: {map_name}",
        f"Matches: {len(matches)}  |  Rounds: {len(rounds)}",
    ]

    # --- Aim, pooled across the map's matches ---
    aim_blocks: list[dict] = []
    for m in matches:
        raw = m.get("aim_stats")
        if not raw:
            continue
        try:
            aim_blocks.append(json.loads(raw) if isinstance(raw, str) else raw)
        except Exception:
            continue

    def _headline(block: str, field: str = "median") -> list[float]:
        out = []
        for a in aim_blocks:
            v = (a.get(block) or {}).get(field)
            if v is not None:
                out.append(v)
        return out

    def _sample(block: str) -> int:
        return sum((a.get(block) or {}).get("n", 0) for a in aim_blocks)

    lines.append("")
    lines.append("=== AIM (per-match medians, averaged; n = engagements measured) ===")
    if aim_blocks:
        ratings = [a["aim_rating"] for a in aim_blocks if a.get("aim_rating") is not None]
        lines.append(f"Aim rating: {_mean(ratings)}/100 over {len(ratings)} matches")
        lines.append(
            f"Shot speed: {_mean(_headline('movement'))} u/s "
            f"(accurate fire needs <85; n={_sample('movement')})"
        )
        lines.append(
            f"Crosshair placement: {_mean(_headline('preaim'))}° off target "
            f"(n={_sample('preaim')})"
        )
        lines.append(
            f"Engagement time: {_mean(_headline('ttk'))}s first shot to kill "
            f"(n={_sample('ttk')})"
        )
        lines.append(
            f"Reaction: {_mean(_headline('reaction'))} ms (diagnostic only, small samples)"
        )
        acc = [a.get("accuracy", {}).get("pooled_pct") for a in aim_blocks]
        acc = [v for v in acc if v is not None]
        if acc:
            lines.append(f"Accuracy: {_mean(acc)}% of bullets hit")

        # Counter-strafe and peek speed: the movement technique detail.
        cs_attempts = sum((a.get("movement") or {}).get("counterstrafe_attempts", 0) for a in aim_blocks)
        cs_good = sum((a.get("movement") or {}).get("counterstrafe_good", 0) for a in aim_blocks)
        if cs_attempts:
            lines.append(
                f"Counter-strafe: {_pct(cs_good, cs_attempts)}% of rifle stops done properly "
                f"({cs_good}/{cs_attempts}); the rest coasted to a halt"
            )
        # Pooled per peek-speed band, which says whether the stop survives speed.
        band_totals: dict[str, list[int]] = {}
        for a in aim_blocks:
            for b in (a.get("movement") or {}).get("counterstrafe_by_peek", []) or []:
                slot = band_totals.setdefault(b["label"], [0, 0])
                slot[0] += b.get("attempts", 0)
                slot[1] += b.get("good", 0)
        for label, (att, good) in band_totals.items():
            if att:
                lines.append(f"  - {label} peeks: {_pct(good, att)}% stopped properly ({good}/{att})")
        peek_zone: dict[str, int] = {}
        peek_n = 0
        for a in aim_blocks:
            for z in (a.get("peek") or {}).get("by_zone", []) or []:
                peek_zone[z["label"]] = peek_zone.get(z["label"], 0) + z.get("n", 0)
                peek_n += z.get("n", 0)
        if peek_n:
            spread = ", ".join(f"{k} {_pct(v, peek_n)}%" for k, v in peek_zone.items())
            lines.append(f"Speed carried into duels: {spread} (n={peek_n})")
    else:
        lines.append("No aim data stored for this map.")

    # --- Utility ---
    lines.append("")
    lines.append("=== UTILITY ===")
    util_blocks: list[dict] = []
    for m in matches:
        raw = m.get("utility_data")
        if not raw:
            continue
        try:
            util_blocks.append(json.loads(raw) if isinstance(raw, str) else raw)
        except Exception:
            continue
    if util_blocks:
        ratings = [u["utility_rating"] for u in util_blocks if u.get("utility_rating") is not None]
        if ratings:
            lines.append(f"Utility rating: {_mean(ratings)}/100")
        spent = sum((u.get("economics") or {}).get("total_spent", 0) for u in util_blocks)
        wasted = sum((u.get("economics") or {}).get("total_wasted", 0) for u in util_blocks)
        if spent:
            lines.append(
                f"Grenade spend: ${spent} bought, ${wasted} never thrown ({_pct(wasted, spent)}% wasted)"
            )
        flashes_thrown = sum((u.get("flash") or {}).get("thrown", 0) for u in util_blocks)
        flashed = sum((u.get("flash") or {}).get("enemies_flashed", 0) for u in util_blocks)
        blind = [(u.get("flash") or {}).get("avg_enemy_blind_duration") for u in util_blocks]
        blind = [b for b in blind if b]
        lines.append(
            f"Flashes: {flashes_thrown} thrown, {flashed} enemies flashed, "
            f"avg blind {_mean(blind)}s"
        )
        # Counting heads treats a 0.3s glance the same as a 4s blind, so the
        # share that actually bought time is the figure worth reasoning over.
        eff = [(u.get("flash") or {}).get("effective_flash_pct") for u in util_blocks]
        eff = [e for e in eff if e is not None]
        if eff:
            lines.append(f"  Flashes blinding an enemy over 1s: {_mean(eff)}%")
        team_flashed = sum((u.get("flash") or {}).get("team_flashed", 0) for u in util_blocks)
        self_flashed = sum((u.get("flash") or {}).get("self_flashed", 0) for u in util_blocks)
        if team_flashed or self_flashed:
            lines.append(f"  Teammates flashed: {team_flashed}; self-flashed: {self_flashed}")
        he = sum((u.get("he") or {}).get("total_damage", 0) for u in util_blocks)
        molly = sum((u.get("molotov") or {}).get("total_damage", 0) for u in util_blocks)
        lines.append(f"Utility damage: {he} HE, {molly} molotov/incendiary")
    else:
        lines.append("No utility data stored for this map.")

    # --- Round behaviour, split by side ---
    lines.append("")
    lines.append("=== ROUND BEHAVIOUR (by side) ===")
    for side in ("CT", "T"):
        side_rounds = [r for r in rounds if (r.get("enriched") or {}).get("side") == side]
        if not side_rounds:
            continue
        n = len(side_rounds)
        wins = sum(
            1 for r in side_rounds
            if (r.get("enriched") or {}).get("round_winner") == side
        )
        opens = [(r.get("enriched") or {}).get("opening_duel") for r in side_rounds]
        ok = sum(1 for o in opens if o and o.get("role") == "opening_kill")
        od = sum(1 for o in opens if o and o.get("role") == "opening_death")
        survived = sum(1 for r in side_rounds if r.get("survived"))
        traded = sum(1 for r in side_rounds if r.get("traded"))
        deaths = sum(1 for r in side_rounds if (r.get("deaths") or 0) > 0)
        buys: dict[str, int] = {}
        for r in side_rounds:
            bt = ((r.get("enriched") or {}).get("economy") or {}).get("buy_type", "?")
            buys[bt] = buys.get(bt, 0) + 1
        clutches = [(r.get("enriched") or {}).get("clutch") for r in side_rounds]
        clutch_n = sum(1 for c in clutches if c)
        clutch_won = sum(1 for c in clutches if c and c.get("won"))

        lines.append(f"{side}: {n} rounds, {_pct(wins, n)}% won")
        lines.append(
            f"  Opening duels: {ok} won, {od} lost "
            f"({_pct(ok + od, n)}% of rounds involved one)"
        )
        lines.append(f"  Survived {_pct(survived, n)}% of rounds; traded on {_pct(traded, deaths)}% of deaths")
        lines.append("  Buys: " + ", ".join(f"{k} {v}" for k, v in sorted(buys.items())))
        if clutch_n:
            lines.append(f"  Clutches: {clutch_won}/{clutch_n} won")

    # --- Utility discipline per round: bought versus actually thrown ---
    bought = 0
    thrown = 0
    for r in rounds:
        e = r.get("enriched") or {}
        items = (e.get("economy") or {}).get("items", []) or []
        bought += sum(
            1 for i in items
            if any(t in str(i).lower() for t in ("grenade", "flashbang", "molotov"))
        )
        thrown += len((e.get("utility") or {}).get("grenades", []) or [])
    if bought:
        lines.append("")
        lines.append(
            f"Grenades bought {bought}, thrown {thrown} "
            f"({_pct(thrown, bought)}% — carrying across rounds makes this approximate)"
        )

    return "\n".join(lines)


def strip_json_fences(text: str) -> str:
    """Unwrap a ```json ... ``` block, if the model wrapped its answer in one.

    Asking for raw JSON does not stop every provider from fencing it anyway,
    and a fenced answer is a correct answer badly packaged — worth unwrapping
    rather than dropping into the prose fallback.
    """
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    parts = text.split("\n", 1)
    if len(parts) < 2:
        return text
    return parts[1].rsplit("```", 1)[0].strip()


def build_role_context(map_name: str, ct_rounds: list, t_rounds: list) -> str:
    """Build a concise prompt with round-by-round positional data for AI role assessment."""
    lines = [
        f"Map: {map_name}",
        f"CT rounds: {len(ct_rounds)}  |  T rounds: {len(t_rounds)}",
        "",
        "=== CT SIDE ROUNDS ===",
    ]
    for r in ct_rounds:
        lines.append(format_round_narrative(r))
    lines.extend(["", "=== T SIDE ROUNDS ==="])
    for r in t_rounds:
        lines.append(format_round_narrative(r))
    return "\n".join(lines)


def matches_with_recorded_play(matches: list[dict]) -> list[dict]:
    """Drop matches the player does not actually appear in.

    A demo belonging to someone else still imports, and lands with zero kills,
    zero deaths and zero damage across a full set of rounds.  Averaged in, one
    of those drags a map's ADR down by a third and invites the assessment to
    call it a weak map on the strength of a match nobody played — which is
    exactly what happened the first time this ran against real data.

    Anything with a single kill, death or point of damage is kept: a genuinely
    terrible match is still a match, and only the total absence of the player
    is evidence they were not there.
    """
    return [
        m for m in matches
        if (m.get("kills") or 0) or (m.get("deaths") or 0) or (m.get("adr") or 0)
    ]


ASSESSMENT_SYSTEM_PROMPT = (
    "You are PULSE_AI, an expert CS2 analyst. You are given everything known about one "
    "player on one map: round-by-round positional narratives, and aggregated measurements "
    "of their aim, utility and round behaviour with the sample size behind each figure.\n\n"
    "Produce one assessment covering two things:\n"
    "1. ROLES — the position they play on each side, from where they fight and die.\n"
    "2. PATTERNS — the habits and tendencies the measurements describe.\n\n"
    "IMPORTANT:\n"
    "- Every claim must trace to something in the data. Do not invent statistics.\n"
    "- Respect sample sizes. Say a pattern is tentative when n is small, and do not build "
    "a tendency out of fewer than ~10 measurements.\n"
    "- Roles must be map-specific and concrete, e.g. 'Pit Anchor', 'Banana Entry', "
    "'Lurk Apartments' — not 'Rifler'.\n"
    "- Prefer the specific over the generic: 'stops cleanly off a walk but coasts on "
    "full-speed peeks' beats 'movement could improve'.\n"
    "- Thresholds worth knowing: rifles are inaccurate above 85 u/s; crosshair placement "
    "under 10 degrees is good and over 20 is poor; wasting over ~20% of grenade spend is high.\n"
    "- A tendency is not automatically a flaw. Name strengths where the numbers show them.\n"
    "- Connect the two halves where the data supports it: a role explains why a tendency "
    "shows up, and a measurement can confirm or contradict the role.\n\n"
    "Respond in EXACTLY this JSON format (no markdown, no extra text):\n"
    "{\n"
    '  "ct_role": {\n'
    '    "name": "<short role name, 2-4 words, map-specific e.g. Pit Anchor, AWP Mid>",\n'
    '    "icon": "<one of: shield, bolt, visibility, anchor, shield_with_heart, sync_alt, precision_manufacturing, swords>",\n'
    '    "description": "<2-3 sentences explaining the role from the actual positions played>"\n'
    "  },\n"
    '  "t_role": {\n'
    '    "name": "<short role name, 2-4 words, map-specific e.g. Banana Entry, Lurk Apartments>",\n'
    '    "icon": "<one of: shield, bolt, visibility, anchor, shield_with_heart, sync_alt, precision_manufacturing, swords>",\n'
    '    "description": "<2-3 sentences explaining the role from the actual positions played>"\n'
    "  },\n"
    '  "aim": {\n'
    '    "name": "<short pattern name, 2-5 words, e.g. Patient But Slow To Stop>",\n'
    '    "icon": "<one of: my_location, target, speed, visibility, timer, crisis, trending_up>",\n'
    '    "description": "<2-3 sentences on their aim tendencies, citing the numbers>",\n'
    '    "tendencies": ["<one short observation>", "<another>", "<optional third>"]\n'
    "  },\n"
    '  "utility": {\n'
    '    "name": "<short pattern name, e.g. Buys Nades, Holds Them>",\n'
    '    "icon": "<one of: local_fire_department, flare, bomb, sensors, savings, crisis, trending_up>",\n'
    '    "description": "<2-3 sentences on their utility tendencies, citing the numbers>",\n'
    '    "tendencies": ["<one short observation>", "<another>", "<optional third>"]\n'
    "  },\n"
    '  "behaviour": {\n'
    '    "name": "<short pattern name, e.g. Opens Early, Fades Late>",\n'
    '    "icon": "<one of: psychology, groups, directions_run, shield, swords, timeline, trending_up>",\n'
    '    "description": "<2-3 sentences on round-level habits by side, citing the numbers>",\n'
    '    "tendencies": ["<one short observation>", "<another>", "<optional third>"]\n'
    "  },\n"
    '  "headline": "<one sentence naming the single most actionable thing>"\n'
    "}"
)


# The career assessment answers the same "how do they play" question over every
# match instead of one map's worth, so it keeps the pattern half and drops the
# roles.  Callouts only mean something on the map they belong to; pooling
# "Palace" and "Banana" into one role would describe nobody.  What replaces
# them is the comparison only the overall view can make — which maps the player
# is actually good on.
OVERALL_KEYS = ("aim", "utility", "behaviour", "maps")


# Reserved storage key: no map is named "__overall__", so the career assessment
# can live in the same file as the per-map ones without a second store.
OVERALL_KEY = "__overall__"


OVERALL_SYSTEM_PROMPT = (
    "You are PULSE_AI, an expert CS2 analyst. You are given one player's whole match "
    "history: their aim, utility and round behaviour measured across every map, a "
    "per-map breakdown, and how their form has moved over time. Sample sizes are given "
    "for every measurement.\n\n"
    "Name the habits and tendencies that hold across their play as a whole, and say "
    "which maps they are actually good on.\n\n"
    "IMPORTANT:\n"
    "- Every claim must trace to a number in the data. Do not invent statistics.\n"
    "- Respect sample sizes. A map played twice says almost nothing; say so rather than "
    "ranking it confidently against a map played twenty times.\n"
    "- Distinguish a real trend from noise. Two matches of better form is not improvement.\n"
    "- Prefer the specific over the generic: 'stops cleanly off a walk but coasts on "
    "full-speed peeks' beats 'movement could improve'.\n"
    "- Thresholds worth knowing: rifles are inaccurate above 85 u/s; crosshair placement "
    "under 10 degrees is good and over 20 is poor; wasting over ~20% of grenade spend is high.\n"
    "- A tendency is not automatically a flaw. Name strengths where the numbers show them.\n\n"
    "Respond in EXACTLY this JSON format (no markdown, no extra text):\n"
    "{\n"
    '  "aim": {\n'
    '    "name": "<short pattern name, 2-5 words, e.g. Patient But Slow To Stop>",\n'
    '    "icon": "<one of: my_location, target, speed, visibility, timer, crisis, trending_up>",\n'
    '    "description": "<2-3 sentences on their aim tendencies, citing the numbers>",\n'
    '    "tendencies": ["<one short observation>", "<another>", "<optional third>"]\n'
    "  },\n"
    '  "utility": {\n'
    '    "name": "<short pattern name, e.g. Buys Nades, Holds Them>",\n'
    '    "icon": "<one of: local_fire_department, flare, bomb, sensors, savings, crisis, trending_up>",\n'
    '    "description": "<2-3 sentences on their utility tendencies, citing the numbers>",\n'
    '    "tendencies": ["<one short observation>", "<another>", "<optional third>"]\n'
    "  },\n"
    '  "behaviour": {\n'
    '    "name": "<short pattern name, e.g. Opens Early, Fades Late>",\n'
    '    "icon": "<one of: psychology, groups, directions_run, shield, swords, timeline, trending_up>",\n'
    '    "description": "<2-3 sentences on round-level habits by side, citing the numbers>",\n'
    '    "tendencies": ["<one short observation>", "<another>", "<optional third>"]\n'
    "  },\n"
    '  "maps": {\n'
    '    "name": "<short summary of their map pool, e.g. One Strong Map, Rest Even>",\n'
    '    "icon": "<one of: map, explore, terrain, public, trending_up, crisis>",\n'
    '    "description": "<2-3 sentences on which maps they perform on and which they do not, '
    'naming the sample behind each>",\n'
    '    "tendencies": ["<one short observation>", "<another>", "<optional third>"]\n'
    "  },\n"
    '  "headline": "<one sentence naming the single most actionable thing across their play>"\n'
    "}"
)


def build_overall_context(matches: list[dict], rounds: list[dict]) -> str:
    """Career context: the measurements, plus per-map and over-time comparisons.

    Positional narratives are deliberately absent.  Every map's callouts are its
    own vocabulary, and pasting all of them together would cost a great many
    tokens to describe a player who does not exist.  What the career view can
    say that a single map cannot is how the maps compare and which way form is
    moving, so that is what gets added.
    """
    parts = [build_patterns_context("all maps", matches, rounds)]

    by_map: dict[str, list[dict]] = {}
    for m in matches:
        by_map.setdefault(m.get("map_name", "unknown"), []).append(m)

    lines = ["=== PER-MAP BREAKDOWN ==="]
    for name, group in sorted(by_map.items(), key=lambda kv: -len(kv[1])):
        ratings = [m["hltv_rating"] for m in group if m.get("hltv_rating") is not None]
        adrs = [m["adr"] for m in group if m.get("adr") is not None]
        wins = sum(1 for m in group if str(m.get("match_result", "")).upper() == "WIN")
        lines.append(
            f"{name}: {len(group)} matches, {_mean(ratings)} rating, "
            f"{_mean(adrs)} ADR, {_pct(wins, len(group))}% won"
        )
    parts.append("\n".join(lines))

    # Form: the oldest half against the newest, which is the coarsest split that
    # cannot be moved by a single good match.
    dated = [m for m in matches if m.get("date")]
    dated.sort(key=lambda m: str(m["date"]))
    if len(dated) >= 4:
        half = len(dated) // 2
        older, newer = dated[:half], dated[half:]

        def _form(group: list[dict]) -> str:
            ratings = [m["hltv_rating"] for m in group if m.get("hltv_rating") is not None]
            adrs = [m["adr"] for m in group if m.get("adr") is not None]
            return f"{_mean(ratings)} rating, {_mean(adrs)} ADR"

        parts.append(
            "=== FORM OVER TIME ===\n"
            f"Earlier {len(older)} matches ({older[0]['date']} to {older[-1]['date']}): {_form(older)}\n"
            f"Recent {len(newer)} matches ({newer[0]['date']} to {newer[-1]['date']}): {_form(newer)}"
        )

    return "\n\n".join(parts)
