"""Grenades and economy.

Needs the raw purchase and detonation frames, so it can only be recomputed
while the demo is still on disk.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.callouts import get_callout, is_map_supported
from src.domain.metrics._shared import (
    _CONFIDENCE_K,
    _find_id_col,
    _median,
)
from src.domain.metrics.context import MetricContext
from src.domain.metrics.registry import ENRICHED_ROUNDS, metric


@metric(
    id="utility.stats",
    label="Utility & Economy",
    group="utility",
    version=1,
    requires={
        ENRICHED_ROUNDS,
        "parsed:item_purchase",
        "parsed:player_blind",
        "parsed:smoke_detonate",
        "parsed:molotov_detonate",
        "parsed:weapon_fire",
    },
    output_key="utility_data",
    description=(
        "Grenades thrown and what they achieved, flash effectiveness, and how "
        "much of the grenade spend was wasted."
    ),
)
def utility_stats(ctx: MetricContext) -> dict[str, Any]:
    return _calculate_utility_stats(
        ctx.enriched_rounds, ctx.parsed, ctx.steam_id, ctx.total_rounds, ctx.map_name,
    )


# Enemy blind time (seconds) below which a flash did not really achieve
# anything.  A quarter of the enemy flashes in the stored matches land under a
# second, which is barely a flinch — counting those the same as a four-second
# blind is what made "enemies flashed" a poor guide to flash quality.
_EFFECTIVE_BLIND_SECONDS = 1.0


# Relative weights for the utility rating.  Smoke placement is deliberately
# absent — see the note at the rating itself.
_UTILITY_RATING_WEIGHTS = {
    "use_rate": 0.35,
    "flash": 0.35,
    "damage": 0.30,
}


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


def _genuine_purchases(purchase_df: pd.DataFrame) -> pd.DataFrame:
    """Purchases we can actually stand behind.

    item_purchase carries two things that are not buys. Refunds come back with
    ``was_sold`` set, and the game periodically re-emits a player's whole
    inventory as a burst of rows sharing one tick with the slot numbers
    restarting at zero. A re-emitted rifle looked exactly like buying a second
    one, which is how the drop detector reported weapons nobody bought.

    Only single-row ticks are kept. Roughly a third of rows sit in bursts and
    not all of them are snapshots, so this discards some real purchases too —
    the trade is deliberate: under-reporting drops beats inventing them.
    """
    if purchase_df.empty:
        return purchase_df
    df = purchase_df
    if "was_sold" in df.columns:
        df = df[df["was_sold"] != True]  # noqa: E712
    id_col = _find_id_col(df, ("steamid", "attacker_steamid", "user_steamid"))
    if id_col is None or "tick" not in df.columns:
        return df
    counts = df.groupby([id_col, "tick"])[id_col].transform("size")
    return df[counts == 1]


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
    _bought_source = purchase_df
    if not purchase_df.empty and "was_sold" in purchase_df.columns:
        _bought_source = purchase_df[purchase_df["was_sold"] != True]  # noqa: E712
    if not _bought_source.empty:
        id_col = _find_id_col(_bought_source, ("steamid", "attacker_steamid", "user_steamid"))
        if id_col:
            name_col = "item_name" if "item_name" in _bought_source.columns else (
                "weapon" if "weapon" in _bought_source.columns else None
            )
            if name_col and "round" in _bought_source.columns:
                player_buys = _bought_source[_bought_source[id_col].astype(str) == sid]
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
    self_flashes = 0
    total_self_blind_duration = 0.0
    enemy_blind_durations: list[float] = []

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
                    # Blinding yourself is not a team flash. Same team by
                    # definition, so the team check alone charged every
                    # self-flash to the player as if they had blinded a mate.
                    if str(brow.get("user_steamid", "")) == sid:
                        self_flashes += 1
                        total_self_blind_duration += dur
                        continue
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
                        enemy_blind_durations.append(dur)

    # Flash assists from death events
    death_df = parsed_data.get("player_death", pd.DataFrame())
    if not death_df.empty and "assistedflash" in death_df.columns:
        fa = death_df[
            (death_df.get("assister_steamid", pd.Series(dtype=str)).astype(str) == sid)
            & (death_df["assistedflash"] == True)  # noqa: E712
        ]
        flash_assists = len(fa)

    flashes_thrown = thrown.get("flashbang", 0)
    effective_flashes = sum(
        1 for d in enemy_blind_durations if d >= _EFFECTIVE_BLIND_SECONDS
    )

    flash_efficiency: dict[str, Any] = {
        "thrown": flashes_thrown,
        "enemies_flashed": enemy_flashes,
        "team_flashed": team_flashes,
        "self_flashed": self_flashes,
        "self_blind_duration": round(total_self_blind_duration, 1),
        "avg_enemy_blind_duration": round(
            total_enemy_blind_duration / enemy_flashes, 1
        ) if enemy_flashes > 0 else 0.0,
        "median_enemy_blind_duration": round(_median(enemy_blind_durations), 2)
        if enemy_blind_durations else None,
        "total_enemy_blind_duration": round(total_enemy_blind_duration, 1),
        "flash_assists": flash_assists,
        "enemies_per_flash": round(
            enemy_flashes / flashes_thrown, 2
        ) if flashes_thrown > 0 else 0.0,
        # A count of enemies flashed treats a 0.3 s glance and a 4 s blind as
        # the same event, and a quarter of these land under a second.  Blind
        # seconds delivered per flash thrown is the figure that tracks whether
        # the flash actually bought anything.
        "effective_flashes": effective_flashes,
        "effective_flash_pct": round(
            effective_flashes / enemy_flashes * 100, 1
        ) if enemy_flashes > 0 else 0.0,
        "blind_seconds_per_flash": round(
            total_enemy_blind_duration / flashes_thrown, 2
        ) if flashes_thrown > 0 else 0.0,
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

    # Check if smokes extinguished *enemy* molotovs (within ~300 unit radius).
    # Own molotovs have to be excluded: smokes and fire both get thrown at the
    # same chokepoints, so counting every molotov in the round credited players
    # for smoking out their own utility.
    molly_extinguishes = 0
    enemy_molly_det = molotov_det_df
    if not molotov_det_df.empty:
        molly_id_col = _find_id_col(
            molotov_det_df, ("user_steamid", "steamid", "attacker_steamid")
        )
        if molly_id_col:
            enemy_molly_det = molotov_det_df[
                molotov_det_df[molly_id_col].astype(str) != sid
            ]
    if smoke_locations and not enemy_molly_det.empty and "x" in enemy_molly_det.columns:
        for sm in smoke_locations:
            for _, mrow in enemy_molly_det.iterrows():
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
                sum(
                    f.get("duration", 0)
                    for f in util.get("flash_instances", [])
                    if not f.get("is_friendly") and not f.get("is_self")
                ),
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
    #                                                                      #
    # Same construction as the aim rating: each component scores off what   #
    # was actually observed, carries weight proportional to how much        #
    # evidence stands behind it, and is dropped entirely when there is no   #
    # evidence rather than being filled in at 50.                           #
    #                                                                      #
    # Smoke placement used to be 20% of this and has been removed.  It      #
    # scored resolved-callout ÷ smokes-thrown, which measures how complete   #
    # our callout map is and whether detonation events survived the parse —  #
    # not whether the smoke was any good.  Across 76 stored matches it       #
    # produced values from 0 to 100 on identical play.  Smoke locations are  #
    # still reported; they are just no longer scored.                        #
    # ------------------------------------------------------------------ #
    util_components: list[tuple[str, float, int]] = []

    if total_bought > 0:
        util_components.append(("use_rate", min(100.0, use_rate), total_bought))

    if flashes_thrown > 0:
        # ~2 s of enemy blindness per flash thrown is a good flash.
        bspf = total_enemy_blind_duration / flashes_thrown
        util_components.append((
            "flash", min(100.0, bspf / 2.0 * 100.0), flashes_thrown,
        ))

    total_dmg_nades = (
        thrown.get("hegrenade", 0) + thrown.get("molotov", 0) + thrown.get("incgrenade", 0)
    )
    if total_dmg_nades > 0:
        avg_dmg = (total_he_damage + total_molly_damage) / total_dmg_nades
        util_components.append((
            "damage", min(100.0, avg_dmg * 2.5), total_dmg_nades,  # 40 dmg/nade = 100
        ))

    rating: float | None = None
    rating_inputs: list[dict[str, Any]] = []
    if util_components:
        weighted_sum = 0.0
        total_weight = 0.0
        for name, score, n in util_components:
            weight = _UTILITY_RATING_WEIGHTS[name] * (n / (n + _CONFIDENCE_K))
            weighted_sum += score * weight
            total_weight += weight
            rating_inputs.append({
                "metric": name, "score": round(score, 1),
                "n": n, "weight": round(weight, 4),
            })
        if total_weight > 0:
            rating = weighted_sum / total_weight
            for item in rating_inputs:
                item["weight_share"] = round(item["weight"] / total_weight, 3)
            # Teamplayer penalty — minimal: minor incidents are normal in CS2.
            # Team flash: only penalise beyond 3 flashes (−0.5 each)
            if team_flashes > 3:
                rating -= (team_flashes - 3) * 0.5
            rating = round(min(100.0, max(0.0, rating)), 1)

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
                        (player_blinds[atk_t] == player_blinds[vic_t])
                        & (player_blinds["user_steamid"].astype(str) != sid)
                    ]

    # Pre-compute per-round weapon drops
    _drop_rounds: dict[int, list[str]] = {}
    _drop_source = _genuine_purchases(purchase_df)
    if not _drop_source.empty:
        id_col = _find_id_col(_drop_source, ("steamid", "attacker_steamid", "user_steamid"))
        if id_col:
            name_col = "item_name" if "item_name" in _drop_source.columns else (
                "weapon" if "weapon" in _drop_source.columns else None
            )
            if name_col and "round" in _drop_source.columns:
                player_buys = _drop_source[_drop_source[id_col].astype(str) == sid]
                for rnd_num in player_buys["round"].unique():
                    rnd_buys = player_buys[player_buys["round"] == rnd_num]
                    slot_items: dict[str, list[str]] = {}
                    for _, row in rnd_buys.iterrows():
                        raw = str(row[name_col]).lower()
                        slot = _WEAPON_SLOT.get(raw)
                        if slot:
                            slot_items.setdefault(slot, []).append(raw)
                    dropped: list[str] = []
                    for items in slot_items.values():
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
        "utility_rating_inputs": rating_inputs,
        "economics": economics,
        "flash": flash_efficiency,
        "he": he_efficiency,
        "molotov": molly_efficiency,
        "smoke": smoke_efficiency,
        "per_round": per_round,
        "utility_rating": rating,
        "teamplayer": teamplayer,
    }
