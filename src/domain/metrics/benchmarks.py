"""Benchmark tiers for the aim and utility figures.

Hand-set thresholds, not calibrated against a population of real players. Each
benchmark carries ``calibration: "heuristic"`` so a consumer can tell the
difference between this and the measured tables in src/domain/calibration/.
"""

from __future__ import annotations

from typing import Any

from src.domain.metrics._shared import (
    _aim_bounds,
    _confidence,
)

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

    Returns a dict of metric_key → {value, tier, unit, n, confidence,
    calibration} where tier is one of "pro", "high_amateur", "average",
    "below_average" and ``calibration`` is always "heuristic" for now — see the
    note above the threshold tables.
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
        # All of these read the median rather than the mean: with a dozen-odd
        # samples per match a single bad engagement was enough to drop a tier.

        # Speed When Shooting (u/s) — lower is better
        mv = aim_stats.get("movement", {})
        if mv.get("median") is not None:
            benchmarks["shot_speed"] = {
                "value": mv["median"],
                "n": mv.get("n"),
                "confidence": mv.get("confidence"),
                "tier": _classify_tier_lower_better(mv["median"], *_aim_bounds("movement")),
                "unit": "u/s",
            }

        # Counter-strafe rate — of the rifle engagements that actually needed
        # a stop, how many were stopped properly rather than coasted.  Graded
        # on its own sample size, which is much smaller than the movement
        # sample it comes from once crouches and non-rifles are excluded.
        cs_rate = mv.get("counterstrafe_rate")
        if cs_rate is not None:
            cs_n = mv.get("counterstrafe_attempts", 0)
            benchmarks["counterstrafe"] = {
                "value": cs_rate,
                "n": cs_n,
                "confidence": _confidence(cs_n),
                "tier": _classify_tier_higher_better(cs_rate, *_aim_bounds("counterstrafe")),
                "unit": "% of rifle stops",
            }

        # Pre-Aim Offset (degrees) — lower is better
        pa = aim_stats.get("preaim", {})
        if pa.get("median") is not None:
            benchmarks["preaim_offset"] = {
                "value": pa["median"],
                "n": pa.get("n"),
                "confidence": pa.get("confidence"),
                "tier": _classify_tier_lower_better(pa["median"], *_aim_bounds("preaim")),
                "unit": "°",
            }

        # Reaction Time (ms) — reported, but flagged: see _AIM_RATING_WEIGHTS.
        rxn = aim_stats.get("reaction", {})
        if rxn.get("median") is not None:
            benchmarks["reaction_time"] = {
                "value": rxn["median"],
                "n": rxn.get("n"),
                "confidence": rxn.get("confidence"),
                "diagnostic_only": True,
                "tier": _classify_tier_lower_better(rxn["median"], *_aim_bounds("reaction")),
                "unit": "ms",
            }

        # Engagement Time to Kill (ms) — lower is better
        ttk = aim_stats.get("ttk", {})
        if ttk.get("median") is not None:
            ttk_ms = round(ttk["median"] * 1000, 0)
            benchmarks["engagement_ttk"] = {
                "value": ttk_ms,
                "n": ttk.get("n"),
                "confidence": ttk.get("confidence"),
                "tier": _classify_tier_lower_better(ttk_ms, *[b * 1000 for b in _aim_bounds("ttk")]),
                "unit": "ms",
            }

    # Every threshold in this function is a hand-set guess.  Marking each entry
    # keeps that visible to consumers instead of leaving the tier looking like
    # a measured comparison.
    for entry in benchmarks.values():
        entry["calibration"] = "heuristic"

    return benchmarks
