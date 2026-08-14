"""Aim and movement.

The one measurement group that needs nothing but the enriched rounds — which
are stored per round in the database. That is what makes it recomputable for a
match whose demo the retention feature deleted: bump the version here and the
value can be rebuilt from SQLite alone.
"""

from __future__ import annotations

from typing import Any

from src.domain.metrics._shared import (
    _AIM_THRESHOLDS,
    _CONFIDENCE_K,
    _PEEK_BUCKETS,
    _confidence,
    _median,
)
from src.domain.metrics.context import MetricContext
from src.domain.metrics.registry import ENRICHED_ROUNDS, metric


@metric(
    id="aim.stats",
    label="Aim & Movement",
    group="aim",
    version=1,
    requires={ENRICHED_ROUNDS},
    output_key="aim_stats",
    description=(
        "Shot speed, crosshair placement, time to damage, reaction, accuracy "
        "and counter-strafe quality, with the sample behind each figure."
    ),
)
def aim_stats(ctx: MetricContext) -> dict[str, Any]:
    return _calculate_aim_stats(ctx.enriched_rounds)


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


def _peek_bucket(speed: float) -> str | None:
    """Which ``_PEEK_BUCKETS`` band *speed* falls in, or None if below them all."""
    for key, _label, lo, hi in _PEEK_BUCKETS:
        if lo <= speed < hi:
            return key
    return None


# The regions of the peek-speed axis, in ascending order.  The chart shades and
# labels these, and the per-match distribution below is reported against the
# same list, so the legend beside a chart names exactly the bands drawn on it.
_PEEK_ZONES: list[dict[str, Any]] = _AIM_THRESHOLDS["peek"]["zones"]


def _peek_zone_index(speed: float) -> int:
    """Which ``_PEEK_ZONES`` region *speed* falls in — what kind of peek it was.

    Unlike ``_peek_bucket`` this never returns None: the bottom region covers
    the speeds that never needed a stop, which are held angles rather than
    peeks but still belong somewhere in the distribution.
    """
    index = 0
    for i, zone in enumerate(_PEEK_ZONES):
        if speed >= zone["at"]:
            index = i
    return index


# Engagement times at or beyond this are not telling us about aim any more —
# they are a fight that broke off and resumed, a reload, or a repositioning.
# Excluded from the median and counted separately, so the figure stays a
# statement about duels rather than about how long the round was.
_TTK_OUTLIER_SECONDS = 1.0


# Relative weights of the components that feed the aim rating.
#
# Two measured metrics are deliberately absent.
#
# Reaction time: at 2-20 samples per match, and with no way to tell a flick
# from an enemy walking into a held crosshair, it is a diagnostic rather than a
# rating input.  What is actually measured here is the gap between the crosshair
# arriving on the enemy and the shot, which is not the same thing as how fast
# the player reacted, so it is reported and not graded.
#
# Counter-strafe rate: it is the best-measured technique stat here, but it
# answers a different question than the rating asks.  The movement component
# already scores the *outcome* — was the shot taken from an accurate state —
# and coasting to a halt produces just as accurate a shot as counter-strafing,
# only slower and more telegraphed.  Feeding both in would score movement
# twice and break the independence the per-dimension weighting depends on, so
# counter-strafe rate is reported and graded on its own instead.
_AIM_RATING_WEIGHTS = {
    "preaim": 0.40,
    "movement": 0.30,
    "ttk": 0.30,
}


# Weapons where counter-strafing is the technique that decides the shot.
# Rifles have a hard accuracy penalty above ~34% of max speed and no way to
# shoot through it, so stopping properly is the whole skill.  SMGs, shotguns
# and pistols carry a far smaller penalty and are routinely fired on the move
# by design, and snipers are a different mechanic again — grading any of them
# on counter-strafe quality measures the weapon, not the player.
_COUNTERSTRAFE_WEAPONS: set[str] = {
    "AK-47", "M4A4", "M4A1-S", "Galil AR", "FAMAS", "SG 553", "AUG",
}


def _summarise(values: list[float], digits: int) -> dict[str, Any]:
    """Central-tendency block shared by every aim metric.

    ``median`` is the headline figure and what the rating and benchmarks read;
    ``avg`` is kept alongside it because it is what earlier versions reported
    and it is still useful for spotting a skewed distribution at a glance.
    """
    n = len(values)
    return {
        "n": n,
        "confidence": _confidence(n),
        "median": round(_median(values), digits),
        "avg": round(sum(values) / n, digits),
        "min": round(min(values), digits),
        "max": round(max(values), digits),
    }


def _calculate_aim_stats(enriched_rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-kill movement, pre-aim, and TTK data into match-level stats.

    Every metric block carries ``median`` (the headline figure), ``avg``,
    ``min``, ``max``, ``n`` and a coarse ``confidence`` label.  The median
    leads because these samples are small and right-skewed.

    Returns a dict with:
      - movement: {speeds: [...], weapons: [...], median, avg, min, max, n,
                   confidence, standing_pct, counterstrafed_pct, stopped_pct,
                   running_pct, low_penalty_weapons: [...],
                   counterstrafe_by_peek: [...]}
      - peek: speed carried into the duel; diagnostic only, never graded
      - preaim: {errors: [...], median, avg, ..., excellent_pct, good_pct,
                 moderate_pct, poor_pct}
      - ttk: {values: [...], median, avg, ..., excluded_outliers}  (seconds)
      - reaction: diagnostic only, never an input to aim_rating
      - aim_rating: 0-100, or None when nothing measurable was found
      - aim_rating_inputs: per-component score, n and the weight it earned
    """
    shot_speeds: list[float] = []
    # Peak speed in the half second before the shot — how fast the player
    # entered the duel.  Kept parallel to the lists around it so the
    # counter-strafe cross-tab can zip the two together.
    peek_speeds: list[float] = []
    kill_weapons: list[str] = []
    movement_qualities: list[str] = []
    movement_crouched: list[bool] = []
    preaim_errors: list[float] = []
    preaim_qualities: list[str] = []
    ttk_values: list[float] = []
    ttk_shots: list[int] = []
    ttk_hits: list[int] = []
    reaction_values: list[float] = []
    reaction_categories: list[str] = []

    # Accuracy per-encounter
    accuracy_values: list[float] = []    # hit_pct per engagement
    accuracy_hits: list[int] = []
    accuracy_shots: list[int] = []
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
        for k in r.get("kills_detail", []):
            # Every kills_detail entry is a kill the player secured, so the
            # aim-scatter outcome is always "kill". (Dying later in the same
            # round does not turn a won duel into a lost one.)
            outcome = "kill"
            weapon = k.get("weapon", "")
            enc: dict[str, Any] = {"outcome": outcome}

            mv = k.get("movement")
            if mv:
                shot_speeds.append(mv["shot_speed"])
                peek_speeds.append(mv["pre_speed"])
                movement_qualities.append(mv["movement_quality"])
                movement_crouched.append(bool(mv.get("crouched")))
                kill_weapons.append(weapon)
                outcomes_mov.append(outcome)
                enc["movement"] = mv["shot_speed"]
                enc["peek"] = mv["pre_speed"]
                if mv.get("stop_ticks") is not None:
                    enc["stop_ticks"] = mv["stop_ticks"]

            pa = k.get("preaim")
            if pa:
                preaim_errors.append(pa["crosshair_error"])
                preaim_qualities.append(pa["preaim_quality"])
                outcomes_preaim.append(outcome)
                enc["preaim"] = pa["crosshair_error"]

            ttd = k.get("ttd")
            # A one-tap registers first_shot_tick == kill_tick, so its
            # engagement time is legitimately 0.  Requiring > 0 dropped exactly
            # the fastest kills a player has, pulling the median upward.
            if ttd and ttd.get("ttk_seconds") is not None:
                ttk_values.append(ttd["ttk_seconds"])
                ttk_shots.append(ttd.get("shots_fired", ttd.get("hits", 1)))
                ttk_hits.append(ttd.get("hits", 1))
                outcomes_ttk.append(outcome)
                # Same exclusion the aggregate applies, or the scatter would
                # plot engagements the median beside it has thrown away.
                if ttd["ttk_seconds"] < _TTK_OUTLIER_SECONDS:
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
                accuracy_hits.append(int(ttd.get("hits", 0)))
                accuracy_shots.append(int(ttd.get("shots_fired", 0)))
                outcomes_acc.append(outcome)
                enc["accuracy"] = acc["hit_pct"]

            encounters.append(enc)

        # Damage encounters (hurt an enemy but did not kill them). A duel is
        # counted as lost ("death") when the player dies to that same enemy
        # shortly after last hitting them; otherwise it stays "damage"
        # (inconclusive — enemy disengaged, was traded, or round ended).
        death = r.get("death_detail")
        death_tick = death.get("tick") if death else None
        killer_sid = death.get("killer_steamid") if death else None

        for d in r.get("damage_encounters", []):
            outcome = "damage"
            if (
                killer_sid
                and d.get("victim_sid") == killer_sid
                and death_tick is not None
                and d.get("last_tick") is not None
                and 0 <= death_tick - d["last_tick"] <= _LOST_DUEL_WINDOW
            ):
                outcome = "death"

            enc = {"outcome": outcome}

            mv = d.get("movement")
            if mv:
                shot_speeds.append(mv["shot_speed"])
                peek_speeds.append(mv["pre_speed"])
                movement_qualities.append(mv["movement_quality"])
                movement_crouched.append(bool(mv.get("crouched")))
                kill_weapons.append(d.get("weapon", ""))
                outcomes_mov.append(outcome)
                enc["movement"] = mv["shot_speed"]
                enc["peek"] = mv["pre_speed"]
                if mv.get("stop_ticks") is not None:
                    enc["stop_ticks"] = mv["stop_ticks"]

            pa = d.get("preaim")
            if pa:
                preaim_errors.append(pa["crosshair_error"])
                preaim_qualities.append(pa["preaim_quality"])
                outcomes_preaim.append(outcome)
                enc["preaim"] = pa["crosshair_error"]

            # Reaction and accuracy do not depend on the duel being won, and
            # these engagements are roughly 40% of a player's shooting.
            # Excluding them measured only the fights that went well.
            rxn = d.get("reaction")
            if rxn and rxn.get("reaction_ms") is not None:
                reaction_values.append(rxn["reaction_ms"])
                reaction_categories.append(rxn["category"])
                outcomes_rxn.append(outcome)
                enc["reaction"] = rxn["reaction_ms"]

            acc = d.get("accuracy")
            if acc and acc.get("hit_pct") is not None:
                accuracy_values.append(acc["hit_pct"])
                first_bullet_hits.append(acc["first_bullet_hit"])
                hitgroup_head += acc.get("head", 0)
                hitgroup_upper += acc.get("upper", 0)
                hitgroup_lower += acc.get("lower", 0)
                hitgroup_total += acc.get("head", 0) + acc.get("upper", 0) + acc.get("lower", 0)
                accuracy_hits.append(int(d.get("hits", 0)))
                accuracy_shots.append(int(d.get("shots_fired", 0)))
                outcomes_acc.append(outcome)
                enc["accuracy"] = acc["hit_pct"]

            encounters.append(enc)

        # Bursts that landed nothing. They carry no movement or crosshair data
        # (there is no victim to measure against), but their bullets belong in
        # the accuracy denominator — they are the engagements that went worst.
        for w in r.get("whiffed_engagements", []):
            shots = int(w.get("shots_fired", 0))
            if shots <= 0:
                continue
            accuracy_values.append(0.0)
            accuracy_hits.append(0)
            accuracy_shots.append(shots)
            first_bullet_hits.append(False)
            outcomes_acc.append("whiff")
            encounters.append({"outcome": "whiff", "accuracy": 0.0})

    n_mov = len(movement_qualities)
    n_aim = len(preaim_qualities)

    movement = {}
    if shot_speeds:
        standing = sum(1 for q in movement_qualities if q == "standing")
        cs = sum(1 for q in movement_qualities if q == "counter-strafed")
        stopped = sum(1 for q in movement_qualities if q == "stopped")
        running = sum(1 for q in movement_qualities if q == "running")
        # Identify which kills used low-penalty weapons
        low_penalty_flags = [
            w in _LOW_PENALTY_WEAPONS for w in kill_weapons
        ]
        running_low = sum(
            1 for q, lp in zip(movement_qualities, low_penalty_flags)
            if q == "running" and lp
        )
        # Counter-strafe rate, scored only where the technique decides the
        # shot: rifles, not crouched, and only engagements that actually
        # needed a stop.  Reporting counter-strafes as a share of *all*
        # engagements buried the signal — running and standing shots, which
        # say nothing about stopping ability, made up most of the denominator.
        #
        # The same engagements are also split by how fast the peek was.  The
        # overall rate hides the question worth asking: a player who stops
        # cleanly off a walk and coasts off every full-speed peek scores the
        # same as one who is merely inconsistent, and only the first has a
        # specific thing to practise.
        cs_attempts = 0
        cs_good = 0
        cs_by_peek: dict[str, list[int]] = {k: [0, 0] for k, _, _, _ in _PEEK_BUCKETS}
        for q, w, crouch, peak in zip(
            movement_qualities, kill_weapons, movement_crouched, peek_speeds,
        ):
            if crouch or w not in _COUNTERSTRAFE_WEAPONS:
                continue
            if q not in ("counter-strafed", "stopped"):
                continue
            good = q == "counter-strafed"
            cs_attempts += 1
            if good:
                cs_good += 1
            bucket = _peek_bucket(peak)
            if bucket:
                cs_by_peek[bucket][0] += 1
                if good:
                    cs_by_peek[bucket][1] += 1

        movement = {
            **_summarise(shot_speeds, 1),
            "speeds": [round(s, 1) for s in shot_speeds],
            "weapons": kill_weapons,
            "low_penalty": low_penalty_flags,
            "crouched": movement_crouched,
            "outcomes": outcomes_mov,
            "crouched_pct": round(
                sum(1 for c in movement_crouched if c) / n_mov * 100, 1
            ) if n_mov else 0,
            "counterstrafe_attempts": cs_attempts,
            "counterstrafe_good": cs_good,
            "counterstrafe_rate": round(cs_good / cs_attempts * 100, 1) if cs_attempts else None,
            "counterstrafe_by_peek": [
                {
                    "bucket": key,
                    "label": label,
                    "min": lo,
                    "max": None if hi == float("inf") else hi,
                    "attempts": cs_by_peek[key][0],
                    "good": cs_by_peek[key][1],
                    "rate": round(cs_by_peek[key][1] / cs_by_peek[key][0] * 100, 1)
                    if cs_by_peek[key][0] else None,
                }
                for key, label, lo, hi in _PEEK_BUCKETS
            ],
            "standing_pct": round(standing / n_mov * 100, 1) if n_mov else 0,
            "counterstrafed_pct": round(cs / n_mov * 100, 1) if n_mov else 0,
            "stopped_pct": round(stopped / n_mov * 100, 1) if n_mov else 0,
            "running_pct": round(running / n_mov * 100, 1) if n_mov else 0,
            "running_total": running,
            "running_low_penalty": running_low,
        }

    # How fast the player was moving on the way into the duel, measured as the
    # peak speed over the half second ending at the first shot.  Reported, never
    # graded and never fed into the rating: a slow approach is correct when
    # holding an angle and wrong when entering a site, and the demo does not say
    # which the player intended.  Its value is as the x-axis for the metrics
    # that *are* graded — above all the counter-strafe, which is the technique
    # that has to absorb whatever speed the peek carried in.
    peek = {}
    if peek_speeds:
        n_peek = len(peek_speeds)
        zone_counts = [0] * len(_PEEK_ZONES)
        for s in peek_speeds:
            zone_counts[_peek_zone_index(s)] += 1
        by_zone = [
            {
                "label": zone["label"],
                "at": zone["at"],
                "n": count,
                "pct": round(count / n_peek * 100, 1),
            }
            for zone, count in zip(_PEEK_ZONES, zone_counts)
        ]
        peek = {
            **_summarise(peek_speeds, 1),
            "values": [round(s, 1) for s in peek_speeds],
            "outcomes": outcomes_mov,
            "diagnostic_only": True,
            # The share of engagements in each region of the axis, so a legend
            # can name the same bands the chart shades.
            "by_zone": by_zone,
            # Headline shortcuts, read off the same counts rather than
            # recomputed: the bottom region is a held angle (never fast enough
            # to need a stop) and the top one is a committed full-speed peek.
            "held_pct": by_zone[0]["pct"],
            "full_pct": by_zone[-1]["pct"],
        }

    preaim = {}
    if preaim_errors:
        exc = sum(1 for q in preaim_qualities if q == "excellent")
        good = sum(1 for q in preaim_qualities if q == "good")
        mod = sum(1 for q in preaim_qualities if q == "moderate")
        poor = sum(1 for q in preaim_qualities if q == "poor")
        preaim = {
            **_summarise(preaim_errors, 1),
            "errors": [round(e, 1) for e in preaim_errors],
            "outcomes": outcomes_preaim,
            "excellent_pct": round(exc / n_aim * 100, 1) if n_aim else 0,
            "good_pct": round(good / n_aim * 100, 1) if n_aim else 0,
            "moderate_pct": round(mod / n_aim * 100, 1) if n_aim else 0,
            "poor_pct": round(poor / n_aim * 100, 1) if n_aim else 0,
        }

    ttk = {}
    if ttk_values:
        # Drop engagements long enough that they stopped being about aim.  The
        # outcomes list runs parallel to the values, so it has to be filtered
        # in lockstep or the scatter plot mislabels every point after the first
        # exclusion.
        kept = [
            (v, o) for v, o in zip(ttk_values, outcomes_ttk)
            if v < _TTK_OUTLIER_SECONDS
        ]
        excluded = len(ttk_values) - len(kept)
        if kept:
            kept_values = [v for v, _ in kept]
            total_shots = sum(ttk_shots)
            total_hits = sum(ttk_hits)
            ttk = {
                **_summarise(kept_values, 3),
                "values": [round(v, 3) for v in kept_values],
                "outcomes": [o for _, o in kept],
                "excluded_outliers": excluded,
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
            **_summarise(reaction_values, 1),
            # Reported for inspection only — see _AIM_RATING_WEIGHTS for why it
            # does not feed the rating.
            "diagnostic_only": True,
            "values": [round(v, 1) for v in reaction_values],
            "outcomes": outcomes_rxn,
            "lightning_pct": round(lightning / n_rxn * 100, 1),
            "fast_pct": round(fast / n_rxn * 100, 1),
            "average_pct": round(average / n_rxn * 100, 1),
            "slow_pct": round(slow / n_rxn * 100, 1),
        }

    accuracy = {}
    if accuracy_values:
        n_acc = len(accuracy_values)
        fb_hit = sum(1 for fb in first_bullet_hits if fb)
        total_acc_hits = sum(accuracy_hits)
        total_acc_shots = sum(accuracy_shots)
        accuracy = {
            **_summarise(accuracy_values, 1),
            # Accuracy is a rate, so the headline pools every bullet rather
            # than averaging per-engagement percentages.  A one-bullet exchange
            # scores 100% and a thirty-round spray that lands ten scores 33%;
            # treating those as two equal observations put the median 20 points
            # above the rate the player actually shot at.
            "pooled_pct": round(total_acc_hits / total_acc_shots * 100, 1)
            if total_acc_shots else None,
            "total_hits": total_acc_hits,
            "total_shots": total_acc_shots,
            "values": [round(v, 1) for v in accuracy_values],
            "outcomes": outcomes_acc,
            "first_bullet_pct": round(fb_hit / n_acc * 100, 1) if n_acc else 0,
            "head_pct": round(hitgroup_head / hitgroup_total * 100, 1) if hitgroup_total else 0,
            "upper_pct": round(hitgroup_upper / hitgroup_total * 100, 1) if hitgroup_total else 0,
            "lower_pct": round(hitgroup_lower / hitgroup_total * 100, 1) if hitgroup_total else 0,
            "head_count": hitgroup_head,
            "upper_count": hitgroup_upper,
            "lower_count": hitgroup_lower,
        }

    # ------------------------------------------------------------------ #
    # Aim rating (0-100)                                                   #
    #                                                                      #
    # Each component scores off the *median* of its samples, then carries  #
    # weight proportional to how much evidence stands behind it.  Absent   #
    # components are dropped and the remaining weights renormalised —      #
    # previously they were filled in at 50, which meant a match where a    #
    # metric could not be measured was rated partly on a constant.  That   #
    # is untenable once these weights become user-adjustable: someone who  #
    # cares mostly about one metric would be scored on a placeholder.      #
    # ------------------------------------------------------------------ #
    components: list[tuple[str, float, int]] = []

    preaim_median = _median(preaim_errors)
    if preaim_median is not None:
        # 0° = 100, 20°+ = 0
        components.append((
            "preaim",
            max(0.0, min(100.0, 100.0 - preaim_median * 5.0)),
            len(preaim_errors),
        ))

    if n_mov:
        good_mov = sum(1 for q in movement_qualities if q != "running")
        components.append(("movement", good_mov / n_mov * 100.0, n_mov))

    ttk_median = _median([v for v in ttk_values if v < _TTK_OUTLIER_SECONDS])
    if ttk_median is not None:
        # 0.15s = 100, 0.8s+ = 0
        components.append((
            "ttk",
            max(0.0, min(100.0, (0.8 - ttk_median) / 0.65 * 100.0)),
            len([v for v in ttk_values if v < _TTK_OUTLIER_SECONDS]),
        ))

    aim_rating: float | None = None
    rating_inputs: list[dict[str, Any]] = []
    if components:
        weighted_sum = 0.0
        total_weight = 0.0
        for name, score, n in components:
            weight = _AIM_RATING_WEIGHTS[name] * (n / (n + _CONFIDENCE_K))
            weighted_sum += score * weight
            total_weight += weight
            rating_inputs.append({
                "metric": name,
                "score": round(score, 1),
                "n": n,
                "weight": round(weight, 4),
            })
        if total_weight > 0:
            aim_rating = round(min(100.0, max(0.0, weighted_sum / total_weight)), 1)
            for item in rating_inputs:
                item["weight_share"] = round(item["weight"] / total_weight, 3)

    return {
        "thresholds": _AIM_THRESHOLDS,
        "aim_rating_inputs": rating_inputs,
        "movement": movement,
        "peek": peek,
        "preaim": preaim,
        "ttk": ttk,
        "reaction": reaction,
        "accuracy": accuracy,
        "aim_rating": aim_rating,
        "encounters": encounters,
    }


# A damage encounter counts as a lost duel ("death" outcome) when the player
# dies to that same enemy within this many ticks of last hitting them.
_LOST_DUEL_WINDOW = 160  # ≈ 2.5 s at 64-tick
