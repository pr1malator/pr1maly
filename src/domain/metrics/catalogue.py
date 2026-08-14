"""What each number means, where it comes from, and how much to trust it.

The registry next door catalogues five *measurements* — the units that get
versioned and recomputed. One analysis produces about a hundred and eighty
numbers, and until now the only place their meaning existed was inside the
function that computed them. The app then showed them with a verdict attached,
which is the part that matters: a player reading "NEEDS WORK" on their crosshair
placement deserves to know what was measured, over how many samples, and
whether the line they fell below was measured from real players or picked by
hand.

So each figure the interface shows gets an entry here: what it measures, how it
is derived, and — where a tier is shown against it — where that tier came from.
One table, read by the API, the interface, and the generated reference in
METRICS.md, so those three cannot drift.

Deliberately at the level of the figure, not the leaf. ``aim_stats.reaction``
is described once; its median, average, range, sample size and confidence are
the same five statistics that every figure carries, explained in STATISTICS
below rather than thirty times over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Where a threshold or a formula came from. This is the distinction the
# interface has never made and should: a tier is either something observed
# about real players, something a third party published, or a line someone drew
# because a line was needed.
MEASURED = "measured"
PUBLISHED = "published"
HEURISTIC = "heuristic"

PROVENANCE: dict[str, str] = {
    MEASURED: (
        "Taken from a real corpus of matches. The numbers behind it can be "
        "regenerated, and the observation count for each cell is recorded."
    ),
    PUBLISHED: (
        "A formula published by someone else, reproduced as specified. Not ours "
        "to change, and not calibrated against this player."
    ),
    HEURISTIC: (
        "These are sensible targets, not percentiles of a player population. A "
        "tier here says where you fall against a line somebody chose, not "
        "against other players."
    ),
}

# Every figure that reports a distribution carries the same five, so they are
# described once instead of on each entry.
STATISTICS: dict[str, str] = {
    "median": "The headline figure. Used because one outlier round cannot move it.",
    "avg": "The mean, kept alongside the median: a wide gap between them means a skewed match.",
    "min": "The best single sample in the match.",
    "max": "The worst single sample in the match.",
    "n": "How many samples the figure is built from.",
    "confidence": (
        "How much weight the sample count supports — low, medium or high. A "
        "figure from four duels is not the same claim as one from forty."
    ),
}


@dataclass(frozen=True)
class FieldSpec:
    """One figure the interface shows."""

    key: str
    """Dotted path into a stored analysis, e.g. ``aim_stats.reaction``."""

    label: str
    metric: str
    """The registry metric that produces it, or ``core`` for the scoreboard."""

    measures: str
    """What the number is, in one sentence."""

    derived: str
    """How it is arrived at, in one sentence."""

    unit: str = ""
    tiers: str | None = None
    """Provenance of the tier or grade shown against it, if any."""

    note: str = ""
    """Anything a reader would otherwise get wrong."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "metric": self.metric,
            "measures": self.measures,
            "derived": self.derived,
            "unit": self.unit,
            "tiers": self.tiers,
            "tiers_meaning": PROVENANCE.get(self.tiers or "", ""),
            "note": self.note,
        }


FIELDS: tuple[FieldSpec, ...] = (
    # -- The scoreboard ----------------------------------------------------
    FieldSpec(
        key="hltv_rating",
        label="HLTV 2.0 Rating",
        metric="core",
        unit="",
        measures="Overall contribution across the match, on the scale where 1.00 is par.",
        derived=(
            "HLTV's published 2.0 formula over kills, deaths, KAST, ADR and "
            "impact, per round."
        ),
        tiers=PUBLISHED,
        note=(
            "The coefficients are HLTV's, applied to matchmaking demos rather "
            "than the professional matches they were fitted on."
        ),
    ),
    FieldSpec(
        key="adr",
        label="Average Damage per Round",
        metric="core",
        unit="damage",
        measures="Damage dealt to enemies per round played.",
        derived=(
            "Damage capped at the health the victim actually had, so overkill "
            "does not inflate it, then divided by rounds."
        ),
    ),
    FieldSpec(
        key="kast",
        label="KAST",
        metric="core",
        unit="%",
        measures="Share of rounds in which you got a kill, an assist, survived, or were traded.",
        derived="Rounds meeting any of the four conditions, over rounds played.",
    ),
    FieldSpec(
        key="kpr",
        label="Kills per Round",
        metric="core",
        unit="",
        measures="Kills divided by rounds played.",
        derived="Total kills over total rounds.",
    ),
    FieldSpec(
        key="dpr",
        label="Deaths per Round",
        metric="core",
        unit="",
        measures="Deaths divided by rounds played.",
        derived="Total deaths over total rounds.",
    ),
    FieldSpec(
        key="impact",
        label="Impact",
        metric="core",
        unit="",
        measures="HLTV's impact term: opening duels and multi-kills weighted above ordinary trades.",
        derived="The published impact expression over kills per round, opening kills and multi-kill rounds.",
        tiers=PUBLISHED,
    ),
    FieldSpec(
        key="round_stats",
        label="Round timeline",
        metric="core",
        unit="",
        measures="Your kills, deaths, assists, damage and survival, round by round.",
        derived="One row per round, taken from the events attributed to you in that round.",
        note=(
            "The source every match-level figure above is aggregated from, and "
            "what the round timeline in the interface draws."
        ),
    ),
    # -- Aim ---------------------------------------------------------------
    FieldSpec(
        key="aim_stats.aim_rating",
        label="Aim Rating",
        metric="aim.stats",
        unit="/100",
        measures="A single figure for mechanical quality in duels.",
        derived=(
            "Crosshair placement 40%, shot speed 30%, engagement time 30%, each "
            "scored against its own tier bounds."
        ),
        tiers=HEURISTIC,
        note=(
            "Reaction time and counter-strafe rate are deliberately not inputs. "
            "Reaction has too few samples per match to grade, and counter-strafe "
            "would score movement twice — shot speed already measures the outcome."
        ),
    ),
    FieldSpec(
        key="aim_stats.movement",
        label="Shot Speed",
        metric="aim.stats",
        unit="u/s",
        measures="How fast you were moving at the moment you fired the killing shot.",
        derived="Your own velocity on the tick of the shot, taken across every kill.",
        tiers=HEURISTIC,
        note="Rifles lose accuracy above roughly a third of running speed, which is where the tiers sit.",
    ),
    FieldSpec(
        key="aim_stats.peek",
        label="Peek Speed",
        metric="aim.stats",
        unit="u/s",
        measures="How fast you were travelling in the half-second before firing.",
        derived="Median velocity over the 0.5s window preceding the shot.",
        note=(
            "Deliberately ungraded. The bands name what kind of peek it was — "
            "held, walk, half speed, full speed — rather than ranking them; a "
            "held angle and a fast peek are different choices, not better and "
            "worse ones."
        ),
    ),
    FieldSpec(
        key="aim_stats.preaim",
        label="Crosshair Placement",
        metric="aim.stats",
        unit="°",
        measures="How far your crosshair was from the enemy when the duel began.",
        derived="Angle between your view direction and the enemy at the moment they became engageable.",
        tiers=HEURISTIC,
        note="The largest single input to the aim rating, because it is the one most under your control.",
    ),
    FieldSpec(
        key="aim_stats.ttk",
        label="Engagement Time",
        metric="aim.stats",
        unit="s",
        measures="How long a duel took from your first damage to the kill.",
        derived="Time between first damage dealt and the killing blow, outliers excluded and counted separately.",
        tiers=HEURISTIC,
    ),
    FieldSpec(
        key="aim_stats.reaction",
        label="Reaction Time",
        metric="aim.stats",
        unit="ms",
        measures="How quickly you fired once your crosshair was on the enemy.",
        derived=(
            "Walks back from the first shot to the last tick your crosshair was "
            "off the enemy; the gap between that and the shot is the reaction."
        ),
        tiers=HEURISTIC,
        note=(
            "Diagnostic only, and never an input to the aim rating. Two to "
            "twenty samples a match, and what is measured is the gap between "
            "your crosshair arriving on the enemy and your shot — which is not "
            "the same as how fast you reacted, because an enemy can walk into a "
            "crosshair you were already holding."
        ),
    ),
    FieldSpec(
        key="aim_stats.accuracy",
        label="Accuracy",
        metric="aim.stats",
        unit="%",
        measures="Share of your shots that hit, with the head and lower-body split.",
        derived="Hits over shots fired in duels, broken down by hitgroup.",
        tiers=HEURISTIC,
    ),
    FieldSpec(
        key="aim_stats.encounters",
        label="Duels",
        metric="aim.stats",
        unit="",
        measures="The individual duels every aim figure above is built from.",
        derived="One record per engagement, carrying its weapon, distance, and each measurement taken.",
        note="This is the raw material: any aim figure can be traced back to the duels behind it.",
    ),
    FieldSpec(
        key="aim_stats.thresholds",
        label="Aim tier bounds",
        metric="aim.stats",
        unit="",
        measures="The cut-offs each aim figure is graded against, and which direction is better.",
        derived="Hand-set bounds, shipped with the analysis so the interface grades against the same lines the backend used.",
        tiers=HEURISTIC,
    ),
    # -- Utility -----------------------------------------------------------
    FieldSpec(
        key="utility_data.utility_rating",
        label="Utility Rating",
        metric="utility.stats",
        unit="/100",
        measures="A single figure for how well your grenades were spent.",
        derived="Combines flash effectiveness, damage from HE and molotov, and how much utility went unused.",
        tiers=HEURISTIC,
    ),
    FieldSpec(
        key="utility_data.flash",
        label="Flashes",
        metric="utility.stats",
        unit="",
        measures="Enemies blinded, for how long, how often a flash achieved anything, and flash assists.",
        derived="Blind events attributed to your flashes, with duration and whether a kill followed.",
    ),
    FieldSpec(
        key="utility_data.he",
        label="HE Grenades",
        metric="utility.stats",
        unit="damage",
        measures="Damage dealt by your HE grenades and how often they landed.",
        derived="Damage attributed to HE detonations, over grenades thrown.",
        tiers=HEURISTIC,
    ),
    FieldSpec(
        key="utility_data.molotov",
        label="Molotovs",
        metric="utility.stats",
        unit="damage",
        measures="Damage dealt by your incendiaries and how often they landed.",
        derived="Damage attributed to fire you started, over molotovs thrown.",
        tiers=HEURISTIC,
    ),
    FieldSpec(
        key="utility_data.smoke",
        label="Smokes",
        metric="utility.stats",
        unit="",
        measures="Where you smoke, and how often a smoke put out a molotov.",
        derived="Smoke detonation positions resolved to callouts, plus extinguish events.",
    ),
    FieldSpec(
        key="utility_data.economics",
        label="Utility Economy",
        metric="utility.stats",
        unit="$",
        measures="What you spent on grenades and how much of it was never thrown.",
        derived="Buy value of utility against what was used, per match.",
        tiers=HEURISTIC,
        note="Utility you died holding counts as wasted — the money bought nothing.",
    ),
    FieldSpec(
        key="utility_data.teamplayer",
        label="Team Support",
        metric="utility.stats",
        unit="",
        measures="Flashes and drops that helped teammates, and damage you did to them.",
        derived="Flash assists and weapon drops for teammates, against team damage and team flashes.",
    ),
    FieldSpec(
        key="utility_data.per_round",
        label="Utility by round",
        metric="utility.stats",
        unit="",
        measures="The same utility figures, round by round.",
        derived="One entry per round, so a single expensive round can be told from a pattern.",
    ),
    # -- Impact ------------------------------------------------------------
    FieldSpec(
        key="impact_stats.net_swing_per_round",
        label="Net Impact per Round",
        metric="impact.stats",
        unit="win probability",
        measures="How much your kills and deaths moved your team's chance of winning, per round.",
        derived=(
            "Every kill and death is priced by how the round's win probability "
            "changes with the player count, and the two are netted off."
        ),
        tiers=MEASURED,
        note=(
            "The win-probability table is measured from real matches and its "
            "observation count is recorded per cell; thinly observed states "
            "fall back to a formula."
        ),
    ),
    FieldSpec(
        key="impact_stats.kill_swing_total",
        label="Impact won",
        metric="impact.stats",
        unit="win probability",
        measures="Total win probability your kills added across the match.",
        derived="Sum of the swing attributed to each kill.",
        tiers=MEASURED,
    ),
    FieldSpec(
        key="impact_stats.death_swing_total",
        label="Impact lost",
        metric="impact.stats",
        unit="win probability",
        measures="Total win probability your deaths cost across the match.",
        derived="Sum of the swing attributed to each death.",
        tiers=MEASURED,
    ),
    FieldSpec(
        key="impact_stats.best_kill_swing",
        label="Best kill",
        metric="impact.stats",
        unit="win probability",
        measures="The single kill that moved the round most.",
        derived="Largest win-probability swing among your kills.",
        tiers=MEASURED,
    ),
    FieldSpec(
        key="impact_stats.median_kill_swing",
        label="Typical kill",
        metric="impact.stats",
        unit="win probability",
        measures="What a normal kill of yours was worth this match.",
        derived="Median swing across your kills, so one clutch does not set the figure.",
        tiers=MEASURED,
    ),
    FieldSpec(
        key="impact_stats.per_round",
        label="Impact by round",
        metric="impact.stats",
        unit="",
        measures="The same swing figures, round by round.",
        derived="One entry per round, with the kills and deaths that produced it.",
        tiers=MEASURED,
    ),
    # -- Roles -------------------------------------------------------------
    FieldSpec(
        key="role_data.ct_primary",
        label="Primary CT role",
        metric="roles.positional",
        unit="",
        measures="Where you actually played on defence.",
        derived=(
            "Your position at round start is resolved to a map callout, and the "
            "callouts are scored against the role definitions for that map."
        ),
        note=(
            "Positional, not tactical: it says where you stood, which is not the "
            "same as what your team asked you to do."
        ),
    ),
    FieldSpec(
        key="role_data.t_primary",
        label="Primary T role",
        metric="roles.positional",
        unit="",
        measures="Where you actually played on attack.",
        derived="The same zone scoring, against the attacking role definitions for that map.",
    ),
    FieldSpec(
        key="role_data.roles_ct",
        label="CT role split",
        metric="roles.positional",
        unit="rounds",
        measures="How your defensive rounds divided between roles.",
        derived="Rounds attributed to each role, in order of frequency.",
    ),
    FieldSpec(
        key="role_data.roles_t",
        label="T role split",
        metric="roles.positional",
        unit="rounds",
        measures="How your attacking rounds divided between roles.",
        derived="Rounds attributed to each role, in order of frequency.",
    ),
    # -- Benchmarks --------------------------------------------------------
    FieldSpec(
        key="benchmarks.engagement_ttk",
        label="Engagement time benchmark",
        metric="aim.stats",
        unit="ms",
        measures="Your engagement time placed in a tier.",
        derived="The median engagement time compared against hand-set cut-offs.",
        tiers=HEURISTIC,
    ),
    FieldSpec(
        key="benchmarks.enemies_flashed",
        label="Enemies flashed benchmark",
        metric="utility.stats",
        unit="per 24 rounds",
        measures="How many enemies you blinded, placed in a tier.",
        derived="Enemies flashed scaled to a 24-round match, against per-map cut-offs.",
        tiers=HEURISTIC,
        note="The cut-offs differ by map because the maps do — Dust2 rewards flashes Inferno does not.",
    ),
    FieldSpec(
        key="benchmarks.utility_damage",
        label="Utility damage benchmark",
        metric="utility.stats",
        unit="damage",
        measures="Damage from HE and molotov, placed in a tier.",
        derived="Total utility damage scaled to a 24-round match, against hand-set cut-offs.",
        tiers=HEURISTIC,
    ),
    FieldSpec(
        key="benchmarks.utility_waste_pct",
        label="Utility waste benchmark",
        metric="utility.stats",
        unit="%",
        measures="Share of bought utility you never threw, placed in a tier.",
        derived="Unused utility value over utility bought, against hand-set cut-offs.",
        tiers=HEURISTIC,
    ),
)

_BY_KEY = {field.key: field for field in FIELDS}


def describe(key: str) -> FieldSpec | None:
    return _BY_KEY.get(key)


def keys() -> list[str]:
    return [field.key for field in FIELDS]


def for_metric(metric_id: str) -> list[FieldSpec]:
    return [field for field in FIELDS if field.metric == metric_id]


def as_dicts() -> list[dict[str, Any]]:
    return [field.as_dict() for field in FIELDS]
