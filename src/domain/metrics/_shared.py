"""Small pieces more than one metric needs.

Kept apart from the metrics themselves so that moving a measurement does not
drag a threshold table three other measurements also read. Nothing here is a
metric: these are the shared vocabulary the metrics are written in.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Peek-speed bands: was the counter-strafe held together at speed, or only on
# the slow peeks where there was time for it?  They label the regions of the
# peek-speed axis on the charts as well as splitting the counter-strafe rate.
#
# The floor is _ACCURATE_SPEED — below it the player never had to stop at all,
# and no engagement enters the counter-strafe denominator.  (Written as a
# literal because _ACCURATE_SPEED is defined further down and so cannot be
# referenced at import time; a test pins the two together.)
#
# The upper split is at 180 u/s.  A rifle caps a player at 215-225 rather than
# the 250 a knife allows, so 180 is roughly 80% of the fastest a peek with an
# AK or M4 in hand can be — the range where the stop has to be timed to the
# tick.  The lower split at 130 sits just above shift-walk speed (~112), which
# separates a peek taken at a walk from one taken with real commitment.
_PEEK_BUCKETS: list[tuple[str, str, float, float]] = [
    ("walk", "Walk", 85.0, 130.0),
    ("half", "Half speed", 130.0, 180.0),
    ("full", "Full speed", 180.0, float("inf")),
]

# ---------------------------------------------------------------------------
# Aim metric bands — the single source of truth
#
# These bounds were previously written down three times: once as the per-kill
# quality buckets, once as the benchmark tiers, and once again in the scatter
# plot's JavaScript.  They had drifted apart — crosshair placement was bucketed
# at 5/10/20 but graded at 3/10/25, and the scatter drew bands at yet another
# set — so the same number could be called "good" on one card and "fair" three
# inches to the right.  Everything now reads from here, and the table is
# shipped to the frontend in ``aim_stats["thresholds"]`` so nothing has to
# duplicate it.
#
# ``range`` fixes the axis a chart draws on. Scaling to each match's own
# spread meant an identical distribution looked different from game to
# game, and a good match and a bad one drew the same picture — the only
# cue left was colour. A fixed span makes the shape itself comparable.
#
# The values remain hand-set heuristics; see the note on compute_benchmarks.
# ---------------------------------------------------------------------------
_AIM_THRESHOLDS: dict[str, dict[str, Any]] = {
    "movement": {
        "label": "Shot Speed", "unit": "u/s",
        "bounds": [15, 40, 100], "lower_better": True,
        "range": [0, 250],
    },
    # Peek speed ships with no bounds because it carries no verdict on its own:
    # approaching slowly is right when holding an angle and wrong when entering
    # a site, and nothing in the demo says which one the player meant to do.
    # Charts therefore draw it as a plain axis, and whatever grading appears on
    # a chart it is plotted against comes from the *other* metric — which is the
    # comparison that actually means something.
    #
    # ``zones`` name the regions of that axis instead.  They are categories,
    # not grades — which kind of peek this was, at the same speeds the
    # counter-strafe cross-tab splits on, so a point's position on the chart
    # and its row in the breakdown mean the same thing.  Charts must colour
    # them as a speed ramp, never in the tier palette.
    "peek": {
        "label": "Peek Speed", "unit": "u/s",
        "bounds": [], "lower_better": True,
        "range": [0, 250],
        "zones": [{"at": 0.0, "label": "Held"}] + [
            {"at": lo, "label": label} for _key, label, lo, _hi in _PEEK_BUCKETS
        ],
    },
    # The middle bound must stay equal to _COUNTERSTRAFE_MAX_TICKS, which is
    # defined further down and so cannot be referenced at import time.  A test
    # pins the two together.
    "stop_ticks": {
        "label": "Counter-strafe", "unit": "ticks",
        "bounds": [3, 7, 15], "lower_better": True,
        "range": [0, 32],
    },
    # Not a per-encounter measurement like the rest — an encounter is either a
    # clean stop or it is not — so this never appears on the scatter.  It lives
    # here so the benchmark badge and the per-peek-speed breakdown grade the
    # rate against one set of numbers instead of two.
    "counterstrafe": {
        "label": "Counter-strafe Rate", "unit": "%",
        "bounds": [80, 60, 35], "lower_better": False,
        "range": [0, 100],
    },
    "preaim": {
        "label": "Crosshair Placement", "unit": "°",
        "bounds": [5, 10, 20], "lower_better": True,
        "range": [0, 45],
    },
    "ttk": {
        "label": "Engagement Time", "unit": "s",
        "bounds": [0.4, 0.65, 1.1], "lower_better": True,
        "range": [0, 1.0],
    },
    "reaction": {
        "label": "Reaction Time", "unit": "ms",
        "bounds": [150, 200, 300], "lower_better": True,
        "range": [0, 800],
    },
    "accuracy": {
        "label": "Accuracy", "unit": "%",
        "bounds": [75, 50, 30], "lower_better": False,
        "range": [0, 100],
    },
}

def _aim_bounds(metric: str) -> list[float]:
    return _AIM_THRESHOLDS[metric]["bounds"]

# Shrinkage constant for confidence weighting: a metric measured n times
# carries n / (n + k) of its nominal weight, so a two-sample estimate barely
# moves the rating while a twenty-sample one counts almost fully.  Standard
# empirical-Bayes shrinkage; k ≈ 12 puts the half-way point at a dozen
# engagements, which is roughly what a normal match produces per metric.
_CONFIDENCE_K = 12

def _median(values: list[float]) -> float | None:
    """Median of *values*, or None when empty.

    Used instead of the mean throughout the aim aggregates: these samples are
    few and have a long right tail, so one bad engagement drags a mean well
    away from typical behaviour.
    """
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0

def _confidence(n: int) -> str:
    """Coarse label for how much a sample of *n* engagements can carry."""
    if n >= 20:
        return "high"
    if n >= 8:
        return "medium"
    return "low"

def _find_id_col(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Return the first column from *candidates* that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _build_round_team_map(
    death_df: pd.DataFrame,
    steam_id: str,
    round_end_df: pd.DataFrame,
) -> dict[int, str]:
    """Build a mapping of round number → player team ("CT" or "T").

    Inspects kill/death events to find the player's ``team_num`` for each
    round they appear in.  For rounds with no player events (survived
    without kills), the team is carried forward from the last known round.
    """
    _TEAM_MAP = {2: "T", 3: "CT"}
    if death_df.empty or "round" not in death_df.columns:
        return {}

    sid = str(steam_id)
    round_team: dict[int, str] = {}

    # Collect team observations from kill events (as attacker) and death
    # events (as victim).  Prefer earlier ticks for each round.
    for id_col, team_col in [
        ("attacker_steamid", "attacker_team_num"),
        ("user_steamid", "user_team_num"),
    ]:
        if id_col not in death_df.columns or team_col not in death_df.columns:
            continue
        mask = death_df[id_col].astype(str) == sid
        subset = death_df.loc[mask, ["round", team_col]].dropna()
        for _, row in subset.iterrows():
            rnd = int(row["round"])
            if rnd in round_team:
                continue  # already have data for this round
            team = _TEAM_MAP.get(int(row[team_col]))
            if team:
                round_team[rnd] = team

    if not round_team:
        return {}

    # Fill gaps: for rounds where the player had no events, carry forward
    # from the last known round (team only changes at halftime boundaries).
    all_rounds = sorted(int(r) for r in round_end_df["round"].dropna().unique())
    known_sorted = sorted(round_team.keys())
    last_known: str | None = None
    for rnd in all_rounds:
        if rnd in round_team:
            last_known = round_team[rnd]
        elif last_known is not None:
            round_team[rnd] = last_known

    # Back-fill: if the first few rounds were missing, fill backward from
    # the first known round.
    if all_rounds and all_rounds[0] not in round_team:
        first_known = round_team[known_sorted[0]]
        for rnd in all_rounds:
            if rnd in round_team:
                break
            round_team[rnd] = first_known

    return round_team
