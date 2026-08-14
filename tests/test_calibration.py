"""Tests for the numbers that came from outside the code.

The win-probability table is the one that matters. Its comment argued the table
was trustworthy because certain properties *fell out of the data* rather than
being imposed on it: an even round landing on a coin flip, man advantage
monotone in both directions, planting moving every state toward the T side.

Writing those down as tests showed two of them were overstated. Planting
inverts in five thin cells, and the planted states are not monotone in man
advantage because 5v1 and 5v2 were never observed with a plant and fall back to
a formula. Both exceptions are named below rather than papered over, and the
module comment now says the same thing.

What remains is a real guard: recalibrating against a bad corpus fails loudly
instead of quietly changing every impact score in the app.
"""

from __future__ import annotations

import pytest

from src.domain.calibration import (
    HLTV_COEFFICIENTS,
    WIN_PROB,
    hltv_rating,
    win_probability,
)

# ---------------------------------------------------------------------------
# Win probability — the claimed invariants
# ---------------------------------------------------------------------------


def test_an_even_round_is_a_coin_flip():
    """Claimed: 5v5 no bomb comes out at 0.501 on 793 observations."""
    assert 0.45 < win_probability(5, 5, False) < 0.55


@pytest.mark.parametrize("alive", [4, 3, 2])
def test_every_even_state_is_near_a_coin_flip(alive):
    """Claimed: 4v4, 3v3 and 2v2 all sit within a couple of points of 0.500."""
    assert 0.42 < win_probability(alive, alive, False) < 0.58


def test_unplanted_states_are_monotone_in_man_advantage():
    """More CTs alive never makes the CT side worse off, and vice versa."""
    for t_alive in range(1, 6):
        values = [win_probability(ct, t_alive, False) for ct in range(1, 6)]
        assert values == sorted(values), f"not monotone in CTs at t={t_alive}"
    for ct_alive in range(1, 6):
        values = [win_probability(ct_alive, t, False) for t in range(1, 6)]
        assert values == sorted(values, reverse=True), f"not monotone in Ts at ct={ct_alive}"


def test_planted_states_are_monotone_in_the_t_direction():
    """Fewer Ts alive never makes the CT side worse off, plant or no plant."""
    for ct_alive in range(1, 6):
        values = [win_probability(ct_alive, t, True) for t in range(1, 6)]
        assert values == sorted(values, reverse=True), f"not monotone in Ts at ct={ct_alive}"


# 5v1 and 5v2 were never observed with a plant, so they fall back to the
# unplanted value minus a tenth. 4v1 planted is measured at 1.000 off eleven
# observations, which puts the 5v1 fallback below it. Named here so the
# exception is a decision rather than a gap in the coverage.
_UNMEASURED_PLANTED = {(5, 1), (5, 2)}


def test_planted_states_are_monotone_in_ct_advantage_where_measured():
    for t_alive in range(1, 6):
        values = [
            win_probability(ct, t_alive, True)
            for ct in range(1, 6)
            if (ct, t_alive) not in _UNMEASURED_PLANTED
        ]
        assert values == sorted(values), f"not monotone in CTs at t={t_alive}"


def test_planting_moves_the_round_toward_the_t_side_on_average():
    """True in aggregate and in every well-sampled cell. Five thin cells invert
    by a point or two, which is noise at those counts — see the module comment."""
    deltas = [
        win_probability(ct, t, True) - win_probability(ct, t, False)
        for ct in range(1, 6)
        for t in range(1, 6)
    ]
    assert sum(deltas) / len(deltas) < -0.05, "planting should favour the T side"


@pytest.mark.parametrize("state", [(5, 5), (3, 3), (2, 2), (3, 2), (2, 3)])
def test_planting_favours_the_t_side_in_the_well_sampled_cells(state):
    ct, t = state
    planted, unplanted = win_probability(ct, t, True), win_probability(ct, t, False)
    assert planted <= unplanted + 0.02, f"plant helped CT at {ct}v{t}"


def test_a_decided_round_is_not_a_probability():
    assert win_probability(1, 0, False) == 1.0
    assert win_probability(0, 1, False) == 0.0
    assert win_probability(0, 0, False) == 0.0


def test_every_value_is_a_probability():
    for state, value in WIN_PROB.items():
        assert 0.0 <= value <= 1.0, f"{state} is not a probability"


def test_unmeasured_states_still_return_something_sensible():
    """Rare states fall back rather than dropping the kill out of the swing."""
    assert win_probability(5, 5, False) < win_probability(5, 1, True) <= 1.0
    assert win_probability(2, 5, False) < win_probability(5, 2, False)


@pytest.mark.parametrize("state", sorted(WIN_PROB))
def test_the_table_is_keyed_the_way_it_is_read(state):
    ct, t, planted = state
    assert 1 <= ct <= 5 and 1 <= t <= 5
    assert planted in (0, 1)
    assert win_probability(ct, t, bool(planted)) == WIN_PROB[state]


# ---------------------------------------------------------------------------
# HLTV rating
# ---------------------------------------------------------------------------


def test_hltv_rating_uses_every_coefficient():
    """A dropped term would still produce a plausible-looking number."""
    base = dict(kast=70.0, kpr=0.7, dpr=0.65, impact=1.2, adr=80.0)
    baseline = hltv_rating(**base)
    for field in base:
        moved = hltv_rating(**{**base, field: base[field] + 10})
        assert moved != baseline, f"{field} does not affect the rating"


def test_deaths_are_the_only_term_that_hurts():
    base = dict(kast=70.0, kpr=0.7, dpr=0.65, impact=1.2, adr=80.0)
    baseline = hltv_rating(**base)
    assert hltv_rating(**{**base, "dpr": 0.9}) < baseline
    for better in ("kast", "kpr", "impact", "adr"):
        assert hltv_rating(**{**base, better: base[better] * 1.5}) > baseline


def test_an_empty_performance_is_near_the_intercept():
    assert hltv_rating(0, 0, 0, 0, 0) == pytest.approx(
        HLTV_COEFFICIENTS["intercept"], abs=1e-6
    )


def test_a_typical_performance_lands_in_the_expected_band():
    """Roughly average competitive numbers should read close to 1.0."""
    assert 0.85 < hltv_rating(kast=70.0, kpr=0.68, dpr=0.66, impact=1.1, adr=76.0) < 1.20
