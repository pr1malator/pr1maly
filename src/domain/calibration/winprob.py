"""P(CT wins the round | ct_alive, t_alive, bomb_planted).

Measured, not assumed. The counts beside each cell are the observations behind
it, and the invariants the comment below claims are enforced by
tests/test_calibration.py — so a recalibration against a bad corpus fails
rather than silently changing every impact score.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Round win probability
#
# P(CT wins the round | ct_alive, t_alive, bomb_planted), measured from this
# installation's own demo corpus: 40 demos, ~800 rounds, 6045 observed states.
# Every state below has at least 8 observations behind it.
#
# The table validates against known truths, which is the main reason to trust
# it: 5v5 with no bomb comes out at 0.501 on 793 observations, the even states
# (4v4, 3v3, 2v2) all sit within a couple of points of 0.500, and the unplanted
# states are monotone in man advantage in both directions. None of that was
# imposed — it fell out of the data.
#
# Two claims that were made here before and do NOT survive checking, recorded so
# nobody re-derives them from the comment:
#
#   Planting does not move *every* state toward the T side. It does on average
#   and in every well-sampled cell, but five thin cells invert by a point or
#   two — 1v3 (n=108), 1v4 (n=74), 4v1 (n=11), 4v5 (n=28) and 5v5 (n=39). That
#   is sampling noise at those counts, not a finding about the game.
#
#   The planted states are not monotone in man advantage, because 5v1 and 5v2
#   were never observed with a plant and fall back to the unplanted value minus
#   a tenth. With 4v1 measured at 1.000 off eleven observations, the fallback
#   for 5v1 lands below it.
#
# tests/test_calibration.py encodes what does hold, including those exceptions
# by name, so a recalibration against a bad corpus fails rather than quietly
# changing every impact score in the app.
#
# It is one player's matchmaking pool, so it encodes that pool's tendencies.
# Recalibrate with scripts/calibrate_winprob.py as the corpus grows.
# ---------------------------------------------------------------------------
WIN_PROB: dict[tuple[int, int, int], float] = {
    (1, 1, 0): 0.667,       # n=51
    (1, 1, 1): 0.299,       # n=107
    (1, 2, 0): 0.298,       # n=57
    (1, 2, 1): 0.099,       # n=142
    (1, 3, 0): 0.039,       # n=51
    (1, 3, 1): 0.056,       # n=108
    (1, 4, 0): 0.000,       # n=40
    (1, 4, 1): 0.027,       # n=74
    (1, 5, 0): 0.000,       # n=26
    (1, 5, 1): 0.000,       # n=27
    (2, 1, 0): 0.828,       # n=122
    (2, 1, 1): 0.683,       # n=79
    (2, 2, 0): 0.513,       # n=148
    (2, 2, 1): 0.387,       # n=119
    (2, 3, 0): 0.290,       # n=145
    (2, 3, 1): 0.137,       # n=95
    (2, 4, 0): 0.123,       # n=114
    (2, 4, 1): 0.077,       # n=78
    (2, 5, 0): 0.068,       # n=74
    (2, 5, 1): 0.030,       # n=33
    (3, 1, 0): 0.956,       # n=137
    (3, 1, 1): 0.939,       # n=33
    (3, 2, 0): 0.738,       # n=210
    (3, 2, 1): 0.727,       # n=66
    (3, 3, 0): 0.498,       # n=257
    (3, 3, 1): 0.456,       # n=68
    (3, 4, 0): 0.315,       # n=232
    (3, 4, 1): 0.238,       # n=63
    (3, 5, 0): 0.141,       # n=170
    (3, 5, 1): 0.038,       # n=26
    (4, 1, 0): 0.991,       # n=109
    (4, 1, 1): 1.000,       # n=11
    (4, 2, 0): 0.921,       # n=178
    (4, 2, 1): 0.864,       # n=22
    (4, 3, 0): 0.688,       # n=282
    (4, 3, 1): 0.684,       # n=38
    (4, 4, 0): 0.520,       # n=369
    (4, 4, 1): 0.468,       # n=47
    (4, 5, 0): 0.310,       # n=364
    (4, 5, 1): 0.321,       # n=28
    (5, 1, 0): 1.000,       # n=51
    (5, 2, 0): 0.961,       # n=102
    (5, 3, 0): 0.812,       # n=208
    (5, 3, 1): 0.750,       # n=12
    (5, 4, 0): 0.672,       # n=415
    (5, 4, 1): 0.640,       # n=25
    (5, 5, 0): 0.501,       # n=793
    (5, 5, 1): 0.513,       # n=39
}


def win_probability(ct_alive: int, t_alive: int, bomb_planted: bool) -> float:
    """P(CT wins) for a round state, falling back when the state is unmeasured.

    Falls back to the same man-count without the bomb, and finally to a plain
    function of the man advantage, so that rare states still return something
    monotone rather than dropping the kill out of the swing calculation.
    """
    if ct_alive <= 0:
        return 0.0
    if t_alive <= 0:
        return 1.0

    planted = 1 if bomb_planted else 0
    exact = WIN_PROB.get((ct_alive, t_alive, planted))
    if exact is not None:
        return exact

    no_bomb = WIN_PROB.get((ct_alive, t_alive, 0))
    if no_bomb is not None:
        # Planting is worth roughly a tenth of a round to the T side across the
        # measured states; apply that shift when the planted cell is missing.
        return max(0.0, min(1.0, no_bomb - 0.10)) if planted else no_bomb

    diff = ct_alive - t_alive
    return max(0.0, min(1.0, 0.5 + 0.18 * diff - (0.10 if planted else 0.0)))
