"""The HLTV 2.0 rating approximation.

Published coefficients from a third party, reproduced rather than derived. They
go stale on someone else's schedule, which is the reason they live here with
the other numbers we did not choose.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# HLTV 2.0 Rating formula coefficients (publicly documented approximation).
# Source: https://www.hltv.org/news/20695/introducing-rating-20
# ---------------------------------------------------------------------------
HLTV_COEFFICIENTS = {
    "kast_weight": 0.0073,
    "kpr_weight": 0.3591,
    "dpr_weight": -0.5329,
    "impact_weight": 0.2372,
    "adr_weight": 0.0032,
    "intercept": 0.1587,
}


def hltv_rating(
    kast: float,
    kpr: float,
    dpr: float,
    impact: float,
    adr: float,
) -> float:
    """Apply the HLTV 2.0 rating formula and return a rounded result."""
    c = HLTV_COEFFICIENTS
    rating = (
        c["kast_weight"] * kast
        + c["kpr_weight"] * kpr
        + c["dpr_weight"] * dpr
        + c["impact_weight"] * impact
        + c["adr_weight"] * adr
        + c["intercept"]
    )
    return round(rating, 4)
