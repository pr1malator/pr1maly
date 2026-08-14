"""Numbers that came from outside the code.

Three artifacts with three different provenances and three different staleness
clocks, which is why they sit together rather than beside the metrics that
consume them:

  winprob     measured from this installation's own demo corpus, regenerable
              with scripts/calibrate_winprob.py
  hltv        published coefficients from a third party, not ours to change
  benchmarks  hand-set thresholds, explicitly not calibrated against anything

Keeping them in one place makes "which of our numbers are measured and which
are guessed?" a question you can answer by looking in a directory, rather than
by finding the right comment in a five-thousand-line module.
"""

from src.domain.calibration.hltv import HLTV_COEFFICIENTS, hltv_rating
from src.domain.calibration.winprob import WIN_PROB, win_probability

__all__ = [
    "HLTV_COEFFICIENTS",
    "WIN_PROB",
    "hltv_rating",
    "win_probability",
]
