"""The catalogue has to describe the analysis the app actually produces.

A table of descriptions is worth nothing if it drifts from the code, and it
drifts silently: nobody notices a missing entry, and an entry for a figure that
no longer exists reads exactly like one that does. Both directions are checked
here against the golden analysis, which is a real run of the whole pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.settings import PROJECT_ROOT
from src.domain.metrics import catalogue
from src.domain.metrics.registry import REGISTRY

_GOLDEN = json.loads(
    (PROJECT_ROOT / "tests" / "snapshots" / "analysis_golden.json").read_text(
        encoding="utf-8"
    )
)

# Figures a reader never sees, and does not need described. Each one is here
# for a reason, not because writing an entry was inconvenient.
_NOT_SHOWN = {
    "analyzer_version",       # bookkeeping
    "player_name",            # the player, not a measurement
    "map_name",
    "match_result",
    "team_score",
    "enemy_score",
    "total_rounds",
    "kills",                  # the scoreboard's own columns, self-evident
    "deaths",
    "assists",
    "kd_ratio",
    "rounds_2k",
    "rounds_3k",
    "rounds_4k",
    "rounds_5k",
    "all_players",            # the other nine players' scoreboard rows
    "enriched_rounds",        # the intermediate every metric reads
    "replay_data",            # positions for the 2D viewer, not a figure
    "chat_messages",
    "tags",
    "aim_stats.aim_rating_inputs",       # the rating's own working
    "utility_data.utility_rating_inputs",
    "impact_stats.confidence",           # the shared statistics, described once
    "impact_stats.n",
    "impact_stats.kills_scored",
    "impact_stats.deaths_scored",
    "impact_stats.net_swing_total",
    "role_data.map",
    "role_data.rounds",
    "role_data.ct_summary",
    "role_data.t_summary",
}


def _shown_keys() -> set[str]:
    """Every figure in a stored analysis, at the level the catalogue describes."""
    found: set[str] = set()
    for key, value in _GOLDEN.items():
        if key.startswith("_"):
            continue
        if key in ("aim_stats", "utility_data", "impact_stats", "role_data", "benchmarks"):
            for sub in value:
                if not sub.startswith("_"):
                    found.add(f"{key}.{sub}")
        else:
            found.add(key)
    return found


# ---------------------------------------------------------------------------
# The two directions of drift
# ---------------------------------------------------------------------------


def test_every_figure_the_analysis_produces_is_described():
    """A new number added to the output has to be explained or excluded."""
    undescribed = sorted(_shown_keys() - set(catalogue.keys()) - _NOT_SHOWN)
    assert not undescribed, (
        "these figures are produced but nothing says what they are:\n  "
        + "\n  ".join(undescribed)
        + "\nAdd them to src/domain/metrics/catalogue.py, or to _NOT_SHOWN here "
          "with a reason."
    )


def test_the_catalogue_describes_nothing_that_does_not_exist():
    """An entry for a figure that has been removed is worse than no entry: it
    reads exactly like a live one."""
    phantom = sorted(set(catalogue.keys()) - _shown_keys())
    assert not phantom, f"described but never produced: {phantom}"


def test_nothing_is_both_described_and_excluded():
    overlap = sorted(set(catalogue.keys()) & _NOT_SHOWN)
    assert not overlap, f"listed in both places: {overlap}"


# ---------------------------------------------------------------------------
# Each entry says enough to be worth reading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", catalogue.FIELDS, ids=lambda f: f.key)
def test_each_entry_says_what_it_measures_and_how(field):
    assert len(field.measures) > 20, f"{field.key}: 'measures' says nothing useful"
    assert len(field.derived) > 20, f"{field.key}: 'derived' says nothing useful"
    assert field.measures.endswith("."), f"{field.key}: 'measures' is not a sentence"
    assert field.derived.endswith("."), f"{field.key}: 'derived' is not a sentence"


@pytest.mark.parametrize("field", catalogue.FIELDS, ids=lambda f: f.key)
def test_a_tier_names_where_it_came_from(field):
    """The point of the exercise: a grade shown to a player must be traceable to
    measured data, a published formula, or an admission that it was hand-set."""
    if field.tiers is None:
        return
    assert field.tiers in catalogue.PROVENANCE, (
        f"{field.key} claims provenance {field.tiers!r}, which is not one of "
        f"{sorted(catalogue.PROVENANCE)}"
    )


@pytest.mark.parametrize("field", catalogue.FIELDS, ids=lambda f: f.key)
def test_each_entry_belongs_to_a_real_metric(field):
    assert field.metric == "core" or field.metric in REGISTRY.ids(), (
        f"{field.key} claims to come from {field.metric!r}, which is not registered"
    )


def test_keys_are_unique():
    assert len(catalogue.keys()) == len(set(catalogue.keys()))


def test_the_hand_set_tiers_admit_it():
    """The tier labels read as a comparison against other players. Most of them
    are lines someone drew, and the catalogue is where that is recorded."""
    heuristic = [f.key for f in catalogue.FIELDS if f.tiers == catalogue.HEURISTIC]
    assert heuristic, "no figure admits to a hand-set tier — that cannot be right"
    assert "population" not in catalogue.PROVENANCE[catalogue.HEURISTIC].split("not")[0]


def test_the_statistics_are_explained_once():
    """Every distribution carries these; describing them per figure would be
    thirty copies of the same paragraph."""
    for name in ("median", "avg", "n", "confidence"):
        assert name in catalogue.STATISTICS
        assert len(catalogue.STATISTICS[name]) > 20


def test_the_reference_covers_a_substantial_share_of_the_output():
    """Guards against the catalogue quietly shrinking to a handful of entries
    while _NOT_SHOWN grows to cover the rest."""
    assert len(catalogue.FIELDS) >= 30, f"only {len(catalogue.FIELDS)} figures described"
    described = len(_shown_keys() & set(catalogue.keys()))
    assert described >= len(_shown_keys()) * 0.5, (
        f"only {described} of {len(_shown_keys())} figures in the analysis are described"
    )


def test_the_golden_is_the_thing_being_checked():
    """If the golden ever stops being a full analysis, everything above passes
    for the wrong reason."""
    assert Path(PROJECT_ROOT / "tests" / "snapshots" / "analysis_golden.json").is_file()
    assert len(_shown_keys()) >= 40, f"only {len(_shown_keys())} figures in the golden"


# ---------------------------------------------------------------------------
# The markers in the markup
# ---------------------------------------------------------------------------


def _explained_in_markup() -> dict[str, str]:
    """Every data-explain="..." in the frontend, and which file names it."""
    import re

    from src.config.settings import FRONTEND_DIR

    found: dict[str, str] = {}
    for path in sorted(FRONTEND_DIR.rglob("*")):
        if path.suffix not in (".html", ".js") or "vendor" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for key in re.findall(r"""data-explain=["']([^"'$]+)["']""", text):
            found.setdefault(key, path.name)
    return found


def test_every_marker_in_the_interface_names_a_figure_that_exists():
    """A marker for an unknown key opens onto nothing, and says so only in the
    console — which is the same as saying nothing."""
    unknown = {
        key: where
        for key, where in _explained_in_markup().items()
        if key not in set(catalogue.keys())
    }
    assert not unknown, f"markers naming figures the catalogue does not describe: {unknown}"


def test_the_interface_actually_explains_something():
    """Guards against the markers being dropped in a markup edit and nobody
    noticing, since a missing marker looks exactly like a figure nobody chose
    to explain."""
    assert len(_explained_in_markup()) >= 8, (
        f"only {len(_explained_in_markup())} figures carry an explanation marker"
    )


# ---------------------------------------------------------------------------
# Claims about other people's products
# ---------------------------------------------------------------------------

# Naming another analysis service was doing one job in this codebase: lending
# authority to a threshold by saying somebody else drew it in the same place.
# None of those claims were verified, none of them can be kept current, and the
# reasons they were attached to stand up perfectly well on their own.
#
# HLTV is deliberately absent from this list. "HLTV 2.0 Rating" is the name of
# a published formula this app implements, and attributing it is honesty rather
# than comparison — removing the name would leave a number with no provenance
# at all.
_OTHER_SERVICES = ("leetify", "refrag", "scope.gg", "csstats", "faceit analy")


def test_no_claims_about_what_other_services_do():
    """Reasons here have to stand on their own rather than on someone else's
    authority, and a claim about a third party's product is one nobody in this
    repository can check or keep up to date."""

    from src.config.settings import PROJECT_ROOT

    skip = {".git", "node_modules", "dist", "vendor", ".venv", "__pycache__", "img"}
    offenders: list[str] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in (".py", ".md", ".js", ".html"):
            continue
        if skip & set(path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            for name in _OTHER_SERVICES:
                if name in lowered:
                    rel = path.relative_to(PROJECT_ROOT)
                    offenders.append(f"{rel}:{number}: {line.strip()[:80]}")

    # This test names them itself; that is the one place they belong.
    offenders = [o for o in offenders if "test_catalogue.py" not in o]
    assert not offenders, (
        "these say what another service does, which is not ours to claim:\n  "
        + "\n  ".join(offenders)
    )
