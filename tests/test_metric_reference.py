"""METRICS.md is generated, so it cannot be allowed to drift from the code.

The failure mode of a hand-maintained reference is not that it goes missing —
it is that a description of something that has changed reads exactly like one
that has not. Generating it removes the possibility; this test removes the
possibility of forgetting to regenerate it.
"""

from __future__ import annotations

import pytest

from src.config.settings import PROJECT_ROOT
from src.domain.metrics import catalogue

pytest.importorskip("tools.generate_metric_reference", reason="development tool")

from tools.generate_metric_reference import OUTPUT, render  # noqa: E402


def test_the_reference_exists():
    assert OUTPUT.is_file(), "run python tools/generate_metric_reference.py"


def test_the_reference_is_not_stale():
    """Fails on the commit that changes a figure without regenerating."""
    assert OUTPUT.read_text(encoding="utf-8") == render(), (
        "METRICS.md does not match the catalogue — run "
        "python tools/generate_metric_reference.py"
    )


def test_every_figure_appears_in_it():
    text = OUTPUT.read_text(encoding="utf-8")
    missing = [field.key for field in catalogue.FIELDS if f"`{field.key}`" not in text]
    assert not missing, f"described in the catalogue but not in the reference: {missing}"


def test_it_says_which_grades_were_picked_by_hand():
    """The whole point of the provenance field. If this sentence ever leaves the
    document, a reader has no way to tell a measured tier from an invented one."""
    text = OUTPUT.read_text(encoding="utf-8")
    assert "not percentiles of a player population" in text
    assert "Hand-set" in text


def test_the_readme_points_at_it():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "METRICS.md" in readme, "the reference exists but nothing sends anyone to it"
