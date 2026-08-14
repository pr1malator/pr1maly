"""What each page is wired to do, checked by doing it.

tests/test_frontend_wiring.py checks that the *names* line up: an inline
handler calls something the page defines. That is a real check and it has
caught real breakage, but it cannot tell you whether clicking the button
reaches the function — only that a function of that name exists somewhere.

This one loads each page in a DOM, replaces the page's own functions with
recorders, dispatches the event at every element that carries a handler, and
writes down what ran. The result is a map of

    element  +  event  ->  the functions it calls

and that map is the thing a refactor has to preserve. Moving code between
files, or moving a handler out of markup and into a delegated listener, is
correct exactly when this snapshot does not change.

Needs Node and jsdom, so it skips where they are absent rather than failing:

    cd tools/domtest && npm install

Regenerate deliberately, after reading the diff:

    UPDATE_SNAPSHOTS=1 python -m pytest tests/test_frontend_dom.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from src.config.settings import PROJECT_ROOT

_DOMTEST = PROJECT_ROOT / "tools" / "domtest"
_SNAPSHOT = PROJECT_ROOT / "tests" / "snapshots" / "frontend_dom.json"
_UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"

# There used to be an allow-list here: four handlers that called nothing,
# because they were an expression written straight into an attribute. They are
# named functions now, so every wired element calls something and the exception
# is gone.


def _run_harness() -> dict:
    result = subprocess.run(
        ["node", "wiring.mjs"],
        cwd=_DOMTEST,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, f"the DOM harness failed:\n{result.stderr[-3000:]}"
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def wiring() -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is not installed")
    if not (_DOMTEST / "node_modules" / "jsdom").is_dir():
        pytest.skip("jsdom is not installed — cd tools/domtest && npm install")

    current = _run_harness()
    if _UPDATE:
        _SNAPSHOT.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return current


@pytest.fixture(scope="module")
def snapshot() -> dict:
    if not _SNAPSHOT.is_file():
        pytest.skip("no snapshot yet — UPDATE_SNAPSHOTS=1 python -m pytest")
    return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))


def test_every_page_loads_without_an_error(wiring):
    """Nothing in the snapshot below means anything if the page threw on load."""
    broken = {page: info["errors"] for page, info in wiring.items() if info["errors"]}
    assert not broken, f"pages raised while loading: {json.dumps(broken, indent=2)}"


def test_every_wired_element_actually_calls_something(wiring):
    """A handler that fires nothing is a button that does nothing."""
    silent = [
        f"{page} {handler['element']} on{handler['event']}"
        for page, info in wiring.items()
        for handler in info["handlers"]
        if not handler["fires"]
    ]
    assert not silent, "these elements are wired to nothing:\n  " + "\n  ".join(silent)


def test_the_wiring_is_unchanged(wiring, snapshot):
    """The contract: same element, same event, same functions.

    This is what makes the frontend refactorable. Code can move between files
    and handlers can move out of markup entirely; if this passes, every button
    still does what it did.
    """
    assert wiring.keys() == snapshot.keys(), "the set of pages changed"

    report: list[str] = []
    for page in sorted(snapshot):
        was = {(h["element"], h["event"]): h["fires"] for h in snapshot[page]["handlers"]}
        now = {(h["element"], h["event"]): h["fires"] for h in wiring[page]["handlers"]}

        lost = sorted(f"{el} on{ev} -> {was[(el, ev)]}" for el, ev in was.keys() - now.keys())
        gained = sorted(f"{el} on{ev} -> {now[(el, ev)]}" for el, ev in now.keys() - was.keys())
        changed = sorted(
            f"{el} on{ev}: {was[(el, ev)]} -> {now[(el, ev)]}"
            for el, ev in was.keys() & now.keys()
            if was[(el, ev)] != now[(el, ev)]
        )

        if lost or gained or changed:
            report.append(
                f"{page} is wired differently than it was.\n"
                + ("  no longer wired:\n    " + "\n    ".join(lost) + "\n" if lost else "")
                + ("  newly wired:\n    " + "\n    ".join(gained) + "\n" if gained else "")
                + (
                    "  calls something else:\n    " + "\n    ".join(changed) + "\n"
                    if changed
                    else ""
                )
            )

    # Reported together: a refactor that moves handlers touches every page, and
    # seeing one page's diff at a time turns a review into a guessing game.
    assert not report, (
        "\n".join(report) + "\nIf this is intended, regenerate with UPDATE_SNAPSHOTS=1."
    )


def test_the_snapshot_is_substantive(snapshot):
    """Guards against the harness silently degrading into a no-op."""
    handlers = sum(len(info["handlers"]) for info in snapshot.values())
    wired = sum(
        1 for info in snapshot.values() for h in info["handlers"] if h["fires"]
    )
    assert len(snapshot) >= 5, "fewer pages than the app has"
    assert handlers >= 150, f"only {handlers} handlers captured"
    assert wired >= 150, f"only {wired} handlers actually called anything"


# ---------------------------------------------------------------------------
# The explanation affordance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def explanation(tmp_path_factory) -> dict:
    """Click a "what is this?" marker and read what it says.

    The markers attach to figures that only appear once a match is open, which
    the harness has no way to arrange, so this exercises the mechanism itself.
    The API is served the *real* catalogue response, because the property worth
    checking is that the wording on screen came from
    src/domain/metrics/catalogue.py and not from a second copy in the frontend.
    """
    if not shutil.which("node"):
        pytest.skip("Node is not installed")
    if not (_DOMTEST / "node_modules" / "jsdom").is_dir():
        pytest.skip("jsdom is not installed — cd tools/domtest && npm install")

    os.environ.setdefault("DB_PATH", ":memory:")
    from fastapi.testclient import TestClient

    from api import app

    fixtures = tmp_path_factory.mktemp("fixtures") / "api.json"
    fixtures.write_text(
        json.dumps({"/api/metrics": TestClient(app).get("/api/metrics").json()}),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["node", "explain.mjs"],
        cwd=_DOMTEST,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        env={**os.environ, "DOMTEST_API_FIXTURES": str(fixtures)},
        check=False,
    )
    assert result.returncode == 0, f"the explanation harness failed:\n{result.stderr[-3000:]}"
    return json.loads(result.stdout)


def test_naming_a_figure_gets_you_a_marker(explanation):
    """An element that names a figure is given something to click, wherever it
    was rendered — including when it is the added node itself, which is how
    most of this interface is built."""
    assert explanation["marker_added"], "no explanation marker was added"
    assert explanation["marker_action"] == "explain"


def test_the_marker_explains_the_figure(explanation):
    assert explanation["popover_shown"], "clicking the marker showed nothing"
    text = explanation["text"]
    assert "Crosshair Placement" in text
    assert "How far your crosshair was from the enemy" in text, (
        "the explanation does not carry the catalogue's wording"
    )


def test_the_explanation_admits_a_hand_set_grade(explanation):
    """The reason the whole catalogue exists. This figure is graded against
    thresholds somebody chose, and a player reading a verdict on their aim
    should be told that rather than left to assume a population percentile."""
    text = explanation["text"]
    assert "Hand-set" in text
    assert "not percentiles of a player population" in text
