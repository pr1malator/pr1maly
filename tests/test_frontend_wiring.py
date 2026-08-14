"""Checks that the frontend is wired up, without opening a browser.

There are no browser tests, so a broken button shows up when someone clicks it.
Markup names an action and JavaScript registers one, and whether those two sets
agree is a static property — so it can be checked without running anything.

The pages used to carry 191 inline ``onclick`` attributes instead, each naming
a function that had to exist as a global by the time it fired. The check for
that is still here: it should keep finding nothing, and if an inline handler
comes back, the test below says so.

The behavioural half of this lives in tests/test_frontend_dom.py, which loads
the pages in a DOM and clicks things.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.config.settings import FRONTEND_DIR

PAGES = sorted(p.name for p in FRONTEND_DIR.glob("*.html"))

# onclick="foo(...)" and friends. Also picks up `onclick="a(); b()"`.
_HANDLER_ATTR = re.compile(
    r"""\bon(?:click|change|input|submit|keyup|keydown|blur|focus)\s*=\s*["']([^"']+)["']""",
    re.I,
)
# A bare call, not a method: `doThing(` needs to be a global, but
# `document.getElementById(` and `event.preventDefault()` do not.
_CALL = re.compile(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(")

# Anything the page can reach without declaring it.
_BUILTIN = {
    "alert", "confirm", "prompt", "event", "this", "return", "if", "for",
    "typeof", "void", "new", "delete", "window", "document", "console",
    "setTimeout", "setInterval", "clearTimeout", "clearInterval", "fetch",
    "parseInt", "parseFloat", "Number", "String", "Boolean", "Array", "Object",
    "JSON", "Math", "Date", "encodeURIComponent", "decodeURIComponent",
    "requestAnimationFrame", "localStorage", "sessionStorage", "location",
    "open", "close", "print", "focus", "blur", "scrollTo", "Promise",
}

_DEFINITION = re.compile(
    r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"          # function foo(
    r"|^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*="          # const foo =
    r"|^\s*window\.([A-Za-z_$][\w$]*)\s*="                      # window.foo =
    r"|\bwindow\.([A-Za-z_$][\w$]*)\s*=",                       # ... anywhere
    re.M,
)


def _page_text(name: str) -> str:
    return (FRONTEND_DIR / name).read_text(encoding="utf-8", errors="replace")


def _scripts_loaded_by(name: str) -> list[str]:
    """Local files the page pulls in with <script src=...>."""
    text = _page_text(name)
    srcs = re.findall(r"""<script[^>]*\bsrc\s*=\s*["']([^"']+)["']""", text, re.I)
    return [s.split("?")[0] for s in srcs if not s.startswith(("http://", "https://", "//"))]


_IMPORT = re.compile(
    r"""^\s*import\s+(?:[^'"]*?\bfrom\s+)?["'](\.[^"']+)["']""", re.M
)


def _sources_for(name: str) -> list[tuple[str, str]]:
    """The page and all the code it runs, as (label, text).

    Two hops are needed, because the frontend uses both kinds of script. The
    page's <script src> tags give the classic ones; the module entry point
    reaches the rest through imports, and a check that stopped at the tags
    would silently see a fraction of the code. It would not fail — it would
    find nothing, which is worse.
    """
    sources = [(name, _page_text(name))]
    seen: set[Path] = set()

    def follow(path: Path, label: str) -> None:
        path = path.resolve()
        if path in seen or not path.is_file():
            return
        seen.add(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        sources.append((label, text))
        for specifier in _IMPORT.findall(text):
            target = (path.parent / specifier).resolve()
            follow(target, str(target.relative_to(FRONTEND_DIR)).replace("\\", "/"))

    for src in _scripts_loaded_by(name):
        follow(FRONTEND_DIR / src, src)
    return sources


def _names_defined_for(name: str) -> set[str]:
    """Every name the page could resolve: its own scripts plus what it loads."""
    sources = [text for _, text in _sources_for(name)]

    defined: set[str] = set()
    for text in sources:
        for match in _DEFINITION.finditer(text):
            defined.add(next(g for g in match.groups() if g))
    return defined


def _handlers_used_by(name: str) -> set[str]:
    used: set[str] = set()
    for attr in _HANDLER_ATTR.findall(_page_text(name)):
        used.update(_CALL.findall(attr))
    return used - _BUILTIN


@pytest.mark.parametrize("page", PAGES)
def test_every_inline_handler_resolves_to_something_the_page_defines(page):
    """A button whose handler is not defined does nothing, silently."""
    missing = sorted(_handlers_used_by(page) - _names_defined_for(page))
    assert not missing, (
        f"{page} has inline handlers calling functions it never defines: {missing}. "
        f"If these moved into an ES module, the module must assign them to window "
        f"— module scope is not global scope."
    )


_ACTION_ATTR = re.compile(
    # data-action="foo" in markup, and the two ways JavaScript sets the same
    # attribute on an element it built.
    r"""\bdata-action\s*=\s*["']([^"'$]+)["']"""
    r"""|\.dataset\.action\s*=\s*["']([^"'$]+)["']"""
    r"""|setAttribute\(\s*["']data-action["']\s*,\s*["']([^"'$]+)["']"""
)
_REGISTERED = re.compile(r"registerActions\(\{(.*?)\}\)", re.S)
# js/actions.js provides these two itself.
_BUILT_IN = {"preventDefault", "stopPropagation"}


def _actions_used_by(name: str) -> set[str]:
    """Action names in the page's markup and in the markup its scripts build."""
    used: set[str] = set()
    for label, text in _sources_for(name):
        # The mechanism documents itself with examples; those are not usages.
        if label.endswith("js/actions.js"):
            continue
        for groups in _ACTION_ATTR.findall(text):
            for value in groups if isinstance(groups, tuple) else (groups,):
                used.update(value.split())
    return used - _BUILT_IN


def _actions_registered_for(name: str) -> set[str]:
    registered: set[str] = set()
    for _, text in _sources_for(name):
        for block in _REGISTERED.findall(text):
            registered.update(re.findall(r"([A-Za-z_$][\w$]*)\s*[,:}]", block))
    return registered


@pytest.mark.parametrize("page", PAGES)
def test_every_action_the_markup_names_is_registered(page):
    """The replacement for "does this global still exist".

    A button whose data-action nothing registers does nothing when clicked, and
    says so only in the console. This is the static half of that check; the
    behavioural half is tests/test_frontend_dom.py.
    """
    missing = sorted(_actions_used_by(page) - _actions_registered_for(page))
    assert not missing, (
        f"{page} names actions nothing registers: {missing}. Add them to a "
        f"registerActions({{...}}) call in the file that defines them."
    )


@pytest.mark.parametrize("page", PAGES)
def test_no_action_is_registered_that_nothing_calls(page):
    """A registration for markup that no longer exists is dead weight, and the
    next person cannot tell it from something still in use."""
    unused = sorted(_actions_registered_for(page) - _actions_used_by(page))
    assert not unused, f"{page} registers actions no markup names: {unused}"


@pytest.mark.parametrize("page", PAGES)
def test_no_inline_event_handlers_remain(page):
    """Inline handlers are why the frontend could not use modules: they are
    evaluated against global scope, so everything they call has to be global."""
    leftover = re.findall(
        r"""\son(?:click|change|input|submit|keyup|keydown)\s*=\s*["']""",
        _page_text(page),
    )
    assert not leftover, (
        f"{page} still has {len(leftover)} inline handler(s); use data-action"
    )


@pytest.mark.parametrize("page", PAGES)
def test_every_script_the_page_loads_exists(page):
    for src in _scripts_loaded_by(page):
        assert (FRONTEND_DIR / src).is_file(), f"{page} loads missing {src}"


@pytest.mark.parametrize(
    "script",
    sorted(
        str(p.relative_to(FRONTEND_DIR)).replace("\\", "/")
        for p in FRONTEND_DIR.rglob("*.js")
        if "vendor" not in p.parts
    ),
)
def test_shared_scripts_parse(script):
    """Catches a syntax error in a file no Python test would otherwise load."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, "--check", str(FRONTEND_DIR / script)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"{script} does not parse:\n{result.stderr}"


_LOADED_EXTERNAL = re.compile(
    r"""<(?:script|link)[^>]*\b(?:src|href)\s*=\s*["'](https?://[^"']+)["']""", re.I
)


@pytest.mark.parametrize("page", PAGES)
def test_no_page_loads_a_resource_from_the_internet(page):
    """The README promises "no external services required".

    Stylesheets, fonts and scripts fetched from a CDN make that false: with no
    network the app renders unstyled and its icon buttons show their names as
    text. <a href> links out are fine — those are the user choosing to leave.
    """
    external = _LOADED_EXTERNAL.findall(_page_text(page))
    assert not external, (
        f"{page} loads {external} from the network; vendor it under "
        f"frontend/vendor/ instead"
    )


def test_the_icon_stylesheet_still_makes_icon_names_render_as_icons():
    """Icons are written as words — <span ...>show_chart</span> — and become
    glyphs through an OpenType ligature.

    Three things have to survive vendoring for that to work: the @font-face, a
    font-family on the class, and the 'liga' feature. Vendoring by pulling out
    the @font-face rules dropped the class rule that carries the other two, and
    every icon on every page rendered as its own name — SHOW_CHART where a
    chart icon belongs, and the play button on the 2D replay reading "play".

    The pages set only font-variation-settings themselves, so they cannot
    make up for a stylesheet that has lost the rest.
    """
    css = (FRONTEND_DIR / "vendor" / "symbols.css").read_text(encoding="utf-8")

    assert "@font-face" in css, "no @font-face: the font is never loaded"
    assert ".material-symbols-outlined" in css, (
        "the icon class rule is missing — elements fall back to the body font"
    )
    assert "liga" in css, (
        "the ligature feature is missing — icon names render as literal text"
    )
    class_rule = css[css.index(".material-symbols-outlined"):]
    assert "font-family" in class_rule.split("}")[0], (
        "the icon class does not set font-family"
    )


@pytest.mark.parametrize("page", PAGES)
def test_a_page_using_icons_links_the_icon_stylesheet(page):
    """Otherwise its icons render as their own names, which is how the 2D
    replay's play control came to read as the word "play"."""
    uses_icons = any(
        "material-symbols-outlined" in text for _, text in _sources_for(page)
    )
    if not uses_icons:
        pytest.skip(f"{page} uses no icons")
    assert "vendor/symbols.css" in _page_text(page), (
        f"{page} uses icon spans but never links the stylesheet that maps them"
    )


@pytest.mark.parametrize("sheet", ["symbols.css", "text.css"])
def test_vendored_stylesheets_reference_only_local_fonts(sheet):
    path = FRONTEND_DIR / "vendor" / sheet
    assert path.is_file(), f"vendor/{sheet} is missing"
    css = path.read_text(encoding="utf-8")
    for url in re.findall(r"url\(([^)]+)\)", css):
        cleaned = url.strip("'\" ")
        assert not cleaned.startswith("http"), f"vendor/{sheet} still fetches {cleaned}"
        assert (path.parent / cleaned).is_file(), f"vendor/{sheet} points at missing {cleaned}"


def test_pages_agree_on_where_the_api_lives():
    """calibrate.html used to omit the /api suffix the other pages add, so its
    requests went to the wrong place."""
    bases: dict[str, str] = {}
    for page in PAGES:
        for label, text in _sources_for(page):
            for match in re.finditer(r"""const\s+API\s*=\s*([^;\n]+)""", text):
                bases[label] = match.group(1).strip()
    assert bases, "no page declares an API base any more"
    unique = set(bases.values())
    assert len(unique) == 1, f"pages disagree on the API base: {bases}"


def test_debug_drawing_stays_inside_a_debug_function():
    """The overlay on the match minimap is a deliberate tool, not a leftover.

    It prints position and zone counts and a calibration grid, and it is
    reached only through a button labelled "Debug: show all event positions".
    What would be wrong is that drawing happening anywhere else, so the check
    is that every DEBUG draw sits inside a function whose name says so.
    """
    seen = 0
    for page in PAGES:
        for label, text in _sources_for(page):
            lines = text.splitlines()
            for number, line in enumerate(lines):
                if "fillText('DEBUG" not in line and 'fillText("DEBUG' not in line:
                    continue
                seen += 1
                enclosing = next(
                    (
                        lines[i]
                        for i in range(number, -1, -1)
                        if re.match(r"^\s*(?:async\s+)?function\s", lines[i])
                    ),
                    "",
                )
                assert "ebug" in enclosing, (
                    f"{label}:{number + 1} draws a DEBUG overlay from "
                    f"{enclosing.strip()!r}, which is not a debug function"
                )
    # The overlay exists; a version of this test that finds nothing to check is
    # passing for the wrong reason.
    assert seen, "no DEBUG drawing found at all — is this still looking in the right place?"
