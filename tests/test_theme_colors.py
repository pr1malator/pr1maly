"""Every colour the interface asks for has to exist.

The bug this is written against: a commit replaced 255 palette classes with
seven role names — good, bad, caution, warn, info, accent, muted — so a theme
could speak to them, and defined --s-* tokens for all seven in every theme. It
never named them in the Tailwind config. Tailwind emits a rule only for a class
it can resolve, so every one of those 137 class names produced no colour at
all: the kill/death legend, the aim card, the T-side bar, the sidebar. Nothing
failed and nothing logged, and a page with no colour still looks like a page.

A missing colour is invisible in exactly the way a missing test is. Both halves
are checked here: the name has to be in the Tailwind config, and the CSS
variable it resolves to has to be defined by every theme, because a colour that
exists in one theme and not another is the same failure with extra steps.
"""

from __future__ import annotations

import re

import pytest

from src.config.settings import FRONTEND_DIR

_CONFIG = (FRONTEND_DIR / "js" / "tailwind-config.js").read_text(encoding="utf-8")
_THEME_CSS = (FRONTEND_DIR / "theme.css").read_text(encoding="utf-8")

# bg-primary, text-on-surface-variant/70, border-white/10, ring-primary ...
# class="..." and className = "...", which is where colour utilities live.
_CLASS_LIST = re.compile(r"""class(?:Name)?\s*=\s*["'`]([^"'`]*)["'`]""")

_COLOUR_CLASS = re.compile(
    r"\b(?:bg|text|border|ring|from|via|to|fill|stroke|decoration|outline|accent|caret|divide|shadow)"
    r"-([a-z][a-z0-9-]*)(?:/\d+)?\b"
)

# Tailwind's own palette and keywords, which need no declaration from us.
_BUILT_IN_PALETTE = {
    "slate", "gray", "grey", "zinc", "neutral", "stone", "red", "orange",
    "amber", "yellow", "lime", "green", "emerald", "teal", "cyan", "sky",
    "blue", "indigo", "violet", "purple", "fuchsia", "pink", "rose",
    "white", "black", "transparent", "current", "inherit", "none", "auto",
}

# Utilities that share the same prefixes but name something other than a colour.
_SIDE_OR_OFFSET = re.compile(r"^(?:[btlrxyse]|offset)-\d+$")

_NOT_COLOURS = {
    "center", "left", "right", "top", "bottom", "middle", "justify", "start",
    "end", "clip", "ellipsis", "wrap", "nowrap", "balance", "pretty",
    "xs", "sm", "base", "lg", "xl", "2xl", "3xl", "4xl", "5xl", "6xl", "7xl",
    "solid", "dashed", "dotted", "double", "hidden", "collapse", "separate",
    "opacity", "1", "2", "3", "4", "5", "6", "7", "8", "0",
    "b", "t", "l", "r", "x", "y", "s", "e", "inner", "md",
}


def _declared_in_config() -> set[str]:
    return set(re.findall(r'"([a-z][a-z0-9-]*)":\s*"rgb\(', _CONFIG))


def _colour_names_used() -> dict[str, str]:
    """Colour name -> the first file that uses it."""
    used: dict[str, str] = {}
    for path in sorted(FRONTEND_DIR.rglob("*")):
        if path.suffix not in (".html", ".js") or "vendor" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        classes = " ".join(_CLASS_LIST.findall(text))
        for name in _COLOUR_CLASS.findall(classes):
            if name in _BUILT_IN_PALETTE or name in _NOT_COLOURS:
                continue
            if _SIDE_OR_OFFSET.match(name) or name.startswith("gradient-"):
                continue
            if name.split("-")[0] in _BUILT_IN_PALETTE:  # amber-300, red-500
                continue
            used.setdefault(name, str(path.relative_to(FRONTEND_DIR)).replace("\\", "/"))
    return used


def test_every_colour_the_markup_asks_for_is_declared():
    """Tailwind emits nothing for a class it cannot resolve, so the element
    simply has no colour — which looks like a design choice."""
    declared = _declared_in_config()
    missing = {
        name: where
        for name, where in _colour_names_used().items()
        if name not in declared
    }
    assert not missing, (
        "these colours are used but not declared in js/tailwind-config.js:\n  "
        + "\n  ".join(f"{name}  (first seen in {where})" for name, where in sorted(missing.items()))
    )


@pytest.mark.parametrize("name", sorted(_declared_in_config()))
def test_every_declared_colour_resolves_to_a_variable_every_theme_defines(name):
    """A colour defined for one theme and not another fails only for the people
    using the other theme, which is the hardest kind of report to act on."""
    variable = re.search(rf'"{re.escape(name)}":\s*"rgb\(var\((--[a-z0-9-]+)\)', _CONFIG)
    assert variable, f"{name} is declared in a form this test cannot follow"

    themes = re.findall(r"^(:root|\[data-theme=\"[a-z]+\"\])\s*\{", _THEME_CSS, re.M)
    blocks = re.findall(r"^(?::root|\[data-theme=\"[a-z]+\"\])\s*\{(.*?)^\}", _THEME_CSS, re.S | re.M)
    assert len(blocks) >= 4, "theme.css no longer looks like a set of theme blocks"

    without = [
        theme
        for theme, body in zip(themes, blocks, strict=False)
        if f"{variable.group(1)}:" not in body
    ]
    assert not without, f"{name} resolves to {variable.group(1)}, undefined in: {without}"


def test_the_semantic_colours_are_present():
    """The seven roles, named individually because these are the ones that were
    missing and a regression here is invisible on the page."""
    declared = _declared_in_config()
    for name in ("good", "bad", "caution", "warn", "info", "accent", "muted"):
        assert name in declared, f"{name} is used by the markup and must be declared"


def test_the_check_is_looking_at_something():
    used = _colour_names_used()
    assert len(used) >= 20, f"only {len(used)} colour names found — is the scan working?"
