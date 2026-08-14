"""Which callouts indicate which positional role, per map and side.

The data lives in ``role_zones/*.json``. It was a 262-line Python literal, and
the failure mode it has is quiet: a role lists callout names, those names are
matched against the labels the callout lookup produces, and a name that matches
nothing simply contributes no score. The role does not error — it just stops
being detected, and nothing downstream notices.

tests/test_role_zones.py cross-checks every name here against the callout zone
files for the same map, which turns that silent failure into a red test.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROLE_ZONES_DIR = Path(__file__).parent / "role_zones"


@lru_cache(maxsize=1)
def role_zones() -> dict[str, dict[str, dict[str, list[str]]]]:
    """``{map: {side: {role: [callout, ...]}}}`` for every map with a file."""
    loaded: dict[str, dict[str, dict[str, list[str]]]] = {}
    for path in sorted(ROLE_ZONES_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        loaded[doc["map"]] = doc["sides"]
    return loaded


def roles_for(map_name: str, side: str) -> dict[str, list[str]]:
    """The roles defined for one side of one map, or empty when unsupported."""
    return role_zones().get(map_name, {}).get(side, {})


def supported_maps() -> tuple[str, ...]:
    return tuple(sorted(role_zones()))
