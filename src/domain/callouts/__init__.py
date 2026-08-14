"""Map callout zones, as data.

Each map is a JSON file under ``zones/`` listing named rectangles in Source
engine world units. They were 300 lines of Python literals; being data means a
wrong or missing callout is a data fix, checkable by the tests in
tests/test_callout_zones.py without anyone reading code.

Order is load-bearing. Zones are tested in sequence and the first match wins, so
a specific zone has to appear before the larger one enclosing it — "B Apartments
Entrance" before "B Apartments". JSON arrays preserve order, and the loader does
not sort.
"""

from __future__ import annotations

import json
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

ZONES_DIR = Path(__file__).parent / "zones"

# (label, min_x, max_x, min_y, max_y) — the shape the lookup has always used.
Zone = tuple[str, float, float, float, float]


@cache
def _load(map_name: str) -> tuple[Zone, ...]:
    path = ZONES_DIR / f"{map_name}.json"
    if not path.is_file():
        return ()
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    return tuple(
        (z["label"], z["min_x"], z["max_x"], z["min_y"], z["max_y"])
        for z in doc["zones"]
    )


@lru_cache(maxsize=1)
def supported_maps() -> tuple[str, ...]:
    """Every map with a zone file, sorted."""
    return tuple(sorted(p.stem for p in ZONES_DIR.glob("*.json")))


def zones_for(map_name: str) -> tuple[Zone, ...]:
    """The ordered zones for *map_name*, or empty when it is not supported."""
    return _load(map_name)


def zone_document(map_name: str) -> dict[str, Any]:
    """The raw file, including the area headings and provenance the tuples drop."""
    path = ZONES_DIR / f"{map_name}.json"
    if not path.is_file():
        return {}
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
    return doc
