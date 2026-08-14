"""
Map callout coordinate system.

Translates in-game (X, Y) coordinates to human-readable CS2 callout names.
Each map defines a list of named zones as (min_x, max_x, min_y, max_y).
Coordinates are checked in order; the first match wins, so place more
specific / smaller zones before their enclosing larger ones.
"""

from __future__ import annotations

from typing import Any

from src.domain.callouts import supported_maps, zones_for

# ---------------------------------------------------------------------------
# Zone definitions: (label, min_x, max_x, min_y, max_y)
# Derived from official radar images and community callout maps.
# Coordinates use the Source engine world units from demoparser2.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Zone definitions live in src/domain/callouts/zones/*.json — one file per map,
# listing named rectangles in Source engine world units.
#
# They are data because that is what they are: 175 hand-placed rectangles
# derived from official radar images. A wrong or missing callout is a data fix,
# and tests/test_callout_zones.py checks them without anyone reading code.
#
# Order is load-bearing and the loader preserves it: zones are tested in
# sequence and the first match wins, so a specific zone must come before the
# larger one enclosing it.
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# Pixel-space label overrides (1024×1024 radar image).
# For maps without overrides, labels use geometric centres of zone boundaries.
# ---------------------------------------------------------------------------

_LABEL_PIXEL_OVERRIDES: dict[str, dict[str, tuple[float, float]]] = {
    "de_mirage": {
        "B Site": (232, 287),
        "Bench": (145, 276),
        "B Van": (182, 181),
        "Market": (238, 445),
        "Market Door": (175, 401),
        "Kitchen": (426, 250),
        "B Apartments": (397, 190),
        "B Short": (478, 303),
        "A Site": (556, 784),
        "A Default": (583, 782),
        "Firebox": (528, 772),
        "Tetris": (625, 632),
        "Stairs": (543, 632),
        "A Ramp": (725, 657),
        "Jungle": (482, 637),
        "Ticket": (463, 837),
        "CT Spawn": (310, 714),
        "Snipers Nest": (414, 463),
        "Connector": (512, 546),
        "Chair": (680, 524),
        "Top Mid": (727, 470),
        "Mid Window": (417, 471),
        "Window": (418, 449),
        "Catwalk": (536, 428),
        "Short": (480, 357),
        "Mid": (575, 469),
        "Underpass": (443, 340),
        "Ladder Room": (418, 371),
        "T Spawn": (897, 367),
        "T Ramp": (908, 284),
        "A Palace": (659, 801),
        "A Main": (400, 562),
        "B Apartments Entrance": (458, 227),
        "A Side": (618, 791),
        "B Side": (255, 228),
        "Mid Area": (606, 468),
        "T Area": (729, 242),
    },
}


def get_callout(map_name: str, x: float, y: float) -> str:
    """Return the callout name for the given (x, y) coordinate.

    Returns ``"unknown"`` if no zone matches or the map is not supported.
    """
    zones = zones_for(map_name)
    if not zones:
        return "unknown"
    for label, min_x, max_x, min_y, max_y in zones:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return label
    return "unknown"


def is_map_supported(map_name: str) -> bool:
    """Check if we have callout data for this map."""
    return map_name in supported_maps()


# ---------------------------------------------------------------------------
# Radar transform constants  (from game .txt files, 1024×1024 radar images)
# Formula:  pixel_x = (game_x - pos_x) / scale
#           pixel_y = (pos_y - game_y) / scale
# ---------------------------------------------------------------------------

_MAP_RADAR: dict[str, dict[str, float]] = {
    "de_mirage":   {"pos_x": -3230, "pos_y": 1713, "scale": 5.00},
    "de_dust2":    {"pos_x": -2476, "pos_y": 3239, "scale": 4.40},
    "de_inferno":  {"pos_x": -2087, "pos_y": 3870, "scale": 4.90},
    "de_anubis":   {"pos_x": -2796, "pos_y": 3328, "scale": 5.22},
    "de_nuke":     {"pos_x": -3453, "pos_y": 2887, "scale": 7.00},
    "de_ancient":  {"pos_x": -2953, "pos_y": 2164, "scale": 5.00},
    "de_overpass": {"pos_x": -4831, "pos_y": 1781, "scale": 5.20},
    "de_vertigo":  {"pos_x": -3168, "pos_y": 1762, "scale": 4.00},
    "de_train":    {"pos_x": -2308, "pos_y": 2078, "scale": 4.082077},
    "de_cache":    {"pos_x": -2000, "pos_y": 3250, "scale": 5.50},
}


def get_radar_config(map_name: str) -> dict[str, float] | None:
    """Return radar transform config for a map, or None."""
    return _MAP_RADAR.get(map_name)


def game_to_pixel(map_name: str, x: float, y: float) -> tuple[float, float] | None:
    """Convert game-world (x, y) to radar pixel (px, py) on a 1024×1024 image."""
    cfg = _MAP_RADAR.get(map_name)
    if not cfg:
        return None
    s = cfg["scale"]
    px = (x - cfg["pos_x"]) / s
    py = (cfg["pos_y"] - y) / s
    return (round(px, 1), round(py, 1))


def get_zone_center(map_name: str, callout: str) -> tuple[float, float] | None:
    """Return the pixel-space label position for a named callout zone.

    Uses manually curated pixel overrides when available, falling back
    to the geometric centre of the classification boundary.
    """
    overrides = _LABEL_PIXEL_OVERRIDES.get(map_name, {})
    for key, pos in overrides.items():
        if key.lower() == callout.lower():
            return pos

    zones = zones_for(map_name)
    cfg = _MAP_RADAR.get(map_name)
    if not zones or not cfg:
        return None
    for label, min_x, max_x, min_y, max_y in zones:
        if label.lower() == callout.lower():
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            s = cfg["scale"]
            px = (cx - cfg["pos_x"]) / s
            py = (cfg["pos_y"] - cy) / s
            return (round(px, 1), round(py, 1))
    return None


def get_all_zones_pixel(map_name: str) -> list[dict[str, Any]] | None:
    """Return all zones for a map converted to 1024×1024 pixel rects.

    Each dict has: ``label``, ``px1``, ``py1`` (top-left), ``px2``, ``py2``
    (bottom-right), ``cx``, ``cy`` (center).  Used by the schematic radar
    renderer on the frontend.
    """
    zones = zones_for(map_name)
    cfg = _MAP_RADAR.get(map_name)
    if not zones or not cfg:
        return None

    # Categories for zone colouring on the frontend
    _SITE_LABELS = {"A Site", "B Site", "A Default"}
    _SPAWN_LABELS = {"CT Spawn", "T Spawn"}
    _MID_LABELS = {"Mid", "Top Mid", "Catwalk", "Short", "Mid Area",
                   "Alt Mid", "Mid Doors", "Xbox", "Connector"}

    overrides = _LABEL_PIXEL_OVERRIDES.get(map_name, {})

    result = []
    s = cfg["scale"]
    for label, min_x, max_x, min_y, max_y in zones:
        px1 = round((min_x - cfg["pos_x"]) / s, 1)
        py1 = round((cfg["pos_y"] - max_y) / s, 1)  # max_y → top
        px2 = round((max_x - cfg["pos_x"]) / s, 1)
        py2 = round((cfg["pos_y"] - min_y) / s, 1)  # min_y → bottom
        if label in overrides:
            cx, cy = overrides[label]
        else:
            cx = round((px1 + px2) / 2, 1)
            cy = round((py1 + py2) / 2, 1)

        if label in _SITE_LABELS:
            cat = "site"
        elif label in _SPAWN_LABELS:
            cat = "spawn"
        elif label in _MID_LABELS:
            cat = "mid"
        elif any(t in label for t in ("T ", "T_")):
            cat = "t_area"
        else:
            cat = "zone"

        result.append({
            "label": label, "cat": cat,
            "px1": px1, "py1": py1, "px2": px2, "py2": py2,
            "cx": cx, "cy": cy,
        })
    return result
