"""
Map callout coordinate system.

Translates in-game (X, Y) coordinates to human-readable CS2 callout names.
Each map defines a list of named zones as (min_x, max_x, min_y, max_y).
Coordinates are checked in order; the first match wins, so place more
specific / smaller zones before their enclosing larger ones.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Zone definitions: (label, min_x, max_x, min_y, max_y)
# Derived from official radar images and community callout maps.
# Coordinates use the Source engine world units from demoparser2.
# ---------------------------------------------------------------------------

_MIRAGE_ZONES: list[tuple[str, float, float, float, float]] = [
    # Recalibrated 2026-03-28 from hand-placed pixel positions on official radar.
    # Game coords derived via: gx = px*5 - 3230, gy = 1713 - py*5
    # Format: (label, min_x, max_x, min_y, max_y).
    # Zones ordered specific-first; first match wins.

    # ---- B site area ----
    ("B Site",         -2300, -1850,      0,   550),
    ("Bench",          -2750, -2300,     50,   650),
    ("B Van",          -2550, -2050,    550,  1100),
    ("Market",         -2350, -1700,   -850,  -250),
    ("Market Door",    -2600, -1700,   -600,  -250),
    ("Kitchen",        -1400,  -850,    200,   530),
    ("B Apartments Entrance", -1200, -650, 300, 900),
    ("B Apartments",   -1500,  -950,    500,  1050),
    ("B Short",        -1100,  -550,     50,   500),

    # ---- A site area ----
    ("Firebox",         -850,  -475,  -2400, -2050),
    ("A Default",       -550,  -100,  -2450, -2100),
    ("A Palace",        -100,   900,  -2550, -2100),
    ("A Site",          -750,   100,  -2550, -1950),
    ("Ticket",         -1200,  -600,  -2750, -2200),

    # ---- Between A and mid ----
    ("Jungle",         -1100,  -600,  -1750, -1250),
    ("Stairs",          -700,  -300,  -1700, -1200),
    ("Tetris",          -350,   200,  -1700, -1200),
    ("A Ramp",           100,   700,  -1850, -1300),

    # ---- CT area ----
    ("CT Spawn",       -2050, -1300,  -2200, -1500),

    # ---- Window / snipers cluster ----
    ("Mid Window",     -1400,  -900,   -850,  -630),
    ("Snipers Nest",   -1400,  -900,   -630,  -560),
    ("Window",         -1400,  -900,   -560,  -150),

    # ---- Connector / transition ----
    ("A Main",         -1500,  -950,  -1350,  -850),
    ("Connector",       -950,  -400,  -1300,  -750),
    ("Chair",            -50,   450,  -1150,  -700),

    # ---- Mid area ----
    ("Top Mid",          150,   700,   -900,  -350),
    ("Catwalk",         -800,  -250,   -600,  -200),
    ("Ladder Room",    -1400, -1050,   -400,   100),
    ("Underpass",      -1300,  -920,   -250,   300),
    ("Short",           -920,  -550,   -300,   200),
    ("Mid",             -650,   -50,   -900,  -400),

    # ---- T side ----
    ("T Spawn",          850,  1700,   -500,   200),
    ("T Ramp",          1000,  1700,    100,   600),

    # ---- Fallback / broad ----
    ("A Side",         -1300,   800,  -2800, -1200),
    ("B Side",         -2800,  -550,   -300,  1300),
    ("Mid Area",        -900,   800,  -1000,   300),
    ("T Area",            50,  1800,   -400,  1000),
]

_DUST2_ZONES: list[tuple[str, float, float, float, float]] = [
    # ---- A site ----
    ("A Site",          900,  1500,   2100,  2800),
    ("A Short (Cat)",   500,  1000,   1400,  2100),
    ("A Long",         1200,  2200,    200,  1400),
    ("A Long Doors",   1200,  2200,   -500,   200),
    ("A Pit",          1500,  2000,   2200,  2800),
    ("A Car",          1300,  1800,   1800,  2200),
    ("Goose",           700,  1000,   2400,  2800),
    ("A Ramp",          500,   900,   2100,  2600),
    ("CT Spawn",        200,   900,   2800,  3400),

    # ---- Mid ----
    ("Mid Doors",       -50,   500,    -50,   500),
    ("Mid",             -50,   500,    500,  1400),
    ("Catwalk",         200,   700,   1400,  2100),
    ("Lower Tunnels",  -800,  -200,   -800,    50),
    ("Upper Tunnels",  -800,     0,   -800, -1200),
    ("Xbox",            100,   500,    500,   900),

    # ---- B site ----
    ("B Site",         -800,    50,   1400,  2200),
    ("B Tunnels",     -1400,  -600,    400,  1200),
    ("B Doors",        -600,   100,   1200,  1600),
    ("B Window",       -500,    50,   2200,  2700),
    ("B Back Site",    -800,  -200,   2200,  2800),
    ("B Car",          -200,   200,   1800,  2200),

    # ---- T side ----
    ("T Spawn",        -400,   400,  -2400, -1600),
    ("T Mid",          -200,   400,  -1200,  -500),
    ("Outside Long",    800,  1700,  -1200,  -200),

    # ---- Fallback ----
    ("A Side",          500,  2400,   1200,  3500),
    ("B Side",        -1500,   200,   1200,  3000),
    ("Mid Area",       -200,   700,   -200,  1400),
]

_INFERNO_ZONES: list[tuple[str, float, float, float, float]] = [
    # Calibrated from actual demo position data (2026-03).
    # Format: (label, min_x, max_x, min_y, max_y) in Source engine units.
    # Zones are ordered specific-first; first match wins.

    # ---- A site ----
    ("A Site",         1800,  2400,    200,   800),
    ("Pit",            2100,  2700,   -600,   300),
    ("Balcony",        1600,  2000,    800,  1200),
    ("Library",        1200,  1700,    800,  1400),
    ("Arch",           1000,  1500,    200,   800),
    ("Graveyard",      2200,  2700,    600,  1100),
    ("Truck",          1600,  2250,   -100,   400),

    # ---- Mid ----
    ("Mid",             600,  1300,   -100,   800),
    ("Alt Mid",        -300,   700,    400,  1000),
    ("Underpass",       400,   900,   -600,    50),
    ("Top Mid",         400,   900,    800,  1400),

    # ---- B site ----
    ("B Site",          100,   700,  -1200,  -600),
    ("Banana",          100,   800,  -2200, -1200),
    ("Oranges",          50,   400,  -1600, -1200),
    ("Car",              50,   500,  -1200,  -800),
    ("Dark",            -200,   200,  -1100,  -700),
    ("CT",               50,   600,   -700,  -200),
    ("Construction",    -400,   200,  -1400,  -900),
    ("New Box",          50,   500,   -900,  -500),
    ("Coffins",         500,   900,  -1000,  -500),

    # ---- T side ----
    ("T Spawn",        -200,   400,  -3000, -2200),
    ("T Apartments",    800,  1800,   1200,  2100),
    ("Apartments",      100,  1800,   1400,  2800),
    ("Second Mid",      100,  1200,   1000,  2000),
    ("Boiler",         1000,  1400,    400,   800),

    # ---- Fallback ----
    ("A Side",         1000,  2800,   -600,  1600),
    ("B Side",         -600,  1000,  -2400,  -200),
    ("T Area",         -400,   800,  -3400,  -1800),
    ("T Approach",      200,   800,   2000,  3500),
]

_ANUBIS_ZONES: list[tuple[str, float, float, float, float]] = [
    ("A Site",         -800,  -200,   1400,  2000),
    ("A Main",         -400,   200,    600,  1400),
    ("A Bridge",       -800,  -200,    800,  1400),
    ("A Connector",    -600,     0,   2000,  2600),

    ("Mid",            -200,   600,     50,   600),
    ("Top Mid",         200,   800,   -600,    50),

    ("B Site",          600,  1400,   1400,  2000),
    ("B Main",          400,  1000,    600,  1400),
    ("B Pillar",        800,  1200,   1600,  2000),
    ("B Connector",     200,   800,   2000,  2600),

    ("CT Spawn",       -200,   600,   2400,  3000),
    ("T Spawn",        -200,   600,  -1200,  -400),

    ("A Side",        -1200,   200,    600,  2800),
    ("B Side",          200,  1600,    600,  2800),
]

_NUKE_ZONES: list[tuple[str, float, float, float, float]] = [
    ("A Site",         -300,   400,  -1200,  -500),
    ("Hut",             400,   800,  -1200,  -700),
    ("Heaven",         -800,  -200,  -1400,  -800),
    ("Hell",           -600,     0,   -800,  -400),
    ("Squeaky",         400,   800,   -700,  -300),
    ("Main",           -200,   400,   -500,   200),
    ("Lobby",          -600,   200,    200,   800),
    ("Ramp",            -50,   500,    800,  1600),
    ("Outside",         600,  1800,   -600,   600),
    ("Secret",          400,  1000,   1400,  2200),
    ("B Site",         -300,   400,  -1200,  -500),  # Below A
    ("T Spawn",        -400,   400,   1800,  2800),
    ("CT Spawn",       -400,   400,  -2400, -1600),
]

_ANCIENT_ZONES: list[tuple[str, float, float, float, float]] = [
    # Calibrated from de_ancient radar: pos_x=-2953, pos_y=2164, scale=5.
    # Game coords: gx = px*5 - 2953, gy = 2164 - py*5
    # Specific zones first; first match wins.

    # ---- A site area ----
    ("A Site",         -1200,  -550,  -1900, -1200),
    ("A Main",          -550,   200,  -1600,  -900),
    ("A Bridge",       -1300,  -550,  -1200,  -600),

    # ---- B site area ----
    ("B Site",          -600,   200,    600,  1300),
    ("B Pillar",        -200,   400,    400,   800),

    # ---- B approaches ----
    ("B Main",           200,   900,    400,  1100),
    ("B Connector",     -800,  -100,     50,   600),

    # ---- Mid ----
    ("Mid",             -600,   200,   -600,   100),
    ("Top Mid",          200,   900,   -600,   200),

    # ---- A connector / donut ----
    ("A Connector",    -1400,  -700,   -600,    50),

    # ---- Spawns ----
    ("CT Spawn",       -1800, -1100,   -400,   400),
    ("T Spawn",          800,  1600,   -600,   200),

    # ---- Fallback ----
    ("A Side",         -1800,   200,  -2000,  -600),
    ("B Side",          -900,   900,     50,  1400),
    ("Mid Area",        -800,   900,   -700,   200),
]

_OVERPASS_ZONES: list[tuple[str, float, float, float, float]] = [
    # Calibrated from de_overpass radar: pos_x=-4831, pos_y=1781, scale=5.2.
    # Game coords: gx = px*5.2 - 4831, gy = 1781 - py*5.2
    # Specific zones first; first match wins.

    # ---- A site area ----
    ("A Site",         -2200, -1400,   -750,    50),
    ("A Long",         -1400,  -400,   -750,   200),

    # ---- Upper / Mid / Toilets ----
    ("Toilets",        -2500, -1800,    100,   800),
    ("Party",          -3100, -2500,   -200,   500),
    ("Balloons",       -3500, -2800,    200,   800),

    # ---- Connector ----
    ("Connector",      -2700, -2000,   -400,   200),

    # ---- Heaven / Sniper (overlook B) ----
    ("Heaven",         -3200, -2500,  -1600,  -900),
    ("Sniper",         -2500, -1800,  -1400,  -800),

    # ---- B site area ----
    ("B Site",         -3500, -2600,  -2200, -1500),
    ("Pillar",         -3200, -2700,  -1700, -1300),
    ("Barrels",        -2600, -2100,  -2100, -1600),
    ("Pit",            -3600, -3000,  -1600, -1000),

    # ---- B approaches ----
    ("B Short",        -2600, -1800,  -1100,  -400),
    ("Water",          -3300, -2600,   -900,  -200),
    ("Sandbags",       -2200, -1700,  -1600, -1000),
    ("Monster",        -3800, -3200,  -2400, -1700),

    # ---- Mid ----
    ("Mid",            -2200, -1400,    200,   900),

    # ---- Spawns ----
    ("CT Spawn",       -2000, -1200,  -1800, -1000),
    ("T Spawn",        -1300,  -400,    800,  1600),

    # ---- Fallback ----
    ("A Side",         -2500,  -400,   -800,   300),
    ("B Side",         -3800, -2000,  -2500,  -800),
    ("Mid Area",       -2800, -1200,    100,  1000),
]

# Map name → zone list
_MAP_ZONES: dict[str, list[tuple[str, float, float, float, float]]] = {
    "de_mirage":   _MIRAGE_ZONES,
    "de_dust2":    _DUST2_ZONES,
    "de_inferno":  _INFERNO_ZONES,
    "de_anubis":   _ANUBIS_ZONES,
    "de_nuke":     _NUKE_ZONES,
    "de_ancient":  _ANCIENT_ZONES,
    "de_overpass": _OVERPASS_ZONES,
}

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
    zones = _MAP_ZONES.get(map_name)
    if not zones:
        return "unknown"
    for label, min_x, max_x, min_y, max_y in zones:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return label
    return "unknown"


def is_map_supported(map_name: str) -> bool:
    """Check if we have callout data for this map."""
    return map_name in _MAP_ZONES


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

    zones = _MAP_ZONES.get(map_name)
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
    zones = _MAP_ZONES.get(map_name)
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
